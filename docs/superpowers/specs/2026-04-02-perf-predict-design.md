# Performance Estimation System Design (DAOP)

## 1. Problem & Goal

**Problem**: 用户在下载或加载 LLM 模型前，无法知道该模型在本机上的推理性能。如果性能太差（如 decode < 10 tok/s），下载 GB 级权重是浪费时间和带宽。

**Goal**: 提供一个轻量级的性能预估系统，基于模型元数据 + 本机硬件 profile，估算 prefill 和 decode 的 tokens/sec。

**Project code name**: DAOP (Dynamic AI Offloading Protocol)。CLI 命令以 `daop-` 为前缀。

**Non-goals (MVP)**:
- 从 registry 只拉 GGUF header（MVP 下载完整 GGUF，同 `ollama run`）
- 实时 GPU/CPU utilization 感知
- 多用户并发推理预测
- llamarunner (C++) 架构支持
- 自动推荐最优量化方案

## 2. Architecture Overview

系统分三个阶段，对应三个核心模块：

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. Benchmark    │────>│  2. Profile      │────>│  3. Estimate     │
│  (跑一次)        │     │  (存两个文件)     │     │  (每个模型秒出)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
   ollama daop-bench       raw + profile           ollama daop-estimate
   ~1-5 min               ~/.ollama/bench/         model → tok/s
```

- **Benchmark 模块**: 枚举本机所有 backend，跑硬件特征化 + 算子校准。输出 raw data 文件。
- **Profile 模块**: 处理 raw data -> 生成 profile 文件。支持 viewer 查看。
- **Estimate 模块**: 读模型元数据 -> 构图 -> 遍历融合后的图 -> 用 profile 估算性能。输出结构化结果 + summary。

## 3. Theoretical Foundation: Roofline Model

### 3.1 核心公式

对于任意计算操作，性能受两个硬件限制之一约束:

```
T_compute = FLOPs / peak_FLOPS(backend, compute_dtype)
T_memory  = bytes_moved / peak_bandwidth(backend)
T_predicted = max(T_compute, T_memory)
```

**Arithmetic Intensity** (AI) = FLOPs / bytes_moved (单位: FLOP/byte)

- AI < balance_point → memory-bound, 实际延迟 ≈ T_memory
- AI > balance_point → compute-bound, 实际延迟 ≈ T_compute

**Balance Point** = peak_FLOPS / peak_bandwidth (单位: FLOP/byte)

### 3.2 校准因子 η

真实 kernel 达不到理论峰值。η 表示实际吞吐与理论峰值的比例:

```
T_actual = T_predicted / η    (η ∈ (0, 1])
```

η 从 microbenchmark 数据反推:

```
η = T_predicted / T_measured
  = max(FLOPs/peak_FLOPS, bytes/peak_BW) / T_measured
```

对同一 (op, backend, dtype) 的多个 benchmark 点取中位数得到稳定的 η。

**注意**: η 可能在不同 intensity regime 下不同（例如 MUL_MAT 在 memory-bound 区域 η=0.78，compute-bound 区域 η=0.62）。如果发现单个 η 无法覆盖全 intensity 范围（自适应追加后 variance 仍高），可以扩展为 piecewise η:
- η_memory: intensity < balance_point 时使用
- η_compute: intensity > balance_point 时使用

MVP 先用单个 η，如果估算准确度不够再切换到 piecewise。

### 3.3 Per-op FLOPs 和 bytes 计算

| Op | FLOPs | Bytes moved | 说明 |
|----|-------|-------------|------|
| MUL_MAT [M,K]×[K,N] | 2×M×K×N | element_size(A)×M×K + element_size(B)×K×N + 4×M×N | 量化 weight 读取用实际 element_size |
| MUL_MAT_ID | 同 MUL_MAT × num_active_experts | 同上 × num_active_experts | MoE 场景 |
| FLASH_ATTN_EXT Q[B,H,S_q,D] × K,V[B,H,S_kv,D] | 2×B×H×S_q×S_kv×D | B×H×(S_q×D + 2×S_kv×D + S_q×D)×elem_size | FLOPs=O(S²D), IO=O(S×D) — FA 不 materialize S×S matrix |
| RMS_NORM [N,M] | 3×N×M (sqr+mean+rsqrt+mul) | 2×N×M×elem_size (read+write) + weight | reduce 操作 |
| LAYER_NORM [N,M] | 5×N×M | 2×N×M×elem_size + weight + bias | 类似 RMS_NORM |
| SOFTMAX [N,M] | 4×N×M (max+sub+exp+sum+div) | 2×N×M×elem_size | reduce |
| ADD/MUL/DIV/NEG [N] | N | 3×N×elem_size (2 read + 1 write) | elementwise |
| SILU/GELU/SIGMOID/TANH [N] | ~5×N | 2×N×elem_size | activation, 几个 FP ops |
| GLU (SWIGLU/GEGLU) [N] | ~6×N | 3×N×elem_size (gate+up read, out write) | fused gate×activation, 常见于 FFN |
| GET_ROWS [N_idx, D] | 0 | N_idx×D×elem_size (random read) + N_idx×D×4 (write f32) | 随机访问 pattern |
| ROPE [B,H,S,D] | 6×B×H×S×(D/2) | 2×B×H×S×D×elem_size | sin/cos + rotate |
| CONT/CPY/CONCAT | 0 | 2×total_bytes (read+write) | 纯内存搬运 |
| CONV_2D [Cout,Cin,Kh,Kw,Oh,Ow,B] | 2×Cout×Cin×Kh×Kw×Oh×Ow×B | kernel + input + output bytes | 标准卷积 |
| SSMConv [D,K,N] | D×K×N | (D×K + D×N + D×N)×elem_size | 1D 因果卷积 |
| SolveTri [N,N] | N×N×N/3 (近似) | 2×N×N×elem_size | 三角求解 |
| CumSum [N] | N | 2×N×elem_size | 累积和 |
| SumRows [N,M] | N×M | N×M×elem_size + M×elem_size | reduce |
| TopK [N,K] | N×log(K) (近似) | N×elem_size + K×(elem_size+4) | partial sort |
| EXP/SQRT/SQR/SIN/COS [N] | N | 2×N×elem_size | unary math |
| L2Norm [N,M] | 3×N×M | 2×N×M×elem_size | sqr+sum+rsqrt+mul |
| Softplus [N] | ~3×N | 2×N×elem_size | log(1+exp(x)) |
| Tri/Diag [N,N] | 0 | N×N×elem_size (write) | mask 生成 |
| VIEW/RESHAPE/PERMUTE | 0 | 0 | 零开销, 跳过 |

注意: Flash Attention 的 FLOPs/bytes 是近似值。实际 tiling 策略决定真实 IO，η 校准会补偿误差。

## 4. Benchmark Module

### 4.1 硬件特征化

对每个 (backend, dtype) 组合测两个值:

| 测量项 | 方法 |
|--------|------|
| Peak FLOPS | 大方阵 MUL_MAT (如 4096×4096)，取最高吞吐。per dtype: fp16, fp32, bf16, int8 等 |
| Peak Bandwidth | 大 tensor CONT/CPY（纯搬数据），取最高吞吐。per backend（与 dtype 无关） |

### 4.1.1 Interconnect Bandwidth 测量

在硬件特征化阶段，对每对 backend 测量数据传输带宽（详见 Section 14）。
此步骤在 Peak FLOPS/BW 测量之后、算子发现之前执行。

### 4.2 算子校准: 两层策略

**第一层: 预定义常见算子集（初始 `daop-bench`，无需本地模型）**

首次运行 `daop-bench` 时，用户可能没有任何本地模型。此时校准以下预定义算子集，
用预设 shape 覆盖绝大多数 transformer 架构的核心算子:

| 算子 | 校准的 dtype 组合 |
|------|------------------|
| MUL_MAT | fp16, fp32, bf16, q4_0, q4_K, q5_K, q6_K, q8_0 |
| FLASH_ATTN_EXT | fp16, bf16 |
| RMS_NORM | fp32 |
| SOFTMAX | fp32 |
| SILU, GELU | fp32 |
| GLU (SWIGLU) | fp32 |
| ADD, MUL | fp32 |
| ROPE | fp32, fp16 |
| GET_ROWS | fp16, q4_0, q8_0 |
| CONT/CPY | fp32, fp16 |
| CONV_2D | fp32, fp16 |

**第二层: 图驱动发现（`daop-bench --update`，需要本地模型 GGUF）**

当用户有本地模型后，通过构图发现预定义集之外的算子:

1. 对指定模型（或所有本地模型）各构一次 prefill 图 + decode 图
2. 遍历图，收集所有唯一 (op, backend, compute_dtype, weight_dtype) 四元组
3. 与 profile 中已有的 operator entries 对比，找出缺失项
4. 对缺失项跑校准 benchmark

**目标覆盖的架构** (确保对主流模型开箱即用):

| 架构 | 注册名 | 覆盖的特殊算子 |
|------|--------|---------------|
| Llama | llama | 基础 transformer |
| Qwen3 | qwen3, qwen3moe | MoE routing (TopK, MulmatID, SumRows) |
| Qwen3.5 | qwen35, qwen35moe | DeltaNet (SSMConv, SolveTri, CumSum, L2Norm, Softplus, Diag, Tri) |
| Gemma3 | gemma3 | GELU, post-norm, Scale |
| DeepSeek2 | deepseek2 | MLA attention, ChunkSections, Sigmoid (covers V3/R1) |
| MLLama | mllama | Conv2D, LayerNorm, Tanh, cross-attention |
| Qwen2.5-VL | qwen25vl | Conv2D (temporal), Argsort, vision RoPE |
| Qwen3-VL | qwen3vl | multimodal vision pipeline |
| DeepSeek OCR | deepseekocr | Conv2D (SAM), vision pipeline |

### 4.3 单算子 benchmark 构造

通过 Go 层的 GGML backend 接口构造单算子图:

```go
// 伪代码
func benchmarkOp(backend ml.Backend, op OpType, shapes [][]int64, dtype ml.DType) BenchResult {
    ctx := backend.NewContext()

    // 创建输入 tensor (random data)
    a := ctx.Zeros(dtype, shapes[0]...)
    b := ctx.Zeros(dtype, shapes[1]...)

    // 构建单算子图
    output := a.Mulmat(ctx, b)  // 或其他 op
    ctx.Forward(output)

    // Warmup
    for i := 0; i < 3; i++ {
        ctx.Compute(output)
    }

    // 计时
    start := time.Now()
    for i := 0; i < reps; i++ {
        ctx.Compute(output)
    }
    // sync (确保 GPU 完成)
    elapsed := time.Since(start)

    return BenchResult{LatencyUs: elapsed.Microseconds() / reps}
}
```

关键: 需要在 Compute 后做 backend sync（如 CUDA 的 stream synchronize），否则计时不准。
当前 `ComputeWithNotify` 已经有同步机制可以复用。

### 4.4 Benchmark 点选取策略

对每个 (op, backend, dtype)，选择 5 个初始 benchmark 点，使其 arithmetic intensity 跨越 Roofline 的不同区域。

先从硬件特征化得到 balance point BP = peak_FLOPS / peak_BW。然后选 5 个点的 intensity 分别在:

```
目标 intensity: ~BP×0.1, ~BP×0.5, ~BP×1.0, ~BP×2.0, ~BP×10.0
```

对于 MUL_MAT [M,K]×[K,N]:
- intensity ≈ 2×M×N×K / (elem_A×M×K + elem_B×K×N + 4×M×N)
- 固定 K=4096, 调整 N 来控制 intensity:
  - N=1 → 极低 intensity (decode-like)
  - N=32 → 低 intensity
  - N=256 → 中等 intensity (接近 balance point)
  - N=1024 → 高 intensity
  - N=4096 → 极高 intensity (方阵)

对于其他 memory-bound op (如 ADD, RMS_NORM)，intensity 固定（不随 size 变），5 个点用不同的 tensor size:
- 小 (1024), 中小 (64K), 中 (1M), 大 (16M), 极大 (64M) 元素

### 4.5 自适应追加

跑完初始 5 点后:

1. 算 η 的标准差 σ
2. 如果 σ / mean(η) > 0.10 (变异系数 > 10%)，追加点
3. 追加策略: 在 η 波动最大的 intensity 区间插入新点
4. 最多追加到 10 个点
5. 最终 η = 中位数 (对 outlier 稳健)

### 4.6 GEMM 特殊处理

MUL_MAT / MUL_MAT_ID 按 (backend, weight_quant_type) 细分校准:

- 常见量化: f16, f32, bf16, q4_0, q4_1, q5_0, q5_1, q8_0, q4_K, q5_K, q6_K, q3_K, iq4_nl
- 每种量化的 dequant 开销不同，η 可能差异显著
- 第一层 (Section 4.2) 校准常见量化类型 (q4_0, q4_K, q5_K, q6_K, q8_0 等)
- 第二层 (graph-driven) 发现模型实际使用的量化类型，补测缺失项

## 5. Profile Storage

### 5.1 文件位置

```
~/.ollama/bench/
  raw-20260401-103000.json       # 原始测量数据
  raw-20260402-143500.json       # 第二次跑的 raw
  profile.json                   # 处理后的 profile (当前)
```

### 5.2 Raw Data Schema

```json
{
  "version": 1,
  "timestamp": "2026-04-01T10:30:00Z",
  "hardware": {
    "backends": [
      {
        "name": "cuda",
        "device": "NVIDIA GeForce RTX 4090",
        "driver": "560.35",
        "compute_capability": "8.9",
        "vram_bytes": 25769803776
      },
      {
        "name": "cpu",
        "device": "13th Gen Intel Core i9-14900K",
        "cores": 24,
        "ram_bytes": 68719476736
      }
    ]
  },
  "hardware_benchmarks": [
    {"backend": "cuda", "dtype": "fp16", "test": "peak_flops", "value": 82.6e12, "unit": "FLOPS"},
    {"backend": "cuda", "dtype": "fp32", "test": "peak_flops", "value": 41.3e12, "unit": "FLOPS"},
    {"backend": "cuda", "dtype": "bf16", "test": "peak_flops", "value": 82.6e12, "unit": "FLOPS"},
    {"backend": "cuda", "dtype": "int8", "test": "peak_flops", "value": 165.2e12, "unit": "FLOPS"},
    {"backend": "cuda", "test": "peak_bandwidth", "value": 1008e9, "unit": "bytes/sec"},
    {"backend": "cpu", "dtype": "fp32", "test": "peak_flops", "value": 1.2e12, "unit": "FLOPS"},
    {"backend": "cpu", "test": "peak_bandwidth", "value": 52e9, "unit": "bytes/sec"}
  ],
  "operator_benchmarks": [
    {
      "op": "MUL_MAT",
      "backend": "cuda",
      "compute_dtype": "fp16",
      "weight_dtype": "q4_0",
      "points": [
        {
          "input_shapes": [[4096, 4096], [4096, 1]],
          "output_shape": [4096, 1],
          "flops": 33554432,
          "bytes_moved": 8396800,
          "intensity": 4.0,
          "latency_us": 12.5,
          "reps": 200,
          "stddev_us": 0.8
        }
      ]
    }
  ],
  "interconnect_benchmarks": [
    {"from": "cuda:0", "to": "cpu", "bandwidth": 25.6e9, "latency_us": 2.1, "test_size_bytes": 67108864}
  ]
}
```

### 5.3 Profile Schema

```json
{
  "version": 1,
  "generated_from": ["raw-20260401-103000.json"],
  "generated_at": "2026-04-01T10:35:00Z",
  "hardware": {
    "backends": [
      {
        "name": "cuda",
        "device": "NVIDIA GeForce RTX 4090",
        "peak_flops": {
          "fp16": 82.6e12,
          "fp32": 41.3e12,
          "bf16": 82.6e12,
          "int8": 165.2e12
        },
        "peak_bandwidth": 1008e9,
        "balance_points": {
          "fp16": 81.9,
          "fp32": 41.0,
          "bf16": 81.9,
          "int8": 163.9
        }
      }
    ]
  },
  "operators": [
    {
      "op": "MUL_MAT",
      "backend": "cuda",
      "compute_dtype": "fp16",
      "weight_dtype": "q4_0",
      "eta": 0.62,
      "eta_variance": 0.003,
      "num_points": 7
    }
  ],
  "interconnects": [
    {"from": "cuda:0", "to": "cpu", "bandwidth": 25.6e9},
    {"from": "cpu", "to": "cuda:0", "bandwidth": 25.6e9}
  ]
}
```

### 5.4 版本管理

- 每次 `ollama daop-bench` 生成新的 `raw-YYYYMMDD-HHMMSS.json`
- 同时更新 `profile.json`
- 旧 raw 文件永不自动删除，用于对比和回溯
- `ollama daop-bench --update` 生成增量 raw 文件，合并更新 profile.json
- Profile 里 `generated_from` 字段追溯来源（多次 update 时为数组）

## 6. Estimate Module

### 6.1 输入模式

支持两种输入方式:

**Model ID 模式** (推荐):
```bash
ollama daop-estimate qwen3:8b-q4_0 --input-length 1024 --output-length 256
```

模型解析复用 Ollama 现有逻辑:
1. 检查本地是否已有该模型的 GGUF 文件 (`~/.ollama/models/`)
2. 如果本地有 → 直接读取 GGUF header（几十 KB，不加载权重数据）
3. 如果本地没有 → 下载完整 GGUF（同 `ollama run` 的下载流程），然后读 header

从 GGUF header 获取:
- `general.architecture` → 架构名
- hyperparameters: `block_count`, `embedding_length`, `attention.head_count` 等
- tensor 列表: name + shape + dtype → 创建 tensor 描述符供 `model.Forward()` 使用

**GGUF 文件模式** (用于自定义/未发布模型):
```bash
ollama daop-estimate ./custom-model.gguf --input-length 1024 --output-length 256
```

直接读取指定 GGUF 文件的 header。

**未来优化** [HIGH TODO]: Model ID 模式在本地没有模型时，改为只下载 GGUF header（几十 KB）而非完整文件（GB 级），实现真正的"下载前预估"。

### 6.2 估算流程

```
estimate(model_path, input_length, output_length, profile):

  1. 加载 profile.json
  2. 检查架构是否受支持 (Ollama Go 注册的架构)

  --- Prefill 估算 ---
  3. max_batch = max_batch_size 参数 (默认 512，可通过 --batch-size 覆盖)
     n_batches = ceil(input_length / max_batch)
  4. prefill_total = 0
     for i in 0..n_batches:
       batch_len = min(max_batch, input_length - i * max_batch)
       kv_len = min((i + 1) * max_batch, input_length)
       graph = build_graph(model_meta, batch_size=batch_len, kv_cache_len=kv_len)
       fused_graph = split_and_optimize(graph)  // backend fusion
       prefill_total += estimate_graph(fused_graph, profile)

  --- Decode 估算 ---
  5. 采样 3+ 个 KV cache 位置 (decode 时 batch=1，MUL_MAT intensity 基本恒定，
     但 Flash Attention 的 cost 与 S_kv 成正比，因此需要在不同 KV 长度下采样取平均):
       positions = [input_length, input_length + output_length/2, input_length + output_length]
     for pos in positions:
       graph = build_graph(model_meta, batch_size=1, kv_cache_len=pos)
       fused_graph = split_and_optimize(graph)
       decode_latencies.append(estimate_graph(fused_graph, profile))
     avg_decode_per_token = mean(decode_latencies)

  6. 汇总结果
```

### 6.3 estimate_graph 逻辑

```
estimate_graph(fused_graph, profile):
  total_latency = 0
  op_stats = {}

  for node in fused_graph.nodes:
    if node.op in [VIEW, RESHAPE, PERMUTE]:
      continue  // 零开销

    flops = compute_flops(node.op, node.shapes, node.dtypes)
    bytes = compute_bytes(node.op, node.shapes, node.dtypes)
    intensity = flops / bytes if bytes > 0 else infinity

    backend = node.assigned_backend
    compute_dtype = node.compute_dtype
    weight_dtype = node.weight_dtype

    peak_flops = profile.peak_flops[backend][compute_dtype]
    peak_bw = profile.peak_bandwidth[backend]
    bp = peak_flops / peak_bw

    t_compute = flops / peak_flops
    t_memory = bytes / peak_bw
    t_theoretical = max(t_compute, t_memory)
    bound = "compute" if intensity > bp else "memory"

    // 查 η — 两种缺失情况:
    key = (node.op, backend, compute_dtype, weight_dtype)
    if key in profile.operators:
      eta = profile.operators[key].eta
    elif can_compute_flops(node.op):
      // 已知算子但未校准: 用 η=1.0 (乐观估计，Roofline 理论值)
      eta = 1.0
      warnings.add("uncalibrated op: " + key)
    else:
      // 完全未知算子: 按纯 memory-bound 保守估算
      t_theoretical = bytes / peak_bw  // 忽略 compute，只算 bandwidth
      eta = 1.0
      warnings.add("unknown op: " + key)
    t_actual = t_theoretical / eta

    total_latency += t_actual
    op_stats[key].total += t_actual
    op_stats[key].count += 1
    op_stats[key].bound_counts[bound] += 1

  return total_latency, op_stats
```

### 6.4 图构建 (无权重)

复用 Ollama 现有机制。`model.New()` 解析 GGUF header 创建 tensor 描述符 (data=NULL)，
模型通过 `backend.Get("blk.0.attn_q.weight")` 按名字查找 tensor，Forward() 用这些 tensor 构图。

关键路径:

```go
// 1. 创建模型描述符 (解析 GGUF header, no_alloc=true, 张量 data=NULL)
model, err := model.New(ggufPath, params)

// 2. 创建 KV cache (空)
cache := kvcache.NewInputCache(model, ...)

// 3. 构建计算图
ctx := model.Backend().NewContext()
fakeBatch := createFakeBatch(batchSize, kvCacheLen)
output, err := model.Forward(ctx, fakeBatch)

// 4. 触发 split + optimize (不执行计算)
ctx.Forward(output).Reserve()

// 5. 遍历融合后的图节点
//    需要在 ml/backend/ggml 包中新增导出 API，因为 ctx.graph 是内部字段。
//    接口契约: 返回每个节点的 (op, backend, shape, compute_dtype, weight_dtype)
nodes := ctx.GraphNodes()  // 新增方法，内部通过 CGo 调 ggml_graph_n_nodes/node
for _, node := range nodes {
    // node.Op: "MUL_MAT", "RMS_NORM", ...
    // node.Backend: "cuda", "cpu", ...
    // node.Shape: [4]int64
    // node.ComputeDtype, node.WeightDtype: "f16", "q4_0", ...
    // ... 估算该节点
}
```

**重要实现细节**: Reserve() 最后调 `ggml_backend_sched_reset()` 会清除 `node_backend_ids`
(每个节点的 backend 分配)。因此需要以下策略之一:

- **方案 A (推荐)**: 在 Reserve 内部、reset 之前，通过新增的 Go/CGo API 捕获 split 结果
  并存储到 Context 上。数据: 每个图节点的 `backend_id`（int，对应 scheduler 的 backend 数组索引）。
  调用者通过 `ctx.GraphNodes()` 获取包含 backend 信息的节点列表。
  需要修改 `ml/backend/ggml/ggml.go` 的 Reserve 实现：在 `ggml_backend_sched_reserve()` 返回后、
  `reset()` 之前，遍历 `sched->node_backend_ids` 保存到 Go slice。
- **方案 B**: 不调 Reserve，而是手动调 `split_graph` + `graph_optimize`，在 reset 前
  遍历图。需要新增 CGo binding。
- **方案 C**: 用简化的启发式确定 backend 分配 — 根据 GPULayers 参数，前 N 层在 GPU，
  其余在 CPU。不如 A/B 精确，但最简单。

推荐 **方案 A**: 修改量最小，且复用了 GGML scheduler 的真实分配逻辑。

### 6.5 Split 配置支持

对于 CUDA + iGPU 或 GPU + CPU offload 场景:

- Reserve 的 `split_graph()` 按 layer -> backend 分配
- 每个图节点有 `backend_id`，用对应 backend 的 profile 数据
- 跨 backend 数据传输: 当相邻节点在不同 backend 上时，插入传输开销估算:
  ```
  transfer_time = tensor_bytes / profile.interconnects[from_backend][to_backend].bandwidth
  ```
  Interconnect bandwidth 在硬件特征化阶段实测 (Section 4.1.1 / Section 14)

## 7. Output Format

### 7.1 结构化输出 (Go struct)

```go
type EstimateResult struct {
    Model        string          `json:"model"`
    Backends     []BackendInfo   `json:"backends"`
    InputLength  int             `json:"input_length"`
    OutputLength int             `json:"output_length"`
    MaxBatchSize int             `json:"max_batch_size"`
    Prefill      PhaseEstimation `json:"prefill"`
    Decode       PhaseEstimation `json:"decode"`
    Warnings     []string        `json:"warnings,omitempty"`
    Summary      string          `json:"summary"`
}

type BackendInfo struct {
    Name           string  `json:"name"`
    Device         string  `json:"device"`
    PeakFLOPS      float64 `json:"peak_flops"`
    PeakBandwidth  float64 `json:"peak_bandwidth"`
    BalancePoint   float64 `json:"balance_point"`
}

type PhaseEstimation struct {
    TotalLatencyMs float64       `json:"total_latency_ms"`
    TokensPerSec   float64       `json:"tokens_per_sec"`
    TTFTMs         float64       `json:"ttft_ms,omitempty"`      // Time To First Token (prefill only)
    NumBatches     int           `json:"num_batches,omitempty"`
    Bottleneck     string        `json:"bottleneck"`
    TopOps         []OpBreakdown `json:"top_ops"`
}

type OpBreakdown struct {
    Op            string  `json:"op"`
    Backend       string  `json:"backend"`
    ComputeDtype  string  `json:"compute_dtype"`
    WeightDtype   string  `json:"weight_dtype,omitempty"`
    Count         int     `json:"count"`
    TotalMs       float64 `json:"total_ms"`
    Percentage    float64 `json:"percentage"`
    BoundBreakdown string `json:"bound_breakdown"`
}
```

### 7.2 人类可读输出

```
Model: qwen3:8b-q4_0 | Backend: cuda (RTX 4090) + cpu
Input: 1024 tokens | Output: 256 tokens | Max batch: 512

Hardware Profile
────────────────
Backend    Device        Dtype   Peak FLOPS     Peak BW       Balance Point
cuda       RTX 4090      fp16    82.6 TFLOPS    1008 GB/s     81.9 FLOP/byte
cuda       RTX 4090      fp32    41.3 TFLOPS    1008 GB/s     41.0 FLOP/byte
cpu        i9-14900K     fp32    1.2 TFLOPS     52 GB/s       23.1 FLOP/byte

Prefill (1024 tokens, 2 batches of 512)
───────────────────────────────────────
  Estimated: 320ms total, 3200 tok/s, TTFT ≈ 320ms
  Bottleneck: compute-bound
  Top ops (grouped by op+dtype):
    Op              Dtype  Count  Total ms  Bound breakdown
    MUL_MAT         q4_0   96     250ms     72x mem (180ms) + 24x compute (70ms)
    FLASH_ATTN_EXT  f16    32      45ms     32x memory
    RMS_NORM        f32    64      10ms     64x memory
    ...

Decode (avg over 256 positions)
───────────────────────────────
  Estimated: 10.5ms/tok, 95 tok/s
  Bottleneck: memory-bound
  Top ops:
    Op              Dtype  Count  Total ms  Bound breakdown
    MUL_MAT         q4_0   96     8.7ms    96x memory
    FLASH_ATTN_EXT  f16    32     1.2ms    32x memory
    ...

Warnings:
  ⚠ Missing ops: SolveTri(cuda,f32), CumSum(cuda,f32)
    Run: ollama daop-bench --update --model qwen35:8b

Summary: qwen3:8b-q4_0 | input=1024 | output=256
  Prefill: ~3200 tok/s (batch=512, 2 batches, TTFT ≈ 320ms)
  Decode:  ~95 tok/s (memory-bound by VRAM bandwidth)
```

`--json` flag 输出完整 EstimateResult JSON。
`--detail` flag 展开每个 op 实例的 shape, intensity, latency。

## 8. CLI Interface

所有命令以 `daop-` 为前缀，归属 DAOP 项目命名空间。

```bash
# === Benchmark ===
ollama daop-bench                                  # 全量 benchmark (~1-5 min)
ollama daop-bench --backends cuda,cpu              # 只跑指定 backend
ollama daop-bench --update                         # 扫描所有本地模型，补测缺失 op
ollama daop-bench --update --model qwen35:8b       # 只补测指定模型的缺失 op

# === Profile Viewer ===
ollama daop-bench view                             # profile 概览
ollama daop-bench view --detail                    # 所有 op 的 η 详情
ollama daop-bench view --compare raw-A.json raw-B.json  # 对比两次 raw

# === Estimate ===
ollama daop-estimate qwen3:8b-q4_0                              # Model ID, 默认 input=512, output=128
ollama daop-estimate qwen3:8b-q4_0 --input-length 2048 --output-length 512
ollama daop-estimate ./custom.gguf --input-length 1024           # GGUF 文件
ollama daop-estimate qwen3:8b-q4_0 --json                        # 结构化输出
ollama daop-estimate qwen3:8b-q4_0 --detail                      # 展开每个 op 实例
```

## 9. Edge Cases

| 场景 | 处理 |
|------|------|
| 没跑过 bench 就 estimate | 报错: "请先运行 `ollama daop-bench`" |
| Model ID 本地无 GGUF | 自动下载完整 GGUF（同 `ollama run`），未来优化为只下载 header |
| 模型架构 Ollama Go 不支持 | 报错: "架构 X 暂不支持估算 (仅 llamarunner 支持)" |
| Profile 缺少某 op | warning + 该 op 用 η=1.0 按 peak bandwidth 保守估算 |
| Profile 缺少某 quant type | warning + 用最接近的量化类型近似 |
| 多 GPU split | 按 Reserve 的 split 结果，每个节点用所在 GPU 的 profile |
| GPU + CPU offload | 部分 layer 在 CPU 上，用 CPU profile + 加入传输开销 |
| η variance 高 (>10% CV) | daop-estimate 输出标记该 op 估算置信度低 |
| input_length > max_batch | 自动分多 batch 计算，每 batch 的 KV cache 长度不同 |
| output_length 未指定 | 默认 128 |
| input_length 未指定 | 默认 512 |
| 多模态模型 (mllama, qwen25vl 等) | MVP 只估算 text 部分。Vision encoder 估算标记为 future work |

## 10. Package Structure

```
perf/                          # 新 package, 性能预估核心逻辑
  bench.go                     # Benchmark 模块: 硬件特征化 + 算子校准
  profile.go                   # Profile 读写 + 处理
  estimate.go                  # Estimate 模块: 构图 + 估算
  roofline.go                  # Roofline 计算: FLOPs, bytes, intensity
  ops.go                       # Per-op 的 FLOPs/bytes 计算规则
  types.go                     # 共享类型定义 (EstimateResult, Profile, etc.)
  viewer.go                    # Profile viewer (TUI/print)

cmd/
  cmd.go                       # 注册 ollama daop-bench / ollama daop-estimate 子命令
```

## 11. Key Code Integration Points

| 需要交互的现有代码 | 位置 | 用途 |
|-------------------|------|------|
| Backend 初始化 + device 枚举 | `ml/backend/ggml/ggml.go:54-69` | 发现可用 backend |
| Tensor 接口 (构造单 op 图) | `ml/backend.go:130-241` | benchmark 单算子 |
| model.New() + Forward() | `model/model.go:161`, 各架构 `model.go` | 构建完整计算图 |
| Reserve (split + optimize) | `runner/ollamarunner/runner.go:1069` | 图优化 + 内存规划 |
| GGML graph 遍历 API | `ggml/include/ggml.h` 的 `ggml_graph_n_nodes/node` | 遍历融合后的图 |
| ComputeWithNotify | `ml/backend/ggml/ggml.go:814` | benchmark 计时 |
| DeviceInfo | `ml/device.go:276` | 硬件信息 |
| GGUF 解析 | `fs/ggml/gguf.go` | 读 GGUF header |

## 12. Assumptions & Risks

**Assumptions**:
- GGML 算子执行顺序: 节点按拓扑排序逐个执行，无算子间并行
- η 在同一硬件上相对稳定 (同 op, 同 backend, 同 dtype)
- Roofline 模型对 LLM 推理的主要算子 (特别是 GEMM) 是一个好的近似
- graph_optimize (fusion) 在 Reserve 和实际推理时的行为一致

**Risks**:
- Flash Attention 的 FLOPs/bytes 近似可能不准，高度依赖 η 校准补偿
- 极小 tensor 操作的 kernel launch overhead 可能主导延迟 (Roofline 不建模)
- 未来 GGML 添加算子间并行会使求和模型失效
- 量化 dequant 的开销在 Roofline 框架内难以精确建模，依赖 η
- Model ID 模式在本地无模型时需下载完整 GGUF（GB 级），真正的"下载前预估"待 GGUF header-only 下载实现

## 13. `ollama daop-bench --update` 流程

当用户运行 `ollama daop-bench --update [--model <id>]` 时:

```
1. 加载现有 profile.json
2. 确定要扫描的模型:
   - 指定了 --model: 只构该模型的图 (需要本地 GGUF)
   - 未指定: 扫描所有本地可用模型 (~/.ollama/models/)
3. 对每个模型构 prefill 图 + decode 图
4. 遍历图收集所有 (op, backend, compute_dtype, weight_dtype) 四元组
5. 与 profile.json 中已有的 operator entries 对比
6. 找出缺失的四元组 → 这就是需要补测的
7. 对缺失的四元组跑校准 benchmark (同 Section 4.3-4.5 流程)
8. 生成新的 raw 文件 (增量，只含本次补测的数据)
9. 合并更新 profile.json (保留已有的 + 追加新的)
```

## 14. PCIe / Interconnect Bandwidth 测量

Split 场景下需要估算跨设备传输开销。在硬件特征化阶段额外测量:

```
对每对 (backend_A, backend_B):
  1. 在 backend_A 分配 tensor
  2. 拷贝到 backend_B
  3. 测量传输 bandwidth (GB/s)

常见场景:
  - CPU ↔ CUDA:  PCIe bandwidth (~12-32 GB/s depending on gen/lanes)
  - CUDA ↔ CUDA: NVLink or PCIe
  - CPU ↔ Vulkan: PCIe
```

存入 profile.json:
```json
"interconnects": [
  {"from": "cuda:0", "to": "cpu", "bandwidth": 25.6e9},
  {"from": "cpu", "to": "cuda:0", "bandwidth": 25.6e9}
]
```
