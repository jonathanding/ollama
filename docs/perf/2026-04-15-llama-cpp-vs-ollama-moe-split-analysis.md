# llama.cpp `--cpu-moe` vs Ollama MoE Split — 深度对比分析

**日期：** 2026-04-15  
**硬件：** Intel Arrow Lake 265K · Windows 11 · 128 GB DDR5 6400 MT/s · RTX 3090 (24 GB VRAM, PCIe 4.0 x16)  
**模型：** Qwen3-Coder-Next 80B Q4_K_M (`~52 GB GGUF`)  
**分支：** `moe-split-cpu-phase1`  
**对比目标：** llama.cpp `master` (commit tree at 2026-04-15)

---

## 1. 背景与动机

社区多位用户报告，llama.cpp 的 `--n-cpu-moe` / `--cpu-moe` 功能在 MoE 模型上可带来显著的 decode 性能提升：

> RTX 5080 16GB VRAM + Qwen3 Coder Next 80B：使用 `--n-cpu-moe` 后 decode 从 10 t/s 提升至 32 t/s (3x)。

> RTX 5080 16GB + Qwen3.5 35B-a3b-q4_K_M (23GB)：标准卸载 ~10 t/s，`--cpu-moe` 后 ~70 t/s。

Ollama 的 `moe-split-cpu-phase1` 分支实现了类似机制（所有 dense 层放 GPU，MoE expert 权重按 VRAM 余量分 GPU/CPU），但在 RTX 3090 24 GB + 80B 模型场景下，**prefill 和 decode 均未观察到统计显著的性能改善**。本文档详细对比两种实现的机制差异，并分析性能差异的根因。

---

## 2. llama.cpp `--cpu-moe` 实现机制

### 2.1 参数定义与传递

**参数定义**（`common/arg.cpp:2292-2312`）：

```cpp
// --cpu-moe：全部 MoE expert 权重放 CPU
add_opt(common_arg(
    {"-cmoe", "--cpu-moe"},
    "keep all Mixture of Experts (MoE) weights in the CPU",
    [](common_params & params) {
        params.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
    }
).set_env("LLAMA_ARG_CPU_MOE"));

// --n-cpu-moe N：前 N 层的 MoE expert 权重放 CPU
add_opt(common_arg(
    {"-ncmoe", "--n-cpu-moe"}, "N",
    "keep the MoE weights of the first N layers in the CPU",
    [](common_params & params, int value) {
        for (int i = 0; i < value; ++i) {
            static std::list<std::string> buft_overrides;
            buft_overrides.push_back(llm_ffn_exps_block_regex(i));
            params.tensor_buft_overrides.push_back(
                {buft_overrides.back().c_str(), ggml_backend_cpu_buffer_type()});
        }
    }
).set_env("LLAMA_ARG_N_CPU_MOE"));
```

**正则表达式**（`common/common.h:983-991`）：

```cpp
const char * const LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

inline std::string llm_ffn_exps_block_regex(int idx) {
    return string_format("blk\\.%d%s", idx, LLM_FFN_EXPS_REGEX);
}
```

匹配张量名：`blk.N.ffn_up_exps`, `blk.N.ffn_down_exps`, `blk.N.ffn_gate_exps`, `blk.N.ffn_gate_up_exps` 及 `_ch_` 变体。

**参数转换**（`common/common.cpp:1436-1441`）：

```cpp
// tensor_buft_overrides → llama_model_params.tensor_buft_overrides
mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
```

**核心数据结构**（`include/llama.h:281-284`）：

```cpp
struct llama_model_tensor_buft_override {
    const char * pattern;               // 正则表达式
    ggml_backend_buffer_type_t buft;    // 目标 buffer type（CPU）
};
```

### 2.2 张量 Buffer Type Override

**模型加载时的 Override 应用**（`src/llama-model-loader.cpp:1156-1180`）：

```cpp
if (tensor_buft_overrides) {
    std::string tensor_name = tn.str();
    for (const auto * overrides = tensor_buft_overrides; overrides->pattern != nullptr; ++overrides) {
        std::regex pattern(overrides->pattern);
        if (std::regex_search(tensor_name, pattern)) {
            if (overrides->buft == ggml_backend_cpu_buffer_type()) {
                // CPU override 时考虑 extra buffer types（包括 host buffer）
                buft = select_weight_buft(hparams, t_meta, op, buft_list_cpu);
                if (use_mmap) {
                    LLAMA_LOG_WARN("tensor overrides to CPU are used with mmap enabled"
                                   " - consider using --no-mmap for better performance\n");
                }
            } else {
                buft = overrides->buft;
            }
            break;
        }
    }
}
```

**CPU Buffer Type 优先级列表**（`src/llama-model.cpp:526-585`，`make_cpu_buft_list()`）：

```
ACCEL buffer (BLAS) > Host buffer (CUDA pinned!) > Extra buffer > CPU buffer
```

当 `--no-mmap` + `--cpu-moe` 一起使用时，MoE 权重可被分配到 **CUDA pinned host buffer**，拷贝带宽从 ~12 GB/s 提升到 ~25 GB/s。但使用 mmap 时，llama.cpp 显式降级为 pageable：

```cpp
// src/llama-model-loader.cpp:1190-1198
if (use_mmap && buft == ggml_backend_dev_host_buffer_type(buft_dev)) {
    buft = ggml_backend_dev_buffer_type(cpu_dev);  // 降级为 pageable
}
```

### 2.3 op_offload 调度

**Scheduler 层面**（`ggml/src/ggml-backend.cpp:919`）：

```cpp
if (sched->op_offload && src_backend_id == sched->n_backends - 1
    && ggml_backend_buffer_is_host(src->buffer)) {
    for (int b = 0; b < src_backend_id; b++) {
        if (ggml_backend_supports_op(sched->backends[b], tensor)
            && ggml_backend_offload_op(sched->backends[b], tensor)) {
            return b;  // offload 到 GPU
        }
    }
}
return src_backend_id;  // 留在 CPU
```

**关键：llama.cpp 最新版中 scheduler 层面没有 `batch_size` 门控。** 决定权完全委托给设备层的 `offload_op()` 回调。

**CUDA 设备层**（`ggml/src/ggml-cuda/ggml-cuda.cu:5084-5087`）：

```cpp
static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;  // 默认 32，可通过 GGML_OP_OFFLOAD_MIN_BATCH 环境变量配置
}
```

其中 `get_op_batch_size` 对 `MUL_MAT_ID` 返回 `op->ne[2]`（= batch size）。

**结果**：
- **Prefill**（batch ≥ 32）：MoE op 被 offload 到 GPU，权重自动拷贝
- **Decode**（batch = 1 < 32）：MoE op 留在 CPU 直接执行

### 2.4 Selective Expert Copy

当 MoE 权重被 offload（prefill）时，scheduler 执行 **selective expert copy**，只拷贝被激活的 expert 而非全部（`ggml/src/ggml-backend.cpp:1515-1599`）：

```cpp
// when offloading MoE weights, we can reduce the amount of data copied
// by copying only the experts that are used
if (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID) {
    // 1. 读取 expert selection ids
    // 2. 用 bitset 标记激活的 experts
    // 3. 合并连续的 expert 做批量拷贝
    copy_experts(first_id, last_id);
}
```

---

## 3. Ollama MoE Split 实现机制

### 3.1 核心架构

Ollama 的 MoE split 通过 Go 层的权重追踪和分配实现，最终通过 `BackendParams.MoEGPULayers` 传递给 GGML 后端。

**数据结构**：

| 文件 | 结构 | 说明 |
|---|---|---|
| `ml/device.go:156` | `DeviceMemory.MoEWeights []uint64` | 每层 MoE expert 权重大小 |
| `ml/backend.go:72` | `BackendParams.MoEGPULayers` | MoE 权重在 GPU 的层索引 |
| `llm/server.go:483` | `LoadRequest.MoEGPULayers` | 传递给 runner 的 MoE 分配 |

**MoE Tensor 识别**（`ml/backend/ggml/ggml.go:150-155`）：

```go
var moeExpertRE = regexp.MustCompile(`\.ffn_(up|down|gate)_(ch_)?exps(\.weight)?$`)
```

### 3.2 两轮分配策略（`llm/server.go:983-1152`）

```
第一轮：检查所有 dense 权重能否放入 GPU
  → dense 总量 ~3.9 GiB < 可用 VRAM → 继续

第二轮：贪心分配 MoE 层到 GPU
  → 从后向前，尽可能多的 MoE 层放入剩余 VRAM
  → OLLAMA_MOE_GPU_LAYERS 可覆盖
```

### 3.3 张量路由

**在 `createTensors()` 中**（`ml/backend/ggml/ggml.go:400-411`）：

```go
if layerIndex >= 0 {
    bts := layers[layerIndex].bts          // dense 路由（GPU）
    if isMoEExpertTensor(t.Name) && len(params.MoEGPULayers) > 0 {
        bts = moeLayers[layerIndex].bts    // MoE 路由（可能 CPU）
    }
    createTensor(tensor{source: t}, bts, layerIndex)
}
```

### 3.4 op_offload 调度

**Ollama 内嵌的 `ggml-backend.cpp:865`**：

```cpp
if (sched->op_offload
    && (sched->batch_size < 0 || sched->batch_size >= 32)   // ← Ollama 特有的 batch_size 门控
    && src_backend_id == sched->n_backends - 1
    && ggml_backend_buffer_is_host(src->buffer)) {
```

**Ollama 内嵌的 `ggml-cuda.cu:4941-4944`**：

```cpp
static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const int min_batch_size = 32;                    // ← 硬编码，不可配置
    return get_op_batch_size(op) >= min_batch_size;
}
```

---

## 4. 关键差异总结

### 4.1 机制差异

| 维度 | llama.cpp `--cpu-moe` | Ollama MoE Split |
|---|---|---|
| **实现层级** | C++ 模型加载器（`tensor_buft_overrides`） | Go GGML 后端（`assignMoELayer`） |
| **张量路由方式** | 正则表达式匹配 → 覆盖 buffer type | Go 代码 `isMoEExpertTensor()` → 分配 buffer |
| **Scheduler batch_size 门控** | **无**（最新版已移除） | **有**：`batch_size < 0 \|\| batch_size >= 32` |
| **CUDA offload_op min_batch** | 可通过 `GGML_OP_OFFLOAD_MIN_BATCH` 配置 | **硬编码 32**，不可配置 |
| **CPU Buffer 类型** | `--no-mmap` 时可用 pinned host buffer | 始终 pageable（mmap 或标准 CPU） |
| **op_offload 效果（Decode）** | 不触发（batch=1 < 32） | 不触发（batch=1 < 32） |
| **op_offload 效果（Prefill）** | 触发（batch ≥ 32） | 触发（batch ≥ 32） |

### 4.2 运行时等价性

尽管实现路径不同，**两者产生的最终运行时状态完全一致**：

1. MoE expert 张量在 CPU buffer（pageable，mmap 场景下）
2. Dense/attention 张量在 GPU buffer
3. 同一份 ggml scheduler 代码处理计算图分割和数据拷贝
4. 同一份 CUDA `offload_op()` 回调决定是否 offload
5. 同一份 selective expert copy 逻辑（`MUL_MAT_ID` 场景）

**Decode 时两者行为完全相同**：MoE op 在 CPU 执行，attention op 在 GPU 执行，scheduler 自动处理 GPU↔CPU splits。

---

## 5. Decode Benchmark 结果

### 5.1 测试配置

| 配置 | `OLLAMA_MOE_GPU_LAYERS` | max_tokens | 备注 |
|---|---|---|---|
| `baseline-moe-split-disabled-output-200` | `0`（标准路由） | 200 | Bug 2+3 修复后 |
| `moe-split-enabled-output-200` | 未设置（auto） | 200 | 17 层 MoE on GPU |

### 5.2 层分配对比

**标准布局**（日志）：

```
GPULayers:  20 [Layers: 28..47]     → 层 28-47 整层在 GPU
MoEGPULayers: 20 [Layers: 28..47]   → 与 GPULayers 相同
GPU weights: 19.9 GiB    CPU weights: 28.3 GiB
GPU KV: 1.1 GiB          CPU KV: 1.5 GiB
GPU graph: 884.1 MiB      CPU graph: 270.6 MiB
```

**MoE Split**（日志）：

```
GPULayers:  49 [Layers: 0..48]      → 所有层 dense 在 GPU
MoEGPULayers: 17 [Layers: 31..47]   → 层 31-47 的 MoE 也在 GPU
GPU weights: 17.9 GiB    CPU weights: 30.3 GiB
GPU KV: 2.6 GiB          CPU KV: 0 GiB（全 GPU）
GPU graph: 800.3 MiB      CPU graph: 8.0 MiB
```

### 5.3 数据

| 指标 | 标准布局 | MoE Split | 差异 |
|---|---|---|---|
| **gen_tps (mean)** | **18.09 t/s** | **17.61 t/s** | **-2.7%** |
| gen_tps stddev | 0.07 | 0.19 | — |
| gen_tps CV | 0.39% | 1.09% | — |
| gen_ms (mean, 200 tok) | 11,058 ms | 11,357 ms | +299 ms |
| prefill_ms (mean) | 2,095 ms | 2,061 ms | -34 ms (<1σ) |
| VRAM used | 23.46 GiB | 22.80 GiB | -0.66 GiB |

**注**：MoE split 的 epoch 4 出现异常（eval_count=2, gen_tps=32.9），在 stats 计算中被排除（因为仅生成 2 个 token，不代表稳态性能），mean/median 基于其余 5 个有效 epoch。

### 5.4 结论

**MoE split 在 decode 阶段未产生性能改善，甚至略慢 2.7%（~0.5 t/s）。** 差异在 ~2.6σ 水平，接近统计显著但方向为负。

---

## 6. Decode 无改善的根因分析

### 6.1 Qwen3-Coder-Next 的 Hybrid Attention 架构

**Qwen3-Coder-Next 80B 不是标准的 Transformer**。它采用 **Hybrid Attention** 架构，混合使用两种 attention 类型（`convert/convert_qwen3next.go:49-51`，`model/models/qwen3next/model.go:497-549`）：

- **Full Attention 层**（标准二次注意力）：使用完整的 Q/K/V 投影 + causal attention + KV cache
- **Linear Attention 层**（Gated Delta Net / SSM）：使用递归状态替代 KV cache，计算复杂度 O(n) 而非 O(n²)

层类型由 `full_attention_interval` 决定（`convert/convert_qwen3next.go:218`）：

```go
isRecurrent[i] = (i+1) % fullAttentionInterval != 0
```

以 `full_attention_interval=4` 为例（典型配置），48 层中：
- **Full Attention**：层 3, 7, 11, ... 47 → **12 层**（25%）
- **Linear Attention (GatedDeltaNet)**：其余 **36 层**（75%）

这对 MoE split 的 decode 收益分析有根本性影响。

### 6.2 Linear Attention 在 CPU 上本来就很快

**Full Attention 层**（`model/models/qwen3next/attention.go`）的 decode 开销：
- Q/K/V 投影 → Attention Score → Output → 需要读取 KV cache
- CPU 上：受限于内存带宽，~1.0 ms/层
- GPU 上：kernel launch + matmul，~0.15-0.20 ms/层
- **CPU→GPU 收益：~0.8 ms/层**

**Linear Attention 层**（`model/models/qwen3next/deltanet.go`）的 decode 开销：
- SSM QKV 投影 → 1D Conv → Beta/Alpha 门控 → 递归状态更新 → Output
- **不需要 KV cache**，只维护固定大小的递归状态
- 权重更小（比 Full Attention 少 ~2-3x）
- CPU 上：O(1) per token，计算量小，~0.3-0.5 ms/层
- GPU 上：kernel launch overhead 相对于少量计算来说很高，~0.1-0.15 ms/层
- **CPU→GPU 收益：仅 ~0.2-0.35 ms/层**

### 6.3 修正后的理论收益

标准布局中层 0-27 在 CPU（28 层），其中约 75% 是 Linear Attention（~21 层），25% 是 Full Attention（~7 层）：

```
Full Attention 层从 CPU→GPU:   7 × 0.8 ms  =  5.6 ms
Linear Attention 层从 CPU→GPU: 21 × 0.3 ms =  6.3 ms
────────────────────────────────────────────────────
理论总收益:                                    11.9 ms
```

**Hybrid architecture 使得 attention 从 CPU→GPU 的收益仅 ~12 ms，远低于全 Full Attention 模型的 ~22 ms 估算。**

### 6.4 理论代价：Split 切换开销

MoE split 改变了计算图的 split 结构。

**标准布局 decode 时的 splits**：

```
Split 1 [GPU]: 层 28-47 全部 ops
Split 2 [CPU]: 层 0-27 全部 ops（attention + MoE）
Split 3 [GPU]: output 层
总计: ~3 个 splits
```

**MoE split decode 时的 splits**：

对于每个 CPU-MoE 层（0-30），该层的计算被拆成两个 backend：

```
Layer N:
  [GPU split]: Norm → Attention/DeltaNet → Residual Add
  → sync + copy hidden state (GPU→CPU, ~16 KB)
  [CPU split]: FFN Norm → MoE Gate → MUL_MAT_ID → Residual Add  
  → sync + copy result (CPU→GPU, ~16 KB)
Layer N+1:
  [GPU split]: ...
```

31 个 CPU-MoE 层 × 每层 2 次 backend 切换 = **~62 次 split 切换**。

每次 split 切换的固定开销：
- `ggml_backend_synchronize()` — CUDA 流全量同步：~50-200 μs
- Hidden state 数据拷贝：~16 KB，带宽不是瓶颈，~10 μs

```
总 split 开销 = 62 × (100-200 μs)
             ≈ 6.2 - 12.4 ms/token
```

### 6.5 收支平衡分析

```
理论收益（attention/SSM CPU→GPU）:  ~11.9 ms
理论代价（split sync 开销）:        ~9.3 ms（62 × 150 μs）
理论代价（MoE 新增 CPU 层）:        3 层 MoE 从 GPU→CPU
                                    ≈ 3 × 0.8 ms ≈ 2.4 ms
────────────────────────────────────────────────────
理论净收益:                         ~0.2 ms → 基本持平
```

**理论预测与实测（-1.5 ms，即 -2.7%）高度吻合。** Hybrid attention 架构大幅削弱了 attention CPU→GPU 的收益，使其几乎被 split 切换开销完全抵消。

### 6.6 其他加剧因素

1. **DDR5-6400 双通道高带宽**：实际 ~75-90 GB/s，CPU 上的 Linear Attention 层本来就很快。注意：Arrow Lake 265K **不支持 AVX-512**（仅 AVX2），但 SSM 递归状态更新对 SIMD 宽度要求不高

2. **GPU kernel launch overhead**：batch=1 时每个小 kernel 都有 ~5-10 μs 的 launch 开销，49 层 × 多个 kernel 累计可观

3. **KV cache 位置变化**：标准布局 1.5 GiB KV 在 CPU，MoE split 后 2.6 GiB KV 全在 GPU，增加 GPU 内存压力

4. **GPU 带宽竞争**：MoE split 后 GPU 需服务 49 层 attention/SSM 权重 + 17 层 MoE 权重 + 2.6 GiB KV cache，而标准布局只需服务 20 层

---

## 7. 为什么社区报告 3x 提升而本实验无提升

### 7.1 硬件-模型配比差异 + 架构差异

社区报告的典型场景与本实验存在**双重差异**：硬件配比不同，且模型架构不同。

**社区场景**：RTX 5080 16GB + Qwen3-Coder-Next 80B  
**本实验**：RTX 3090 24GB + Qwen3-Coder-Next 80B

| 因素 | 社区场景 (16GB + 80B) | 本实验 (24GB + 80B) |
|---|---|---|
| 标准布局 GPU 层数 | ~49 层中 ~8 层 (~16%) | ~49 层中 ~20 层 (~41%) |
| 标准布局 CPU attention 层 | ~41 层 | ~28 层 |
| 其中 Full Attention 在 CPU | ~10 层 | ~7 层 |
| 其中 Linear Attention 在 CPU | ~31 层 | ~21 层 |
| CPU-MoE 层（split 后） | ~41 层 | ~31 层 |
| Split 切换次数 | ~82 次 | ~62 次 |
| Split 总开销 | ~12.3 ms | ~9.3 ms |
| Full Attn CPU→GPU 节省 | 10 × 0.8 = **8.0 ms** | 7 × 0.8 = **5.6 ms** |
| Linear Attn CPU→GPU 节省 | 31 × 0.3 = **9.3 ms** | 21 × 0.3 = **6.3 ms** |
| MoE CPU→GPU 损失 | 8 × 0.8 = **6.4 ms** | 3 × 0.8 = **2.4 ms** |

但社区报告的是从 10 t/s → 32 t/s（3x 提升），而上述表格中理论净收益仍然不大。说明社区场景中，**标准布局的基线极度恶劣**（很多 Full Attention 层被迫在 CPU 上执行），而 16 GB VRAM 的标准布局把大量层整层留在 CPU，这些层的 attention 成为绝对瓶颈。

### 7.2 基线越差，收益越大

社区 16 GB VRAM 的标准布局**极度不利**：

- 只有 ~16% 层在 GPU → decode 时 ~84% 的 attention/SSM 计算在 CPU
- CPU 层中的 MoE 权重浪费大量内存带宽（996 MiB/层但只激活 2 个 expert）
- Full Attention 层在 CPU 上尤其慢（需要读取 KV cache + 完整 Q/K/V 投影）
- 标准 decode ~10 t/s → 每 token ~100 ms

MoE split 后：
- 100% attention/SSM 在 GPU → 消除了 CPU attention 这个最大瓶颈
- CPU 只跑 MoE 的 `MUL_MAT_ID`，batch=1 时只读 ~15 MiB 激活权重
- 新 decode ~32 t/s → 每 token ~31 ms

从 100 ms 到 31 ms 的跨越，核心是因为标准布局的基线太差（大量 Full Attention 在 CPU），split 后消除了这个瓶颈。

**而在 24 GB 场景**：基线 55 ms/token 已经不差（41% 层在 GPU），CPU attention 负担较轻，且 75% 是 Linear Attention（本来就快），MoE split 的 attention 收益被 split 切换开销抵消。

### 7.3 跨工具对比的不确定性

部分社区报告是 **KoboldCpp vs Ollama** 的跨工具对比，可能包含工具层面的差异：
- BLAS 库选择不同
- mmap 默认值不同（KoboldCpp 可能默认 `--no-mmap`，此时 CPU 权重使用 pinned host buffer，op_offload 拷贝更快）
- 线程调度策略不同
- llama.cpp 版本差异（可能包含额外优化）

---

## 8. 两个版本的 ggml 差异

在调查过程中发现 Ollama 内嵌的 ggml 代码与 llama.cpp 最新版存在以下差异：

### 8.1 Scheduler batch_size 门控

**Ollama** (`ml/backend/ggml/ggml/src/ggml-backend.cpp:865`)：

```cpp
if (sched->op_offload
    && (sched->batch_size < 0 || sched->batch_size >= 32)
    && src_backend_id == sched->n_backends - 1
    && ggml_backend_buffer_is_host(src->buffer))
```

**llama.cpp** (`ggml/src/ggml-backend.cpp:919`)：

```cpp
if (sched->op_offload
    && src_backend_id == sched->n_backends - 1
    && ggml_backend_buffer_is_host(src->buffer))
```

llama.cpp 最新版 **移除了 `batch_size` 门控**。`ggml_backend_sched_set_batch_size()` 函数本身也已从 llama.cpp 中移除。

**影响**：在当前实现下无影响（因为 CUDA `offload_op` 仍有 batch_size ≥ 32 的硬门控），但如果未来修改 `offload_op` 的阈值，Ollama 的额外门控会成为阻碍。

### 8.2 CUDA offload_op 配置能力

**Ollama** (`ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu:4941-4944`)：

```cpp
static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const int min_batch_size = 32;    // 硬编码
    return get_op_batch_size(op) >= min_batch_size;
}
```

**llama.cpp** (`ggml/src/ggml-cuda/ggml-cuda.cu:5084-5087`)：

```cpp
static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;  // 可配置
}
```

**初始化**（`ggml-cuda.cu:5258`）：

```cpp
const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH")
    ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;
dev_ctx->op_offload_min_batch_size = min_batch_size;
```

**影响**：llama.cpp 允许用户通过 `GGML_OP_OFFLOAD_MIN_BATCH=1` 让 decode (batch=1) 也触发 offload（将 MoE 拷贝到 GPU 计算）。Ollama 不支持此配置。

---

## 9. 关键代码位置索引

### llama.cpp

| 文件 | 行号 | 说明 |
|---|---|---|
| `common/arg.cpp` | 2292-2312 | `--cpu-moe` / `--n-cpu-moe` CLI 定义 |
| `common/common.h` | 983-991 | MoE regex 模式和辅助函数 |
| `common/common.cpp` | 1436-1441 | `tensor_buft_overrides` 转换 |
| `include/llama.h` | 281-284 | `llama_model_tensor_buft_override` 结构 |
| `src/llama-model-loader.cpp` | 1156-1180 | Override 匹配和 buffer type 覆盖 |
| `src/llama-model-loader.cpp` | 1190-1198 | mmap + host buffer 降级逻辑 |
| `src/llama-model.cpp` | 526-585 | `make_cpu_buft_list()` CPU buffer 优先级 |
| `ggml/src/ggml-backend.cpp` | 919 | op_offload 调度（无 batch_size 门控） |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 5084-5087 | CUDA `offload_op`（可配置 min_batch） |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 5258 | `GGML_OP_OFFLOAD_MIN_BATCH` 环境变量 |
| `ggml/src/ggml-backend.cpp` | 1515-1599 | Selective expert copy |

### Ollama

| 文件 | 行号 | 说明 |
|---|---|---|
| `ml/backend/ggml/ggml.go` | 150-155 | `moeExpertRE` MoE tensor 识别 |
| `ml/backend/ggml/ggml.go` | 231-248 | `assignMoELayer()` MoE 层 buffer 分配 |
| `ml/backend/ggml/ggml.go` | 400-411 | 张量路由决策（MoE vs dense） |
| `ml/backend/ggml/ggml.go` | 875-877, 890-891 | `SetBatchSize` → `ggml_backend_sched_set_batch_size` |
| `llm/server.go` | 983-1152 | `buildLayout()` 两轮 MoE split 分配 |
| `envconfig/config.go` | 226 | `OLLAMA_MOE_GPU_LAYERS` 定义 |
| `ml/backend/ggml/ggml/src/ggml-backend.cpp` | 865 | op_offload 调度（有 batch_size 门控） |
| `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu` | 4941-4944 | CUDA `offload_op`（硬编码 32） |

---

## 10. 结论与后续方向

### 10.1 核心结论

1. **llama.cpp `--cpu-moe` 与 Ollama MoE split 的底层执行机制等价**：两者都把 MoE expert 权重放 CPU、dense/attention 放 GPU，由同一套 ggml scheduler 处理
2. **Decode 无改善的根因有两层**：
   - **Hybrid Attention 架构削弱了收益**：Qwen3-Coder-Next 约 75% 的层是 Linear Attention (GatedDeltaNet)，在 CPU 上 O(1) per token、权重小、本来就快，从 CPU→GPU 的收益远小于标准 Full Attention 层
   - **Split 切换开销抵消了剩余收益**：62 次 backend 同步（~9.3 ms）与 ~12 ms attention/SSM 收益几乎完全对消
3. **社区 3x 报告成立但场景不同**：16 GB VRAM 的标准布局极度不利（~84% 层在 CPU），大量 Full Attention 在 CPU 是绝对瓶颈，MoE split 消除了这个瓶颈

### 10.2 后续方向

| 方向 | 说明 | 预期影响 |
|---|---|---|
| **同步 llama.cpp 的 ggml 变更** | 移除 scheduler batch_size 门控；CUDA offload_op 支持 `GGML_OP_OFFLOAD_MIN_BATCH` 配置 | 对齐上游能力，为未来优化预留空间 |
| **Pinned host buffer for CPU-MoE** | 在 `assignMoELayer()` 中使用 GPU host buffer type 替代 CPU buffer type | Prefill 拷贝带宽 12→25 GB/s |
| **减少 split 切换次数** | 探索图融合（将相邻的 GPU attention + CPU MoE 合并为单个 split） | Decode 减少同步开销 |
| **Async prefetch pipeline** | CUDA 双缓冲 staging buffer，GPU 计算层 N 时异步预取层 N+1 的 MoE | Prefill 理论 -40%，decode 有条件改善 |
| **面向 16 GB VRAM 用户优化** | 在小 VRAM 场景下 MoE split 的 ROI 最高，优先适配该场景 | 社区影响最大 |
