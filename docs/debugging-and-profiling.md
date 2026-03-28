# Ollama + llama.cpp 调试与性能分析手册

本文档汇总了 Ollama 和底层 llama.cpp / ggml 栈中所有可用的调试、日志与性能分析手段。

---

## 目录

1. [能力边界一览](#能力边界一览)
2. [Ollama 层：环境变量](#ollama-层环境变量)
3. [llama.cpp 层：CLI flags](#llamacpp-层cli-flags)
4. [llama.cpp 层：运行时 API](#llamacpp-层运行时-api)
5. [ggml 层：计算图打印](#ggml-层计算图打印)
6. [ggml 层：per-node eval 回调](#ggml-层per-node-eval-回调)
7. [GPU 后端专项](#gpu-后端专项)
   - [Vulkan — GGML_VK_PERF_LOGGER（★ 最详细的内置 per-op 计时）](#vulkan--ggml_vk_perf_logger)
   - [CUDA 环境变量与编译开关](#cuda-环境变量与编译开关)
   - [Metal（Apple Silicon）](#metalapple-silicon)
8. [编译期 debug 宏](#编译期-debug-宏)
9. [Ollama Bench 工具](#ollama-bench-工具)
10. [外部 GPU Profiler（终极手段）](#外部-gpu-profiler终极手段)
11. [典型工作流示例](#典型工作流示例)

---

## 能力边界一览

| 想看什么 | 能否开箱即用 | 方式 |
|---|---|---|
| 模型加载时间 | ✅ | `OLLAMA_DEBUG=1` / `--verbose` |
| Prefill / Decode 总速度 (token/s) | ✅ | 默认输出 / API 响应字段 |
| 哪些层在 GPU / CPU | ✅ | `OLLAMA_DEBUG=1` |
| 总 VRAM / RAM 用量 | ✅ | `OLLAMA_DEBUG=1` |
| KV cache 大小与类型 | ✅ | `OLLAMA_DEBUG=1` |
| 计算图节点类型 + 张量形状 | ⚠️ 需插桩 | `ggml_graph_print()` + 重编译 |
| **每个节点的执行时间** — Vulkan | ✅ 运行时环境变量 | `GGML_VK_PERF_LOGGER=1` |
| **每个节点的执行时间** — CUDA/Metal | ❌ 需外部工具 | Nsight / Instruments |
| 单个 kernel 的 FLOPS/内存带宽 | ❌ | Nsight Compute |
| 多模型、多配置对比 | ✅ | `ollama-bench` / `llama-bench` |

---

## Ollama 层：环境变量

### `OLLAMA_DEBUG`（最常用）

```bash
OLLAMA_DEBUG=1 ollama serve
```

支持三档：

| 值 | 日志级别 | 说明 |
|---|---|---|
| `0` / 未设置 | INFO | 默认，只打印关键信息 |
| `1` / `true` | DEBUG | 打印模型加载详情、GPU 分配、KV cache 大小 |
| `2` | TRACE | 极详细，含内部调度细节 |

DEBUG 级别下可以看到：
```
llm_load_tensors: offloaded 32/33 layers to GPU
llm_load_tensors: VRAM used = 4721 MiB
llm_load_tensors: CPU used = 312 MiB
kv_cache: n_ctx = 4096, n_kv_max = 4096, kv_size = 512 MiB (f16)
```

### GPU 设备控制

| 变量 | 平台 | 说明 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | NVIDIA | 限制可见 GPU，如 `0,1` |
| `HIP_VISIBLE_DEVICES` | AMD | 同上（ROCm） |
| `ROCR_VISIBLE_DEVICES` | AMD | 按 UUID 或序号指定 |
| `GGML_VK_VISIBLE_DEVICES` | Vulkan | 按序号指定 Vulkan 设备 |
| `GPU_DEVICE_ORDINAL` | AMD | 按序号指定 |
| `HSA_OVERRIDE_GFX_VERSION` | AMD | 覆盖 GFX 版本号，解决兼容性问题 |
| `OLLAMA_GPU_OVERHEAD` | 所有 | 每块 GPU 预留的 VRAM 字节数（默认 0） |

### 其他有用变量

| 变量 | 说明 |
|---|---|
| `OLLAMA_LLM_LIBRARY` | 强制指定后端库（如 `cuda_v11`），绕过自动检测 |
| `OLLAMA_FLASH_ATTENTION=1` | 开启 Flash Attention（影响 VRAM 用量和速度） |
| `OLLAMA_KV_CACHE_TYPE` | KV cache 量化类型（`f16`、`q8_0`、`q4_0`） |
| `OLLAMA_CONTEXT_LENGTH` | 覆盖默认 context 长度 |
| `OLLAMA_SCHED_SPREAD=1` | 强制将模型分散到所有 GPU（多卡） |
| `OLLAMA_NUM_PARALLEL` | 并行请求槽数 |
| `OLLAMA_VULKAN=1` | 启用实验性 Vulkan 后端 |

---

## llama.cpp 层：CLI flags

当直接使用 `llama-cli` / `llama-server`（不通过 Ollama）时：

```bash
./llama-cli \
  -m model.gguf \
  --verbose \           # 开启详细日志
  --log-format json \   # 日志格式：text（默认）或 json
  -ngl 99 \             # offload 层数到 GPU
  -p "Hello" -n 100
```

`--verbose` 典型输出：
```
llm_load_print_meta: general.name     = Llama 3.1 8B
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_embd           = 4096
llm_load_tensors: offloaded 32/33 layers to GPU
llm_load_tensors: VRAM used = 4721.23 MiB

llama_perf_context_print: load time       =   523.45 ms
llama_perf_context_print: prompt eval time =  1023.12 ms / 128 tokens (7.99 ms/token, 125.1 tokens/s)
llama_perf_context_print:        eval time =  4521.33 ms /  99 tokens (45.67 ms/token, 21.9 tokens/s)
llama_perf_context_print:       total time =  5544.45 ms / 227 tokens
```

---

## llama.cpp 层：运行时 API

### `llama_perf_context_print()` / `llama_perf_sampler_print()`

推理结束后自动调用（或手动调用），输出**聚合级别**的性能数据：

```c
struct llama_perf_context_data {
    double t_start_ms;    // 启动时间戳
    double t_load_ms;     // 模型加载耗时
    double t_p_eval_ms;   // prefill 总耗时
    double t_eval_ms;     // decode 总耗时
    int32_t n_p_eval;     // prefill token 数
    int32_t n_eval;       // decode token 数
};
```

这是 Ollama 在 API 响应的 `eval_duration`、`prompt_eval_duration` 字段的数据来源。

### `llama_print_system_info()`

打印所有已加载后端的 feature 列表：
```
CUDA : ARCHS = 70;75;80;86;89;90 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128 | ...
```

### `llama_model_print_metadata()`

打印模型架构参数（层数、head 数、维度等）：在加载时通过 `--verbose` 自动触发。

---

## ggml 层：计算图打印

### `ggml_graph_print()`

打印整个计算图的节点列表，包含：算子类型、张量名称、形状（ne）、字节数。

**⚠️ 不包含执行时间和设备信息。**

示例输出：
```
=== FORWARD GRAPH ===
n_nodes = 867
 - node   0: RESHAPE       [4096, 1, 1, 1]  name = inp_embd
 - node   1: GET_ROWS      [ 128, 1, 1, 1]  name = inp_pos
 - node   2: RMS_NORM      [4096, 1, 1, 1]  name = norm_0
 - node   3: MUL_MAT       [4096,4096,1,1]  name = wq_0
 - node   4: MUL_MAT       [4096,4096,1,1]  name = wk_0
 - node   5: ROPE          [4096, 1, 1, 1]  name = kq_0
 ...
```

**如何触发：**

需要在 `llama.cpp` 源码中手动添加调用，在计算图执行前插入：

```cpp
// 在 src/llama-context.cpp 的 llama_decode_internal() 中
// 找到 ggml_backend_sched_graph_compute() 调用前
ggml_graph_print(gf);  // gf 是 struct ggml_cgraph *
```

修改后重新编译。`GGML_DEBUG=1` 环境变量在当前版本中**不会**自动触发此函数。

---

## ggml 层：per-node eval 回调

这是**不修改核心代码**情况下获取 per-node 信息的唯一运行时接口。

### API

```c
// ggml-backend.h
typedef bool (*ggml_backend_sched_eval_callback)(
    struct ggml_tensor * t,   // 当前节点
    bool ask,                  // true=询问是否要计时，false=节点执行完毕
    void * user_data
);

ggml_backend_sched_set_eval_callback(sched, callback, user_data);
```

### 用法（llama.cpp 层）

通过 `llama_context_params.cb_eval` 设置：

```c
llama_context_params cparams = llama_context_default_params();
cparams.cb_eval = [](struct ggml_tensor * t, bool ask, void * user_data) -> bool {
    if (ask) {
        return true; // 对所有节点计时
    }
    // 节点执行完毕，此时可记录时间
    // 注意：此时从 GPU 同步回 CPU 会引入 overhead
    fprintf(stderr, "node: %-20s shape=[%lld,%lld,%lld,%lld]\n",
            t->name, t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    return true;
};
cparams.cb_eval_user_data = nullptr;
```

**Ollama 当前状态：** Ollama 的 LlamaRunner 通过 `OLLAMA_TRACE_DIR` 启用此回调，生成 JSONL trace 文件。OllamaRunner 有基础 pass 级别 tracing。

### GPU 性能影响（重要）

当 eval callback 被设置时，scheduler 的行为发生根本性改变：

| | 无 callback（正常） | 有 callback（tracing） |
|---|---|---|
| **dispatch 方式** | 一次提交整个 graph | 逐 node 提交 |
| **GPU 同步** | 仅在 graph 结束时 | 每个 node 后强制 `synchronize` |
| **GPU 流水线** | kernel 之间可 overlap | 完全串行，无 overlap |
| **推理速度** | 正常 | GPU 模型慢 2x+（500 node × ~50µs sync） |
| **CPU 影响** | 1 次 graph_compute | 503 次 graph_compute，每次创建/销毁线程池 |
| **CPU 多线程** | 正常 | 每次 graph_compute 内部多线程正常工作，per-op 计算时间准确 |

### 数据解读指南

eval callback 测到的 per-op 时间是**每个 op 在隔离执行下的耗时**（GPU 上没有其他 kernel 并行）。

- **有效用途：** 相对排序（找最慢的 op）、模型结构分析（DAG/shape/dtype）、跨模型比较
- **无效用途：** 绝对耗时占比分析（`sum(per_op) > 实际总时间`，因为正常执行中小 op 与大 op 并行）
- **准确性：** 大 op（如 `MUL_MAT`）的测量值接近真实值；小 op 因 sync 开销被放大

### 工具对比与推荐工作流

| | Ollama Trace | GGML_VK_PERF_LOGGER | Nsight Systems |
|---|---|---|---|
| **信息粒度** | 每个 op 实例（tensor name, shape, DAG） | 按 op 类型汇总平均 | 每个 CUDA kernel |
| **GPU 时间准确性** | ⚠️ 近似（强制同步） | ✅ 准确（GPU timestamp） | ✅ 准确（驱动层） |
| **影响执行** | ⚠️ 破坏 GPU 流水线 | ✅ 不影响 | ✅ 不影响 |
| **适用 backend** | 所有 | 仅 Vulkan | 仅 CUDA |

**注意：** `GGML_VK_PERF_LOGGER` 把所有同类 op 混在一起算平均（如 32 个不同大小的 `MUL_MAT` → 一个平均值），丢失了每个具体 op 的 tensor name、shape 等上下文。

**推荐工作流：**
1. **Ollama Trace** — 先跑一次，理解模型结构、找到关键 op / 层
2. **Nsight / VK_PERF_LOGGER** — 对关键 op 做精确 GPU 计时（不影响执行）

---

## GPU 后端专项

### Vulkan — `GGML_VK_PERF_LOGGER`

**★ 这是目前内置的最详细 per-op 计时工具，无需重新编译。**

```bash
# 每次推理后打印一次 per-op 汇总
GGML_VK_PERF_LOGGER=1 ollama run llama3

# 每 N 次推理打印一次（减少输出量）
GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=10 ollama run llama3
```

输出示例（打印到 stderr）：
```
----------------
Vulkan Timings:
MUL_MAT_VEC: 64 x 142.3 us (187.4 GFLOPS/s)
MUL_MAT: 32 x 8921.4 us (412.1 GFLOPS/s)
RMS_NORM: 64 x 12.1 us
ROPE: 64 x 8.4 us
SOFT_MAX: 32 x 6.2 us
ADD: 96 x 3.1 us
Total time: 294821.0 us.
```

**注意：** 仅在 Vulkan 后端（`OLLAMA_VULKAN=1`）下有效。CUDA 和 Metal 后端无对应机制。

#### Vulkan 其他调试变量

| 变量 | 说明 |
|---|---|
| `GGML_VK_PREFER_HOST_MEMORY=1` | 优先使用主机内存（pinned memory） |
| `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` | 禁用主机可见 VRAM |
| `GGML_VK_ALLOW_SYSMEM_FALLBACK=1` | 允许回退到系统内存 |
| `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` | 禁用 Vulkan 计算图优化（调试用） |
| `GGML_VK_DISABLE_COOPMAT=1` | 禁用 cooperative matrix |
| `GGML_VK_DISABLE_COOPMAT2=1` | 禁用 cooperative matrix v2 |
| `GGML_VK_DISABLE_ASYNC=1` | 禁用异步执行 |
| `GGML_VK_DISABLE_F16=1` | 禁用 f16 支持 |
| `GGML_VK_FORCE_MAX_ALLOCATION_SIZE=N` | 强制最大分配字节数 |
| `GGML_VK_SUBALLOCATION_BLOCK_SIZE=N` | 控制子分配块大小 |

#### Vulkan 编译期调试

```bash
cmake -B build -DGGML_VULKAN=ON -DGGML_VULKAN_DEBUG=ON
```

启用后 `VK_LOG_DEBUG()` 宏生效，打印每个 Vulkan API 调用的详细信息（极其冗长）。

---

### CUDA 环境变量与编译开关

#### 运行时环境变量

| 变量 | 说明 |
|---|---|
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` | 启用 CUDA 统一内存（UMA），适合 VRAM 不足时 |
| `GGML_CUDA_INIT=1` | 打印每块 GPU 的初始化详情（设备名、显存大小等） |
| `CUDA_VISIBLE_DEVICES=0,1` | 限制可见 GPU |
| `CUDA_LAUNCH_BLOCKING=1` | 同步执行所有 CUDA kernel（严重降速，仅用于定位崩溃位置） |

#### 编译期开关（需重新编译）

```bash
cmake -B build -DGGML_CUDA=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \        # 强制使用量化矩阵乘法（禁用 cuBLAS）
  -DGGML_CUDA_FORCE_CUBLAS=ON \     # 强制使用 cuBLAS（禁用量化 kernel）
  -DGGML_CUDA_FA_ALL_QUANTS=ON      # Flash Attention 支持所有量化类型
```

这些开关在加载时会打印到日志：
```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    yes
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
```

---

### Metal（Apple Silicon）

Metal 后端没有类似 `GGML_VK_PERF_LOGGER` 的运行时计时开关。

| 方式 | 说明 |
|---|---|
| `MTL_DEBUG_LAYER=1` | 启用 Metal validation layer（OS 级别，非 ggml 变量） |
| `GGML_METAL_NDEBUG=0` | 保留 Metal shader 中的 assert（默认=1 已禁用） |
| Instruments → Metal System Trace | 推荐：GUI 工具，可看到每个 shader dispatch 的精确时间轴 |

使用 Instruments：
```
# 命令行启动 Instruments 抓取
xcrun xctrace record --template "Metal System Trace" \
  --launch -- ./llama-cli -m model.gguf -p "Hello" -n 50
```

---

## 编译期 debug 宏

此 repo 中的特殊机制：`ml/backend/ggml/ggml/src/ggml-cpu/cpu_debug.go`

```go
//go:build debug

package cpu

// #cgo CPPFLAGS: -DOLLAMA_DEBUG
import "C"
```

用 `debug` build tag 编译时，会向 C 代码传入 `-DOLLAMA_DEBUG`，启用 CPU 后端的额外 assert 和日志：

```bash
go build -tags debug ./...
```

---

## Ollama Bench 工具

Ollama 仓库内置了一个 Go 编写的 benchmark 工具（`cmd/bench/`），通过 REST API 与 Ollama 交互，不需要直接调用 llama.cpp。

### 构建

```bash
go build -o ollama-bench ./cmd/bench
```

或直接运行：
```bash
go run ./cmd/bench -model gemma3 -epochs 6
```

### 用法示例

```bash
# 基础 benchmark
./ollama-bench -model gemma3 -epochs 6

# 多模型对比（benchstat 格式）
./ollama-bench -model gemma3,llama3.2 -epochs 6 | tee results.bench
benchstat -col /step results.bench

# 指定 prompt 长度（用于可重现的测试）
./ollama-bench -model gemma3 -epochs 6 -prompt-tokens 512 -max-tokens 200

# CSV 输出（方便导入 Excel / 画图）
./ollama-bench -model gemma3 -epochs 6 -format csv -output results.csv

# 带预热
./ollama-bench -model gemma3 -epochs 6 -warmup 2

# 图文多模态
./ollama-bench -model qwen3-vl -image photo.jpg -epochs 6 -p "Describe this image"
```

### 输出指标

| 指标 | 说明 |
|---|---|
| `prefill` | prompt 处理速度 (ns/token, token/s) |
| `generate` | token 生成速度 (ns/token, token/s) |
| `ttft` | Time to First Token，延迟 (ns) |
| `load` | 模型加载耗时 (ns) |
| `total` | 总请求时间 (ns) |

输出的 model info 注释行还包含：
- `Params`：参数量（如 4.3B）
- `Quant`：量化类型（如 Q4_K_M）
- `Size`：模型总内存字节数
- `VRAM`：GPU 占用内存（`Size - VRAM` 即 CPU 溢出量）

### 与 llama-bench 的区别

| 方面 | `ollama-bench` | `llama-bench` |
|---|---|---|
| 接口 | REST API（通过 Ollama） | 直接调用 llama.cpp |
| 并发/slot 开销 | 包含 Ollama 调度开销 | 不包含 |
| KV cache | Ollama 自动管理 | 手动 `-c` 指定 |
| 代表性 | 更接近真实服务场景 | 更接近纯计算性能上限 |
| 多配置矩阵 | 基础 | 支持 `-p/-n/-b/-ub` 矩阵扫描 |

### llama-bench（上游工具）构建

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON   # 或 -DGGML_METAL=ON
cmake --build build --config Release -j$(nproc)
./build/bin/llama-bench \
  -m model.gguf \
  --ngl 99 \
  -p 128 -n 128 \
  -b 512 \
  -r 5 \
  --output csv
```

---

## 外部 GPU Profiler（终极手段）

当需要 per-kernel 级别的分析时，必须使用平台 profiler。

### NVIDIA — Nsight Systems

```bash
# 抓取完整推理的 timeline
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output report \
  ollama run llama3 <<< "Hello"

# 用 GUI 打开
nsys-ui report.nsys-rep
```

能看到：每个 CUDA kernel（matmul、softmax、rope…）的开始/结束时间、SM 占用、内存拷贝。

### NVIDIA — Nsight Compute

```bash
# 分析单个 kernel 的详细指标
ncu --set full \
  --target-processes all \
  -o profile \
  ./llama-cli -m model.gguf -p "Hello" -n 5
```

能看到：FLOPS/s、内存带宽利用率、roofline 位置、寄存器压力。

### Apple — Instruments

打开 Xcode → Instruments → Metal System Trace，attach 到 `ollama` 进程。

能看到：每个 Metal compute pass 的 GPU 时间线、shader 执行时间、blitter 拷贝。

### AMD — ROCm Profiler

```bash
rocprof --stats -o profile.csv \
  ollama run llama3 <<< "Hello"
```

---

## 典型工作流示例

### 场景 1：快速确认 GPU 是否被用上

```bash
OLLAMA_DEBUG=1 ollama serve 2>&1 | grep -E "offload|VRAM|layer"
```

### 场景 2：比较两种量化类型的速度

```bash
# 下载两个量化版本
ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull llama3.2:3b-instruct-q8_0

./ollama-bench -model llama3.2:3b-instruct-q4_K_M,llama3.2:3b-instruct-q8_0 \
  -epochs 6 -prompt-tokens 256 -max-tokens 100 | tee quant_compare.bench
benchstat -col /name quant_compare.bench
```

### 场景 3：Vulkan 后端 per-op 分析

```bash
OLLAMA_VULKAN=1 GGML_VK_PERF_LOGGER=1 ollama run llama3 <<< "Hello" 2>&1 | grep -A 20 "Vulkan Timings"
```

### 场景 4：CUDA kernel 级别分析

```bash
# 用 Nsight 抓一次短推理
nsys profile --trace=cuda -o llama_profile \
  curl -s http://localhost:11434/api/generate \
    -d '{"model":"llama3","prompt":"Hello","stream":false}'
```

### 场景 5：定位 CPU/GPU 内存溢出

```bash
# ollama-bench 的 VRAM 字段直接显示
./ollama-bench -model llama3 -epochs 1 -format csv | grep "^#"
# 输出：# Model: llama3 | Size: 8589934592 | VRAM: 6442450944
# Size - VRAM = 2147483648 bytes = 2GB 溢出到 CPU
```

### 场景 6：Flash Attention 效果对比

```bash
# 不开 FA
./ollama-bench -model llama3 -epochs 6 > no_fa.bench

# 开 FA
OLLAMA_FLASH_ATTENTION=1 ./ollama-bench -model llama3 -epochs 6 > fa.bench

benchstat no_fa.bench fa.bench
```
