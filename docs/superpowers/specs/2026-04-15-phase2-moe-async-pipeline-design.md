# Phase 2 MoE Async Pipeline — 设计文档

**日期：** 2026-04-15
**硬件：** Intel Arrow Lake 265K · Windows 11 · 128 GB DDR5 6400 MT/s · RTX 3090 (24 GB VRAM, PCIe 4.0 x16)
**模型：** Qwen3-Coder-Next 80B Q4_K_M (~52 GB GGUF)
**分支基础：** `moe-split-cpu-phase1`

---

## 1. 背景与动机

Phase 1 实验结论（`docs/perf/2026-04-14-moe-split-phase1-experiment-report.md`）：

- MoE split 实现正确，32 层 MoE expert 权重留在 CPU，17 层常驻 Video RAM (VRAM)
- Prefill 1024 tokens 基线：**2096 ms**
- Phase 1 无统计显著改善，根本原因：op_offload 的 Host-to-Device (H2D) copy 来源是 pageable mmap 内存，带宽仅 ~14.4 GB/s，且 `cudaMemcpyAsync` 阻塞 CPU 线程，无法与 GPU 计算并行

Micro-benchmark 实测结论（`scripts/cuda-overlap-bench/`）：

| 来源 | H2D 带宽 | Copy Engine / SM 重叠比例 |
|---|---|---|
| pageable mmap | 14.4 GB/s | **0%** |
| `cudaMallocHost` | 25.1 GB/s | **100%** |
| mmap + `cudaHostRegister` | 25.1 GB/s | **100%** |

**关键发现：** `cudaHostRegister` 在 Windows + mmap 内存上完全可用，注册后性能与 `cudaMallocHost` 无差别，且不增加物理内存占用（只锁定已有页面）。

---

## 2. 目标

| 阶段 | 方法 | 预测 prefill | vs 基线 |
|---|---|---|---|
| 方案 A | `cudaHostRegister` + 现有同步 op_offload | ~1838 ms | **-12.3%** |
| 方案 B | `cudaHostRegister` + 双缓冲异步 pipeline | ~1498 ms | **-28.6%** |

预测基于：17 resident 层 × 10.6 ms + 32 pipeline 层 × (copy + compute)。
方案 B 的 copy 和 compute 重叠，每层耗时 = max(41.2 ms, 10.6 ms) = 41.2 ms。

---

## 3. 方案 A 设计：cudaHostRegister

### 3.1 环境变量

**`envconfig/config.go`** 新增：

```go
// MoePinned enables cudaHostRegister for CPU-side MoE expert weight buffers.
// When enabled, the CUDA Copy Engine can DMA directly from mmap memory
// without CPU-side staging, increasing H2D bandwidth from ~14 GB/s to ~25 GB/s.
// Requires MoE split to be active (OLLAMA_MOE_GPU_LAYERS != 0).
// Default: false (original pageable behavior).
MoePinned = Bool("OLLAMA_MOE_PINNED")
```

行为矩阵：

| `OLLAMA_MOE_GPU_LAYERS` | `OLLAMA_MOE_PINNED` | 行为 |
|---|---|---|
| `0` | 任意 | 标准路由，不调用 `cudaHostRegister` |
| `-1` 或 `>0` | 未设置（默认） | MoE split 启用，CPU-MoE 保持 pageable |
| `-1` 或 `>0` | `1` 或 `true` | MoE split 启用，CPU-MoE 权重注册为 pinned |

### 3.2 C 桥接层

新增 **`ml/backend/ggml/moe_pinned.h`**：

```c
// moe_pinned.h — CGo bridge for cudaHostRegister / cudaHostUnregister
// Called from ggml.go after MoE CPU buffer allocation.

#pragma once
#include <stdbool.h>
#include <stddef.h>

// Register a host memory region as pinned (page-locked) so that the CUDA
// Copy Engine can DMA directly from it without CPU-side staging.
// Returns true on success. On failure, the memory remains pageable and
// op_offload will fall back to the original CPU-staging path transparently.
bool moe_pinned_register(void * ptr, size_t size);

// Unregister a previously registered region. Must be called before the
// backing buffer is freed to release the page lock.
void moe_pinned_unregister(void * ptr);
```

新增 **`ml/backend/ggml/moe_pinned.cpp`**（约 30 行）：

```cpp
#include "moe_pinned.h"
#include "ggml/src/ggml-cuda/ggml-cuda.h"  // ggml_backend_cuda_register_host_buffer

bool moe_pinned_register(void * ptr, size_t size) {
    return ggml_backend_cuda_register_host_buffer(ptr, size);
}

void moe_pinned_unregister(void * ptr) {
    ggml_backend_cuda_unregister_host_buffer(ptr);
}
```

`ggml_backend_cuda_register_host_buffer` 已在 `ggml-cuda.cu:4291` 实现，内部调用
`cudaHostRegister(buffer, size, cudaHostRegisterDefault)`，失败时打印警告并返回 false（不 crash）。

### 3.3 Go 层改动：`ml/backend/ggml/ggml.go`

**步骤 1：收集 CPU-MoE buffer 指针**

在 tensor 路由循环（当前约第 400 行）中，当一个 MoE expert tensor 被分配到 CPU buffer type 时，记录该 buffer type 到一个集合：

```go
cpuMoEBufTypes := make(map[C.ggml_backend_buffer_type_t]struct{})

// 在 tensor 路由的 default 分支:
if layerIndex >= 0 {
    bts := layers[layerIndex].bts
    if isMoEExpertTensor(t.Name) && len(params.MoEGPULayers) > 0 {
        bts = moeLayers[layerIndex].bts
        // 如果这层 MoE 在 CPU，记录其 buffer type
        if moeLayers[layerIndex].d == cpuDeviceBufferType.d {
            cpuMoEBufTypes[bts[0]] = struct{}{}
        }
    }
    createTensor(tensor{source: t}, bts, layerIndex)
}
```

**步骤 2：buffer 分配后注册**

在现有的 buffer 分配循环之后（当前约第 462 行）：

```go
// 已有代码:
for bt, c := range ctxs {
    b := C.ggml_backend_alloc_ctx_tensors_from_buft(c, bt)
    bbs[c] = b
}

// 新增: 如果 OLLAMA_MOE_PINNED 且 MoE split 启用，注册 CPU-MoE buffers
var pinnedBuffers []unsafe.Pointer
if params.AllocMemory && envconfig.MoePinned() && len(params.MoEGPULayers) > 0 {
    for bt, c := range ctxs {
        if _, isCPUMoE := cpuMoEBufTypes[bt]; !isCPUMoE {
            continue
        }
        b := bbs[c]
        ptr  := C.ggml_backend_buffer_get_base(b)
        size := C.ggml_backend_buffer_get_size(b)
        if C.moe_pinned_register(ptr, size) {
            pinnedBuffers = append(pinnedBuffers, unsafe.Pointer(ptr))
            slog.Info("moe pinned: registered CPU-MoE buffer",
                "ptr", ptr, "size", format.HumanBytes2(uint64(size)))
        } else {
            slog.Warn("moe pinned: cudaHostRegister failed, falling back to pageable",
                "ptr", ptr, "size", format.HumanBytes2(uint64(size)))
        }
    }
}
```

**步骤 3：cleanup 时 unregister**

在返回的 `*Backend` 结构体的 `Close()` 方法里（或等效的 finalizer），在释放 buffer 之前调用：

```go
for _, ptr := range pinnedBuffers {
    C.moe_pinned_unregister(ptr)
}
```

### 3.4 改动文件汇总

| 文件 | 改动类型 | 改动量 |
|---|---|---|
| `envconfig/config.go` | 新增环境变量 | ~5 行 |
| `ml/backend/ggml/moe_pinned.h` | 新增文件 | ~20 行 |
| `ml/backend/ggml/moe_pinned.cpp` | 新增文件 | ~15 行 |
| `ml/backend/ggml/ggml.go` | 注册/注销逻辑 | ~30 行 |

---

## 4. 方案 B 设计：双缓冲异步 Pipeline

> 方案 B 在方案 A 实测收益确认后实施。本节为预设计，实施前需根据方案 A 实测数据复核。

### 4.1 核心机制

在 VRAM 里分配 2 个 ~1 GB 的 staging buffer（Buffer A / Buffer B）。
GPU 从 Buffer A 计算第 N 层时，copy stream 把第 N+1 层写入 Buffer B，交替使用。

```
timeline (每格 = 10ms):

GPU compute: [Layer 17][Layer 18][Layer 19][Layer 20]...
copy stream:       [L18→B]  [L19→A]  [L20→B]  [L21→A]...

copy (41ms) >> compute (10.6ms) → pipeline 受 PCIe 带宽限制，每层 41ms
```

### 4.2 主要改动点

方案 B 的核心改动在 `ggml-backend.cpp` 的
`ggml_backend_sched_compute_splits()` 函数，在现有的 split 执行循环里增加跨层预取逻辑。具体设计在方案 A 实测后单独展开。

### 4.3 成功标准

| 指标 | 目标 |
|---|---|
| Nsight Systems 确认 copy/compute 重叠比例 | ≥ 70% |
| Prefill 1024 tokens | ≤ 1550 ms |

---

## 5. 方案 A 成功标准

| 指标 | 目标 |
|---|---|
| 日志确认 "moe pinned: registered" | 32 个 CPU-MoE buffer |
| Prefill 1024 tokens | ≤ 1900 ms（预测 1838ms，留 3% 余量） |
| Decode gen_tps | 不低于基线（方案 A 不改变 decode 路径） |

---

## 6. 风险

| 风险 | 可能性 | 影响 | 缓解 |
|---|---|---|---|
| 某些 VRAM 不足时 `cudaHostRegister` 静默失败 | 低（本机 128 GB DDR5） | 回退到 pageable，无崩溃 | 日志已区分 registered vs fallback |
| Windows 版本差异导致注册失败 | 低（CUDA 13.2 + Win11 已验证） | 同上 | `OLLAMA_MOE_PINNED` 默认关闭，不影响其他用户 |
| 方案 A 实测未达 -12% | 中（ggml 调度开销可能抵消） | 重新评估方案 B 必要性 | 方案 A 失败不影响代码正确性，可直接回退 |

---

## 7. 相关文档

- Phase 0 实验结果：`docs/perf/2026-04-10-kv-cache-quantization-evaluation.md`
- Phase 1 实验结果：`docs/perf/2026-04-14-moe-split-phase1-experiment-report.md`
- llama.cpp 对比分析：`docs/perf/2026-04-15-llama-cpp-vs-ollama-moe-split-analysis.md`
- MoE split 原始设计：`docs/perf/2026-04-10-moe-split-prefill-optimization.md`
- Micro-benchmark 代码：`scripts/cuda-overlap-bench/`
