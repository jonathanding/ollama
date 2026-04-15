# CUDA Overlap Benchmark

测试 GPU Copy Engine 与 Streaming Multiprocessor (SM) 的并行可行性，为 MoE 异步 pipeline（Phase 2）提供实测依据。

包含两个独立的 benchmark：

| 工具 | 目的 |
|---|---|
| `cuda_overlap_bench` | 测量 Host-to-Device (H2D) 带宽和 copy/compute 重叠比例 |
| `cuda_hostregister_bench` | 测试 `cudaHostRegister` 在 malloc 和 Windows mmap 内存上的可行性 |

---

## 环境要求

- CUDA Toolkit 12.x 或 13.x（含 cuBLAS）
- CMake 3.18+
- 支持 Ampere 架构的 NVIDIA GPU（compute capability 8.0 或 8.6，即 RTX 30xx / A100）
- Windows 或 Linux

---

## 编译

```bash
cd scripts/cuda-overlap-bench
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

编译成功后 `build/` 目录下会生成两个可执行文件：
- `cuda_overlap_bench`（Linux）或 `cuda_overlap_bench.exe`（Windows）
- `cuda_hostregister_bench`（Linux）或 `cuda_hostregister_bench.exe`（Windows）

---

## 运行

### Benchmark 1：H2D 带宽与重叠测量

```bash
./cuda_overlap_bench        # Linux
.\cuda_overlap_bench.exe    # Windows
```

运行时间约 3–5 分钟（Phase 4/5 各跑 28 次重叠测量）。

### Benchmark 2：cudaHostRegister 可行性测试

```bash
./cuda_hostregister_bench        # Linux
.\cuda_hostregister_bench.exe    # Windows
```

运行时间约 5–8 分钟。需要写入临时文件（`C:\Windows\Temp\` 或 `/tmp/`），运行结束后自动删除。

---

## 测试流程说明

### cuda_overlap_bench 的五个阶段

**Phase 1 — 内存分配与 warm-up**
分配 ~996 MB pageable 内存和 ~996 MB pinned 内存（通过 `cudaMallocHost`），对 pageable 内存执行全量 `memset`，确保所有物理页面都已加载到 RAM，模拟模型已运行一段时间后的稳态（消除 page fault 干扰）。

**Phase 2 — H2D 带宽基准（同步）**
分别对 pageable 来源和 pinned 来源执行 10 次同步 `cudaMemcpy`，计算平均带宽（GB/s）。这是串行基线，不涉及并行。

**Phase 3 — GPU 计算时间基准**
用 `cublasSgemm` 模拟单层 MoE FFN 的三次矩阵乘法（gate + up + down），执行 50 次取平均。矩阵尺寸基于 Qwen3-Coder-Next 80B 的实际参数（M=1024 tokens，K=2048 embedding dim，N=5120）。

> **注意：** benchmark GEMM 时间（~2-3 ms）低于真实单层计算时间（~10.6 ms），因为 ggml scheduler 的调度开销无法被 `cublasSgemm` 重现。VERDICT 部分的 prefill 预测使用 Phase 1 实测的 10.6 ms/层，而非 benchmark GEMM 时间。

**Phase 4 — 重叠测量（pageable 来源）**
模拟 28 层 pipeline。每层：在 copy stream 发射 `cudaMemcpyAsync`（pageable → VRAM Buffer B），同时在 compute stream 发射 `cublasSgemm`（使用 Buffer A）。用 CUDA event 精确记录四个时间戳（copy_start、copy_end、gemm_start、gemm_end），计算每层的实际重叠比例。

**Phase 5 — 重叠测量（pinned 来源）**
与 Phase 4 相同，唯一区别是 H2D copy 来源换为 pinned 内存。

---

### 重叠比例的计算方法

每层记录四个 CUDA event 时间戳（均在同一 GPU 时钟上）：

```
copy stream:   copy_start ─────────────── copy_end
compute stream:      gemm_start ─── gemm_end

wall_ms  = max(copy_end, gemm_end) - min(copy_start, gemm_start)
serial   = copy_ms + gemm_ms
saved    = serial - wall_ms
overlap% = saved / gemm_ms × 100   （夹紧到 [0, 100]）
```

分母用 `gemm_ms` 是因为理论最优情况是 GEMM 完全藏在 copy 等待时间里（saved = gemm_ms = 100%）。

---

### cuda_hostregister_bench 的三组测试

**Reference — cudaMallocHost**
作为最优参考：直接用 `cudaMallocHost` 分配 pinned 内存，测量带宽和重叠比例。

**Source A — malloc + cudaHostRegister**
用 `malloc` 分配 pageable 内存，warm-up 后调用 `cudaHostRegister` 就地注册为 pinned，测量注册是否成功、注册后带宽和重叠比例。

**Source B — Windows mmap + cudaHostRegister**
创建临时文件并用 `CreateFileMapping` / `MapViewOfFile` 映射进虚拟地址空间，模拟 Ollama 加载模型权重的实际路径，调用 `cudaHostRegister` 测试注册可行性。**这是最关键的测试**——如果 Source B 失败，Phase 2 需要走备选的 CPU-side pinned staging buffer 路径。

---

## 结果解读

### cuda_overlap_bench

**H2D 带宽（Phase 2）**

| 期望值 | 含义 |
|---|---|
| pageable ~12–15 GB/s | 正常，CUDA 内部 staging 开销 |
| pinned ~22–25 GB/s | 正常，接近 PCIe 4.0 x16 理论上限 |

**重叠比例（Phase 4/5）**

| 重叠比例 | 含义 | Phase 2 策略 |
|---|---|---|
| pageable ≈ 0% | `cudaMemcpyAsync` 阻塞 CPU 线程，copy 和 compute 完全串行 | 必须解决来源 pinned 问题 |
| pinned ≈ 100% | Copy Engine 和 SM 独立并行，硬件层面无障碍 | Phase 2 双缓冲可行 |
| pinned < 60% | 硬件或驱动存在限制，需进一步调查 | Phase 2 效果存疑 |

**VERDICT 预测**

输出里的 prefill 预测公式：

```
resident 层（MoE 常驻 VRAM）:   N_resident × 10.6 ms
pipeline 层（MoE 需 H2D copy）: N_pipeline × max(copy_ms, 10.6 ms)  [Phase 2 双缓冲]
```

`10.6 ms` 来自 Phase 1 实测的单层计算时间，包含 attention/SSM + MoE FFN + ggml 调度开销。

### cuda_hostregister_bench

**关注点：Source B (mmap) 的注册结果**

```
cudaHostRegister(mmap): SUCCESS   → Phase 2 可用 cudaHostRegister，无需额外内存
cudaHostRegister(mmap): FAILED    → 需要 CPU-side pinned staging buffer（2 GB）
```

注册成功时，检查 Source B 的带宽和重叠比例是否与 Reference 一致（应相差 < 5%）。如果带宽一致但重叠比例低于 Reference，说明 Windows 内存管理器对 mmap 注册有额外限制。

---

## 在不同 GPU 上运行

`CMakeLists.txt` 默认编译 `sm_80`（A100）和 `sm_86`（RTX 3090/3080）。如需在其他 GPU 上运行，修改 `CMakeLists.txt` 中的 `CUDA_ARCHITECTURES`：

| GPU 系列 | compute capability | CUDA_ARCHITECTURES 值 |
|---|---|---|
| RTX 20xx (Turing) | 7.5 | `75` |
| RTX 30xx (Ampere) | 8.6 | `86` |
| RTX 40xx (Ada) | 8.9 | `89` |
| H100 (Hopper) | 9.0 | `90` |

---

## 与 Phase 2 实现的关系

benchmark 结果直接决定 Phase 2 的实现策略：

```
cuda_overlap_bench:
  pinned overlap = 100%
      → Phase 2 双缓冲在硬件层面可行

cuda_hostregister_bench:
  Source B (mmap) SUCCESS + overlap ≈ 100%
      → Phase 2 方案 A: cudaHostRegister（无额外内存开销）
        OLLAMA_MOE_PINNED=1 启用，模型加载后对 CPU-MoE buffer 调用 cudaHostRegister

  Source B FAILED
      → Phase 2 方案 A 备选: CPU-side pinned staging buffer（2 GB cudaMallocHost）
        推理时先 CPU memcpy mmap → pinned staging，再 DMA 到 VRAM
```

完整设计文档：`docs/superpowers/specs/2026-04-15-phase2-moe-async-pipeline-design.md`
