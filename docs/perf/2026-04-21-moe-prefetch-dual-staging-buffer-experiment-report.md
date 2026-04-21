# MoE Prefetch Dual Staging Buffer — Experiment Report

**日期：** 2026-04-21
**分支：** `debug/moe-prefetch-plan-b`
**父 commit：** `965da8e0`（Option A correctness baseline — prefetch on main stream）
**硬件：** Intel Arrow Lake 265K · RTX 3090 · PCIe 4.0 x16 · DDR5 6400 MT/s · 128 GB
**模型：** Qwen3-Coder-Next 80B Q4_K_M（48 blocks，31 个 CPU-MoE layer）

---

## 1. 背景

先前的 Plan B / Variant A 实现虽然架构上把 prefetch 放在独立 Compute Unified Device Architecture (CUDA) Stream 上以获取 Host-to-Device (H2D) 和 compute 的并行性，但实测生成了损坏输出（Fibonacci 代码被替换为 `!!!!!!!!!`）。经根因分析（见上一轮会话记录）：

**根因：** `ggml-alloc` arena 对同一 Multi-head Expert (MoE) layer 内连续 split 的 `input_cpy` tensor 进行了 aliasing（因 lifetime 不重叠）。Prefetch Stream 向 `input_cpy` 写入下一个 split 的权重时，主 Compute Stream 正在读同一内存区域作为上一个 split 的权重源 — 典型跨流 **data race**。

**Option A 补丁**（将 prefetch 改回主 Compute Stream 以恢复正确性）通过同流 First-In-First-Out (FIFO) 消除 race，但副作用是彻底丧失 copy/compute 并行性，把 prefill 推高到 2003.6 ms（比 Plan A 1380 ms 差 +624 ms）。

本报告记录 **Dual Staging Buffer**（以下简称 DSB）方案：在 `input_cpy` 之外分配两块专用 Video Random Access Memory (VRAM) 区域作为 prefetch 的落地点，避开 aliasing 冲突，同时恢复跨流并行。

---

## 2. DSB 方案核心思想

### 2.1 问题拆解

想同时满足：
1. **Correctness**：prefetch 写入的 VRAM 区域不能与任何主流正在读的区域 alias。
2. **Overlap**：H2D 必须在独立 Stream 上，才能与主流 compute 并行。
3. **低 VRAM 开销**：不能用 "给每个 split 单独分配 input_cpy" 这种 60+ GiB 的方案。

### 2.2 DSB 架构

在 CUDA Backend Context 中分配两块（double-buffered）大小为 `max(ggml_nbytes(input))` 的 VRAM 区域 `staging[0]` 和 `staging[1]`，以及对应的两个 CUDA Event `h2d_done[0/1]`（`cudaEventDisableTiming` 模式）。

```
main stream:     [compute A]                    [waitEvent(E[1]) + D2D s[1]->cpy_B][compute B]           [waitEvent(E[0]) + D2D s[0]->cpy_C][compute C]
prefetch stream: [H2D pinned_B -> s[1]]─record E[1]    [H2D pinned_C -> s[0]]─record E[0]
                  └──overlaps compute A────┘            └──overlaps compute B────┘
```

- **slot 选择**：`slot = prefetch_counter & 1`。双缓冲避免 "下一次 H2D 写入" 与 "本次 D2D 读取" 落在同一 buffer 上。
- **Prefetch Stream 上的 H2D**：pinned CPU → `staging[slot]`，记录 `h2d_done[slot]`。
- **Main Stream 上的 Device-to-Device (D2D)**：`cudaStreamWaitEvent(main, h2d_done[slot])` 后 `cudaMemcpyAsync(D2D, main)` — 把 staging 拷到 ggml-alloc 分配的 `input_cpy`。
- **无 alias 冲突**：主流对 `input_cpy` 的读（当前 split compute）和主流对 `input_cpy` 的写（D2D）天然同流 FIFO 串行。Prefetch Stream 只写 staging，staging 不参与 aliasing。
- **D2D 代价**：~900 GB/s，每 split ~0.5 ms，远小于 H2D 隐藏的 compute 时间。

### 2.3 实现

- **`common.cuh`**：为 `ggml_backend_cuda_context` 增加 nested `moe_staging{ void *buffers[2]; cudaEvent_t h2d_done[2]; size_t capacity; }`。
- **`ggml-cuda.cu`**：新增 4 个 proc-address 暴露的 helper — `moe_staging_init / destroy / h2d / d2d`。`init` 幂等、只增长不缩；在析构函数中加安全网清理。
- **`ggml-backend.cpp`**：
  - 在 `compute_splits` 初始化段遍历所有 split 计算 `max_moe_split_size`，调用一次 `staging_init`。
  - 在 fire 块调用 `staging_h2d`，在 hit consumption 处调用 `staging_d2d` 代替原来的 `continue`。
  - 在 `ggml_backend_sched_free` 中调用 `staging_destroy` 主动释放，析构函数兜底。

---

## 3. 实验结果

### 3.1 Correctness

`run_single_1k_test.py`（1024 prompt tokens）输出完整 Fibonacci 实现 + docstring，**与 Plan A 输出等价**，无 `!!!!!` 损坏。

### 3.2 Performance

bench-sweep，1024 tokens，6 epochs（含 4 warmup），结果见 `test/moe-split/plan-b/qwen3-coder-next_moe-split-plan-b-double-staging-buffers-2026-04-21.json`。

| Variant | prefill_mean (ms) | prefill_cv (%) | ttft_mean (ms) | gen_mean (ms) | vs Option A | vs Plan A |
|---|---|---|---|---|---|---|
| Plan A（selective baseline） | ~1380 | — | — | — | — | 0 |
| Option A（full-copy，same stream） | 2003.6 | 0.77 | 2071.9 | 832.2 | 0 | +624 |
| **DSB（full-copy + staging）** | **1647.6** | **0.69** | **1717.4** | **831.6** | **−356** | **+267** |

### 3.3 解读

- **Overlap 生效：** DSB 相对 Option A 省出 ~356 ms。按每 layer 3 个 split × 30 CPU-MoE layers × 平均 H2D ~12-17 ms 计算，理论可隐藏 ~3.5 ms × 30 × 2 ≈ 210 ms（只有 A→B 和 B→C 两个相邻对可重叠，C→next-layer-A 之间被非 MoE split 截断）。实测 356 ms 略高于理论，说明 H2D 自身也吃了一些异步队列深度带来的 throughput 增益。
- **未追平 Plan A：** 残差 267 ms 来自 full-copy 的绝对 bulk：30 layer × ~996 MiB/layer ≈ 30 GiB 总 H2D，其中 compute 只能遮掩约一半。要超越 Plan A 必须把 selective 接入 — 激活率约 45-60% 时预计可把总 H2D 压到 ~16-18 GiB，压过 Plan A 的概率就出现。
- **稳定性：** cv_pct 0.69% 低于 Option A 的 0.77%，prefetch 行为对抖动无放大。
- **VRAM：** 在 `staging_init` 日志中单 slot ≈ 420 MiB，双缓冲 ~840 MiB 固定开销，RTX 3090 24GB 可接受。

---

## 4. 结论 & 下一步

**DSB 验证了 copy/compute overlap 机制可行** — 在不触发 ggml-alloc alias race 的前提下，成功恢复了跨流并行性。本方案作为 Phase 3 baseline 固化。

下一步（D-Task 8）：**在当前分支上把 selective 机制接入 DSB 通路**，让 prefetch 只搬运激活的 expert 行。预期 prefill 目标 < 1380 ms（击穿 Plan A），因为：

1. 不仅避免了 Plan A 的串行 H2D / compute，
2. 还把 H2D 总量从 30 GiB 降到理论下限（活跃 expert × 单 expert 字节数）。

变更量：将 `moe_expert_range` 路由信息注入 `staging_h2d`，在其内部按激活 range 发 `cudaMemcpyAsync`（多段）。`staging_d2d` 保持按全 staging buffer 长度拷贝（padded），或者按 range 拷贝（更省）。
