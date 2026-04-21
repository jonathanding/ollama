# MoE Prefetch Dual Staging Buffer — Experiment Report

**日期：** 2026-04-21
**分支：** `feat/moe-split-prefetch`
**诊断起点 commit：** `56315791`（位于 `feat/moe-split-cpu`，Plan B + Variant A selective prefetch on independent stream — 带 Data Race 的不正确实现版本，`!!!` 输出在此版本被观测到）
**开发/优化起点 commit：** `a33239f6`（Plan B 全量拷贝原始实现的末端状态 — 只含基础 Plan B，不含 Variant A；从 Plan B 开始模型输出即损坏。本分支基于此 commit 进行 bug 修复和性能优化）
**硬件：** Intel Arrow Lake 265K · RTX 3090 · PCIe 4.0 x16 · DDR5 6400 MT/s · 128 GB
**模型：** Qwen3-Coder-Next 80B Q4_K_M（48 blocks，31 个 CPU-MoE layer）

---

## 背景

Phase 2 Plan B 以及它的 selective 加强版 Variant A 虽然把 lookahead prefetch 放在独立 Compute Unified Device Architecture (CUDA) Stream 上以获取 Host-to-Device (H2D) 和 compute 的并行性，但实测生成了损坏输出（Fibonacci 生成代码的单元测试输出结果为错误的 `!!!!!!!!!`）。这份实验报告按照 "先定位 → 再避险 → 再优化" 的顺序分三个阶段：

1. **Data Race 验证**：确认跨 Stream write / read `input_cpy` 导致的 ggml-alloc alias race 是错误根因。
2. **Full-Tensor Dual Staging Buffer (DSB) Copy**：用双缓冲 Video Random Access Memory (VRAM) 分离 prefetch 写入和 compute 读取，恢复 copy/compute 并行同时保证正确性。
3. **Selective DSB Copy**：在 DSB 通路上接入 Variant A 的 per-range 拷贝逻辑，让 prefetch 只搬运激活的 expert 字节窗口。

---

## 1. Data Race 验证

### 1.1 现象

`run_single_1k_test.py` 在 Plan B / Variant A 下输出 `!!!!!!!!!` 而非正常 Fibonacci 代码。Plan A（selective 非 prefetch）同 prompt 输出正常。排除了模型权重、路由、tokenizer 差异。

### 1.2 可疑点定位

Plan B/Variant A 的独立 Prefetch Stream 在 `compute N` 正在跑时，为 `split N+1` 的 CPU-MoE 权重发出 `cudaMemcpyAsync(H2D, input, input_cpy, prefetch_stream)`。主 Compute Stream 随后使用 `input_cpy` 作为 `ggml_backend_graph_compute_async` 的输入。两条 Stream 之间除了一个 Event 同步以外没有任何 fence — 典型的跨 Stream 读写场景。

### 1.3 ggml-alloc aliasing

`sched->n_copies = 1`（非 parallel 模式）下，`ggml-alloc` 为了节省 VRAM，会让同一层 3 个连续 split 的 `input_cpy` **指向同一段 VRAM 偏移**（因为它们的 lifetime 在 graph view 下不重叠）。结果 Prefetch Stream 写入的 "split N+1 的 input_cpy" 和主流正在读的 "split N 的 input_cpy" 落在 **同一内存地址**。

### 1.4 Option A（同流 FIFO 序列化）验证

作为对照实验，把 prefetch 改回主 Compute Stream（commit `965da8e0`）：

- **结果：** Fibonacci 输出恢复正常，证明 race 确是根因。
- **副作用：** 跨 Stream 并行性彻底丢失，prefill_mean 从 Variant A 的 ~1720 ms 恶化到 **2003.6 ms**（比 Plan A 1380 ms 差 +624 ms），因为主流必须串行等待 H2D 再做 compute。
- **结论：** 同流序列化能消除 race 但不能作为最终方案 — 需要一种"既能避险又能保留并行"的方法。

---

## 2. Full-Tensor DSB Copy

### 2.1 核心思路

问题的要害不是 "prefetch 在独立 Stream 上" 本身，而是 "prefetch 向 `input_cpy` 写入"。`input_cpy` 属于 ggml-alloc arena，无法避免 aliasing。解决办法：**不要让 prefetch 碰 `input_cpy`**。为 prefetch 专门分配两块 VRAM staging buffer，prefetch 只写 staging，主流再用 Device-to-Device (D2D) 把 staging 拷到 `input_cpy`。

### 2.2 架构

在 CUDA Backend Context 中分配两块（double-buffered）大小为 `max(ggml_nbytes(input))` 的 VRAM 区域 `staging[0]` 和 `staging[1]`，以及对应的两个 CUDA Event `h2d_done[0/1]`（`cudaEventDisableTiming` 模式）。

```
main stream:     [compute A]                    [waitEvent(E[1]) + D2D s[1]->cpy_B][compute B]           [waitEvent(E[0]) + D2D s[0]->cpy_C][compute C]
prefetch stream: [H2D pinned_B -> s[1]]─record E[1]    [H2D pinned_C -> s[0]]─record E[0]
                  └──overlaps compute A────┘            └──overlaps compute B────┘
```

- **slot 选择**：`slot = prefetch_counter & 1`。双缓冲避免 "下一次 H2D 写入" 与 "本次 D2D 读取" 落在同一 buffer 上。
- **Prefetch Stream 上的 H2D**：pinned CPU → `staging[slot]`，记录 `h2d_done[slot]`。
- **Main Stream 上的 D2D**：`cudaStreamWaitEvent(main, h2d_done[slot])` 后 `cudaMemcpyAsync(D2D, main)` — 把 staging 拷到 ggml-alloc 分配的 `input_cpy`。
- **无 alias 冲突**：主流对 `input_cpy` 的读（当前 split compute）和主流对 `input_cpy` 的写（D2D）天然同流 First-In-First-Out (FIFO) 串行。Prefetch Stream 只写 staging，staging 不参与 aliasing。
- **D2D 代价**：~900 GB/s，每 split ~0.5 ms，远小于 H2D 隐藏的 compute 时间。

### 2.3 实现要点

- **`common.cuh`**：为 `ggml_backend_cuda_context` 增加 nested `moe_staging{ void *buffers[2]; cudaEvent_t h2d_done[2]; size_t capacity; }`。
- **`ggml-cuda.cu`**：新增 4 个 proc-address 暴露的 helper — `moe_staging_init / destroy / h2d / d2d`。`init` 幂等、只增长不缩；在析构函数中加安全网清理。
- **`ggml-backend.cpp`**：
  - 在 `compute_splits` 初始化段遍历所有 split 计算 `max_moe_split_size`，调用一次 `staging_init`。
  - 在 fire 块调用 `staging_h2d`，在 hit consumption 处调用 `staging_d2d` 代替原来的 `continue`。
  - 在 `ggml_backend_sched_free` 中调用 `staging_destroy` 主动释放，析构函数兜底。

提交于 `2fa2a2da`。

### 2.4 验证

- **正确性：** `run_single_1k_test.py` 输出完整 Fibonacci 实现 + docstring，与 Plan A 输出等价。
- **性能：** 见 `test/moe-split/plan-b/qwen3-coder-next_moe-split-plan-b-double-staging-buffers-2026-04-21.json`。

| Variant | prefill_mean (ms) | prefill_cv (%) | ttft_mean (ms) | gen_mean (ms) |
|---|---|---|---|---|
| Plan A（selective 无 prefetch） | ~1380 | — | — | — |
| Option A（full-copy 同流） | 2003.6 | 0.77 | 2071.9 | 832.2 |
| **Full-tensor DSB** | **1647.6** | **0.69** | **1717.4** | **831.6** |

相对 Option A 省出 356 ms — overlap 机制生效。距离 Plan A 还差 267 ms，残差来自 full-copy 的绝对 bulk：30 layer × ~996 MiB/layer ≈ 30 GiB 总 H2D，compute 只能遮掩其中约一半。

---

## 3. Selective DSB Copy

### 3.1 动机

Full-Tensor DSB 证明并行机制能工作，但无法击穿 Plan A，原因是 H2D 量太大。Plan A 之所以快，是因为它只 D2H 路由 ids 然后 **按激活 expert 的字节窗口逐段拷贝**（Q4_K_M 下每层约 45-75% 激活，平均每层 ~500 MiB 而非 ~996 MiB）。现在把这个 selective 逻辑嫁接到 DSB 通路上：prefetch 提前干同样的事，省掉 Plan A 串行 H2D 与 compute 的等待。

### 3.2 关键问题：prefetch 需要的 ids 从哪来？

每层 3 个 CPU-MoE split（A = `ffn_gate_exps`，B = `ffn_up_exps`，C = `ffn_down_exps`）共享同一个 ids tensor（router 的输出）。当 **split A 正在 compute** 时（fire 发生在它提交之后），router 已经跑过，ids 在 VRAM 里；Plan A 在 A 的 `input_copy` 块里已经把 ids D2H 下来并建好 `used_ids` bitset。因此 A→B、B→C 的 prefetch 可以直接复用 `used_ids`。**跨层**（比如 layer 2 C → layer 3 A）情况下，layer 3 的 router 还没跑，ids 未知 —— 此时必须回退到 full-tensor 拷贝。

这给出了一个干净的分支条件：**当 `next_split.ids_tensor == prev_ids_tensor` 时走 selective，否则走 full。**

### 3.3 实现要点

- **`ggml-cuda.cu`**：
  - 新增 `struct moe_expert_range { int32_t first_id; int32_t last_id; }`。
  - 新增 `ggml_backend_cuda_moe_staging_h2d_ranges`：在 Prefetch Stream 上为每个 range 发 `cudaMemcpyAsync(H2D, src+offset, staging+offset, bytes)`，全部提交后一次 `cudaEventRecord(h2d_done[slot])`。H2D 把激活 expert 写到 staging 中**与 CPU pinned buffer 相同的偏移**。
  - 新增 `ggml_backend_cuda_moe_staging_d2d_ranges`：主流 `cudaStreamWaitEvent` 后镜像发 N 个 D2D，**使用完全相同的 ranges**。
  - 两个函数都保留 Plan A main path 的 `padding_end` 行为以满足 Matrix-Matrix Quantized (MMQ) kernel 的 padding 假设。

- **`ggml-backend.cpp`**：
  - 为 staging slot 增加 ranges 缓存：`prefetch_ranges[2][1024]`、`prefetch_n_ranges[2]`。Fire 时 `memcpy` 进缓存，consume 时读出并镜像下发。`n_ranges == 0` 表示该 slot 走了 full-tensor 路径。
  - Fire block：`next_ids == prev_ids_tensor` 且 `used_ids` 非空时，用 Plan A 的 merge 算法（同 `copy_experts` 循环）把 bitset 折叠成连续 range 数组，调 `fn_staging_h2d_ranges`。失败或不适用则回退到 `fn_staging_h2d`（full）。
  - Hit consumption：先看 `prefetch_n_ranges[slot]`，>0 → `fn_staging_d2d_ranges(ranges, n)`；==0 → `fn_staging_d2d(全量)`。两条路径完成后都清空 ranges 缓存。
  - 每次 fire 发 DEBUG 日志：`moe prefetch fired: split=X slot=Y mode=full|selective ids=<ptr> bytes=<N> ranges=<M>`。

### 3.4 关键 fix：`prev_ids_tensor` 时序

首次编译后实测 **所有 split 都 fire 为 `mode=full`**，prefill 与 Full-Tensor DSB 持平（1646 ms）。日志里 `next_ids != prev_ids_tensor` 永远为真。根因：Full-Tensor DSB 的 D2D 消费路径在 `input_copy` 块里直接 `continue`，跳过了 ids 解析代码；于是同层 B、C 的 fire block 看到的 `prev_ids_tensor` 还是上一次**非 prefetch-hit split** 的旧值（可能根本不是 MoE split），匹配失败。

**Fix：** 把 ids 解析（`ids_tensor` 取出 + `used_ids` 重建 + `prev_ids_tensor` 赋值）**移到** hit-consume D2D 之前。这样即使 prefetch 命中，在 `continue` 跳过选择拷贝之前，`prev_ids_tensor` 也已经被更新为当前层的 ids。参考 Variant A 的同 pattern 修复 `56315791`。两处改动合并在 commit `e7dd9347`。

### 3.5 验证

- **正确性：** `run_single_1k_test.py` 输出完整 Fibonacci 实现（与 Plan A 和 Full-Tensor DSB 等价）。
- **日志验证（`moe-split-selective-dual-staging-buffer-log.txt`，11 次请求）：**
  - 总 prefetch fire：**1116** 次
  - `mode=selective`：**1488** × （fire+D2D 双行）= 744 selective fires
  - `mode=full`：**744** × 双行 = 372 full fires
  - **2:1 比例**完美吻合设计：每层 3 个 split（A/B/C），A 跨层 full，B/C 同层 selective。
  - 示例片段（layer 2）：
    - `split=5 mode=full bytes=301989888 ranges=1`（288 MiB 全量，A 跨层）
    - `split=6 mode=selective bytes=72548352 ranges=96`（69 MiB，~75% 激活，B 同层复用）
    - `split=7 mode=selective bytes=105799680 ranges=96`（100 MiB，C 同层复用）
    - `split=8 mode=full`（下一层 A，跨层回 full）
- **性能：** 见 `test/moe-split/plan-b/qwen3-coder-next_moe-split-plan-b-selective-double-staging-buffers-2026-04-21.json`。

---

## 汇总对比

| Variant | prefill_mean (ms) | prefill_cv (%) | vs Plan A |
|---|---|---|---|
| Plan A（selective，无 prefetch） | ~1380 | — | 0 |
| Option A（full-copy，同 Compute Stream） | 2003.6 | 0.77 | **+624** |
| Full-Tensor DSB | 1647.6 | 0.69 | +267 |
| **Selective DSB** | **1249.3** | **1.90** | **−131** |

### 解读

- **Selective DSB 同时收获两重增益：** 相对 Full-Tensor DSB 省 398 ms（H2D 总量从 ~30 GiB 缩到 ~13-16 GiB），相对 Plan A 省 ~131 ms（compute/H2D 并行）。
- **击穿 Plan A：** 证明 lookahead prefetch 的价值不是 "把 H2D 隐藏掉就够了"，而是 "隐藏 + 减量" 复合作用。
- **cv 上升到 1.9%：** 高于 Full-Tensor DSB 的 0.69%，来自 selective 路径每 split 的 `cudaMemcpyAsync` 数量（最多 128 次）对 PCIe 调度和 driver queue 深度更敏感。仍远低于 5% 阈值。
- **VRAM：** `staging 2x420 MiB`（单 slot 以最大的 `ffn_down_exps = 420 MiB` 为准），共 840 MiB 固定开销。RTX 3090 24GB 可接受。

---

## 未来方向

- **进一步压低残差：** 当前 A split 必须 full（跨层 router 未知）。如果能在 **layer N 的非 MoE split**（router 自身的 compute）结束后、`ids` 写入 VRAM 的瞬间就把它 D2H 下来，A split 的 prefetch 也能 selective → 预期再省 ~60-80 ms（3 × full 88 MiB × 30 layer = 264 GiB 的 H2D 变成 selective）。工程成本：需要 scheduler 识别 router split 并在其 compute 之后加一次 lazy 的 ids D2H。
- **Parallel mode (`n_copies > 1`)：** 本次所有测试都在非 parallel 模式（`n_copies = 1`）。parallel 模式 arena aliasing 规则不同，可能需要另测。
- **不同激活率的敏感性：** 观测到的激活率在 40-75% 之间浮动（随 prompt 变化），低激活率下 selective 优势更大。建议后续做 selective_ratio × prefill_ms 的 sweep。

---

## 关键 Commit

- `965da8e0` — Option A 正确性基线（prefetch 放回主 Compute Stream，用于验证 race）
- `2fa2a2da` — Full-Tensor DSB 实现
- `e7dd9347` — Selective DSB + `prev_ids_tensor` 时序修复
