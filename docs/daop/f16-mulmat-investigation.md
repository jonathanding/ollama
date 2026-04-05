# Estimate 18x 偏差 — 完整调查报告

> daop-estimate qwen3:1.7b 预测 1338ms/tok，实际 75ms/tok，偏差 18x。
> 本文档记录根本原因分析的完整数据、推理过程和结论。

## 0. 决定性证据：GPU Timestamp 验证 (2026-04-05)

通过 `GGML_VK_PERF_LOGGER=1` 启用 ggml-vulkan 内置的 GPU timestamp query，
获得了每个 op 在 GPU 上的**真实执行时间**（排除所有 CPU 端 dispatch overhead）。

### Decode (N=1) GPU Timestamp 数据

```
Vulkan Timings (qwen3:1.7b decode, steady state):
  ADD:                                     1 x    7.5 us
  CPY:                                     1 x    3.5 us
  FLASH_ATTN_EXT (128,16,1,1)...:        28 x  235.0 us
  GET_ROWS:                                2 x    4.9 us
  GLU:                                    28 x    4.7 us
  MUL_MAT_VEC f16  m=1024 n=1 k=2048:   28 x   83.3 us  (50.3 GFLOPS/s)
  MUL_MAT_VEC q4_K m=1024 n=1 k=2048:   28 x   58.8 us  (71.3 GFLOPS/s)
  MUL_MAT_VEC q4_K m=2048 n=1 k=2048:   29 x  111.6 us  (75.2 GFLOPS/s)
  MUL_MAT_VEC q4_K m=2048 n=1 k=6144:   14 x  289.3 us  (87.0 GFLOPS/s)
  MUL_MAT_VEC q4_K m=6144 n=1 k=2048:   56 x  351.7 us  (71.5 GFLOPS/s)
  MUL_MAT_VEC q6_K m=2048 n=1 k=6144:   14 x  267.3 us  (94.1 GFLOPS/s)
  MUL_MAT_VEC q6_K m=151936 n=1 k=2048:  1 x 7935.0 us  (78.4 GFLOPS/s)
  RMS_NORM_MUL (2048,1,1,1):             57 x   12.4 us
  RMS_NORM_MUL_ROPE (128,16,1,1):        28 x   12.0 us
  RMS_NORM_MUL_ROPE (128,8,1,1):         28 x    6.8 us
  SET_ROWS:                               56 x    4.4 us
  Total time: 54310 us  (~54ms GPU)
```

### GPU 实际 vs 我们的预测

| Op | Count | GPU μs/op | 我们预测 μs/op | 偏差倍数 | 原因 |
|----|-------|-----------|---------------|---------|------|
| f16 MUL_MAT | 28 | **83** | 28,312 | **341x** | benchmark 测 MUL_MAT, 推理用 MUL_MAT_VEC; 加上 dispatch overhead |
| RMS_NORM+MUL | 57 | **12** (fused) | 1,303+2,046 | **279x** | op fusion: 推理中两个 op 合成一个 kernel; dispatch overhead |
| ADD | 1 | **8** | 1,102 | **138x** | dispatch overhead + 推理中大部分 ADD 被融合到 MUL_MAT_ADD |
| FLASH_ATTN_EXT | 28 | **235** | uncalibrated | — | — |
| GLU | 28 | **5** | uncalibrated | — | — |
| SET_ROWS | 56 | **4** | uncalibrated | — | — |

### 根本原因确认

benchmark 的 wall-clock 测量与 GPU 真实时间的差异来自**四个独立问题**：

1. **Dispatch overhead (~1ms per op)**：benchmark 的 `ctx.Compute(out)` 对每个 op 执行完整的
   command buffer 创建 → submit → fence wait 流程，绝大部分时间是 CPU 端等待
2. **MUL_MAT vs MUL_MAT_VEC**：N=1 时 ggml 自动选择 `MUL_MAT_VEC`（向量-矩阵乘法专用 kernel），
   但 benchmark 测的是通用 `MUL_MAT` kernel。f16 MUL_MAT_VEC 83μs vs MUL_MAT 的 25,978μs
3. **Op fusion**：ggml Vulkan 自动融合相邻 op（RMS_NORM+MUL → 一个 kernel，RMS_NORM+MUL+ROPE → 一个 kernel，
   MUL_MAT+ADD → MUL_MAT_ADD）。我们的 estimate 把每个 op 独立计算
4. **f16 shader 并不慢**：GPU timestamp 显示 f16 MUL_MAT_VEC 83μs，达到 50 GFLOPS/s，
   效率合理。之前以为 f16 shader 极慢是错误的——25ms 几乎全是 dispatch overhead

## 1. 问题总览

| 指标 | 预测 | 实际 | 偏差 |
|------|------|------|------|
| Decode latency | 1338ms/tok | 75ms/tok | 17.8x |
| Tokens/sec | 1 tok/s | 13.29 tok/s | — |

预测误差的构成：

| 来源 | 预测贡献 | 占比 | 误差性质 |
|------|---------|------|---------|
| f16 MUL_MAT 28x | 793ms | 59% | 主要问题 |
| 1D ops (MUL+RMS_NORM+ADD+ROPE) | 468ms | 35% | 次要问题 |
| 其他 (q4_K MUL_MAT, FLASH_ATTN 等) | 77ms | 6% | 可能合理 |

## 2. 根本原因：三个独立问题

### 问题 A: f16 MUL_MAT — ~~Intel iGPU f16 shader 确实极慢~~ **已被 GPU timestamp 推翻**

> **更正 (2026-04-05)**：GPU timestamp 显示 f16 MUL_MAT_VEC 在 GPU 上只需 **83μs**（50 GFLOPS/s），
> 效率完全正常。之前的 25,978μs 测量值中，**99.7% 是 CPU 端 dispatch overhead**。
> 之前 "减去 1ms dispatch 后仍然 25ms" 的推理是错误的——dispatch overhead 对 f16 MUL_MAT
> 远不止 1ms（可能 ~25ms），而且 N=1 时推理实际走 MUL_MAT_VEC 不是 MUL_MAT。

**f16 benchmark wall-clock 25,978μs 的分解：**
- GPU 实际执行时间（MUL_MAT kernel at M=K=4096）：未知（需要在 benchmark 中启用 GPU timestamp 验证）
- Vulkan dispatch overhead（command buffer + submit + fence）：占绝大部分
- 可能还包含 MUL_MAT（通用 kernel）vs MUL_MAT_VEC（N=1 专用 kernel）的性能差异

**之前的证据链（已失效）：**

f32 和 f16 在 M=K=4096, N=1 的 wall-clock 对比：

| | f16 | f32 |
|---|---|---|
| wall-clock 测量 | 25,978μs | 3,224μs |
| 减去 ~1ms dispatch | ~24,978μs | ~2,224μs |
| 减 dispatch 后 BW 效率 | 3.5% | 79% |

这个分析假设 dispatch overhead ≈ 1ms（基于 1D ops 的测量），但这个假设对 MUL_MAT 可能不成立。
MUL_MAT 的 Vulkan dispatch 可能涉及更多的 descriptor set allocation、pipeline binding、
memory barrier 等步骤，overhead 可能远大于 1D ops。

**关键教训**：不能用一种 op 的 dispatch overhead 推断另一种 op 的 dispatch overhead。

### 问题 B: 1D ops — benchmark 测量值 99.9% 是 dispatch overhead

**决定性证据：profile.json 中 MUL(f32, Vulkan) 的参考曲线**

| 元素数 | 数据量 | 测量值 (μs) | stddev |
|--------|--------|------------|--------|
| 1,024 | 4KB | 1,073 | 22 |
| 11,026 | 43KB | 1,078 | 4 |
| 16,384 | 64KB | 1,096 | 56 |
| 25,583 | 100KB | 1,068 | 10 |
| **262,144** | **1MB** | **1,068** | 8 |
| 1,278,229 | 4.9MB | 2,170 | 295 |
| 2,822,557 | 10.8MB | 4,943 | 138 |
| 67,108,864 | 256MB | 74,483 | 2,135 |

**从 1K 到 262K 元素（数据量 256 倍变化），延迟完全不变（~1073μs）。** 这不是推测——这是 profile 里的实测数据。1MB 数据的 MUL 和 4KB 数据的 MUL 花一样的时间。

延迟从 ~500K 元素开始增长，说明：
- **< 500K 元素**：100% 是 per-dispatch 固定开销（~1ms）
- **> 500K 元素**：开始看到真正的 GPU 计算/传输时间

Decode 时的 1D ops shape 只有 2048 和 1024 — 深深处于"平坦区"。所有 ~1ms 的预测都是纯 dispatch overhead。

**Vulkan 执行路径确认**（来自 ggml-vulkan.cpp 代码分析）：

Benchmark 路径（单 op graph）:
```
vkBeginCommandBuffer → vkCmdDispatch（1个op）→ vkEndCommandBuffer 
→ vkQueueSubmit → vkWaitForFences
```

实际推理路径:
```
vkBeginCommandBuffer → vkCmdDispatch × ~100个ops → vkEndCommandBuffer 
→ vkQueueSubmit → vkWaitForFences
→ 重复（每 ~100 个 node 或 ~100MB matmul 数据提交一批）
```

实际推理每 ~100 个 node 做一次 submit，所以 per-op dispatch overhead 被稀释 ~100 倍。

### 问题 C: Benchmark 数据质量 — 严重的离群值污染插值

MUL 参考曲线中的离群值：

| 元素数 | 延迟 (μs) | stddev (μs) | CV | 判断 |
|--------|-----------|-------------|-----|------|
| 4,993 | 4,693 | **5,060** | **108%** | 数据无效 |
| 24,347 | 15,959 | 772 | 4.8% | 可疑 |
| 118,715 | 4,607 | **4,068** | **88%** | 数据无效 |
| 578,861 | 12,410 | **8,689** | **70%** | 数据无效 |

这些点的 stddev > latency，说明测量完全不可靠。但 adaptive sampler 仍然接受了它们。

**对估计的影响**：MUL 在 N=2048 时做 log-log 插值，使用 N=1024 (1073μs) 和 N=4993 (4693μs) 两个邻居。N=4993 是坏点（CV=108%），把 N=2048 的插值从 ~1073 拉到了 **2046μs**（多算了 ~1ms）。

实际上从数据看，N=2048 应该是 ~1073μs（和 N=1024 到 N=262144 一样在平坦区内）。

## 3. 综合影响分析

如果修正上述三个问题，预测会变成什么样？

**乐观估计（假设所有 overhead 和质量问题都修正后）：**

| Op 类别 | 当前预测 | 修正后估计 | 说明 |
|---------|---------|-----------|------|
| f16 MUL_MAT 28x | 793ms | **?ms** | 取决于 f16 shader 真实 GPU 时间 |
| MUL 113x | 204ms | **< 1ms** | 2048 元素 GPU 计算时间 ≈ 0 |
| RMS_NORM 113x | 142ms | **< 1ms** | 同上 |
| ADD 56x | 62ms | **< 1ms** | 同上 |
| ROPE 56x | 61ms | **< 1ms** | 同上 |
| q4_K MUL_MAT ~154x | (prefill only) | — | decode 主要贡献 |
| FLASH_ATTN_EXT 28x | (小量) | — | — |

1D ops 从 ~468ms 降到 ≈ 0，这消除了 35% 的误差。

f16 MUL_MAT 的 "真实 GPU 时间" 是未知数。两种可能：
- **如果 f16 shader 在 GPU 端也慢**（3.5% BW 效率是真的）：28 × 3143μs = 88ms
- **如果 f16 shader GPU 端正常**（类似 f32 的 79% 效率）：28 × 139μs = 3.9ms

这个差异巨大，需要用 GPU timestamp 验证。

## 4. 关键发现：ggml-vulkan 已有 GPU Timestamp 支持

在 `ggml-vulkan.cpp` 中发现：

- **Line 13156-13180**：当 `vk_perf_logger_enabled=true` 时，创建 `vk::QueryType::eTimestamp` query pool
- **Line 13304**：在 op 执行前后写入 `writeTimestamp()`
- **Line 13345-13349**：从 query pool 读取结果，计算 per-op GPU 时间：
  ```cpp
  (timestamps[i] - timestamps[i-1]) * device.properties.limits.timestampPeriod
  ```

**这正是我们需要的。** GPU timestamp 直接给出 op 在 GPU 上的真实执行时间，完全排除 CPU 端的 command buffer 创建、queue submit、fence wait 等开销。

## 5. 推荐方案（基于 GPU Timestamp 验证结果更新）

GPU timestamp 验证揭示了 benchmark 和 estimate 需要解决的四个问题。按优先级排序：

### P0: Benchmark 需要测 GPU 时间，不是 wall-clock 时间

**现状**：benchmark 用 `time.Since(ctx.Compute(out))` 测量，包含 Vulkan dispatch overhead (~1ms for 1D ops)。
**目标**：测 GPU 端实际执行时间。

**方案**：
- `GGML_VK_PERF_LOGGER` 只输出到 stderr，无法从 Go 获取数据
- 需要在 ggml C 层新增 API，允许外部读取 per-op GPU timestamp
- 或者：benchmark 改为构建多 op graph（稀释 dispatch overhead），用 wall-clock 总时间 / op 数
- 或者：用 eval callback 在 op 前后记录时间（但这仍然是 CPU 端时间）

### P0: 处理 MUL_MAT vs MUL_MAT_VEC 的差异

**现状**：benchmark 测的是 `MUL_MAT`（通用矩阵乘法），但 N=1 时推理实际用 `MUL_MAT_VEC`（向量-矩阵乘法）。
两者性能差异巨大：f16 MUL_MAT = 25,978μs vs f16 MUL_MAT_VEC = 83μs (at similar shape)。

**方案**：benchmark 需要同时测量 MUL_MAT 和 MUL_MAT_VEC，estimate 根据 N 选择正确的 kernel。

### P1: 处理 Op Fusion

**现状**：estimate 把每个 op 独立计算，但推理中 ggml Vulkan 会自动融合：
- RMS_NORM + MUL → `RMS_NORM_MUL` (12μs, 而非 1303+2046=3349μs)
- RMS_NORM + MUL + ROPE → `RMS_NORM_MUL_ROPE`
- MUL_MAT + ADD → `MUL_MAT_ADD`

**方案**：estimate 需要识别可融合的 op pattern，并用融合后的性能数据预测。

### P2: 补充未校准 op

**现状**：SET_ROWS, GLU, CPY 未校准。GPU timestamp 显示它们都 < 5μs/op，影响不大。

**方案**：在 benchmark 中新增这些 op 的测量，或直接用常量估计（< 10μs）。

## 6. 数据附录

### A. 硬件

```
Intel iGPU (Intel Graphics), Vulkan
peak_tops_f16: 54.9 GFLOPS
peak_tops_f32: 58.6 GFLOPS
peak_bw: 36.5 GB/s
```

### B. f16 效率常量（来自 profile.json）

```
MUL_MAT_f16: compute_eff=0.876, bw_eff=0.035, overhead=25059μs
MUL_MAT_f32: compute_eff=1.009, bw_eff=0.570, overhead=2052μs
```

注意：f16 的 bw_eff 和 overhead 来自单个数据点（N=1）。

### C. Decode 图 f16 MUL_MAT 详情

28 个完全相同的 op：
- Name: node_13, node_50, ..., node_986（每 36 个 node 一个）
- InputShapes: `[[2048, 1024, 1, 1], [2048, 1, 1, 1]]`
- Extracted: M=1024, K=2048, N=1
- Weight: 2048×1024 f16 = 4.0MB
- 每个预测 28,312μs

### D. Vulkan 执行架构

```
ggml_backend_vk_graph_compute (line 13127)
  ├── 每 ~100 nodes 或 ~100MB matmul 做一次 batch submit
  ├── ggml_vk_build_graph (line 11727)
  │     ├── ggml_vk_ctx_begin → vkBeginCommandBuffer
  │     ├── ggml_vk_mul_mat / ggml_vk_op_* → ggml_vk_dispatch_pipeline
  │     │     └── vkCmdBindPipeline + vkCmdDispatch（记录到 command buffer）
  │     └── 当 submit=true: ggml_vk_ctx_end → vkEndCommandBuffer
  ├── ggml_vk_compute_forward → ggml_vk_submit → vkQueueSubmit
  └── ggml_vk_synchronize → vkWaitForFences

Benchmark 每个 op 独立走完整流程 → dispatch overhead 不被分摊
实际推理 ~100 ops 共享一个 submit → dispatch overhead 稀释 ~100x
```

### E. GPU Timestamp 已有实现

```
ggml-vulkan.cpp:
  Line 13156-13180: vk_perf_logger_enabled → QueryPool(eTimestamp)
  Line 13304: writeTimestamp() per op
  Line 13345-13349: getQueryPoolResults() → per-op GPU time
```

## 7. 修复历史

| 日期 | 修复 | 效果 |
|------|------|------|
| Session 11 | ensureLibraryPath | cpu_only → full_offload |
| Session 11 | props.library | Vulkan0 → Vulkan |
| Session 11 | FlashAttention 启用 | f16 MUL_MAT 84x→28x, 2808→1338ms |
| Session 11 | FLASH_ATTN_EXT dtype 匹配 | 不再 uncalibrated |

## 8. 跨 Backend 对比调研 (2026-04-05)

调研了 CUDA 和 CPU backend，确认四个根本原因的普适性：

| 问题 | Vulkan | CUDA | CPU |
|------|--------|------|-----|
| Dispatch overhead | ~1ms/op（严重） | 很低（CUDA Graphs 批量执行） | ~0.001ms（可忽略） |
| MUL_MAT_VEC | 有，341x 差异 | 有（cuBLAS GEMV vs GEMM） | 差异小（同一代码路径，BLAS 自动选） |
| Op fusion | 14 种规则 | 有（`GGML_CUDA_DISABLE_FUSION`） | **无** |
| Wall-clock 准确性 | 低（需 GPU timestamp） | 中等 | **高**（直接可用） |

### CPU Backend
- 无 dispatch overhead：直接函数调用 + optional thread barrier (~100-500ns)
- 无 op fusion：每个 op 独立执行
- MUL_MAT 有 BLAS (llamafile_sgemm) 自动选 GEMV/GEMM，但无独立 shader
- **现有 wall-clock benchmark 足够准确**

### CUDA Backend
- CUDA Graphs 大幅降低 dispatch overhead（单次 `cudaGraphLaunch` 执行整图）
- 有 op fusion（`ggml_cuda_mm_fusion_args_host`），可用 `GGML_CUDA_DISABLE_FUSION` 控制
- 无现成 perf logger（无 `GGML_CUDA_PERF_LOGGER`），需手动 CUDA events 或 Nsight
- **以后再做，优先解决 Vulkan**

### Op Fusion 技术细节（Vulkan）

融合发生在 `ggml_backend_vk_graph_compute()` 的 dispatch 阶段（line 13222-13281），
**无法在 `PlanGraph()` 阶段获取融合图**。融合依赖运行时状态：
- 设备能力标志（`ctx->device->add_rms_fusion`）
- Buffer 对齐（`get_misalign_bytes()`）
- 预分配内存大小
- 前序融合状态（`ctx->num_additional_fused_ops`）

**常见融合规则（LLM 推理）**：

| Pattern | 融合后 | 约束 |
|---------|--------|------|
| MUL_MAT + ADD | MUL_MAT_ADD | mat-vec only, types match |
| RMS_NORM + MUL | RMS_NORM_MUL | all f32, contiguous, no broadcast |
| RMS_NORM + MUL + ROPE | RMS_NORM_MUL_ROPE | 额外 ROPE 约束 |
| ADD + ADD + ... | MULTI_ADD | same shape, f32, no misalignment |
| ADD + RMS_NORM | (partial) | 需 prealloc buffer, single row |

**结论**：融合图不可提前获取，改用 Go 侧 pattern-based 融合模拟（规则有限且稳定）。

### MUL_MAT_VEC 技术细节（Vulkan）

选择发生在 `ggml_vk_mul_mat()`（line 7439）：
```cpp
if (dst->ne[1] <= mul_mat_vec_max_cols && ...) {  // mul_mat_vec_max_cols = 8
    ggml_vk_mul_mat_vec_q_f16(ctx, ...);  // 专用 VEC kernel
} else {
    ggml_vk_mul_mat_q_f16(ctx, ...);       // 通用 MUL_MAT kernel
}
```

- GGML 图中只有 `MUL_MAT`，VEC 变体在图层不可见
- Perf logger 中 "MUL_MAT_VEC" 是显示名（line 1596: `name += "_VEC"`）
- 当前 benchmark 的 N=1 测量值被 dispatch overhead 完全淹没

## 9. 下一步

1. [x] 调研 vk_perf_logger 的启用方式 → `GGML_VK_PERF_LOGGER=1` 环境变量
2. [x] 手动验证 GPU timestamp vs wall-clock → **确认 dispatch overhead 是主因，发现 op fusion 和 MUL_MAT_VEC 两个新问题**
3. [x] 跨 backend 调研 → CPU 无需特殊处理，CUDA 以后再做，Vulkan 优先
4. [x] 调研 op fusion 可行性 → 无法提前获取融合图，改用 pattern-based 模拟
5. [ ] 设计新的 benchmark + estimate 方案
6. [ ] 实现方案并重新 benchmark
7. [ ] 修改 estimate 以处理 op fusion + MUL_MAT_VEC
8. [ ] 重新验证 estimate 精度
