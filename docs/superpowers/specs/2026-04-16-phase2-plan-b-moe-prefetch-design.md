# Phase 2 Plan B — MoE Expert 预取流水线设计文档

**日期：** 2026-04-16  
**硬件：** Intel Arrow Lake 265K · Windows 11 · 128 GB DDR5 6400 MT/s · RTX 3090 (24 GB Video RAM (VRAM), PCIe 4.0 x16)  
**模型：** Qwen3-Coder-Next 80B Q4_K_M (~52 GB GGUF)  
**前置条件：** Plan A（`OLLAMA_MOE_PINNED=1`）必须激活

---

## 1. 背景

Phase 2 Plan A 实验结论（`docs/perf/2026-04-15-moe-split-phase2-plan-a-experiment-report.md`）：

- `cudaHostRegister` 将 Host-to-Device (H2D) 带宽从 14.4 GB/s 提升至 25.1 GB/s（1.74×）
- Prefill 1024 tokens：**2096.3 ms → 1391.7 ms（-33.6%）**
- **改善完全来自带宽提升，无 copy/compute 重叠**

Plan A 的执行模型（`ggml_backend_sched_compute_splits`，`ggml-backend.cpp:1480`）是严格串行的：

```
split N:
  ggml_backend_synchronize(input_backend)   ← 阻塞
  copy_experts(N)                           ← ~21 ms，Copy Engine DMA
  ggml_backend_graph_compute_async(N)       ← ~10.6 ms，GPU compute
  [repeat N+1...]
```

31 个 CPU-MoE split 中，每层 copy（21 ms）与 compute（10.6 ms）完全串行，理论上 copy 时间可以被前一层的 compute 隐藏。

**Plan B 目标：** 在提交 compute N 之后，立刻在独立 CUDA copy stream 上预取 N+1 层的 expert 权重，使 copy N+1 与 compute N 真正并行。

---

## 2. 成功标准

| 指标 | 目标 |
|---|---|
| Prefill 1024 tokens 均值 | < **1360 ms**（= Plan A 均值 1391.7 ms − 1σ 31.3 ms） |
| 测量稳定性 CV | < 5% |
| Decode gen_tps | 不低于 Plan A 基线（18.50 t/s） |

---

## 3. 架构总览

### 3.1 执行模型对比

```
Plan A（串行）：
  split N:   [copy N: 21ms][compute N: 10.6ms]
  split N+1: [copy N+1: 21ms][compute N+1: 10.6ms]
  ...
  31 CPU 层合计 ≈ 31 × 31.6 ms = 980 ms

Plan B（流水线，全量预取）：
  split N:   [copy N: 21ms][compute N: 10.6ms]
                              └─ prefetch N+1 ─────────────────→ [copy N+1: 38ms]
  split N+1:                                                      [wait done][compute N+1: 10.6ms]
                                                                               └─ prefetch N+2 ──→ ...
  31 CPU 层合计 ≈ 21 + 30 × max(38, 10.6) = 21 + 1140 = 1161 ms
```

**注意：** 本次实验采用全量预取（Section 9.1 策略 1），每层传输全部 ~975 MiB expert 权重（38 ms），而非当前按激活 expert 精确 copy 的 21 ms。流水线将每层的 copy 等待从 21 ms 降为 0（与前层 compute 重叠），但全量传输多出 17 ms，两者相抵后的净收益需要实验验证。ggml scheduler 调度开销（约 230 ms）仍然存在，成功标准为 prefill < 1360 ms。

### 3.2 关键洞察

**不需要额外 VRAM staging buffer。**

ggml 调度器已在 VRAM 中为每个 split 的输入分配了 `input_cpy` tensor（`tensor_copy(input, split_backend_id, sched->cur_copy)`，`ggml-backend.cpp:1497`）。Plan B 只是把写入 `input_cpy->data` 的时机，从"compute N 开始前"提前到"compute N 提交后立刻"，无需额外 VRAM。

**Plan A（pinned memory）是 Plan B 的硬性前提。**

预取在 compute N 运行期间通过独立 copy stream 提交。若 source 内存（CPU-MoE buffer）是 pageable，CUDA runtime 会在 CPU 侧做 staging（阻塞 CPU 线程），使 `cudaMemcpyAsync` 调用本身变成阻塞，重叠消失。Pinned source 确保 Copy Engine 可以真正异步 DMA。

---

## 4. 数据结构

Plan B 改动完全封装在 `ggml_backend_sched_compute_splits` 函数内，不修改 `ggml_backend_sched` 结构体，避免影响 ggml 其他路径。

**函数内局部变量：**

```c
// Plan B lookahead 状态（ggml_backend_sched_compute_splits 局部）
cudaStream_t moe_prefetch_stream = NULL;  // 独立 copy stream，与 compute stream 分离
cudaEvent_t  moe_prefetch_event  = NULL;  // 记录预取 copy 完成时间点
bool         prefetch_pending    = false; // 是否有待同步的预取
int          prefetch_split_id   = -1;   // 预取对应的 split id（用于验证）
```

**初始化：**

```c
// 仅在 OLLAMA_MOE_PREFETCH=1 时创建，失败则退化到 Plan A 串行行为
bool prefetch_enabled = (getenv("OLLAMA_MOE_PREFETCH") != NULL);
if (prefetch_enabled) {
    if (cudaStreamCreate(&moe_prefetch_stream) != cudaSuccess ||
        cudaEventCreate(&moe_prefetch_event)   != cudaSuccess) {
        prefetch_enabled = false;  // 静默退化
    }
}
```

---

## 5. 同步协议

### 5.1 辅助函数：识别 CPU-MoE split

```c
// 判断 split_id 对应的 split 是否是 CPU-MoE split（需要预取）
static bool is_moe_cpu_split(ggml_backend_sched_t sched, int split_id) {
    if (split_id < 0 || split_id >= sched->n_splits) return false;
    struct ggml_backend_sched_split *split = &sched->splits[split_id];
    if (split->n_inputs == 0 || split->graph.n_nodes == 0) return false;

    struct ggml_tensor *node = split->graph.nodes[0];
    for (int i = 0; i < split->n_inputs; i++) {
        struct ggml_tensor *input = split->inputs[i];
        if (ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
            ggml_backend_buffer_is_host(input->buffer) &&
            node->src[0] == tensor_copy(input, split->backend_id, sched->cur_copy) &&
            node->op == GGML_OP_MUL_MAT_ID) {
            return true;
        }
    }
    return false;
}
```

### 5.2 辅助函数：发起预取

```c
// 把 split next_id 的 MoE expert weights 异步拷贝到对应的 input_cpy tensor
// 使用独立 copy stream，不等待完成
static void fire_moe_prefetch(ggml_backend_sched_t sched, int next_id,
                               cudaStream_t stream,
                               const std::vector<int32_t>& ids,
                               const std::vector<ggml_bitset_t>& used_ids) {
    struct ggml_backend_sched_split *split = &sched->splits[next_id];
    // 找到 MoE weight input
    struct ggml_tensor *node = split->graph.nodes[0];
    for (int i = 0; i < split->n_inputs; i++) {
        struct ggml_tensor *input = split->inputs[i];
        struct ggml_tensor *input_cpy = tensor_copy(input, split->backend_id, sched->cur_copy);
        if (!(ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
              ggml_backend_buffer_is_host(input->buffer) &&
              node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)) {
            continue;
        }
        // 复用当前的 used_ids，或重新获取（若 ids_tensor 不同）
        // 对每个 activated expert group，调用 cudaMemcpyAsync 到 input_cpy->data
        // 使用 moe_prefetch_stream 而非默认 compute stream
        const int64_t n_expert   = input->ne[2];
        const size_t  expert_size = input->nb[2];
        // ... group consecutive experts, cudaMemcpyAsync per group ...
        break;
    }
}
```

### 5.3 主循环修改

```c
for (int split_id = 0; split_id < sched->n_splits; split_id++) {
    struct ggml_backend_sched_split *split = &sched->splits[split_id];

    // ── 阶段 1：若本层已被预取，等待预取完成，跳过常规 copy ──
    bool skip_copy = false;
    if (prefetch_enabled && prefetch_pending && split_id == prefetch_split_id) {
        cudaEventSynchronize(moe_prefetch_event);  // 等 copy 完成
        prefetch_pending = false;
        skip_copy = true;   // input_cpy 已有数据，跳过 copy_experts
    }

    // ── 阶段 2：常规 input copy（未被预取的情况）──
    for (int input_id = 0; input_id < split->n_inputs; input_id++) {
        // ... 原有逻辑 ...
        // 若 skip_copy && 是 MoE weight input，跳过 copy_experts 调用
    }

    // ── 阶段 3：提交 compute N ──
    if (!sched->callback_eval) {
        enum ggml_status ec = ggml_backend_graph_compute_async(
            split_backend, &split->graph, sched->batch_size);
        if (ec != GGML_STATUS_SUCCESS) {
            // 清理 prefetch 资源后返回
            if (prefetch_enabled) {
                cudaStreamSynchronize(moe_prefetch_stream);
                cudaStreamDestroy(moe_prefetch_stream);
                cudaEventDestroy(moe_prefetch_event);
            }
            return ec;
        }
    }
    // callback_eval 路径不启用预取（保留 debug 语义）

    // ── 阶段 4：compute N 提交后，预取 N+1 ──
    if (prefetch_enabled && !sched->callback_eval) {
        int next_id = split_id + 1;
        if (is_moe_cpu_split(sched, next_id)) {
            fire_moe_prefetch(sched, next_id, moe_prefetch_stream, ids, used_ids);
            cudaEventRecord(moe_prefetch_event, moe_prefetch_stream);
            prefetch_pending    = true;
            prefetch_split_id   = next_id;
        }
    }

    // record event（原有逻辑）
    if (split->n_inputs > 0) {
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_record(
                sched->events[split_backend_id][sched->cur_copy], split_backend);
        }
    }
}

// ── 清理 ──
if (prefetch_enabled) {
    if (prefetch_pending) {
        cudaEventSynchronize(moe_prefetch_event);  // 排空最后一个预取
    }
    cudaStreamDestroy(moe_prefetch_stream);
    cudaEventDestroy(moe_prefetch_event);
}
```

---

## 6. Edge Cases

| 场景 | 处理方式 |
|---|---|
| N+1 不是 CPU-MoE split（GPU split 或最后一层） | `is_moe_cpu_split` 返回 false，不预取，正常串行 |
| N+1 有多个 input（非 MoE weight 的普通 input） | `fire_moe_prefetch` 只预取 MoE weight input，其余 input 保持原路径 |
| `cudaStreamCreate` 或 `cudaEventCreate` 失败 | `prefetch_enabled = false`，静默退化到 Plan A 串行行为 |
| `callback_eval` 路径（debug/compare 模式） | 跳过预取，完全走原有串行路径，保留 debug 语义 |
| 预取写入 `input_cpy->data` 时 compute N 还未完成上一轮的数据 | 不会发生：`input_cpy` 是 per-split 独立分配的 tensor（`tensor_copy` 按 split_id 和 cur_copy 索引），不跨 split 复用 |
| `n_copies > 1`（`parallel=true` 模式）| Plan B 不在此模式下运行（Ollama 当前固定 `parallel=false`，`n_copies=1`） |
| 最后一个 CPU-MoE split 预取后无后续 split | `prefetch_pending` 在循环结束后通过 `cudaEventSynchronize` 排空 |

---

## 7. 实现范围

| 文件 | 改动类型 | 估计行数 |
|---|---|---|
| `envconfig/config.go` | 新增 `MoePrefetch = Bool("OLLAMA_MOE_PREFETCH")` | ~5 行 |
| `llm/server.go` | `MoePrefetch()` 为 true 时注入 `OLLAMA_MOE_PREFETCH=1` 到 runner 子进程 env | ~5 行 |
| `ml/backend/ggml/ggml/src/ggml-backend.cpp` | 核心实现：`ggml_backend_sched_compute_splits` 内的预取逻辑 | ~120 行 |

**不需要修改 `ggml_backend_sched` 结构体**，不影响 ggml 其他路径。

---

## 8. 测试方法

**启动命令：**

```
set OLLAMA_MOE_GPU_LAYERS=-1
set OLLAMA_MOE_PINNED=1
set OLLAMA_MOE_PREFETCH=1
set OLLAMA_DEBUG=1
ollama serve
```

**Benchmark 配置：** 与 Phase 1/Plan A 完全相同：6 epoch（4 warmup），1024 tokens prefill，batch size 1024，最多生成 16 tokens。

**对照组：**

| 测试名 | 配置 |
|---|---|
| `baseline-moe-split-disabled` | 标准路由（已有数据，2096.3 ms） |
| `moe-split-plan-a` | `OLLAMA_MOE_PINNED=1`，无预取（已有数据，1391.7 ms） |
| `moe-split-plan-b` | `OLLAMA_MOE_PINNED=1` + `OLLAMA_MOE_PREFETCH=1` |

---

## 9. 风险

| 风险 | 可能性 | 影响 | 缓解 |
|---|---|---|---|
| `ids` / `used_ids` 在预取时不可用（N+1 的 token 路由在 compute N 结束后才确定） | **高**（关键约束） | 预取无法提前确定要 copy 哪些 expert | 见下方 §9.1 |
| ggml scheduler 调度开销（~230 ms）掩盖预取收益 | 中 | 实测改善低于理论值 | 接受：成功标准已设定为统计显著改善（>1σ） |
| `cudaMemcpyAsync` 在独立 stream 与 compute stream 之间的隐式同步 | 低（已验证 pinned memory 支持真正并行） | 重叠比例低于预期 | Nsight Systems 验证重叠比例 |

### 9.1 ids 可用性问题（最高优先级）

**问题：** Qwen3-Next 的 MoE routing 在每层 compute 中产生 token→expert 映射（`ids_tensor`）。预取 N+1 层需要知道 N+1 层激活了哪些 expert，但 `ids_tensor` 要等 N 层 compute 完成后才有值。

**两种应对策略：**

**策略 1（保守）— 全量预取**  
不依赖 `ids_tensor`，直接对 N+1 层的全部 expert 做 `cudaMemcpyAsync`（~975 MiB per layer）。  
- 优点：简单，不需要读取 `ids_tensor`  
- 缺点：传输量是实际激活量的约 1.85×（当前每层实测约 525 MiB）  
- 带宽：975 MiB / 25.1 GB/s ≈ 38 ms（vs 当前 21 ms）  
- 预测效果：若 compute time（10.6 ms）能被 copy（38 ms）的一部分隐藏，仍有改善

**策略 2（精确）— 先全量 prefetch，compute 结束后按 ids 修正**  
- 预取阶段：全量 copy N+1 到 input_cpy（同策略 1）  
- Compute N 结束后：read `ids_tensor`，确认哪些 expert 已经在 input_cpy 中，不需要再 copy  
- 本质上与策略 1 相同，只是在语义上保留了"按需确认"的框架，便于未来优化

**本次实验采用策略 1（全量预取）**，原因：
1. 实现简单，适合验证 copy/compute 重叠是否有收益
2. 即使全量 copy（38 ms）> 当前串行 copy（21 ms），只要 copy/compute 重叠能隐藏超过 `38 - 21 = 17 ms`，就有净收益
3. 若全量预取无净收益，可评估策略 2 或放弃 Plan B

---

## 10. 相关文档

- Phase 2 原始设计文档：`docs/superpowers/specs/2026-04-15-phase2-moe-async-pipeline-design.md`
- Phase 2 Plan A 实验报告：`docs/perf/2026-04-15-moe-split-phase2-plan-a-experiment-report.md`
- ggml-backend.cpp 核心函数：`ggml_backend_sched_compute_splits`（`ggml-backend.cpp:1480`）
- Micro-benchmark 代码：`scripts/cuda-overlap-bench/`
