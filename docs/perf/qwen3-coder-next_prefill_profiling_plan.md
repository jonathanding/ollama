# qwen3-coder-next Prefill 性能差距 Profiling 实验计划

**分支**: `experiment/prefill-gap-analysis`（基于 `feat/disable-ollama-runner`）
**日期**: 2026-05-15
**前置文档**:
- [qwen3-coder-next_runner_comparison.md](./qwen3-coder-next_runner_comparison.md)
- [qwen3-coder-next_prefill_gap_analysis.md](./qwen3-coder-next_prefill_gap_analysis.md)

---

## 一、背景与目标

### 已有数据

在 RTX 3090 + qwen3-coder-next Q4_K_M（49 层 hybrid attention/recurrent + MoE）下：

| 配置 | ollama runner | llama runner | gap |
|---|---|---|---|
| batch=1024, prompt=1024（**单 ubatch**） | 463 t/s / 2088 ms | 627 t/s / 1561 ms | **+35% / -526 ms** |
| batch=1024, prompt=4096（4 ubatch） | 485 t/s | 612 t/s | +26% |
| batch=512, prompt=4096（8 ubatch） | 321 t/s | 436 t/s | +36% |
| decode（per-token） | 18 t/s | 11–12 t/s | -45 ~ -64% |

### 已排除假设

**`can_reuse` 不是主因**（至少对单 ubatch case 完全无效）。

证据：
- 实测 `LLAMA_GRAPH_REUSE_DISABLE=1` vs `=0`，batch=1024/prompt=1024 配置下 prefill 时间一致（1561 ms vs 1565 ms，CV<3%）
- 此配置只生成 1 个 ubatch，C++ 端无前驱图可复用，reuse 机制不起作用
- 但 ollama vs llama 的 ~500 ms 差距依然存在 → gap 来自其他来源

`LLAMA_GRAPH_REUSE_DISABLE=1` 时 ollama 子进程日志确实出现 `graph reuse disabled`，确认环境变量已正确传到 runner（log: `tests/llama_runner`）。

### 目标 workload

**只关注 batch=1024 / prompt=1024 单 ubatch 配置**（qwen3-coder 代码补全场景的主流输入长度）。优化结论应在此配置下产生。其他配置作为副带验证。

### 实验目标

**定量分解 ~500 ms gap 的来源**，找出 ollama runner 单 ubatch prefill 路径上 C++ 不付而 Go 必须付的开销，按收益/工作量排序，给出可执行的优化方向。

---

## 二、待验证候选（从 prefill_gap_analysis.md 修订）

### 候选 A：图节点数 / split 数差异（最高嫌疑）

**假设**：Go 路径下 `qwen3next.Model.Forward` 产生的 graph 比 C++ `llm_build_qwen3next` 多出节点（额外 Contiguous/Reshape/Permute），或 hybrid offload（20 GPU + 29 CPU）下 sched_split_graph 切出更多 splits，导致更多 H2D/D2H 拷贝和更长 GPU compute。

**验证手段**：
- ollama 端打印 `ggml_graph_n_nodes(c.graph)` 和 `ggml_backend_sched_get_n_splits(sched)`（`ml/backend/ggml/ggml.go:852` 已有 slog.Debug，改为 Info 即可）
- llama 端在 `llama-context.cpp:861` `graph_compute` 调用前加一行 `LLAMA_LOG_INFO` 打印同样数据
- 对比两边数字。**判读**：
  - nodes Go > C++ 30%+ → 图本身更复杂
  - splits Go > C++ 50%+ → offload 切分更碎、GPU compute 之间穿插更多 sync

### 候选 B：CGO 构图阶段累积开销

**假设**：每个 ml op（Mulmat、Add、Mul、Slice、Reshape、Permute、Contiguous 等）都跨一次 CGO，单 ubatch prefill 累计可能 5k–15k 次。Windows 下 CGO 单次开销 1–5 μs，累计 5–75 ms。

**验证手段**：
- 在 `model.Forward` 调用前后打 timestamp（`runner/ollamarunner/runner.go:624`）
- 在 `Compute` 调用前后打 timestamp（`runner/ollamarunner/runner.go:716`）
- 区分"构图阶段"和"GPU compute 阶段"耗时

**判读**：构图阶段 < 50 ms → 候选 B 不重要；构图阶段 > 200 ms → 候选 B 是主要因素。

### 候选 C：sched_alloc_graph 开销（每 batch 必付）

**假设**：每个 batch 都触发 `ggml_backend_sched_split_graph`（拓扑分析 + 每节点查 backend 归属 + 分配中间 tensor buffer）。在 hybrid offload 下这步特别重。

**验证手段**：在 `ml/backend/ggml/ggml.go:825` 的 `ggml_backend_sched_graph_compute_async` 内部分阶段计时（需要在 C 侧加桩）。可先看候选 B 的"GPU compute 阶段"耗时是否远大于 GPU 实际算时间，间接推断。

### 候选 D：Context 生命周期（NewContext / Close 每 batch 一对）

**假设**：每 batch `ggml_init` + `ggml_free`，分配/释放 8–15 MB 元数据 buffer，10–30 ms。

**验证手段**：在 `Backend.NewContext` 和 `Context.Close` 前后打 timestamp。

### 候选 E：Floats() 同步成本（仅多 ubatch case 相关）

**假设**：每 ubatch 的 `Floats()` 调用 `sched_synchronize` 强制阻塞。在多 ubatch case 下中间 ubatch 也同步是浪费。

**对单 ubatch case 不适用**——单 ubatch 必然要等 GPU 完成才能采样，sync 不可省。

仅在做多 ubatch 验证时关注。

---

## 三、实验计划

### Phase 1：核心 timing 探针 + nodes/splits 统计

**说明**：原计划的 Phase 0（零代码改动看 ollama nodes/splits）已合并入 Phase 1——单独看 ollama 的图统计无对照价值，且 `slog.Debug` 输出会被海量调试日志淹没；直接和 timing 探针一起在同一次 rebuild 里搞定。

**目的**：在 ollama runner 上分阶段量化单 ubatch prefill 的耗时归属。

**修改文件清单**（最小侵入）：

#### 1. `runner/ollamarunner/runner.go`

在 `forwardBatch` 和 `computeBatch` 中加 timing 探针，**仅在环境变量 `OLLAMA_PREFILL_PROFILE=1` 时启用**，避免污染常规路径：

- `forwardBatch:492` 之前打 `t_forward_start`
- `forwardBatch:624` 之前打 `t_model_forward_start`，之后打 `t_model_forward_end`
- `forwardBatch` 末尾打 `t_forward_end`
- `computeBatch:715` 之前打 `t_input_inject_start`
- `computeBatch:716` `ComputeWithNotify` 调用前后打 `t_compute_start` / `t_compute_end`
- `computeBatch:723` `Floats()` 之后打 `t_floats_end`
- `computeBatch:641` `Close()` 前后打 `t_close_start` / `t_close_end`

#### 2. `ml/backend/ggml/ggml.go`

- `Context.Close`（line 1008）前后打时间戳
- `ComputeWithNotify`（line 814）：将 `sched_graph_compute_async`、`sched_reset`、`sync()`（即 `sched_synchronize`）三段拆开计时
- `NewContext`（line 663）的 `ggml_init` 前后打时间戳
- 把 `slog.Debug("compute graph", "nodes", ...)` 升级为 `slog.Info`（已存在于 line 852）

#### 3. C++ 端：llama runner 对照打印

为了拿到 llama runner 的 nodes/splits 数字，**只在该 branch 内**临时加一行打印（不要 commit 到 main）：

- `llama/llama.cpp/src/llama-context.cpp:861` 的 `graph_compute` 调用之前，加：
  ```cpp
  LLAMA_LOG_INFO("%s: graph nodes=%d splits=%d ubatch_n_tokens=%d\n",
      __func__, ggml_graph_n_nodes(res->get_gf()),
      ggml_backend_sched_get_n_splits(sched.get()), ubatch.n_tokens);
  ```
- 这要 rebuild llama runner，预期一次性工作。

#### 4. 输出格式

每次 prefill 输出一行结构化日志（便于脚本解析）：

```
PREFILL_PROFILE batch_id=N n_inputs=1024 n_outputs=1
  forward_total=X.XX ms
    new_context=X.XX ms
    model_forward=X.XX ms
    other=X.XX ms
  compute_total=X.XX ms
    input_inject=X.XX ms
    compute_async=X.XX ms
    sync=X.XX ms
    floats_d2h=X.XX ms
  close=X.XX ms
  graph_nodes=N graph_splits=N
```

### Phase 2：基线测量

**步骤**：
1. 切到 `experiment/prefill-gap-analysis` 分支，rebuild ollama
2. 启动 ollama serve（环境变量 `OLLAMA_PREFILL_PROFILE=1`、`OLLAMA_DEBUG=1`）
3. 在 ollama runner 模式下跑：
   ```
   bench-sweep run --model qwen3-coder-next --sizes 1024 --batch-size 1024 --epochs 6 --warmup 4 --name profile-ollama-bs1024
   ```
4. 收集 stderr 日志中的 `PREFILL_PROFILE` 行
5. 切到强制 llama runner 模式（`OLLAMA_FORCE_LLAMA_RUNNER=1` 或对应 env），rebuild 后跑同样命令，收集 llama 端的 nodes/splits 数字
6. 把数据填入"Phase 2 结果"小节

**预期结果格式**：

| 阶段 | ollama runner (ms) | llama runner (ms) | 差值 |
|---|---|---|---|
| new_context | ? | (N/A) | ? |
| model_forward (构图) | ? | ? | ? |
| sched_alloc + compute | ? | ? | ? |
| sync | ? | ? | ? |
| floats d2h | ? | ? | ? |
| close | ? | (N/A) | ? |
| **总计** | ~2080 | ~1560 | ~520 ms |

| 图统计 | ollama | llama |
|---|---|---|
| n_nodes | ? | ? |
| n_splits | ? | ? |

### Phase 3：归因 + 选择优化方向

根据 Phase 2 数据决策：

| 主导阶段 | 解释 | 下一步 |
|---|---|---|
| `model_forward` 构图 > 200 ms | CGO 累积 + 节点构造 | 调研图缓存（不是 reuse 整套机制，只是节点 handle 缓存） |
| `compute_async` GPU 实际算时长 > C++ 多很多 | 节点数或 split 数差异导致 GPU 实际算更慢 | 看 `n_nodes` / `n_splits` 对比，定位 hot 路径 |
| `n_splits` Go ≫ C++ | offload 切分更碎，H2D 多 | 调研 schedule split 决策机制差异 |
| `n_nodes` Go ≫ C++ | qwen3next Go 实现产生更多 op | 在 Go 模型代码里找冗余 Contiguous/Permute/Reshape |
| `new_context + close` > 30 ms | 生命周期开销 | Context 跨 batch 复用（独立小改动） |
| `sync` > GPU compute 时间多 | 同步等待时间过长 | 与 forwardBatch/computeBatch 异步 pipeline 设计有关 |
| 全部阶段都比 C++ 慢 5–10% | 系统性 CGO + Go runtime 开销 | 接受现状，重新评估优化 ROI |

### Phase 4（可选）：补充验证多 ubatch 场景

对 batch=512 / prompt=4096（8 ubatch）配置重测一遍，看：
- 多 ubatch 下 `Floats()` sync 在中间 ubatch 是否成为额外瓶颈（候选 E）
- `n_nodes` 和 `n_splits` 是否随 ubatch 变化
- ollama 是否每个 ubatch 都重做 `new_context + close`（已知答案：是，runner.go:492、641）

仅在 Phase 3 决策需要时进行。

---

## 四、实施约束

### 改动原则

1. **所有改动在 `experiment/prefill-gap-analysis` 分支**，不动 `feat/disable-ollama-runner`，方便后续切回干净版本对比
2. **timing 探针通过 env 开关控制**（`OLLAMA_PREFILL_PROFILE=1`），关闭时零成本
3. **不写任何"修复"代码**，只写诊断代码。优化方案在 Phase 3 后再单独建分支
4. **不重构现有代码**，仅在已有调用点前后插入计时调用
5. C++ 侧的诊断打印要明确标注 "EXPERIMENT ONLY, DO NOT MERGE"

### 编译/运行环境

- Go 端 rebuild：使用现有 `scripts/rebuild_go_windows.ps1`
- C++ 端 rebuild：使用 `scripts/build_cuda_incremental.ps1`（构建 ggml-cuda.dll / ggml-base.dll，参考用户记忆中的 build 规则）
- 测试机器跑预编译的 `bench-sweep.exe`

### 数据记录

- 所有原始日志保存到 `tests/prefill-profiling/<phase>-<runner>-<config>.log`
- 关键数据填入本文档对应 Phase 小节
- 决策点（Phase 3 之后）单独写一份决策摘要（同目录新增 `.md`）

---

## 五、Phase 2 结果（待填）

详见 Phase 1 输出格式，将完整数据放置于此。

---

## 六、变更日志

| 日期 | 变更 | 原因 |
|---|---|---|
| 2026-05-15 | 初稿 | 三组 reuse 实验确认 can_reuse 在单 ubatch 场景无收益，重新规划 profiling 方向 |
| 2026-05-15 | 删除 Phase 0，合并至 Phase 1 | 单独看 ollama 端 nodes/splits 无对照价值，且 slog.Debug 输出会被淹没 |
