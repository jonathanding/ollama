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

## 二、待验证候选（已按 Phase 4 结果定案）

本节保留 Phase 1 之前的初步猜测（按 prefill_gap_analysis.md 修订），用于追溯思路演进。Phase 2/4 实测已对所有候选定案。

| 候选 | 假设 | Phase 2/4 结论 |
|---|---|---|
| A. 图节点数 / split 数差异 | Go 实现可能产生更多节点或更碎的 splits | **否**：实测 Go 节点数(17.5k) < C++(34.6k)；splits 数(595 vs 612)接近 |
| B. CGO 构图累积开销 | 5k-15k 次 CGO 累计 5-75 ms | **否**：实测 model_forward = 6.5 ms，可忽略 |
| C. sched_alloc_graph 开销 | hybrid offload 下 split + alloc 慢 | **否**：每 batch 重做 alloc 但开销小，且 §6.1 证据 1 显示 gap 全在 compute_total |
| D. Context 生命周期 | 每 batch ggml_init+ggml_free 10-30 ms | **否**：new_context+close < 5 ms |
| E. Floats() 同步成本 | 多 ubatch 中间也 sync 是浪费 | 仅多 ubatch 场景相关，本目标 workload (单 ubatch) 不适用 |
| **真因（Phase 4 锁定）** | — | **CPU 层权重 plain malloc → pageable H2D 14 GB/s + 部分 attention 在 CPU 算**；详见 §7.4 |

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

## 五、Phase 2 结果（2026-05-15）

### 5.1 测试条件

- **配置**：batch_size=1024 / prompt=1024 / max_tokens=16 / epochs=6 / warmup=4
- **bench-sweep prefill_ms**：ollama 2093 ms / llama 1569 ms（差 524 ms / +33%）
- **环境变量**：`OLLAMA_PREFILL_PROFILE=1`
- **原始日志**：`test/llama-runner/{ollama,llama}-runner-prefill-profile.txt`

### 5.2 阶段耗时对照（prefill batch 平均，去掉首次 warm-up）

| 阶段 | ollama (ms) | llama (ms) | 差值 (ms) | 备注 |
|---|---:|---:|---:|---|
| **forward_total** | 144 | n/a | - | 含 `<-pendingBatch.computeStartedCh` 等待，不是真开销 |
| └ new_context | <0.5 | n/a | - | time.Now 精度限制，实际亚毫秒 |
| └ model_forward（Go 构图） | 6.5 | (含在 decode) | - | **远小于预期，CGO 累积不是主因** |
| └ forward_other | 138 | n/a | - | channel 等待为主 |
| **compute_total** | 2125 | 1580 | **+545** | **gap 几乎全部在此** |
| └ compute_outer | 2125 | (decode) | - | 内含 `graph_compute_async` 全部工作 |
| └ floats | <0.5 | <0.5 | - | sync 在 compute_outer 内部已完成 |
| └ close | 2.2 | n/a | - | 可忽略 |
| **bench-sweep prefill_ms** | 2093 | 1569 | 524 | 与 compute_total 相符 |

### 5.3 图统计对照（prefill）

| 指标 | ollama | llama | 差值 |
|---|---:|---:|---|
| graph_nodes | 17,556 | 34,628 | **ollama 少一半** |
| graph_splits | 595 | 612 | 几乎相同 |

**这与原假设相反**：Go 模型实现产生的图节点反而比 C++ 少（约 50%）；splits 数也几乎一致。但 ollama 仍然慢 545 ms。

### 5.4 Decode 阶段对照（n_tokens=1，去除等待异常值）

| 指标 | ollama | llama |
|---|---:|---:|
| 总耗时 | ~95–130 ms | ~95 ms |
| graph_nodes | 5,875 | 6,362 |
| **graph_splits** | **3** | **49** |

**反向发现**：decode 阶段 ollama splits 仅为 llama 的 1/16。这正解释了 ollama decode 反而快 ~50% 的现象——splits 少 → backend 间 sync 少 → 总时间短。

### 5.5 graph_reuse 实测命中率（llama runner）

C++ 端 `PREFILL_PROFILE_LLAMA_GRAPH` 日志显示：

| 类别 | reused=1 命中数 |
|---|---|
| Prefill batches | **0 / 10** |
| Decode batches | **0 / 146** |

**`can_reuse` 在此 workload 下一次都未命中**。这彻底解释了之前 `LLAMA_GRAPH_REUSE_DISABLE=0/1` 实验毫无差异的结果——本来就没有 reuse 行为可被禁用。

## 六、Phase 3：根因定位

### 6.1 关键证据链

1. **gap 几乎全部（545/524 ms）在 `compute_total` / `lc.Decode` 内部**，不在 Go 层任何阶段
2. **图节点数 ollama < llama**（17.5k vs 34.6k）但 ollama 反而慢——节点数不是直接因子
3. **splits 数几乎相同**（595 vs 612）——backend 切分粒度不是主因
4. **Decode 阶段 ollama splits 只有 3**（vs llama 49）——这不是优化更好，而是**ollama 在 decode 时 CPU 层走 CPU backend，不需要在 splits 间切换**；llama 把 CPU 层都送 GPU 算，每层都是一次跨 backend split。两种执行模式的副作用，不是优劣
5. **can_reuse 0% 命中率**——porting 该机制无收益

## 七、Phase 4：GPU 利用率对照 + 旧实验交叉验证（2026-05-15）

Phase 3 给出了"gap 在 compute_splits 内部"的方向，Phase 4 通过三条独立证据链把根因**收紧到一行代码差异**。

### 7.1 GPU 利用率曲线对照

**采样方法**：`scripts/profile-gpu-prefill.ps1` 用 `nvidia-smi --query-gpu` 50ms 采样 GPU util / power / memory ctrl util / SM clock，与 bench-sweep 同时运行。

**原始 PNG**：`test/llama-runner/gpu-profile/{ollama,llama}-runner-*/gpu-trace.png`

| 指标 | ollama runner | llama runner |
|---|---|---|
| Prefill 期间 GPU util peak | ~60% | **~85%** |
| GPU util 形状 | 中频锯齿（多峰谷） | 持续高位平台 |
| GPU 功耗 peak | 220W | **245W** |
| 单 prefill 周期长度 | ~3.0s | ~3.0s（含 16 decode token） |
| 单 prefill GPU 累计高电平时长 | ~1.5s 但分散 | ~1.5s 但连续 |

**关键解读**：

- **ollama 的 GPU util 呈 20-60% 反复振荡的锯齿**——典型的"GPU 跑一段 → 等 CPU 算下一段 → GPU 再跑"模式，GPU 在 prefill 期间从未达到 80%
- **llama 的 GPU util 是 80-90% 持续平台**——GPU 在 prefill 期间几乎一直在跑，**说明所有 layer 的计算都在 GPU**

GPU 功耗也吻合：llama 持续把 GPU 推到 245W（接近 RTX 3090 stress），ollama 只到 220W 且常波动。

### 7.2 Task Manager 截图反向佐证

| 指标 | ollama | llama |
|---|---|---|
| Dedicated GPU memory | 22.9 / 24.0 GB | 21.4 / 24.0 GB |
| **Shared GPU memory** | **0.1 GB** | **29.4 GB** |

llama runner 占用了 ~30 GB shared GPU memory（即 GPU 通过 PCIe DMA 访问的 host pinned memory）；ollama 几乎没用——说明 ollama 的"CPU 层权重"是 plain malloc 在系统 RAM，不是 pinned。

### 7.3 旧实验对照（feat/moe-split-prefetch 分支，docs/perf/2026-04-22-moe-split-prefill-full-experiment-report.html）

旧实验的关键阶段与本次 llama runner 对照：

| 实验 | 配置 | prefill_mean (ms) | 与 baseline 差 |
|---:|---|---:|---:|
| Stage 1（ollama baseline） | 默认 hybrid offload，20 GPU + 29 CPU 层（**CPU 层在 CPU 算**） | 2096 | — |
| Stage 2（MoE split, **pageable**） | dense 全 GPU + MoE expert pageable，触发 ggml op_offload，**GPU 算** | 2060 | -36 |
| **Stage 3（MoE split + pinned, selective）** | dense 全 GPU + MoE expert **pinned host**，selective copy_experts | **1392** | **-704** |
| **Stage 4（DSB + pinned, full tensor prefetch）** | dense 全 GPU + MoE expert pinned，**full** copy + 独立 prefetch stream | **1648** | -448 |
| **当前 llama runner 实测** | CUDA_Host buffer (29.8 GiB pinned) + ggml 标准 op_offload | **1597** | -499 |
| 当前 ollama runner 实测 | 默认 hybrid，与 Stage 1 一致 | 2117 | -- |

**Stage 4 (1648 ms) ≈ llama runner (1597 ms)**——差距仅 51 ms。两者执行模式完全一致：
- 把"应在 CPU 上的权重"放在 **pinned host memory**
- 触发**完整**的 expert 权重 H2D
- 让 **GPU 计算 mul_mat**

51 ms 的微差完全可以用实现细节解释（Stage 4 用 DSB + 独立 stream 做 overlap，能藏 ~210 ms compute；llama runner 走 ggml 标准 path 在 splits 间串行）。

**Stage 1 (2096 ms) ≈ 当前 ollama runner (2117 ms)**——差距 21 ms（噪声范围内），证明 ollama runner 的执行模式跟 baseline 完全一致："CPU 层在 CPU 算"。

### 7.4 op-by-op backend 分配的精确还原（基于 GGML_SCHED_DEBUG=2 dump）

> 本节是从 `GGML_SCHED_DEBUG=2` dump 的实际图分配（`test/llama-runner/sched-dump.log`，
> 单 prefill batch=1024 prompt=1024）直接读出来的事实，**取代了之前所有基于代码静态读的推测**。
> dump 的 op 总分布：CUDA0 占 10264 个 node、CPU 占 164 个 node（共 10428 node, 573 splits）。

#### 7.4.1 模型权重的物理位置

48 个 block 中：
- **blk.0–27（前 28 层）**：所有权重（attention/SSM/MoE）的 buffer 在 plain CPU malloc。这些权重每次执行前要 H2D 到 CUDA0
- **blk.28–47（后 20 层）**：所有权重在 CUDA0 dedicated VRAM
- 这与日志 `model weights CUDA0 19.9 GiB / CPU 28.3 GiB` 完全一致

dump 中识别 CPU-resident 的方法：在 `## SPLIT #N: CUDA0 # M inputs: [..., blk.X.attn_q.weight, ...]` 这种行里，weight 出现在 inputs 列表 = 它需要从别的 backend 拷过来 = 它不在 CUDA0 上。

#### 7.4.2 各类 op 实际跑在哪个 backend（dump 直接事实）

按 op 类型整理，每行注明在 CPU-resident 层（blk.0–27）的处理：

| Op 类型 | 实际 backend | 说明 |
|---|---|---|
| **MUL_MAT**（attention Q/K/V/output projection、deltanet in_proj/out_proj、MoE shared_expert、router） | **CUDA0** | weight 在 CPU buffer 但 op_offload 命中（batch=1024 ≥ 32），sched 把 op 分到 CUDA0；每次执行前 weight H2D |
| **MUL_MAT_ID**（MoE expert 矩阵乘 gate_exps / up_exps / down_exps） | **CUDA0**（144/144 全部）| 同上，op_offload 命中。`compute_splits` 内部对 mul_mat_id 还有 selective copy（只搬激活的 expert，约 63%）|
| **FLASH_ATTN**（attention 内部 Q×K^T + softmax + scores×V 一体融合）| **CUDA0**（12/12 全部，每个 attention 层一个）| ollama 用 fused flash_attn_ext op，**不是**两个独立 mul_mat。每次执行前要 H2D **整段 KV cache**（128M leaf_79）+ K/V views |
| **SET_ROWS**（K/V 写入 KV cache）| **CPU**（14 个）+ **CUDA0**（10 个）| 跟着 KV cache buffer 走。CPU 层 KV cache buffer 在 CPU → SET_ROWS 落 CPU；GPU 层落 CUDA0。对 CPU 层每次要 D2H 把刚算好的 K/V 写回 CPU |
| **SOLVE_TRI**（chunked delta net 三角矩阵求解）| **CUDA0**（35/36）+ **CPU**（1，仅 blk.0）| 几乎全部在 CUDA0。blk.0 那 1 个落 CPU 是 sched expand pass 把整段 deltanet 计算从 CPU 端的 mask（FILL/TRI/DIAG）传染过去的副作用 |
| **SSM_CONV**（gated delta net 卷积）| **CUDA0**（36/36） | 全部 GPU，CUDA 支持此 op |
| **CUMSUM**（gated delta net 累积和）| **CUDA0**（36/36）| 同上 |
| **GET_ROWS**（token embedding lookup） | **CPU**（1）+ **CUDA0**（50） | 第一个 GET_ROWS 是 token_embd（embedding lookup）必须在 CPU；其他 GET_ROWS 是 MoE expert id select，在 CUDA0 |
| **FILL / TRI / DIAG**（attention/deltanet mask 准备）| **CPU**（4 个） | CUDA backend 不支持这几个 op，整段 mask 准备只能 CPU 跑。结果再 H2D 到 CUDA0 给后面用 |
| **CPY**（视图复制） | **CPU**（42）+ **CUDA0**（31） | 跟着 src/dst buffer 走；CPU 上的 CPY 大多是 KV cache 写回的辅助拷贝 |
| **CONT / RESHAPE / VIEW / PERMUTE** | 跟随上下游 | view-class op 没有实际计算，sched 跟随相邻节点决定 |
| **RMS_NORM, MUL, ADD, UNARY, ROPE, SOFT_MAX, GLU, SCALE, REPEAT, CONCAT, SUB, DIV, ARGSORT, SUM_ROWS** | **绝大部分在 CUDA0** | 这些是激活上的小 op，sched expand pass 倾向把它们传染到 CUDA0；少数 ADD/MUL/SUB/CPY 在 CPU（在 blk.0 的 CPU split 内部 + CPU 上跑的几个支持 op） |

**总结**：CPU 上跑的 164 个 node 中，绝大部分集中在 blk.0 的一个 30-input CPU split（deltanet 整层被传染到 CPU）+ 12 个 attention 层的 SET_ROWS 写回 KV cache + 输入侧 token embedding。**CPU 层的 weight matmul（attention proj / MoE expert / shared expert）全部在 CUDA0 上跑**，每次执行前从 plain CPU buffer H2D 一份。

#### 7.4.3 H2D 路径的代价（这才是 524 ms gap 的来源）

每个 prefill batch 中，对 CPU-resident 的 28 层每层都要：

| 阶段 | 数据量（每层） | 走 H2D 的次数 | ollama 实际带宽 |
|---|---|---|---|
| Attention 层 weight H2D（attn_q/k/v/output + norm 等） | ~14 MB（attention 层） | 每层每 batch 一次 | pageable 14 GB/s + driver staging |
| MoE expert weight H2D（gate/up/down_exps）| ~996 MB / 层 × ~63% selective | 每层每 batch 一次 | pageable 14 GB/s |
| KV cache H2D（attention 层的 leaf_79 整段）| 128 MB | 每个 attention 层每 batch 一次（7 个 CPU 层 attention）| pageable 14 GB/s |
| K/V D2H 写回 KV cache（SET_ROWS） | 几 MB | 每 attention 层一次 | 同上 |

每次 `cudaMemcpyAsync(HostToDevice)` 在 ollama-cuda.cu:752 之后立刻跟 `cudaStreamSynchronize`——**完全 FIFO 串行，无 overlap**。

llama runner 把所有"应在 CPU 上"的 buffer 改成 cuda_host pinned（`cudaMallocHost`），同样的 `cudaMemcpyAsync` 走 pinned 路径，**有效带宽 25 GB/s** 且 driver 不再做 staging 中转。**仅这一项 1.78× 带宽改善**已经能解释 524 ms 大部分 gap。

### 7.5 三条独立证据汇合

| 证据线 | 结论 |
|---|---|
| GPU 利用率曲线 | ollama 锯齿 60% / llama 平台 85% → ollama 有 CPU attention 节拍 |
| Task Manager Shared GPU memory | ollama 0.1 GB / llama 29.4 GB → llama 把 ~30 GB 映射成 PCIe-accessible pinned |
| 旧实验 Stage 4 ≈ llama runner（1648 vs 1597 ms） | "全量 H2D + pinned + GPU compute" 模式精确还原 llama 性能 |
| 代码 grep | ollama 跳过 `ggml_backend_dev_host_buffer_type(gpu_dev)` |
| ggml 内部日志 | llama 有 `CUDA_Host model buffer = 29804.92 MiB`，ollama 没有 |

### 7.6 内存账本对账

ollama 主进程在加载日志中报告两端的 `model weights / kv cache / compute graph`，但**两端的字段含义不完全等价**——这一节澄清差异，避免后续被误读。

#### 7.6.1 model weights —— 两端可比，差额由层分配解释

| | ollama runner | llama runner |
|---|---|---|
| GPU model weights | 19.9 GiB（CUDA0 dedicated） | 18.9 GiB（CUDA0 dedicated）|
| CPU model weights | 28.3 GiB（plain malloc） | 29.1 GiB（**CUDA_Host pinned**）|

数值接近——差额来自不同 runner 各自的 GPU layer 数（ollama 20 / llama 18）。

#### 7.6.2 KV cache —— ollama 多 ~1.8 GiB，来自 checkpoint 机制

| | ollama runner | llama runner |
|---|---|---|
| CUDA0 KV | 1.1 GiB | 349 MiB |
| CPU KV | 1.5 GiB | 494 MiB |
| **总和** | **2.6 GiB** | **0.84 GiB** |

差额 ~1.8 GiB **不是测量错误**，是 ollama 端独有的 **Recurrent checkpoint 机制**：

- `kvcache/recurrent.go:14` 默认 `DefaultCheckpointCount = 24`
- 每个 sequence slot × 每个 recurrent layer 维护 24 份 conv state + 24 份 recurrent state 副本
- 用途：sequence rollback / 重新生成时无需从头重算 recurrent state
- llama.cpp 端无此机制——KV 估算（`fs/ggml/ggml.go:GraphSize` line 654）只算单份 recurrent state

qwen3-coder-next ~24 层 recurrent，每层每 checkpoint 几 MiB ≈ ~1.8 GiB，**与观察吻合**。

**对 prefill 性能的影响**：checkpoint 写入不在 prefill critical path 上（仅在 reserveCheckpoint 时分配，不占 H2D/compute 时间），不解释 524 ms gap。但是对内存占用是真实开销。

#### 7.6.3 compute graph —— 字段定义不同，"全在 GPU vs hybrid" 是日志错觉

llama runner 报告 `compute graph CUDA0 = 2.2 GiB` 看上去全在 GPU；ollama 报告 `CUDA0 = 884 MiB + CPU = 270 MiB` 看上去 hybrid。**两端含义完全不同**：

| | ollama runner | llama runner |
|---|---|---|
| 数据来源 | `ggml_backend_sched_reserve` 实测每个 backend 的中间张量分配 | ollama 主进程通过 `fs/ggml/ggml.go:GraphSize` 估算公式 + `gqa * kvTotal / 6` fallback heuristic |
| 是否反映真实分配 | ✓ sched 真实数据 | ✗ 保守预留量，给 GPU layout 决策算法用 |
| `server.go:635` 归属 | 按 sched 的 backend split 自然分到 CUDA/CPU | 直接 `Graph = max(partial, full)` 全部归入 GPU 字段，没有 split 概念 |

llama.cpp 内部**实测**的 compute buffer（在 bench-sweep 加载日志里，非 ollama 的内存账本）：

```
llama_context:      CUDA0 compute buffer size =  1000.93 MiB
llama_context:  CUDA_Host compute buffer size =   160.16 MiB
```

合计 **1160 MiB**，与 ollama runner 实测 884 + 270 = 1154 MiB **几乎相等**。

llama.cpp 也是 hybrid（GPU + CUDA_Host），只不过 host 那部分是 pinned。ollama 端因为没用 cuda_host buft，host 那部分是 plain。

**对 prefill 性能的影响**：compute buffer 大小的差异本身不解释性能 gap；它只是被 ollama 主进程的不同账本格式呈现得"看起来不一样"。

### 7.7 hybrid offload 模型下 op 调度通用规则

把 §7.4.2 的 dump 观察抽象成 ggml-backend scheduler 在 hybrid GPU+CPU offload 下的通用调度规则，方便迁移到其他模型分析。前提：prefill batch_size ≥ 32（满足 op_offload 阈值）、CPU device 是最低优先级 backend、CUDA op_offload 开启。

#### 7.7.1 决定 op 跑哪个 backend 的关键变量

| 变量 | 影响 |
|---|---|
| op 是否有 src 是 `WEIGHTS` buffer（`ml/backend/ggml/ggml.go:413` 设置） | 决定走 op_offload 路径还是 expand pass |
| op 类型是否被 CUDA backend `supports_op` 支持 | 决定是否能 offload 到 GPU |
| op 类型是否被 CUDA backend `offload_op` 接受（mul_mat 系要求 `op->ne[1] >= 32`） | 决定 op_offload 路径是否触发 |
| op 的 src 中是否有 tensor 被预先固定在某 backend（pre-allocated KV cache、graph input） | 锁定 op 必须在该 backend 跑 |

#### 7.7.2 通用规则总结表（CPU-resident 层 = weight 在 CPU device buffer）

| Op 类型 | 是 src=WEIGHTS？ | CUDA `supports_op`? | 默认调度结果 | 备注 |
|---|---|---|---|---|
| `MUL_MAT`（weight matmul，如 attn_q/k/v/output、ssm_in/out、router、shared_expert） | ✓ | ✓ | **CUDA**（op_offload 命中，每次 H2D 一份 weight） | qwen3-coder-next、Llama 系都是这条 |
| `MUL_MAT_ID`（MoE expert matmul） | ✓（src[0]=expert weights） | ✓ | **CUDA**（同上，selective copy 只搬激活的 expert）| `compute_splits` 内部 ggml-backend.cpp:1515-1599 selective 路径 |
| `MUL_MAT`（中间张量 × 中间张量，如 chunked deltanet 内部） | ✗（两个 src 都是 activation） | ✓ | **跟随 expand pass**：通常拉到 CUDA，但若上下文是 CPU split 会 fallback CPU | qwen3-coder-next 的 blk.0 deltanet chunk 计算就是被传染到 CPU 的特例 |
| `FLASH_ATTN_EXT`（fused attention） | ✗（K/V 是 KV cache，不是 weight） | ✓（CUDA 支持） | **CUDA**（expand pass + KV cache H2D；如 KV 在 CUDA0 直接用） | ollama 用此 fused op，不像 llama.cpp 也可能拆成 Q×K^T 两个 mul_mat |
| `SET_ROWS`（KV cache 写入） | ✗（dst 是 KV cache） | ✓（CUDA 支持） | **跟随 KV cache buffer**：KV 在 CPU → SET_ROWS 落 CPU；KV 在 CUDA0 → 落 CUDA0 | dst buffer 预分配，sched 走 line 826 dst-binding 路径 |
| `SOLVE_TRI`（chunked delta net 三角矩阵求解） | ✗ | ✓（CUDA 支持） | **CUDA**（expand pass 拉过去），少数 fallback CPU 是相邻 op 触发 expand 副作用 | qwen3-coder-next 特定 |
| `SSM_CONV` / `CUMSUM`（gated delta net 元件） | ✗（输入是 activation） | ✓ | **CUDA**（expand pass） | qwen3-coder-next / Mamba 系列适用 |
| `RMS_NORM`, `MUL`, `ADD`, `SUB`, `UNARY`, `SCALE`, `CONCAT`, `CONT`, `RESHAPE`, `VIEW`, `PERMUTE`, `ROPE`, `SOFT_MAX`, `GLU`, `REPEAT`, `DIV`, `SUM_ROWS`, `ARGSORT` | ✗ | ✓（绝大部分） | **CUDA**（expand pass 倾向 CUDA） | "expand cpu down/up" 显式跳过最低优先级 backend，所以这些激活上的小 op 默认上 CUDA |
| `GET_ROWS`（embedding lookup token_embd） | src[0]=token_embd weight | ✓ | **CPU**（首层 token embd 通常在 CPU device，inputs 默认 CPU `sched->n_backends - 1` 由 line 849 决定） | 第一个 GET_ROWS 是 input embedding；后续 MoE expert id 选择的 GET_ROWS 在 CUDA0 |
| `FILL` / `TRI` / `DIAG`（attention/deltanet 各种 mask 准备） | ✗ | ✗（CUDA 不支持） | **CPU**（CUDA `supports_op = false`）| 后续会作为 split input H2D 到 CUDA0 |
| `MUL_MAT`（output projection，最后一层）| ✓ | ✓ | **CPU**（ollama 主动把 output layer 放 CPU；见 ggml.go:486 "offloading output layer to CPU"）| 即便 op_offload 满足条件也走 CPU——因为 ollama 把 output weight 显式 pin 在 CPU 而不是按 layer offload list 决定 |

#### 7.7.3 留在 CPU 上跑的 op 总结

只有以下情况会真正落到 CPU：

1. **CUDA backend 不支持的 op**（`supports_op = false`）：FILL、TRI、DIAG（mask 准备类）
2. **dst buffer 预分配在 CPU**：KV cache 在 CPU 时的 SET_ROWS、token embedding 输出
3. **被 ollama 显式分配到 CPU 的 op**：output projection（ggml.go:486）
4. **expand pass 副作用**：少数情况下相邻 op 把整个 split 拉到 CPU（如 qwen3-coder-next blk.0 的 deltanet 整层）

剩下所有 CPU-resident 层的 weight matmul、activation matmul、normalization、激活函数等等**全部被 op_offload 移到 GPU 上跑**——只是每次都付一次 H2D 把 weight 从 CPU plain malloc 拷到 GPU input_cpy。

#### 7.7.4 这对优化的指导意义

| 优化策略 | 是否能减少 CPU 上跑的 op 数？ | 是否能减少 H2D 总字节数？ | 是否能减少 H2D 单次延迟？ |
|---|---|---|---|
| §8.1 把 CPU buffer 改成 cuda_host pinned | ❌（op 调度结果不变） | ❌ | ✓（pageable 14 GB/s → pinned 25 GB/s） |
| §8.2.1 把 KV cache 全部放 GPU | ❌ | ✓（attention 层省 128M × N 的 KV H2D） | — |
| §8.2.2 selective copy_experts 进一步 | ❌ | ✓（只搬激活的 expert）| — |
| 把 output layer 也放 GPU | ✓（少几个 CPU MUL_MAT） | ✓ | — |
| 加 fused mask op（合并 FILL/TRI/DIAG） | ✓（如果在 CUDA 实现） | — | — |

**结论**：**524 ms gap 的主因不是"CPU 上跑了多少 op"，而是"H2D 走的是 pageable 而不是 pinned"**。这就是 §8.1 优先级最高的原因。

---

## 八、Phase 5：优化路线

### 8.1 短期目标（追平 llama runner）

**改动核心**：修改 `ml/backend/ggml/ggml.go:160-170`，让 ollama 在构建 `cpuDeviceBufferType.bts` 时**额外**调用 `ggml_backend_dev_host_buffer_type(gpu_dev)` 并把返回的 cuda_host buft **插到列表前列**（参考 `llama-model.cpp:343-349`）。

**核心代码**（约 +12~15 行，单文件）：

```go
// === NEW: prefer GPU's host buffer type (cudaMallocHost pinned)
//          for CPU-side weights. Mirrors llama.cpp's make_cpu_buft_list.
for _, d := range gpus {
    if hostBuft := C.ggml_backend_dev_host_buffer_type(d); hostBuft != nil {
        cpuDeviceBufferType.bts = append(
            []C.ggml_backend_buffer_type_t{hostBuft},
            cpuDeviceBufferType.bts...,
        )
        btDeviceMemory[hostBuft] = &requiredMemory.CPU
        break  // only first GPU's host buft
    }
}
```

**预期效果**：

- **Prefill**: ollama 从 2117 ms → ~1600 ms（追平 llama runner）。机制：CPU 层权重移到 cuda_host pinned，H2D 从 pageable 14 GB/s → pinned 25 GB/s
- **Decode**: 保留 ollama 现有 18 t/s 优势。机制：decode batch_size=1 不满足 `>= 32` op_offload 阈值（ggml-cuda.cu:4940），op 仍跟着 weight 落到 CPU backend；pinned host memory 对 CPU 自身访问无副作用

**潜在副作用**：

| 项 | 影响 | 缓解 |
|---|---|---|
| VRAM 映射成本 | 旧实验 Stage 3 实测从 21.76 → 22.80 GiB（+1 GiB） | 你机器 23.1 GiB free，应可 fit；如不行加 env 开关回退 |
| cudaMallocHost OOM 风险 | 30 GiB 一次性分配可能失败 | hostBuft 分配返回 nullptr 时 createTensor 自动 fallback 到 plain CPU buffer (ml/backend/ggml/ggml.go:245 的 for 循环兜底) |
| 模型加载时间 | cudaMallocHost 30+ GiB 比 malloc 慢一些 | 一次性成本，Stage 3 旧实验未报告问题 |
| 系统级 pinned 限制 | Linux 默认 ulimit -l 可能限制 | 你 128 GB RAM + Windows，无问题；Linux 下需检查 |
| 多 GPU 行为 | 跟 llama.cpp 一致：只取第一个 GPU 的 cuda_host buft | break 取首个，无歧义 |
| KV cache 配置不变 | 1.5 GiB CPU KV 仍在 plain malloc | 这一项是另一个独立优化点（见 §8.2.2）|

### 8.2 中期目标（超过 llama runner）

#### 8.2.1 KV cache 放置策略对齐

ollama 当前 1.5 GiB KV 在 CPU、1.1 GiB 在 GPU；llama 是 494 MiB CPU + 349 MiB GPU。即便 §8.1 让 mul_mat 在 GPU 跑，**CPU 层 attention 的某些 op 仍会因 KV 在 CPU 而落到 CPU backend**。

如果 KV cache 全部放 GPU（或大部分），prefill 还能再省一点（具体多少需实测，但旧实验 Stage 4 看 GPU 利用率说明 KV 在哪不太敏感）。

实施位置：`kvcache/causal.go:464` `c.ctxs[c.curLayer] = c.backend.NewContextSize(2).Layer(c.curLayer)` —— 让这条根据层归属在 GPU/CPU 间选择，而不是统一走"层的默认 backend"。需进一步阅读 `Layer()` 实现。

#### 8.2.2 Selective copy_experts (Stage 3 路线)

旧实验 Stage 3（1392 ms）比 llama runner（1597 ms）快 205 ms，靠的是 **selective copy_experts**（每层只搬当前 batch 真正激活的 expert，约 63%）。

如果 §8.1 已经追平了 llama runner（ollama ~1600 ms），上 selective 可能再省 200ms 进入 1400 ms 区间。

但实施工作量较大（需要在 ggml-backend.cpp 端针对 mul_mat_id 做特化），不是 1.x 阶段目标。

#### 8.2.3 Recurrent checkpoint 内存优化

ollama 端 24 个 checkpoint 占 ~1.8 GiB（§7.6.2），不影响 prefill 性能但占系统 RAM。如果用户场景不依赖 sequence rollback，可以加 env 关闭（`OLLAMA_RECURRENT_CHECKPOINTS=0` 之类），节省 ~1.8 GiB。

### 8.3 不应做的（已被数据排除）

- ❌ porting `can_reuse`：实测 0% 命中
- ❌ 优化 model.Forward CGO：6.5 ms 可忽略
- ❌ Context 生命周期复用：< 5 ms 可忽略
- ❌ 减少 Go 模型节点数：ollama 已比 llama 少一半还慢

---

## 九、Phase 6：buft 改动实测结果（2026-05-18）

### 9.1 改动

`ml/backend/ggml/ggml.go` 在 cpuDeviceBufferType.bts 前列加 `ggml_backend_dev_host_buffer_type(gpu_dev)`，让"分配到 CPU"的权重落在 CUDA pinned host buffer 而非 plain malloc。

通过 `OLLAMA_PINNED_HOST_BUFFER=1` 启用（默认关，opt-in）。commit `9cdb55d3`，+42 行。

### 9.2 性能对比（batch=1024, prompt=1024, max_tokens=16, epochs=6, warmup=4）

| 配置 | prefill_ms | prefill_tps | gen_tps | 备注 |
|---|---:|---:|---:|---|
| ollama runner，pinned OFF | 2061 ms | 475 t/s | 18 t/s | 基线（与之前 2117 ms 在噪声内） |
| **ollama runner，pinned ON（run 1）** | **1473 ms** | **665 t/s** | **18 t/s** | -588 ms vs OFF |
| **ollama runner，pinned ON（run 2 复测）** | **1469 ms** | **667 t/s** | **18 t/s** | CV 极低，可复现 |
| llama runner（参考） | 1591 ms | 616 t/s | 11 t/s | 用作对照 |

**关键结论**：

1. **目标命中且超出预期**：原计划目标是"追平 llama runner"，实测 ollama runner pinned ON 比 llama runner **还快 118 ms / 8% (prefill)**
2. **decode 完全不退化**：18 t/s 三个测试一致——op_offload 的 `batch_size>=32` 阈值正确保护 decode 路径
3. **复测 CV 低**（CV 1.5–2.0%）：改动效果稳定，不是抽样误差
4. **绝对最优**：ollama runner pinned ON 现在 prefill 比 llama runner 快 8%，decode 比 llama runner 快 64%——全方位 dominant

### 9.3 为什么超过 llama runner（推测）

原本预期是"追平"，实测超出 ~118 ms。可能原因：

1. **图节点数优势在没有 H2D 瓶颈后浮现**：ollama 的 Go model.Forward 节点数 17.5k vs llama 34.6k——少一半。当 H2D 不是瓶颈，更少节点意味着更少 sched 调度开销 + 更少 kernel launch
2. **Decode 优势的副作用**：ollama 的 KV 配置（1.1 GiB GPU + 1.5 GiB CPU）虽然在某些 attention layer 多了 H2D，但其他层路径可能更短

但这个 +8% 的具体来源**目前没有直接 profile 数据支撑**，只是观察推测。如果后续 PR review 时有人问，需要再做一次 GPU 利用率曲线对比 + 节点级 timing。

### 9.4 待补充验证

下面这些尚未做、不影响当前结论但 PR 提交前应做：

- [ ] GPU 利用率曲线（pinned ON 后是否变成 80%+ 平台，跟 llama runner 一样）
- [ ] Task Manager Shared GPU memory（pinned ON 后是否出现 ~28 GiB）
- [ ] Long prompt 多 ubatch 验证（batch=1024/prompt=4096，4 个 ubatch）
- [ ] Long prompt 小 batch 验证（batch=512/prompt=4096，8 个 ubatch）
- [ ] VRAM 压力测试：在更紧的 VRAM 配置下确认 cudaMallocHost OOM fallback 路径正确工作

---

## 十、Next Steps

### ✓ Step 1（已完成 2026-05-18）：实现 buft 改动

commit `9cdb55d3`。`OLLAMA_PINNED_HOST_BUFFER` env 开关，默认关。+42 行单文件改动。

### ✓ Step 2（已完成 2026-05-18）：性能验证

见 §九 Phase 6 数据。**目标命中**（prefill 2061 → 1473 ms），且**超出预期**（比 llama runner 快 8%），decode 不退化（18 t/s）。

### Step 3（待做）：补充验证 + GPU 利用率对照

为 PR 提交准备完整数据：

1. **跑 GPU 利用率曲线**：用 `scripts/profile-gpu-prefill.ps1` 在 pinned ON 模式下重新采，确认 GPU util 从锯齿 60% 变成 80%+ 平台
2. **Task Manager Shared GPU memory**：截图确认 pinned ON 后出现 ~28 GiB shared，证实 pinned 真实生效
3. **Long prompt 多 ubatch**：跑 batch=1024/prompt=4096（4 ubatch）和 batch=512/prompt=4096（8 ubatch），看 pinned 改动在多 ubatch 下也工作
4. **VRAM 压力测试**：在更紧的 VRAM 配置（比如手动调整 GPU layers 让 VRAM 占用接近上限）下确认 cudaMallocHost OOM fallback 路径正确

### Step 4（待做）：评估 PR 提交

如果 Step 3 全部通过：

1. 跨平台兼容性确认：
   - Linux：默认 ulimit -l 通常足够，但用户可能受限
   - AMD ROCm：HIP backend 也实现了 host_buffer_type（需确认 hip.h 等价路径）
   - Apple Silicon Metal：不适用（unified memory）
   - 仅 CPU：无 GPU 时 for 循环空跑，行为退化为原版
2. 决定提交策略：
   - 当前 default off（opt-in）—— 安全
   - 建议先 PR opt-in 版本，观察 1-2 周反馈再考虑 default on
3. PR 描述卖点：
   - "对齐 llama.cpp 的 make_cpu_buft_list 行为"（最小风险）
   - "qwen3-coder-next prefill +40%（实测 1473 vs 2061 ms）"（强收益）
   - "decode 不变（保留 ollama 优势）"
   - 跨模型测试一组主流模型（llama 3.x、deepseek、qwen 各类）确认普适

### Step 5（可选）：§8.2 中期优化是否还有必要？

**§8.2.1 KV cache 放置**：原本目标是"如果 §8.1 后 GPU util 还没达 85%，做 KV 放置 GPU"。
- 当前 ollama runner pinned ON 已经比 llama runner 快——除非 Step 3 测出 GPU util 还没满，否则**这个方向 ROI 大幅下降**

**§8.2.2 Selective copy_experts**：原本目标是"超过 llama runner"。
- 已经超过了——这个方向**优先级降为最低**，除非有特定用户场景需要榨干性能

### 不进入 next steps 的方向（保留）

| 方向 | 原因 |
|---|---|
| 完整 profiling Phase 1.5（split 内部 timing）| 已被 §7.4 dump + §7.5 三条证据汇合替代 |
| porting can_reuse | 0% 命中，无收益 |
| 测 100% GPU offload 对比 | 24 GB VRAM 装不下 52 GB 模型，物理不可行 |
| 优化 Go 模型节点数 / CGO | model_forward 才 6.5 ms，方向错了 |

## 十一、变更日志

| 日期 | 变更 | 原因 |
|---|---|---|
| 2026-05-15 | 初稿 | 三组 reuse 实验确认 can_reuse 在单 ubatch 场景无收益，重新规划 profiling 方向 |
| 2026-05-15 | 删除 Phase 0，合并至 Phase 1 | 单独看 ollama 端 nodes/splits 无对照价值，且 slog.Debug 输出会被淹没 |
| 2026-05-15 | 填入 Phase 2 数据 + Phase 3 归因 | 实测：gap 几乎全在 compute_total（545 ms），与图构建/CGO/生命周期无关；can_reuse 实测 0% 命中；ollama 图节点反而比 llama 少一半；推荐 Phase 1.5 走 split 内部 timing + 100% GPU offload 对照 |
| 2026-05-15 | 加入 Phase 4 GPU 利用率分析 + 旧实验对照（Stage 4 ≈ llama runner）；把 root cause 收紧到 `ml/backend/ggml/ggml.go:160-170` 缺失 `ggml_backend_dev_host_buffer_type(gpu_dev)` 调用；写出 Phase 5 优化路线 | 三条独立证据线（GPU util、shared memory、Stage 4 对照）汇合，无需进一步 split 内部 timing |
| 2026-05-15 | 修订 §7.4 为 §7.4.1-7.4.3（精确的 sched 决策树还原 + ollama 默认 op_offload 实际命中、慢在 H2D 带宽和同步行为）；新增 §7.6 内存账本对账（KV checkpoint +1.8 GiB / compute graph 估算 vs 实测）；§二 候选清单压缩为 1 表；删除 §6.2 / §6.3 已过时段落（与 §7.4 / §8.3 重复）；新增 §十 Next Steps | 之前的"plain CPU buffer 让 op_offload 失效"是过度简化；正确表述是 op_offload 在 ollama 默认就触发，gap 来自 pageable vs pinned 的 H2D 带宽差 + cuda_buffer_set_tensor 内部 sync。澄清 KV/compute graph 内存账本以避免误读 |
| 2026-05-15 | 用 GGML_SCHED_DEBUG=2 dump 直接观察 op 调度结果，重写 §7.4 为基于 dump 的事实（不再是代码静态读推测）；新增 §7.7 hybrid offload 通用调度规则总结表 | 之前 §7.4.1-7.4.3 靠代码推断"很可能在 CUDA"——dump 直接确认 attention mul_mat、FLASH_ATTN、MoE expert mul_mat_id 全部在 CUDA0 跑；CPU 上只有 4 类 op（CUDA 不支持的 mask op、KV cache 写回、token embedding、output projection、blk.0 deltanet 的 expand 副作用）。524 ms gap 的真因 = H2D 走 pageable 14 GB/s 而非 pinned 25 GB/s |
| 2026-05-18 | 实施 §8.1 的 buft 改动（commit 9cdb55d3）；新增 §九 Phase 6 实测结果；更新 §十 Next Steps 把 Step 1+2 标完成 | 实测命中预期 + 超出（pinned ON 后 ollama prefill 1473 ms < llama runner 1591 ms）；decode 18 t/s 不退化；§8.2 中期优化优先级因此下降 |
