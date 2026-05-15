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

### 7.4 代码层根因 —— "CPU 层在 CPU 算 vs GPU 算" 的精确还原

> 注意：这一节是基于代码静态读 + ggml-backend scheduler 决策路径的精确还原，
> 修正了之前"plain CPU buffer 让 op_offload 失效"的过度简化。

#### 7.4.1 ollama 默认配置下 mul_mat 跑在哪？—— 按 op 类型不同

走 `ggml_backend_sched_backend_id_from_cur`（ggml-backend.cpp:824）的决策树，对每个 op：

**op = `mul_mat` / `mul_mat_id`（CPU 层 MoE expert 矩阵乘，prefill batch=1024）**：

1. 找 weight 在哪：weight 在 plain CPU buffer，CUDA `supports_buft(plain_cpu_buft) = false`（`ggml-cuda.cu:4922`，要求 `is_cuda(buft)`，integrated GPU 才允许 `is_cuda_host`），CPU `supports_buft = true` → `src_backend_id = CPU backend = sched->n_backends - 1`
2. 进入 op_offload 判断（line 865）：
   - `op_offload = true`（ollama 在 ggml.go:389 硬编码）✓
   - `batch_size = 1024 ≥ 32` ✓
   - `src_backend_id == sched->n_backends - 1` ✓
   - `is_host(plain_cpu_buft) = true`（`ggml_backend_cpu_buffer_type_is_host` 返回 true，cpu-backend.cpp:2314）✓
3. 进入内层循环（line 866-871）尝试更高优先级 backend：
   - CUDA backend `supports_op(mul_mat) = true`
   - CUDA backend `offload_op(mul_mat with batch=1024) = true`（line 4940-4943，要求 `op->ne[1] >= 32`）
   - **→ 该 mul_mat 节点的 `node_backend_id = 0`（CUDA）**
4. compute_splits 执行该 split 时（ggml-backend.cpp:1480）：
   - `cpy_tensor_async(input_backend=CPU, split_backend=CUDA, ...)` 检查 line 2927-2933 → src 不是 cuda buffer → **返回 false**
   - 走 fallback（line 1601-1611）：先 `synchronize(CPU)` + `synchronize(CUDA)` + `tensor_copy`
   - `tensor_copy` 调 `cuda_buffer_set_tensor`（ggml-cuda.cu:748-754）：
     ```cpp
     cudaMemcpyAsync(..., cudaMemcpyHostToDevice, cudaStreamPerThread);
     cudaStreamSynchronize(cudaStreamPerThread);   // ← 立刻 sync！
     ```
   - 因为 src 是 plain pageable memory，`cudaMemcpyAsync` 内部要先 driver staging 到 pinned 中转，再 DMA → **有效带宽 ~14 GB/s**

**op = `attention` 系列（attention 层的 K/V 读、softmax 之类，CPU 层）**：

- attention 的 src 之一是 KV cache。ollama 把 KV cache 1.5 GiB 放在 CPU device（plain CPU buffer），attention op 跟着 KV 走 → `src_backend_id = CPU`
- 但 KV 不是 weight（`buffer->usage != WEIGHTS`），line 862 的 if 不触发
- 落到 line 854-876 的循环之外，最终 backend 由 expand pass（line 1019 注释）决定
- expand pass 显式说 "cpu will never be used unless **weights are on cpu**"——但 KV 在 CPU 算 weight 的"实际位置"——所以**CPU 层 attention 中相关 op 实际落在 CPU backend**
- 这解释了 `compute graph device=CPU size=270.6 MiB`：CPU 上确实有真实的 compute buffer，给 attention 层的中间张量

**op = `mul_mat`（GPU 层）**：weight 在 CUDA0 VRAM，op 直接 GPU 跑，无 H2D。

#### 7.4.2 llama runner 的差异 —— buffer type 不同改变全链路

llama.cpp `make_cpu_buft_list`（llama-model.cpp:321-380）在构建 cpu_buft_list 时**额外**调用 `ggml_backend_dev_host_buffer_type(gpu_dev)`，把 CUDA pinned host buft（底层 `cudaMallocHost`）放在列表前列。所有"分配到 CPU"的权重实际落在 **CUDA_Host pinned buffer**。

进入 sched 决策树后：

- weight 在 cuda_host buft，CUDA `supports_buft(cuda_host) = false`（discrete GPU；line 4922 要求 integrated）→ `src_backend_id = CPU`
- op_offload 同样满足条件（is_host(cuda_host) = true）→ mul_mat **同样被分配到 CUDA backend**
- compute_splits 执行 H2D 时：src buffer 是 cuda_host（pinned）→ `cudaMemcpyAsync` 跳过 driver staging，**直接 DMA → 有效带宽 ~25 GB/s**
- KV cache 配置不同（见 §7.6.2），llama 把更多 KV 放在 CUDA0 VRAM（CUDA0 KV 349 MiB / CPU KV 494 MiB）→ 对应层 attention 也能在 GPU 跑

#### 7.4.3 关键代码路径汇总（按调用顺序）

| 步骤 | 文件:行 | ollama 行为 | llama 行为 |
|---|---|---|---|
| 构建 cpu buft 列表 | `ml/backend/ggml/ggml.go:160-170` 或 `llama-model.cpp:321-380` | **只用 plain CPU buffer**（不调 `ggml_backend_dev_host_buffer_type(gpu)`） | **优先 CUDA_Host pinned**，plain CPU 兜底 |
| 权重分配到的 buffer 类型 | `ggml-cuda.cu:1276` `cuda_host_buffer_type_alloc_buffer` | plain `malloc` | `cudaMallocHost` |
| sched 决策 op 跑哪个 backend | `ggml-backend.cpp:824-879` | mul_mat → CUDA（op_offload 命中） | mul_mat → CUDA（同样命中）|
| H2D 拷贝路径 | `ggml-cuda.cu:748-754` `buffer_set_tensor` | pageable → driver staging → DMA，**~14 GB/s** | pinned → DMA，**~25 GB/s** |
| H2D 同步行为 | `ggml-cuda.cu:752-753` | `cudaMemcpyAsync + cudaStreamSynchronize`（每次 sync） | 同样路径，但 pinned 让单次 sync 等待时间显著缩短 |
| KV cache 放置 | `kvcache/recurrent.go` / `llama_context` | 1.5 GiB 在 CPU → 对应层 attention 在 CPU 算 | 494 MiB 在 CPU，绝大多数在 CUDA0 → attention 在 GPU 算 |

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

## 九、Next Steps

按建议顺序执行：

### Step 1（本周）：实现 §8.1 的 buft 改动

1. 在 `experiment/prefill-gap-analysis` 分支或新开 `feat/cpu-buft-pinned-host` 分支
2. 实现 §8.1 的 ~15 行改动
3. 加 env 开关 `OLLAMA_PINNED_HOST_BUFFER`（默认 enabled，便于 A/B 回退）
4. Build + smoke test：跑一次 short prompt 推理，确认模型加载日志出现 `CUDA_Host model buffer size = ...` 这一行

### Step 2（本周）：性能验证

1. 跑 bench-sweep 对比新版 ollama runner vs 当前 ollama runner vs llama runner（同样的 batch=1024 / prompt=1024 / max_tokens=16 / epochs=6 / warmup=4）
2. 跑 `scripts/profile-gpu-prefill.ps1` 重新采 GPU 利用率曲线，确认从锯齿变为 80%+ 平台
3. 比较 prefill_ms：目标是从 2117 ms → ~1600 ms
4. 比较 decode tps：目标是仍保持 18 t/s（不退化）
5. 跑 long-prompt（4096 token）多 ubatch 配置，验证 §8.1 改动在多 ubatch 下也工作

### Step 3（验证 OK 后）：考虑提交上游

如果性能数据吻合，且 decode 不退化、VRAM fit、长 prompt 也工作：

1. 评估改动的可观察副作用（pinned memory 对低端硬件、Linux ulimit、AMD ROCm 等的兼容性）
2. 整理 PR 描述（突出"对齐 llama.cpp 的 make_cpu_buft_list 行为"作为最小风险卖点）
3. 决定是默认开还是 env opt-in（建议先 env opt-in 一段时间，观察用户反馈）

### Step 4（可选）：进入 §8.2 中期优化

仅在 Step 1-3 都成功且收益已经获得社区/团队认同后再考虑。优先级：

1. KV cache 放置（§8.2.1）—— 如果 §8.1 后看到 GPU 利用率仍未达到 llama 的 85% 平台
2. Selective copy_experts（§8.2.2）—— 实施工作量大，目标是超过 llama runner

### 不进入 next steps 的方向

| 方向 | 原因 |
|---|---|
| 完整 profiling Phase 1.5（split 内部 timing）| 已被 §7.4 代码追溯 + §7.5 三条证据汇合替代，没必要再做 |
| porting can_reuse | 0% 命中，无收益 |
| 测 100% GPU offload 对比 | 24 GB VRAM 装不下 52 GB 模型，物理不可行 |
| 优化 Go 模型节点数 / CGO | 实测 model_forward 才 6.5 ms，方向错了 |

## 十、变更日志

| 日期 | 变更 | 原因 |
|---|---|---|
| 2026-05-15 | 初稿 | 三组 reuse 实验确认 can_reuse 在单 ubatch 场景无收益，重新规划 profiling 方向 |
| 2026-05-15 | 删除 Phase 0，合并至 Phase 1 | 单独看 ollama 端 nodes/splits 无对照价值，且 slog.Debug 输出会被淹没 |
| 2026-05-15 | 填入 Phase 2 数据 + Phase 3 归因 | 实测：gap 几乎全在 compute_total（545 ms），与图构建/CGO/生命周期无关；can_reuse 实测 0% 命中；ollama 图节点反而比 llama 少一半；推荐 Phase 1.5 走 split 内部 timing + 100% GPU offload 对照 |
| 2026-05-15 | 加入 Phase 4 GPU 利用率分析 + 旧实验对照（Stage 4 ≈ llama runner）；把 root cause 收紧到 `ml/backend/ggml/ggml.go:160-170` 缺失 `ggml_backend_dev_host_buffer_type(gpu_dev)` 调用；写出 Phase 5 优化路线 | 三条独立证据线（GPU util、shared memory、Stage 4 对照）汇合，无需进一步 split 内部 timing |
| 2026-05-15 | 修订 §7.4 为 §7.4.1-7.4.3（精确的 sched 决策树还原 + ollama 默认 op_offload 实际命中、慢在 H2D 带宽和同步行为）；新增 §7.6 内存账本对账（KV checkpoint +1.8 GiB / compute graph 估算 vs 实测）；§二 候选清单压缩为 1 表；删除 §6.2 / §6.3 已过时段落（与 §7.4 / §8.3 重复）；新增 §十 Next Steps | 之前的"plain CPU buffer 让 op_offload 失效"是过度简化；正确表述是 op_offload 在 ollama 默认就触发，gap 来自 pageable vs pinned 的 H2D 带宽差 + cuda_buffer_set_tensor 内部 sync。澄清 KV/compute graph 内存账本以避免误读 |
