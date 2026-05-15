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

## 六、Phase 3：根因定位与优化方向

### 6.1 关键证据链

1. **gap 几乎全部（545/524 ms）在 `compute_total` / `lc.Decode` 内部**，不在 Go 层任何阶段
2. **图节点数 ollama < llama**（17.5k vs 34.6k）但 ollama 反而慢——节点数不是直接因子
3. **splits 数几乎相同**（595 vs 612）——backend 切分粒度不是主因
4. **Decode 阶段 ollama splits 只有 3**（vs llama 49）——证明 ollama 在简单 batch 上的 split 优化是好的，**但在 prefill 大 batch 上未达到同等优势**
5. **can_reuse 0% 命中率**——porting 该机制无收益

### 6.2 真正的 root cause 候选

gap 来自 ggml backend 内部的 `compute_splits`（ggml-backend.cpp:1480）执行——具体说，**595 splits 之间的 H2D/D2H 拷贝、sync 点、与 GPU compute 的交互模式**在两端不同。即使 splits 数量接近，每个 split 包含的 nodes、归属哪个 backend、需要拷贝哪些张量都不同。

可能的细分原因：

| 候选 | 说明 |
|---|---|
| **A. CPU split 比例不同** | 49 层 hybrid offload 下，ollama 和 llama 给 CPU 跑的层组合可能不同；CPU 跑的层是慢节拍 |
| **B. MoE expert weight 拷贝模式不同** | `mul_mat_id` 的 expert offload heuristic（ggml-backend.cpp:1515-1599）有"only used experts"优化；可能 ollama 触发更多 expert 拷贝 |
| **C. Node fusion 程度不同** | ollama 节点少一半暗示更多 fusion，但**也可能某些 fused op 在 partial offload 下走 CPU 路径慢于 llama 端的非 fused 但全在 GPU 上的 op** |
| **D. KV cache 路径** | 49 层中包含 attention 层与 deltanet 层；Go 与 C++ 实现的 KV cache 写入策略可能不同 |

### 6.3 不应做的优化方向（已被数据排除）

- ❌ **porting `can_reuse`**：实测 0% 命中率，无收益
- ❌ **优化 model.Forward 的 CGO 累积**：6.5 ms 构图开销可忽略
- ❌ **Context 生命周期复用**：new_context + close 加起来 < 5 ms
- ❌ **简化 Go 模型节点数**：ollama 已经比 llama 节点少一半还慢，方向相反

## 七、Phase 4：GPU 利用率对照 + 旧实验交叉验证（2026-05-15）

Phase 3 用代码静态读得出"split 内部执行模式不同"的猜测，Phase 4 通过两条独立证据链把猜测**收紧到一行根因**。

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

### 7.4 代码层证据

**根因锁定：CPU 层 weight buffer type 选择**。

#### llama.cpp 端（正确的做法）

`llama/llama.cpp/src/llama-model.cpp:321-380` `make_cpu_buft_list`：

```cpp
static buft_list_t make_cpu_buft_list(const std::vector<ggml_backend_dev_t> & devices, ...) {
    buft_list_t buft_list;

    // ...先加 ACCEL buffer...

    // add a host buffer type
    // storing the tensors in a host buffer is useful when the processing of large batches
    // is offloaded to a GPU device, since it reduces the time spent on data transfers
    if (!no_host) {
        for (auto * dev : devices) {                                    // ← devices 含 GPU dev
            ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);  // ← 拿 CUDA pinned host buft
            if (buft) {
                buft_list.emplace_back(dev, buft);                      // ← 优先级靠前
                break;
            }
        }
    }

    // 后面才是真正的 plain CPU buffer fallback
    // ...
}
```

`ggml_backend_dev_host_buffer_type(cuda_dev)` 返回 `ggml_backend_cuda_host_buffer_type()`（cuda.cu:1291），**底层是 `cudaMallocHost`**（cuda.cu:1264）。

运行日志可见：
```
load_tensors: CUDA_Host model buffer size = 29804.92 MiB
```

#### ollama 端（缺失的做法）

`ml/backend/ggml/ggml.go:160-170`：

```go
cpuDeviceBufferType := deviceBufferType{d: ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)}
for _, d := range append(accels, append(gpus, cpus...)...) {
    switch ggml_backend_dev_type(d) {
    case GGML_BACKEND_DEVICE_TYPE_CPU,
         GGML_BACKEND_DEVICE_TYPE_ACCEL:                 // ← 只对 CPU/ACCEL device
        bt := ggml_backend_dev_buffer_type(d)            // ← 用 plain CPU malloc
        cpuDeviceBufferType.bts = append(cpuDeviceBufferType.bts, bt)
    }
    // GPU device 被 switch 跳过 → 完全不调用 ggml_backend_dev_host_buffer_type(gpu_dev)
}
```

**ollama runner 完全跳过了 `ggml_backend_dev_host_buffer_type(gpu_dev)` 这条调用**。CPU 层的 weight 被存到 plain malloc 的 host memory，`is_host` 虽然返回 true，但 buft 不是 cuda_host buft，不参与 cuda backend 的 op_offload 路径——op 跟着 weight 走，落到 CPU backend 上跑。

运行日志可见：
```
compute graph device=CUDA0 size=884.1 MiB     ← 只有 GPU 层的中间张量
compute graph device=CPU size=270.6 MiB        ← CPU 上有 270 MiB compute buffer
```
（无 `CUDA_Host model buffer` 这一行）

### 7.5 三条独立证据汇合

| 证据线 | 结论 |
|---|---|
| GPU 利用率曲线 | ollama 锯齿 60% / llama 平台 85% → ollama 工作不连续 |
| Task Manager Shared GPU memory | ollama 0.1 GB / llama 29.4 GB → llama 把 ~30 GB 映射到 GPU 可访问的 pinned memory |
| 旧实验 Stage 4 ≈ llama runner（1648 vs 1597 ms） | 完整 H2D + pinned + GPU compute 的执行模式精确还原 llama runner 性能 |
| 代码 grep | ollama 跳过 `ggml_backend_dev_host_buffer_type(gpu_dev)` |
| ggml 内部日志 | llama 有 `CUDA_Host model buffer = 29804.92 MiB`，ollama 没有 |

**结论**：500ms gap 的全部来源 = "**ollama 把 CPU 层权重放在 plain malloc → CPU 上跑 mul_mat**" vs "**llama 把 CPU 层权重放在 cudaMallocHost pinned → GPU 通过 PCIe DMA 拉取 → GPU 上跑 mul_mat**"。

---

## 八、Phase 5：优化路线

### 8.1 短期目标（追平 llama runner）

修改 `ml/backend/ggml/ggml.go:160-170`，让 ollama 在构建 `cpuDeviceBufferType.bts` 时也调用 `ggml_backend_dev_host_buffer_type(gpu_dev)` 并放在优先位（参考 `llama-model.cpp:343-349`）。

预期效果：ollama prefill 从 2117 ms 拉到 ~1600 ms，与 llama runner 持平，但**保留 ollama 的 decode 优势**（decode batch_size=1 不触发 op_offload，依然是 CPU 算单 token，仍是 ollama 18 t/s vs llama 11 t/s）。

但需要注意几个潜在副作用：
1. **VRAM 占用增加**：CUDA driver 为 pinned 页建立映射会占额外 dedicated VRAM（旧实验 §4.4 提到 +1 GB 量级）。需在 24 GB VRAM 内验证 fit
2. **冷启动慢**：cudaMallocHost 30+ GB 系统内存，模型加载时间会延长
3. **Decode 模式适用性**：要确保 decode 路径（batch_size<32）不会被 op_offload 错误触发

### 8.2 中期目标（**超过** llama runner）

旧实验 Stage 3 的 selective copy_experts (1392 ms) 比 llama runner (1597 ms) 还快 ~205 ms。如果 ollama 复用 Stage 3 的 selective 优化，可能比 llama runner 还快。

但这不是 1.x 阶段目标——先追平再说。

### 8.3 不应做的（已被数据排除）

- ❌ porting `can_reuse`：实测 0% 命中
- ❌ 优化 model.Forward CGO：6.5 ms 可忽略
- ❌ Context 生命周期复用：< 5 ms 可忽略
- ❌ 减少 Go 模型节点数：ollama 已比 llama 少一半还慢

## 九、变更日志

| 日期 | 变更 | 原因 |
|---|---|---|
| 2026-05-15 | 初稿 | 三组 reuse 实验确认 can_reuse 在单 ubatch 场景无收益，重新规划 profiling 方向 |
| 2026-05-15 | 删除 Phase 0，合并至 Phase 1 | 单独看 ollama 端 nodes/splits 无对照价值，且 slog.Debug 输出会被淹没 |
| 2026-05-15 | 填入 Phase 2 数据 + Phase 3 归因 | 实测：gap 几乎全在 compute_total（545 ms），与图构建/CGO/生命周期无关；can_reuse 实测 0% 命中；ollama 图节点反而比 llama 少一半；推荐 Phase 1.5 走 split 内部 timing + 100% GPU offload 对照 |
| 2026-05-15 | 加入 Phase 4 GPU 利用率分析 + 旧实验对照（Stage 4 ≈ llama runner）；把 root cause 收紧到 `ml/backend/ggml/ggml.go:160-170` 缺失 `ggml_backend_dev_host_buffer_type(gpu_dev)` 调用；写出 Phase 5 优化路线 | 三条独立证据线（GPU util、shared memory、Stage 4 对照）汇合，无需进一步 split 内部 timing |
