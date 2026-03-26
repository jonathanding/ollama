# Profile-Guided Inference Tuning Brainstorm — 会话状态

> **用途**: 会话 compact 或中断后恢复用。把本文件路径发给 Claude 即可继续。
> **最后更新**: 2026-03-24
> **阶段**: **长期方向，已归档**。方案 C（混合方案）已采纳，brainstorming 完成到第 5 步（Present design 的 Section 1 已展示）。待 perftune-agent 成熟 + barrier tracing 就绪后再推进详细 spec。

---

## 1. 用户的核心想法

在 ollama 中打桩（instrumentation），收集算子/图级别的 profiling 数据，然后根据 profiling 结果**微调各层参数**，实现类似编译器 PGO (Profile-Guided Optimization) 的效果——但目标是推理运行时参数，而非编译优化。

关键词：**图/算子级别的 profile guided optimization**

## 2. 已完成的上下文调研

通过 3 个并行 subagent 完成了全面参数扫描，发现 ollama/ggml 有 **50+ 可调参数**，分三层：

### 第 1 层：Ollama Go 层（运行时可调，不需重编译）

| 参数 | 默认值 | 影响 |
|------|--------|------|
| `num_gpu` / GPU layers | auto | **最大影响** — 决定 CPU/GPU split |
| `num_ctx` | auto | 上下文大小，影响 KV cache 内存 |
| `num_batch` | 512 | batch size，影响 prefill 吞吐 |
| `num_thread` | P-core 数 | CPU 线程数 |
| `OLLAMA_FLASH_ATTENTION` | auto | Flash Attention 开关 |
| `OLLAMA_KV_CACHE_TYPE` | f16 | KV cache 量化（q4/q8/f16） |
| `OLLAMA_NUM_PARALLEL` | 1 | 并发请求数 |
| `use_mmap` | auto | 内存映射模型文件 |

### 第 2 层：ggml C 层（线程池参数 + 编译时常量）

| 参数 | 默认值 | 影响 |
|------|--------|------|
| `poll` | 50 | 线程唤醒轮询强度（0=纯sleep, 100=激进polling） |
| `prio` | NORMAL | 线程调度优先级（LOW→REALTIME 五档） |
| `cpumask` | 全零 | CPU affinity（方向 2 讨论的内容） |
| `strict_cpu` | false | 是否严格绑定 CPU |
| `chunk_size` (matmul) | 16/64 | MatMul work-stealing 粒度 |
| `blck_0`/`blck_1` | 16 | MatMul tiling 块大小 |
| `GGML_SCHED_MAX_COPIES` | 4 | pipeline parallelism tensor 副本数 |
| `GGML_SOFT_MAX_UNROLL` | 4 | softmax 展开因子 |
| `GGML_VEC_DOT_UNROLL` | 2 | vec_dot 展开因子 |

### 第 3 层：GPU Kernel 层（编译时常量）— 暂缓

CUDA block sizes, MMQ_ITER_K, Vulkan BM/BN/BK tiling 等。记录在案，除非 low-hanging fruit 否则不做。

### 关键代码位置

| 文件 | 内容 |
|------|------|
| `api/types.go` | Options/Runner 结构体，用户可设参数 |
| `envconfig/config.go` | OLLAMA_* 环境变量 |
| `llm/server.go` | LoadRequest，参数传递链 |
| `runner/ollamarunner/runner.go` | Runner 初始化，batch 处理 |
| `ml/backend/ggml/ggml/include/ggml.h:2700-2707` | threadpool_params 结构体 |
| `ml/backend/ggml/ggml/src/ggml-cpu/ggml-cpu.c` | 线程池、barrier、chunk_size、matmul |
| `ml/backend/ggml/ggml/src/ggml-backend.cpp` | scheduler, SCHED_MAX_COPIES |

## 3. 用户澄清的范围和约束

### 问题 1 回答：PGO 作用范围

**答案：A + B 层为主，C 层记录但暂缓（除非有 low-hanging fruit）**

### 场景约束

- **单请求**：不考虑 batch，不考虑多请求并发（多请求排队）
- **GPU 场景**：CUDA 独显、Intel B60 独显、Intel iGPU (Vulkan) 三种
- **模型**：Qwen3.5 27B 或更大的 70B 量化模型，**一般会 split**
- **不局限于 CPU offload latency**：看数据说话，哪里有机会就优化哪里
- **可以改 ollama/llama.cpp**：加 instrumentation 收集 profiling 数据帮助参数决策

### 问题 2 回答：PGO 工作流

三步走策略：

1. **Expert-driven 快攻**：凭代码分析判断哪些参数有调整机会
2. **Observability 基建**：搭建内部可观测机制
3. **Data-driven 第二轮**：基于观测数据筛选更多参数

### 问题 3 回答：Observability 现状与缺口

**已有（完整体系）**：
- `llm/profiler/` — per-op tracing（JSONL 格式，op/tensor/shape/dtype/backend/nanosecond timestamps）
- `tools/trace-analyzer/` — Python CLI（summary, compare, report）+ React SPA 可视化（DAG/Timeline/Compare/Hotspot）
- LlamaRunner + OllamaRunner 都已集成（`OLLAMA_TRACE_DIR` 控制）
- Vulkan per-op timing：`GGML_VK_PERF_LOGGER=1`
- `ollama-bench`：多模型/多配置对比
- 设计文档：`docs/superpowers/specs/2026-03-19-ollama-profiling-tracing-design.md`
- 可视化设计：`docs/superpowers/specs/2026-03-20-trace-visualization-design.md`
- Trace Replay 设计：`docs/superpowers/specs/2026-03-21-trace-replay-design.md`
- 调试手册：`docs/debugging-and-profiling.md`

**缺口（高优先级 TODO，已记入 `docs/TODO.md`）**：
- **Thread 级 instrumentation**：per-thread compute time vs barrier wait time。当前 per-op 时间包含 barrier 等待但无法拆分。这是调 `num_thread`, `cpumask`, `poll`, `chunk_size` 的前提。
- **Split 边界同步耗时**：`cudaStreamSynchronize` + `cudaMemcpy` 在 eval callback 之外，不可见。

**策略**：先用已有 trace 做第 1 层参数分析，同时并行开发 thread instrumentation。

### 问题 4（隐含）：Data copy 和线程信息

- **Data copy**：`GGML_OP_CPY`/`GGML_OP_DUP` 作为 op 出现在 trace 中有时间。但 split 边界的 `cudaStreamSynchronize` 不可见（在 eval callback 之外）。
- **线程级信息**：完全没有。一个 matmul 给了 10 个线程，不知道 compute vs barrier wait 比例。

## 4. 已提出的方案（Propose Approaches 阶段已完成）

### 方案 C（混合方案，已推荐，用户基本认可）

| 阶段 | 方法 | 参数 |
|------|------|------|
| 第 1 阶段：Expert + Trace 分析 | 跑 baseline trace → 识别瓶颈 → 调 Go 层参数 | `num_gpu`, `num_batch`, `KV cache type`, `flash_attention` |
| 第 2 阶段：Thread instrumentation | 补齐 barrier/thread 数据 → 分析 CPU split 内部 | 新开发 |
| 第 3 阶段：Targeted sweep | 对第 2 层参数做小范围 sweep + trace 验证 | `num_thread`, `cpumask`, `poll`, `chunk_size` |
| 产出：Profile 配置 | 存为 per-model per-hardware 推荐参数集 | — |

### 关键补充：perftune-agent 集成

用户有一个 `c:/workspace/perftune-agent` 项目，基于 OpenCode 实现了 AI 驱动的性能优化 agent 系统：

**perftune-agent 提供**：
- Athanor agent（意图识别 + skill 调度）
- Skill 体系：profiling → analysis → optimization explore 循环
- 长任务管理（long_task_plugin）
- opt-explore skill（评估→规划→执行→发散→报告方法论）
- knowledge/ 知识库

**myollama 提供**：
- Domain knowledge（50+ 参数、三层调优空间）
- Instrumentation（OLLAMA_TRACE_DIR、trace-analyzer）
- Domain-specific skills

**计划的集成 skills**：

| Skill 类型 | Skill 名称 | 职责 |
|-----------|-----------|------|
| profiling | `ollama-trace` | 跑 ollama-bench + OLLAMA_TRACE_DIR 收集 trace |
| analysis | `ollama-trace-summary` | 调用 trace-analyzer summary/compare/report |
| analysis | `ollama-thread-analysis` | 分析 thread/barrier 数据 |
| knowledge | `ollama-tuning-params.md` | 参数清单、调优空间、预期影响 |
| optimization | `ollama-param-sweep` | 对指定参数做小范围 sweep + trace 对比 |

**用户意图**：未来由 perftune-agent 驱动完整 profile→analyze→tune→verify 闭环，myollama 提供 domain knowledge 和 skills。

## 5. 上下文：之前讨论的优化方向

本 brainstorm 建立在之前 CPU offload latency 优化讨论之上：
- **Memo 文件**: `docs/superpowers/plans/2026-03-24-cpu-offload-latency-optimization.md`
- **线程分析报告**: `docs/threading-analysis-report.md`
- **Qwen3 分析**: `docs/qwen3-cpu-offload-lock-analysis.md`
- **TODO 文件**: `docs/TODO.md`（已创建，集中管理所有待办）

已确认的关键事实：
- 单请求 latency 主要来自 ggml barrier spin-wait + backend 切换同步 + CPU 计算慢
- Affinity 管道两端齐全但中间没连接（方向 2，优先探索）
- Weight swap 对 CUDA 独显高潜力，对 UMA iGPU 意义有限（方向 3）
- iGPU 通过 DXGI+PDH 报告 ~16GB 内存，小模型不需要 split

## 6. Brainstorming Checklist 进度

- [x] Explore project context — 3 个 subagent 完成参数扫描
- [x] Offer visual companion — 本话题不涉及视觉问题，跳过
- [x] Ask clarifying questions — 4 个问题已回答
- [x] Propose 2-3 approaches — 方案 A/B/C 已提出，C（混合）被采纳
- [ ] **Present design** — 下一步：展示设计细节，包含 perftune-agent 集成
- [ ] Write design doc
- [ ] Spec review loop
- [ ] User reviews spec
- [ ] Transition to implementation (invoke writing-plans)

## 7. 下一步动作

**已归档为长期方向**。推进条件：
1. perftune-agent 成熟到可以驱动 profile→analyze→tune 闭环
2. Barrier/thread instrumentation 完成（见 `docs/TODO.md`）
3. 第 1 阶段的短平快参数优化（num_gpu sweep、KV cache、flash attention）积累经验

短平快参数优化已并入 CPU offload latency 优化工作流（`2026-03-24-cpu-offload-latency-optimization.md`）。
