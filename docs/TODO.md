# TODO — Ollama 性能优化待办

> 讨论中产生的所有待办事项集中在此。每个 item 标注优先级和来源。
> Claude 在讨论中产生新的 TODO 想法时，应追加到此文件。

---

## 高优先级

### Observability 补充

- [ ] **Thread 级 instrumentation**：在 ggml-cpu.c worker 线程循环中加打点，记录 per-thread 的 compute time vs barrier wait time。这是调 `num_thread`, `cpumask`, `poll`, `chunk_size` 的前提。
  - 来源：PGO brainstorm（2026-03-24）
  - 相关代码：`ggml-cpu.c:549-585`（barrier）、`ggml-cpu.c:2921-2963`（compute loop）

- [ ] **Split 边界同步耗时**：在 `ggml-backend.cpp` 的 split 循环中记录 `ggml_backend_synchronize()` 和跨 backend 拷贝的精确耗时（当前在 eval callback 之外不可见）
  - 来源：PGO brainstorm（2026-03-24）
  - 相关代码：`ggml-backend.cpp:1480-1664`

- [ ] **Barrier tracing + 可视化**：在 thread instrumentation 基础上，将 barrier wait time 数据集成到 trace-analyzer 的 Timeline/Hotspot 视图中，实现可视化分析
  - 来源：PGO brainstorm（2026-03-24）
  - 前置：thread 级 instrumentation 完成
  - 涉及：`tools/trace-analyzer/` CLI + React SPA

### CPU Affinity

- [ ] 确认线程是否真的被调度到 E-core（Process Explorer 验证）
  - 来源：CPU offload latency 讨论（2026-03-24）

- [ ] 测试 affinity 绑定 P-core 后的 latency 变化
  - 来源：CPU offload latency 讨论（2026-03-24）

- [ ] 评估把 affinity 信息从 ollama Go 层传到 ggml 线程池的可行性
  - 来源：CPU offload latency 讨论（2026-03-24）
  - 相关代码：`discover/cpu_windows.go`、`ggml-cpu.c:2431-2462`、`ggml-cpu.c:3040-3044`

### PGO 第 1 轮（Expert-driven）

- [ ] 基于已有 trace 体系，跑 Qwen3 27B split 场景收集 trace
- [ ] 用 trace-analyzer report 生成分析，识别第 1 层（Go 层）参数调优机会
- [ ] 列出第一批值得调的参数并设计实验

---

## 中优先级

### Weight Swapping

- [ ] Weight swap POC：在 ggml 中实现最简单的单轮 swap，验证收益（**仅 CUDA 独显场景**）
  - 来源：CPU offload latency 讨论（2026-03-24）
  - 相关代码：`ggml-backend.cpp:1480-1664`

- [ ] 测量实际的 PCIe 带宽和 CPU 每层计算时间，代入公式计算最优 swap 层数
  - 来源：CPU offload latency 讨论（2026-03-24）

- [ ] 调研 FlexGen 的 overlapping schedule 实现细节
  - 来源：CPU offload latency 讨论（2026-03-24）

### iGPU 验证

- [ ] 在实际 Intel iGPU 机器上确认 DXGI 报告的 SharedSystemMemory 值
  - 来源：CPU offload latency 讨论（2026-03-24）
  - 相关代码：`mem_dxgi_pdh.cpp:234-278`

- [ ] 测量 iGPU Vulkan 场景下的 split 实际开销（是否如预期接近零）
  - 来源：CPU offload latency 讨论（2026-03-24）

---

## PR 回上游

- [ ] 规划 PR 拆分策略：将 dev 上的改动按功能分批 PR 回 ollama/ollama（profiler core → profiler integration → trace-analyzer）(来源: 2026-03-26)
  - 初步思路：PR1 纯 Go profiler 框架，PR2 llama.cpp/runner 集成，PR3 trace-analyzer 或独立 repo
  - 待 brainstorming 细化具体范围和提交顺序

---

## 低优先级 / 未来方向

### PGO 完整系统（长期方向）

- [ ] **Profile-Guided Inference Tuning 完整实现**：在 barrier tracing + perftune-agent 成熟后，推进方案 C（混合方案）的完整调优闭环
  - 来源：PGO brainstorm（2026-03-24）
  - 前置：(1) barrier/thread instrumentation (2) perftune-agent 集成 skills (3) 第 1 轮短平快参数优化经验
  - 状态文件：`docs/superpowers/plans/2026-03-24-pgo-brainstorm-state.md`
  - 方案：Expert+Trace → Thread instrumentation → Targeted sweep → Per-model per-hardware profile configs
  - 集成目标：`c:/workspace/perftune-agent`（Athanor agent + skill 体系驱动自动调优）

### GPU Kernel 层参数调优

- [ ] 调研 CUDA block sizes、`MMQ_ITER_K`、Vulkan `BM/BN/BK` tiling 等编译时常量的调优空间
  - 来源：PGO brainstorm（2026-03-24）
  - 备注：编译时常量，需重编译，复杂度高。除非发现 low-hanging fruit 否则暂缓

---

## 参考文档

- [PGO Brainstorm 状态](superpowers/plans/2026-03-24-pgo-brainstorm-state.md)
- [CPU Offload Latency 优化 Memo](superpowers/plans/2026-03-24-cpu-offload-latency-optimization.md)
- [线程分析报告](threading-analysis-report.md)
- [Qwen3 CPU Offload Lock 分析](qwen3-cpu-offload-lock-analysis.md)
- [调试与 Profiling 手册](debugging-and-profiling.md)
- [Profiling 系统设计](superpowers/specs/2026-03-19-ollama-profiling-tracing-design.md)
- [Trace 可视化设计](superpowers/specs/2026-03-20-trace-visualization-design.md)
