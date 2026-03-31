# Ollama Internals 文档

深入分析 Ollama 内部实现机制，重点关注推理性能优化。

## 文档索引

| 文件 | 内容 | 行数 |
|------|------|------|
| [01-eval-callback-tracing.md](01-eval-callback-tracing.md) | Eval Callback 追踪机制：回调两阶段协议、CGO 桥接、profiler 数据流 | ~280 |
| [02-ggml-backend-scheduler.md](02-ggml-backend-scheduler.md) | GGML Backend Scheduler：图分割算法、split 执行、内存复用、event 同步、MoE op_offload | ~595 |
| [03-ollamarunner-call-chain.md](03-ollamarunner-call-chain.md) | OllamaRunner 完整调用链：主循环、ggml_context、构图机制、batch size、懒同步 | ~298 |
| [04-memory-allocation.md](04-memory-allocation.md) | 内存分配机制：权重分配、计算图 gallocr、运行时 cache、完整生命周期、调用链图 | ~277 |
| [05-cross-library-gpu-mixing.md](05-cross-library-gpu-mixing.md) | 跨 Library GPU 混用：ByLibrary 竞争选举、CUDA+Vulkan 限制、iGPU UMA 价值分析、Vulkan 零拷贝、Phase-Aware Scheduling | ~373 |
| [06-optimization-directions.md](06-optimization-directions.md) | 优化方向分析：KV cache 量化、Speculative Decoding/EAGLE3、MoE 感知拆分、Hybrid 层感知、备选方案 | ~290 |
| [07-community-perf-survey.md](07-community-perf-survey.md) | 社区性能调研（2026-03-31）：Ollama/llama.cpp issues/PRs 汇总、Go engine vs llama.cpp 差距、综合分析与优先级排序 | ~363 |

## 硬件场景

主要针对：**NVIDIA 24GB + Intel iGPU + 128GB DDR5 RAM**，运行 **Qwen3-coder-next 80B (Q4_K_M)**。

## 关键发现速查

- **`OllamaEngineRequired()` 强制 qwen3next 走 Go engine** — 1 行改动即可切换到 llama.cpp runner（[07 §8.8](07-community-perf-survey.md#888-ollama-go-engine-vs-llamacpp-engine-性能差距)）
- **`--n-cpu-moe` 未暴露** — 实测 5-7× MoE decode 加速（[07 §8.2](07-community-perf-survey.md#82-moe-优化)）
- **Vulkan UMA 零拷贝已在上游实现** — 无需数据拷贝（[05 §6.7](05-cross-library-gpu-mixing.md#67-vulkan-uma-零拷贝机制上游已实现)）
- **EAGLE3 对 MoE 模型无效** — 验证阶段 bottleneck（[07 §8.1](07-community-perf-survey.md#81-speculative-decoding--eagle3)）
- **推荐优先级**：切换 llama.cpp runner → `--n-cpu-moe` → KV Q8 → MoE 感知拆分 → Gate fusion → iGPU offload → TurboQuant
