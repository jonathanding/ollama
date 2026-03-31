# Tracing 机制与 OllamaRunner Scheduler 内部实现

> **此文档已拆分为独立文件。** 请查看 [`docs/internals/`](internals/) 目录。

## 文档索引

| 原章节 | 新文件 |
|--------|--------|
| §1 Eval Callback 追踪机制 | [internals/01-eval-callback-tracing.md](internals/01-eval-callback-tracing.md) |
| §2 GGML Backend Scheduler | [internals/02-ggml-backend-scheduler.md](internals/02-ggml-backend-scheduler.md) |
| §3 OllamaRunner 完整调用链 | [internals/03-ollamarunner-call-chain.md](internals/03-ollamarunner-call-chain.md) |
| §4-5 内存分配机制 + 完整调用链图 | [internals/04-memory-allocation.md](internals/04-memory-allocation.md) |
| §6 跨 Library GPU 混用 | [internals/05-cross-library-gpu-mixing.md](internals/05-cross-library-gpu-mixing.md) |
| §7 优化方向分析 | [internals/06-optimization-directions.md](internals/06-optimization-directions.md) |
| §8 社区性能调研 | [internals/07-community-perf-survey.md](internals/07-community-perf-survey.md) |

详细索引和关键发现速查见 [internals/README.md](internals/README.md)。
