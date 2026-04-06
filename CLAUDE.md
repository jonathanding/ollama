# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ollama is a Go-based LLM inference platform with C/C++ native backends for GPU acceleration (CUDA, ROCm, Vulkan, Metal/MLX). It exposes OpenAI- and Anthropic-compatible APIs.

## Build & Run

```bash
# Development build (CPU-only, sufficient for most Go changes)
go run . serve

# Full build with native acceleration (requires CMake 3.21+)
cmake -B build
cmake --build build        # Linux/macOS
cmake --build build --config Release  # Windows

# Then run
go run . serve

# Force rebuild native code if CGO structs are out of sync
go clean -cache
```

## Testing

```bash
go test ./...

# CI uses synctest experiment (catches additional failures)
GOEXPERIMENT=synctest go test ./...

# Single package
go test ./llm/

# Single test
go test ./cmd/ -run TestName
```

## Linting

```bash
golangci-lint run
```

Config in `.golangci.yaml`. Formatters: gofmt, gofumpt. Notable: `errcheck` is disabled.

## Architecture

**Request flow:** CLI (`cmd/`) -> HTTP Server (`server/`, Gin framework) -> LLM Server Manager (`llm/server.go`) -> Runner process (`runner/`)

Key layers:
- **`cmd/`** - Cobra CLI with BubbleTea TUI. Entry point: `main.go` -> `cmd.NewCLI()`
- **`api/`** - REST client and shared types
- **`server/`** - Gin HTTP server, model management endpoints
- **`llm/`** - Runner process lifecycle, memory allocation, device selection
- **`ml/`** - Backend abstraction (device detection, GPU memory)
- **`model/`** - Model loading, inference pipelines, vision/multimodal support
- **`runner/`** - Inference executors: `llamarunner` (llama.cpp), `ollamarunner` (new engine), `mlxrunner` (MLX), `imagegen`
- **`llama/`** - Vendored llama.cpp with local patches (synced via `Makefile.sync`)
- **`openai/`** - OpenAI-compatible API translation
- **`anthropic/`** - Anthropic Messages API translation
- **`parser/`** - Modelfile parsing
- **`convert/`** - Model format conversion

## Commit Message Format

```
<package>: <short description starting with lowercase>
```

The description completes the sentence "This changes Ollama to...". Examples:
- `llm/backend/mlx: support the llama architecture`
- `api: add streaming timeout parameter`

Do not use conventional commit prefixes (feat:, fix:, chore:).

## Key Conventions

- Dependencies in `llama/` are synced from upstream llama.cpp with patches applied via `Makefile.sync`
- Acceleration libraries are loaded from paths relative to the binary: `./lib/ollama` (Windows), `../lib/ollama` (Linux), `.` (macOS), `build/lib/ollama` (dev)
- Go 1.24; uses `sync/errgroup` for concurrency, context-based cancellation throughout
- Testing uses `github.com/stretchr/testify`
- API backward compatibility must be preserved

## Documentation Conventions

- **`docs/internals/`** - 存放对 Ollama/llama.cpp/GGML 内部机制的调查研究文档
  - 鼓励使用 Mermaid 图表辅助说明
  - 代码引用需标注来源（Ollama Go 还是 llama.cpp C++）
  - Mermaid 图中用颜色区分代码归属：🟢 绿色粗边框 (`stroke:#22c55e,stroke-width:3px`) = Ollama Go，🟠 橙色粗边框 (`stroke:#f97316,stroke-width:3px`) = llama.cpp C/C++，subgraph 用同色虚线边框。每张图顶部加图例
- **`docs/TODO.md`** - 存放未来要做的工作。格式：`- [ ] 描述 (来源: YYYY-MM-DD)`
- **`docs/daop/how-to-run-actual-inference.md`** - 如何运行实际推理并采集 per-op GPU 计时的完整流程。包含环境变量 (`OLLAMA_VULKAN=1`, `GGML_VK_PERF_LOGGER=1`, `OLLAMA_FLASH_ATTENTION=1`)、API 调用方式、输出格式解析。**重要**：每次运行需修改 prompt 首字符以避免 prefix cache 命中
