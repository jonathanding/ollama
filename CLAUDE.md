# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```shell
# Run server directly (most common for development)
go run . serve

# Run all tests
go test ./...

# Run a single package's tests
go test ./server/...

# Run tests with synctest experiment (required for some concurrency tests)
GOEXPERIMENT=synctest go test ./...

# Build native acceleration libraries (Windows/Linux/macOS Intel — not needed on Apple Silicon)
cmake -B build
cmake --build build --config Release

# Build MLX engine (macOS Apple Silicon)
cmake -B build --preset MLX
cmake --build build --preset MLX --parallel
cmake --install build --component MLX
```

## Linting

```shell
golangci-lint run
```

The project uses `gofmt`/`gofumpt` for formatting. Linters are configured in `.golangci.yaml`.

## Commit Message Format

```
<package>: <short description starting with lowercase>
```

The package is the most-affected Go package (or directory/filename for non-Go changes). The description should complete the sentence "This changes Ollama to...". Example: `server: fix scheduler eviction order`.

## Shell Command Style

Never use `cd <dir> && <command>` patterns — they trigger a security prompt for bare repository attacks. Instead:
- Git commands: `git -C <path> <subcommand>`
- npm/npx commands: `npx --prefix <path> <command>`
- Other tools: run from the project root with absolute paths, or use tool-specific directory flags

## TODO Tracking

讨论中产生的所有待办事项集中在 [`docs/TODO.md`](docs/TODO.md)。当讨论中出现新的 TODO 想法、优化方向、或待验证事项时，应追加到该文件中（注明优先级和来源）。

## Architecture

### Top-level packages

- **`cmd/`** — CLI entry point (`cobra`-based). `cmd/cmd.go` registers all subcommands. `cmd/launch/` handles `ollama launch <integration>` for Claude Code, Codex, OpenCode, OpenClaw, Cline, Droid, etc.
- **`server/`** — HTTP API server (gin-based). `routes.go` registers all REST endpoints. `sched.go` implements the model scheduler that loads/evicts models from GPU/CPU memory.
- **`llm/`** — Interface to the llama.cpp runner process. Manages the subprocess lifecycle via `server.go`.
- **`ml/`** — Machine learning abstraction layer: `ml/backend/` holds backend implementations (ggml, MLX); `ml/nn/` has reusable neural net building blocks (attention, rope, normalization, etc.).
- **`model/`** — Go-native model implementations. `model/models/` contains per-architecture implementations (llama, gemma3, qwen3, mistral3, deepseek2, mllama, etc.). `model/parsers/` and `model/renderers/` handle model-specific prompt/output parsing.
- **`api/`** — Public Go client library and shared types (`api.Options`, request/response structs).
- **`app/`** — macOS/Windows GUI app wrapper. `app/store/` is SQLite-backed state. `app/tools/` implements tool-calling (web search, browser). `app/wintray/` is the Windows system tray.
- **`anthropic/`** — Anthropic API proxy/translation layer.
- **`convert/`** — Model conversion (Safetensors/PyTorch → GGUF).
- **`fs/ggml/`** — GGUF file format reader.
- **`envconfig/`** — Environment variable configuration (`OLLAMA_*` env vars).
- **`template/`** — Chat prompt templates (Go `text/template`-based).
- **`tokenizer/`** — Tokenizer implementations.
- **`discover/`** — GPU/hardware discovery.
- **`x/`** — Experimental/extended features: `x/imagegen/` for image generation, `x/mlxrunner/` for the MLX inference engine.

### Request flow

1. CLI (`cmd/cmd.go`) or client → HTTP request to `server/routes.go`
2. `server/sched.go` (`Scheduler`) assigns an available model runner or loads a new one
3. For GGUF models: `llm/server.go` manages a `llama.cpp` subprocess via `runner/`
4. For native Go models (`model/models/`): inference runs directly via `ml/` backends
5. For MLX models: `x/mlxrunner/` handles Apple Silicon / NVIDIA GPU dispatch

### Model scheduler

`server/sched.go` maintains a map of loaded `runnerRef`s keyed by model name. Only one model loads at a time (`activeLoading`); concurrent requests to already-loaded models are served in parallel. Models are evicted when VRAM is needed.
