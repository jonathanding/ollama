# Ollama with Tracing

A custom build of Ollama with built-in per-operator tracing, plus a Trace Analyzer toolkit for visualizing and understanding inference performance.

When running LLM inference on different hardware backends (CUDA, Vulkan, CPU), understanding where time is spent is critical for optimization. This package instruments Ollama to capture detailed per-operator execution traces during model inference, and provides interactive tools to explore the results — DAG views, timeline visualizations, comparison reports, and more.

## Installation

Unzip the release package. The resulting directory is a **self-contained Ollama installation** — everything you need is inside it. All commands below assume you have opened a Command Prompt and `cd` into this directory.

Requires Python 3.10+ for the Trace Analyzer. Install it once:

```
pip install tools\trace-analyzer
```

This installs the `ollama-trace-analyzer` CLI command.

## Quick Start

Open a Command Prompt (`cmd`) and `cd` into the release directory.

> **IMPORTANT: You MUST set `OLLAMA_TRACE_DIR` before starting Ollama. Without this, no tracing happens and no trace files are produced!**

The typical workflow is: **collect a trace, generate a summary, visualize in browser**.

```
REM 1. IMPORTANT: Enable tracing by setting the trace output directory
set OLLAMA_TRACE_DIR=C:\traces

REM 2. Start Ollama (this is the tracing-enabled build in this directory)
ollama serve

REM --- In a second Command Prompt window ---

REM 3. Send a request as usual — trace files are written to C:\traces\
ollama run llama3:8b "Hello, world"

REM 4. Generate analysis-ready JSON from the raw trace
ollama-trace-analyzer summary C:\traces\trace_req1.jsonl -o C:\traces\summary.json

REM 5. Launch the interactive visualizer — open http://localhost:8765 in your browser
ollama-trace-analyzer serve --data-dir C:\traces
```

The browser UI provides:
- **DAG View** — Interactive computation graph with layer folding, heatmap coloring, and execution replay
- **Timeline View** — Swimlane visualization of operator execution across passes
- **Compare View** — Side-by-side comparison of traces from different backends/configs
- **Hotspot Panel** — Ranked lists of the most expensive operators and data transfers

### Comparing two backends

```
REM Collect a trace with CUDA
set OLLAMA_TRACE_DIR=C:\traces\cuda
ollama serve
REM   (send a request in another window, then stop the server)

REM Collect a trace with Vulkan
set OLLAMA_TRACE_DIR=C:\traces\vulkan
ollama serve
REM   (send the same request, then stop the server)

REM Generate summaries
ollama-trace-analyzer summary C:\traces\cuda\trace_req1.jsonl -o C:\traces\summary_cuda.json
ollama-trace-analyzer summary C:\traces\vulkan\trace_req1.jsonl -o C:\traces\summary_vulkan.json

REM Generate comparison
ollama-trace-analyzer compare C:\traces\cuda\trace_req1.jsonl C:\traces\vulkan\trace_req1.jsonl --labels "CUDA,Vulkan" -o C:\traces\compare.json

REM Visualize everything
ollama-trace-analyzer serve --data-dir C:\traces
```

### Generating reports for LLM analysis

```
REM Generate a Markdown report, then paste it into Claude for analysis
ollama-trace-analyzer report C:\traces\trace_req1.jsonl -o report.md

REM Or a comparison report
ollama-trace-analyzer report C:\traces\cuda\trace_req1.jsonl --compare C:\traces\vulkan\trace_req1.jsonl --labels "CUDA,Vulkan" -o compare_report.md
```

---

## Reference: Tracing-Enabled Ollama

This is a standard Ollama build with one addition — the `OLLAMA_TRACE_DIR` environment variable:

> **IMPORTANT: `OLLAMA_TRACE_DIR` MUST be set before running `ollama serve`. If it is not set, tracing is completely disabled (zero overhead, no trace files). This is the single switch that enables all tracing functionality.**

```
set OLLAMA_TRACE_DIR=C:\your\trace\output\directory
ollama serve
```

When enabled, each inference request writes a JSONL trace file to the specified directory. Trace files are named `trace_<request-id>.jsonl`. Each line is a JSON object — either a pass boundary event (`pass_start` / `pass_end`) or an operator event with fields like `op`, `name`, `src` (source tensors), `shape`, `dtype`, `backend`, and nanosecond timestamps.

All other Ollama commands and environment variables work exactly as documented at [ollama.com](https://ollama.com).

## Reference: Trace Analyzer CLI

### `ollama-trace-analyzer summary`

Generate a structured summary (JSON) from a raw JSONL trace.

```
ollama-trace-analyzer summary <trace_file> [-o output.json] [--model name] [--pass N]
```

| Option | Description |
|--------|-------------|
| `-o` | Output file (default: stdout) |
| `--model` | Model name to include in metadata |
| `--pass` | Specific pass ID for DAG construction (default: first decode pass) |

### `ollama-trace-analyzer compare`

Compare two traces and produce a diff report (JSON).

```
ollama-trace-analyzer compare <trace_a> <trace_b> --labels "A,B" [-o output.json] [--threshold 10.0]
```

| Option | Description |
|--------|-------------|
| `--labels` | Comma-separated labels for the two traces (required) |
| `-o` | Output file (default: stdout) |
| `--threshold` | Significance threshold in percent (default: 10.0) |

### `ollama-trace-analyzer report`

Generate an LLM-ready Markdown report.

```
ollama-trace-analyzer report <trace_file> [-o report.md] [--compare trace_b] [--labels "A,B"]
```

| Option | Description |
|--------|-------------|
| `-o` | Output file (default: stdout) |
| `--compare` | Second trace file for comparison report |
| `--labels` | Labels for comparison (default: "A,B") |

### `ollama-trace-analyzer serve`

Launch the interactive visualization server.

```
ollama-trace-analyzer serve --data-dir <path> [--port 8765]
```

| Option | Description |
|--------|-------------|
| `--data-dir` | Directory containing summary/compare JSON files (required) |
| `--port` | Server port (default: 8765) |

The `--data-dir` should contain `.json` files produced by the `summary` and `compare` commands. The server provides both the REST API and the browser-based visualization UI.
