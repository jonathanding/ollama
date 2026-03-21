# Ollama Trace Analyzer

Post-process and visualize Ollama inference traces.

## Install

```cmd
cd tools\trace-analyzer
pip install -e .
```

## End-to-End Workflow (Windows)

### 1. Generate a trace

Set `OLLAMA_TRACE_DIR` to enable per-request JSONL trace output, then run a request:

```cmd
set OLLAMA_TRACE_DIR=C:\workspace\myollama\tmp
ollama run llama3.2 "Hello"
```

Trace files appear in the directory as `trace_*.jsonl`.

### 2. Generate summary

```cmd
ollama-trace-analyzer summary C:\workspace\myollama\tmp\trace_xxx.jsonl -o data\summary.json
```

Output:
```
Parsing trace_xxx.jsonl...
  34204 ops across 68 passes
  1 layers, top op: MUL_MAT, wall time: 7147.0ms
  -> data\summary.json (7148658 bytes)
```

### 3. Compare two traces

```cmd
ollama-trace-analyzer compare trace_cuda.jsonl trace_vulkan.jsonl --labels "CUDA,Vulkan" -o data\compare.json
```

### 4. Generate Markdown report

```cmd
ollama-trace-analyzer report C:\workspace\myollama\tmp\trace_xxx.jsonl -o report.md
ollama-trace-analyzer report trace_a.jsonl --compare trace_b.jsonl --labels "A,B" -o compare_report.md
```

### 5. Launch visualization

```cmd
rem Build the React frontend (one-time)
cd web
npm install
npm run build
cd ..

rem Start the server
ollama-trace-analyzer serve --data-dir data\ --port 8765
```

Open http://localhost:8765 in browser.

## CLI Commands

| Command   | Description                          | Key Options                              |
|-----------|--------------------------------------|------------------------------------------|
| `summary` | Single trace → summary.json          | `-o`, `--model`, `--pass`                |
| `compare` | Two traces → compare.json            | `--labels` (required), `--threshold`, `-o` |
| `report`  | Trace → Markdown report              | `--compare`, `--labels`, `-o`            |
| `serve`   | Launch data server + React frontend  | `--data-dir` (required), `--port`        |

## React SPA Views

- **DAG** — Cytoscape.js graph with layer folding, backend/heatmap coloring, tensor search
- **Timeline** — D3.js swimlane chart per pass, zoomable, click to select ops
- **Compare** — Diff table with significance flags + grouped bar chart

## Running Tests

```cmd
python -m pytest tests\ -v
```
