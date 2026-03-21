# Ollama Trace Analyzer

Post-process and visualize Ollama inference traces (JSONL format from Phase 1 profiler).

## Install

```bash
cd tools/trace-analyzer
pip install -e .
```

## Usage

### Generate summary from a trace

```bash
ollama-trace-analyzer summary trace.jsonl -o data/summary.json
```

### Compare two traces

```bash
ollama-trace-analyzer compare cuda_trace.jsonl vulkan_trace.jsonl --labels "CUDA,Vulkan" -o data/compare.json
```

### Generate Markdown report (for LLM analysis)

```bash
ollama-trace-analyzer report trace.jsonl -o report.md
ollama-trace-analyzer report trace.jsonl --compare other.jsonl --labels "A,B" -o compare_report.md
```

### Launch visualization

```bash
# Build the React frontend (one-time)
cd web && npm install && npm run build && cd ..

# Start the server
ollama-trace-analyzer serve --data-dir data/ --port 8765
# Open http://localhost:8765
```

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

```bash
python -m pytest tests/ -v
```
