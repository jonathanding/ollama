# Trace Post-Processing & Visualization Design Spec

**Status**: Draft
**Date**: 2026-03-20
**Depends on**: Phase 1 profiling system (complete, merged to main)

---

## Background & Motivation

### The problem

Phase 1 of the ollama profiling system is complete: LlamaRunner now captures per-operator execution traces as JSONL files (controlled by `OLLAMA_TRACE_DIR`). Each inference request produces one `.jsonl` file containing:

- `pass_start` / `pass_end` events per `llama_decode()` call
- `op` events per GGML compute node: op type, tensor name, source tensor names (DAG edges), output shape, dtype, backend device, nanosecond timestamps

**Raw JSONL files are hard to interpret.** A single inference of llama3:8b generates ~3200 op events per pass, with ~64 passes per request. Manually grepping through thousands of lines is impractical for:

- Understanding the computation graph structure (which operators depend on which)
- Identifying performance bottlenecks (which ops take the most time)
- Spotting unexpected data copies between CPU and GPU
- Comparing behavior across different hardware/software configurations

### Two user needs

1. **Interactive exploration (single trace)**: A friendly web UI to browse the computation DAG, see per-operator timing, understand data flow, and quickly spot bottlenecks — without manually running `jq` commands.

2. **Batch analysis + LLM-assisted insights (multi trace)**: When running the same model across different GPUs, quantization levels, batch sizes, or software versions, we need to systematically compare traces. The raw JSONL is too verbose for LLM context windows. We need structured summaries and diff tables that can be:
   - Fed into LLMs (Claude, GPT) for automated insight extraction
   - Displayed in the web frontend as comparison dashboards

### Design principles

- **Separation of concerns**: Python CLI handles all heavy data processing. React frontend only renders pre-processed JSON. Neither depends on the other's internals.
- **LLM-friendly output**: Markdown reports use clean tables that LLMs can parse. Include `significant` flags and `⚠️` markers to guide LLM attention.
- **Performance**: Use polars (not pandas) for fast data processing of large traces. React uses virtualization for large DAGs.
- **Zero extra infrastructure**: No backend server needed. Python generates static JSON files; React reads them from disk or a simple file server.

---

## Architecture

```
Raw JSONL files (from OLLAMA_TRACE_DIR)
       │
       ▼
┌──────────────────────────────┐
│  Python CLI (trace-analyzer) │
│  ├─ summary  → summary.json │  single trace stats + DAG data
│  ├─ compare  → compare.json │  multi-trace diff with significance
│  ├─ report   → report.md    │  LLM-ready Markdown text
│  └─ serve    → localhost     │  dev server for React frontend
└──────────┬───────────────────┘
           │ JSON files
    ┌──────┴──────┐
    ▼              ▼
 React SPA        LLM
 (browser)        (paste report.md for analysis)
```

**Data flow**:
1. `OLLAMA_TRACE_DIR=/tmp/traces ollama serve` → produces raw `.jsonl` per request
2. `trace-analyzer summary *.jsonl` → produces `summary.json` (per file)
3. `trace-analyzer compare a.jsonl b.jsonl` → produces `compare.json`
4. `trace-analyzer report ...` → produces `report.md`
5. React SPA loads `summary.json` / `compare.json` and renders interactive views
6. `report.md` is pasted into LLM for automated analysis

---

## Component 1: Python CLI (`tools/trace-analyzer/`)

### Tech stack

- Python 3.10+
- **polars** (fast DataFrame library — user preference over pandas)
- **click** (CLI framework)
- **jinja2** (Markdown report templates)
- No heavy ML dependencies

### Commands

```bash
# Single trace summary
trace-analyzer summary trace_req1_123.jsonl -o summary.json

# Multi-trace comparison
trace-analyzer compare trace_cuda.jsonl trace_vulkan.jsonl \
    --labels "CUDA,Vulkan" -o compare.json

# LLM-ready Markdown report (single trace)
trace-analyzer report trace_req1_123.jsonl -o report.md

# LLM-ready Markdown report (comparison)
trace-analyzer report --compare trace_cuda.jsonl trace_vulkan.jsonl \
    --labels "CUDA,Vulkan" -o compare_report.md

# Launch React frontend with data directory
trace-analyzer serve --data-dir /tmp/traces/
```

### Internal modules

| Module | Responsibility |
|--------|----------------|
| `parser.py` | Read JSONL → polars DataFrame. Parse event types, extract fields. |
| `summary.py` | Single-trace analysis: op stats, backend stats, layer stats, copy detection, DAG reconstruction |
| `compare.py` | Multi-trace diff: align ops by name, compute diff%, flag significant differences |
| `dag.py` | Build DAG from op events: `name` → node, `srcs` → edges. Layer grouping by `blk.X` prefix. |
| `report.py` | Jinja2 templates → Markdown output. Single and compare modes. |
| `serve.py` | Simple HTTP server: serves React build + JSON data files |
| `cli.py` | Click entry point wiring all subcommands |

### summary.json schema

```json
{
  "meta": {
    "source_file": "trace_req1_123.jsonl",
    "model": "llama3:8b",
    "total_ops": 3200,
    "total_passes": 64,
    "total_wall_ms": 1250
  },
  "timing": {
    "total_ms": 1250,
    "prefill_ms": 45.2,
    "prefill_tokens": 512,
    "decode_avg_ms": 19.1,
    "per_pass": [
      { "pass": 0, "n_tokens": 512, "wall_ms": 45.2, "n_ops": 1600 },
      { "pass": 1, "n_tokens": 1, "wall_ms": 19.1, "n_ops": 1600 }
    ]
  },
  "op_stats": [
    {
      "op": "MUL_MAT", "count": 640,
      "total_ns": 800000000, "pct_time": 64.0, "avg_ns": 1250000
    },
    {
      "op": "CPY", "count": 32,
      "total_ns": 50000000, "pct_time": 4.0, "avg_ns": 1562500,
      "est_bytes_total": 134217728
    }
  ],
  "backend_stats": [
    { "backend": "CUDA0", "count": 3000, "total_ns": 1200000000, "pct_ops": 93.7, "pct_time": 98.1 },
    { "backend": "CPU", "count": 200, "total_ns": 23000000, "pct_ops": 6.3, "pct_time": 1.9 }
  ],
  "copy_stats": {
    "count": 32,
    "total_ns": 50000000,
    "est_total_bytes": 134217728,
    "copies": [
      {
        "name": "blk.0.ffn_out", "op": "CPY",
        "est_bytes": 4194304, "ns": 1562500, "backend": "CPU"
      }
    ]
  },
  "layer_stats": [
    {
      "layer": "blk.0", "n_ops": 50,
      "total_ns": 38000000, "pct_time": 3.0, "top_op": "MUL_MAT"
    }
  ],
  "dag": {
    "nodes": [
      {
        "id": "blk.0.attn_q", "op": "MUL_MAT", "backend": "CUDA0",
        "ns": 1250000, "shape": [4096, 512], "dtype": "f16", "layer": "blk.0",
        "is_copy": false
      }
    ],
    "edges": [
      { "from": "blk.0.attn_norm", "to": "blk.0.attn_q", "est_bytes": 4194304 }
    ]
  }
}
```

**Data size estimation**: `bytes = product(shape) * dtype_size[dtype]`
where `dtype_size = {"f32": 4, "f16": 2, "bf16": 2, "q4_0": 0.5, "q8_0": 1, ...}`

**Copy detection**: Ops with `op == "CPY"` or `op == "DUP"` are flagged as data copies. The `backend` field on the node plus the `backend` of its source tensor reveals the transfer direction.

### compare.json schema

```json
{
  "labels": ["CUDA", "Vulkan"],
  "meta": [
    { "label": "CUDA", "source_file": "trace_cuda.jsonl", "model": "llama3:8b", "total_wall_ms": 1250 },
    { "label": "Vulkan", "source_file": "trace_vulkan.jsonl", "model": "llama3:8b", "total_wall_ms": 1580 }
  ],
  "timing_diff": {
    "prefill_ms": [45.2, 62.1],
    "decode_avg_ms": [19.1, 24.3],
    "total_ms": [1250, 1580],
    "diff_pct": { "prefill": 37.4, "decode": 27.2, "total": 26.4 }
  },
  "op_diff": [
    {
      "op": "MUL_MAT",
      "values": [
        { "label": "CUDA", "total_ns": 800000000, "count": 640 },
        { "label": "Vulkan", "total_ns": 1020000000, "count": 640 }
      ],
      "diff_pct": 27.5,
      "significant": true
    },
    {
      "op": "RMS_NORM",
      "values": [
        { "label": "CUDA", "total_ns": 120000000 },
        { "label": "Vulkan", "total_ns": 116000000 }
      ],
      "diff_pct": -3.3,
      "significant": false
    }
  ],
  "layer_diff": [
    {
      "layer": "blk.0",
      "values": [
        { "label": "CUDA", "total_ns": 38000000 },
        { "label": "Vulkan", "total_ns": 49000000 }
      ],
      "diff_pct": 28.9,
      "significant": true
    }
  ],
  "copy_diff": [
    {
      "name": "blk.0.ffn_out",
      "values": [
        { "label": "CUDA", "ns": 1562500, "est_bytes": 4194304 },
        { "label": "Vulkan", "ns": 2100000, "est_bytes": 4194304 }
      ],
      "diff_pct": 34.4,
      "significant": true
    }
  ]
}
```

**Significance threshold**: `|diff_pct| > 10%` (configurable via `--threshold` flag).

### Markdown report format

**Single trace report**:
```markdown
# Inference Trace Report: llama3:8b

## Summary
- Total passes: 64, Total ops: 3200
- Wall time: 1250ms (prefill: 45ms/512tok, decode: 19ms/tok)

## Top Operators by Time
| Op       | Count | Total (ms) | % Time | Avg (μs) |
|----------|-------|-----------|--------|----------|
| MUL_MAT  | 640   | 800.0     | 64.0%  | 1250     |
| RMS_NORM | 320   | 120.0     | 9.6%   | 375      |
| CPY      | 32    | 50.0      | 4.0%   | 1563     |

## Backend Distribution
| Backend | Ops  | % Ops | % Time |
|---------|------|-------|--------|
| CUDA0   | 3000 | 93.7% | 98.1%  |
| CPU     | 200  | 6.3%  | 1.9%   |

## Data Transfers
| Tensor          | Est. Size | Time (μs) | Direction  |
|-----------------|-----------|-----------|------------|
| blk.0.ffn_out   | 4.0 MB    | 1563      | GPU→CPU    |

## Per-Layer Breakdown
| Layer  | Ops | Total (ms) | % Time | Top Op   |
|--------|-----|-----------|--------|----------|
| blk.0  | 50  | 38.0      | 3.0%   | MUL_MAT  |
| blk.1  | 50  | 37.5      | 3.0%   | MUL_MAT  |
```

**Comparison report adds diff columns**:
```markdown
## Op Comparison: CUDA vs Vulkan
| Op       | CUDA (ms) | Vulkan (ms) | Diff    | Significant |
|----------|----------|------------|---------|-------------|
| MUL_MAT  | 800.0    | 1020.0     | ⚠️ +27.5% | Yes       |
| RMS_NORM | 120.0    | 116.0      | -3.3%   | No          |
```

The `⚠️` marker helps LLMs focus on significant differences.

---

## Component 2: React SPA (`tools/trace-analyzer/web/`)

### Tech stack

- React 18 + TypeScript
- **Cytoscape.js** — DAG view with compound nodes (layer folding)
- **D3.js** — Timeline swimlane view
- **ECharts** — Compare bar charts
- **Tailwind CSS** — styling

### View 1: DAG View

**Layer folding with Cytoscape.js compound nodes**:

- Parse tensor names to extract layer: `blk.0.attn_q` → layer = `blk.0`
- Tensors outside `blk.X` pattern → top-level group (e.g., `token_embd`, `output_norm`)
- Each layer is a Cytoscape compound (parent) node containing its child op nodes
- Default state: all layers collapsed
  - Collapsed view: ~32 layer nodes + a few top-level nodes
  - Each collapsed node shows: layer name, total time, dominant op type
- Click to expand a layer → reveals ~50 internal op nodes with edges

**Node styling — two modes (toggle button)**:

| Mode | Color mapping | Use case |
|------|--------------|----------|
| Backend (default) | CPU=blue, CUDA=green, Vulkan=orange | Understand device allocation |
| Heatmap | Blue→yellow→red gradient by time | Spot bottlenecks |

Additional node styling:
- Size ∝ execution time (log scale to avoid tiny nodes)
- **Copy ops** (CPY/DUP): red dashed border, always visible regardless of mode
- **Top-5 most expensive ops**: bold border + time label permanently displayed
- Collapsed layer nodes: color reflects **sum of internal ops' time** in heatmap mode

**Edge styling**:
- Width ∝ estimated data size (log scale)
- Hover tooltip: tensor name, shape, dtype, estimated bytes
- Copy edges: dashed red line

**Interactions**:
- Click node → opens detail panel (op, name, shape, dtype, backend, time, sources)
- Zoom/pan (Cytoscape built-in)
- Search box: type tensor name → auto-complete, highlight + center on match
- Double-click collapsed layer → expand; double-click expanded layer header → collapse

### View 2: Timeline View

**D3.js horizontal swimlane chart**:

- X axis = time (nanoseconds, relative to pass start)
- Y axis = passes (one row per pass)
  - Optional: expand a pass into per-layer rows
- Each op = colored rectangle, width = `t_end - t_start`
- Color = op type (default) or backend (toggle)
- Mouse wheel zooms X axis; drag to pan
- Hover → tooltip: op name, type, time, backend
- Click op → linked highlight in DAG view + Hotspot panel

### View 3: Compare View

Loads `compare.json` from Python CLI output.

**Layout**:
- Top: summary cards — side-by-side total wall time, prefill time, decode avg time
- Middle: **Op diff table** — sortable columns, diff% highlighted:
  - Red background: `diff_pct > 10%` (slower)
  - Green background: `diff_pct < -10%` (faster)
  - White: insignificant difference
  - Default sort: by `|diff_pct|` descending → biggest differences first
- Bottom: **Layer diff bar chart** (ECharts grouped bars)
  - One group per layer, bars = each trace's total time
  - Significant diff bars highlighted with bold outline

**DAG diff mode** (advanced feature):
- Load two traces' DAG data
- Render single DAG with the same layout
- Node color = **diff heatmap**: red=slower, white=neutral, green=faster
- Useful for seeing which specific layers/ops got slower or faster

### Hotspot Panel (sidebar, present in all views)

A collapsible right sidebar with three ranking tabs:

| Tab | Content | Sort |
|-----|---------|------|
| Top Ops by Time | All ops ranked by duration | Descending by ns |
| Top Copies by Size | CPY/DUP ops ranked by estimated bytes | Descending by est_bytes |
| Top Copies by Time | CPY/DUP ops ranked by duration | Descending by ns |

Each row shows: rank, op/tensor name, value (time or size), backend.

**Bidirectional linking**:
- Click panel row → DAG centers and highlights the node; Timeline scrolls to time range
- Click node in DAG → panel scrolls to and highlights corresponding row
- In Compare view: panel shows both traces' values and diff%

---

## JSONL Input Format (reference)

Already implemented in Phase 1. The Python CLI reads these event types:

**pass_start**: `{"type":"pass_start","pass":0,"n_tokens":512,"ts":1742123456789}`

**pass_end**: `{"type":"pass_end","pass":0,"n_nodes":0,"ts":1742123456800}`

**op**: `{"type":"op","pass":0,"seq":42,"op":"MUL_MAT","name":"blk.3.attn_q","srcs":["blk.3.attn_norm","blk.3.attn_q.weight"],"shape":[4096,512],"dtype":"f16","backend":"CUDA0","t_start":1742123456790123456,"t_end":1742123456790234567}`

Key fields for visualization:
- `name` + `srcs` → DAG nodes and edges
- `op` → node label and color category
- `backend` → device color
- `t_start`/`t_end` → timing (nanoseconds, CPU-side dispatch)
- `shape` + `dtype` → estimated data size for edge width
- `pass` + `seq` → timeline ordering

**Timing caveat**: `t_start`/`t_end` are CPU-side dispatch times, not actual GPU execution. Accurate for large ops (matmul dominates), approximate for small ops. Pass-level `ts` fields are millisecond precision wall-clock.

---

## File Structure

```
tools/trace-analyzer/
├── pyproject.toml              # Python project: polars, click, jinja2
├── trace_analyzer/
│   ├── __init__.py
│   ├── cli.py                  # Click CLI entry point
│   ├── parser.py               # JSONL → polars DataFrame
│   ├── summary.py              # Single-trace summary → summary.json
│   ├── compare.py              # Multi-trace diff → compare.json
│   ├── dag.py                  # DAG reconstruction (nodes, edges, layer grouping)
│   ├── report.py               # Jinja2 → Markdown reports
│   ├── serve.py                # Dev server for React + data
│   └── templates/
│       ├── single_report.md.j2
│       └── compare_report.md.j2
├── web/
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── DagView.tsx         # Cytoscape.js DAG with layer folding
│   │   │   ├── TimelineView.tsx    # D3 swimlane chart
│   │   │   ├── CompareView.tsx     # Diff tables + ECharts bars
│   │   │   ├── HotspotPanel.tsx    # Sidebar rankings
│   │   │   ├── NodeDetail.tsx      # Op detail popup/drawer
│   │   │   └── ColorToggle.tsx     # Backend / heatmap mode switch
│   │   ├── hooks/
│   │   │   └── useTraceData.ts     # Load & parse summary/compare JSON
│   │   ├── utils/
│   │   │   ├── dagLayout.ts        # Layer folding + compound node logic
│   │   │   ├── colorScale.ts       # Heatmap gradient + backend colors
│   │   │   └── dataSize.ts         # dtype_size map + byte estimation
│   │   └── types/
│   │       └── trace.ts            # TypeScript types for JSON schemas
│   └── public/
│       └── index.html
├── tests/
│   ├── test_parser.py
│   ├── test_summary.py
│   ├── test_compare.py
│   ├── test_dag.py
│   └── fixtures/
│       └── sample_trace.jsonl      # Small fixture for tests
└── README.md                       # Usage instructions
```

---

## Verification Plan

### Python CLI verification

1. **Generate real trace data**:
   ```bash
   OLLAMA_TRACE_DIR=/tmp/traces ollama serve
   # Send inference request, get .jsonl files
   ```
2. **summary command**:
   ```bash
   trace-analyzer summary /tmp/traces/trace_*.jsonl -o summary.json
   # Verify: valid JSON, has meta/timing/op_stats/backend_stats/copy_stats/layer_stats/dag
   ```
3. **report command**:
   ```bash
   trace-analyzer report /tmp/traces/trace_*.jsonl -o report.md
   # Verify: readable Markdown with tables, paste into LLM
   ```
4. **compare command** (use same trace twice as sanity check):
   ```bash
   cp trace_1.jsonl trace_1_copy.jsonl
   trace-analyzer compare trace_1.jsonl trace_1_copy.jsonl --labels "A,B"
   # Verify: all diff_pct ≈ 0%, no significant flags
   ```
5. **Unit tests**: `pytest tools/trace-analyzer/tests/`

### React frontend verification

1. `trace-analyzer serve --data-dir /tmp/traces/` → open browser
2. **DAG View**: layers collapsed? expand one → nodes/edges visible? toggle heatmap? search works?
3. **Timeline View**: passes shown as rows? zoom/pan? hover tooltip?
4. **Compare View**: load compare.json → diff table sorts by significance? bar chart renders?
5. **Hotspot Panel**: click top op → DAG highlights? click DAG node → panel highlights?
6. **Bidirectional linking**: all cross-view highlighting works?

### LLM analysis workflow

1. Generate `report.md` for a real trace
2. Paste into Claude with: "Analyze this inference trace and identify performance bottlenecks"
3. Verify LLM can parse tables and produce meaningful insights
4. Generate `compare_report.md` for two traces
5. Paste into Claude with: "Compare these two configurations and explain the differences"
6. Verify LLM identifies significant diffs marked with ⚠️
