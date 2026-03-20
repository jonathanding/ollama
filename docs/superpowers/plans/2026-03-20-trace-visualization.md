# Trace Post-Processing & Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI + React SPA that transforms raw JSONL inference traces into interactive visualizations and LLM-ready reports.

**Architecture:** Two independent subsystems sharing a JSON contract. Python CLI (`trace-analyzer`) reads raw JSONL, produces `summary.json` / `compare.json` / `report.md`. React SPA consumes these JSON files for interactive DAG, Timeline, Compare views with a Hotspot panel.

**Tech Stack:** Python 3.10+ (polars, click, jinja2) | React 18 + TypeScript (Cytoscape.js, D3.js, Tailwind CSS) | Vite

**Spec:** `docs/superpowers/specs/2026-03-20-trace-visualization-design.md`

---

## File Map

### Python CLI (`tools/trace-analyzer/`)

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata, dependencies, `[project.scripts]` entry point |
| `trace_analyzer/__init__.py` | Package marker |
| `trace_analyzer/parser.py` | JSONL → polars DataFrames (ops_df, passes_df). Skip malformed lines. |
| `trace_analyzer/dag.py` | Build DAG nodes/edges from ops. Layer grouping by `blk.X` prefix. |
| `trace_analyzer/summary.py` | Single-trace analysis → `summary.json` dict |
| `trace_analyzer/compare.py` | Two-trace diff → `compare.json` dict with significance flags |
| `trace_analyzer/report.py` | Jinja2 templates → Markdown strings |
| `trace_analyzer/serve.py` | HTTP server: static files + `/api/files` endpoint |
| `trace_analyzer/cli.py` | Click entry point wiring subcommands |
| `trace_analyzer/templates/single_report.md.j2` | Single trace Markdown template |
| `trace_analyzer/templates/compare_report.md.j2` | Compare Markdown template |
| `tests/fixtures/sample_trace.jsonl` | Small fixture (2 passes, ~10 ops each) |
| `tests/test_parser.py` | Parser unit tests |
| `tests/test_dag.py` | DAG reconstruction tests |
| `tests/test_summary.py` | Summary computation tests |
| `tests/test_compare.py` | Compare diff tests |

### React SPA (`tools/trace-analyzer/web/`)

| File | Responsibility |
|------|---------------|
| `package.json` | Dependencies: react, cytoscape, d3, tailwindcss |
| `vite.config.ts` | Vite config with React plugin |
| `tsconfig.json` | TypeScript config |
| `src/types/trace.ts` | TypeScript interfaces for summary.json / compare.json |
| `src/hooks/useTraceData.ts` | Fetch and parse JSON data files |
| `src/utils/colorScale.ts` | Backend colors + heatmap gradient |
| `src/utils/dataSize.ts` | dtype_size map + byte formatting |
| `src/utils/dagLayout.ts` | Cytoscape element generation with compound nodes |
| `src/components/HotspotPanel.tsx` | Ranked sidebar with tabs |
| `src/components/NodeDetail.tsx` | Op detail drawer |
| `src/components/ColorToggle.tsx` | Backend/heatmap mode switch |
| `src/components/DagView.tsx` | Cytoscape.js DAG with layer folding |
| `src/components/TimelineView.tsx` | D3 swimlane chart |
| `src/components/CompareView.tsx` | Diff table + D3 bar chart |
| `src/App.tsx` | Tab layout wiring all views |

---

## Phase A: Python CLI

### Task 1: Project Scaffolding + Test Fixture

**Files:**
- Create: `tools/trace-analyzer/pyproject.toml`
- Create: `tools/trace-analyzer/trace_analyzer/__init__.py`
- Create: `tools/trace-analyzer/tests/__init__.py`
- Create: `tools/trace-analyzer/tests/fixtures/sample_trace.jsonl`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "trace-analyzer"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "polars>=1.0",
    "click>=8.0",
    "jinja2>=3.1",
]

[project.scripts]
trace-analyzer = "trace_analyzer.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `__init__.py` files**

`trace_analyzer/__init__.py` and `tests/__init__.py`: empty files.

- [ ] **Step 3: Create test fixture `tests/fixtures/sample_trace.jsonl`**

2 passes (pass 0: prefill n_tokens=4, pass 1: decode n_tokens=1), 5 ops each. Must match JSONL format from Phase 1 (`llm/profiler/profiler.go` OpEvent struct).

```jsonl
{"type":"pass_start","pass":0,"n_tokens":4,"ts":1700000000000}
{"type":"op","pass":0,"seq":0,"op":"GET_ROWS","name":"token_embd","srcs":["token_embd.weight"],"shape":[128,4],"dtype":"f16","backend":"CUDA0","t_start":100000000,"t_end":100050000}
{"type":"op","pass":0,"seq":1,"op":"RMS_NORM","name":"blk.0.attn_norm","srcs":["token_embd"],"shape":[128,4],"dtype":"f32","backend":"CUDA0","t_start":100050000,"t_end":100080000}
{"type":"op","pass":0,"seq":2,"op":"MUL_MAT","name":"blk.0.attn_q","srcs":["blk.0.attn_norm","blk.0.attn_q.weight"],"shape":[128,4],"dtype":"f16","backend":"CUDA0","t_start":100080000,"t_end":100280000}
{"type":"op","pass":0,"seq":3,"op":"MUL_MAT","name":"blk.0.ffn_gate","srcs":["blk.0.attn_norm","blk.0.ffn_gate.weight"],"shape":[256,4],"dtype":"f16","backend":"CUDA0","t_start":100280000,"t_end":100500000}
{"type":"op","pass":0,"seq":4,"op":"CPY","name":"blk.0.ffn_out","srcs":["blk.0.ffn_gate"],"shape":[128,4],"dtype":"f32","backend":"CPU","t_start":100500000,"t_end":100600000}
{"type":"pass_end","pass":0,"n_nodes":5,"ts":1700000000001}
{"type":"pass_start","pass":1,"n_tokens":1,"ts":1700000000002}
{"type":"op","pass":1,"seq":0,"op":"GET_ROWS","name":"token_embd","srcs":["token_embd.weight"],"shape":[128,1],"dtype":"f16","backend":"CUDA0","t_start":200000000,"t_end":200030000}
{"type":"op","pass":1,"seq":1,"op":"RMS_NORM","name":"blk.0.attn_norm","srcs":["token_embd"],"shape":[128,1],"dtype":"f32","backend":"CUDA0","t_start":200030000,"t_end":200050000}
{"type":"op","pass":1,"seq":2,"op":"MUL_MAT","name":"blk.0.attn_q","srcs":["blk.0.attn_norm","blk.0.attn_q.weight"],"shape":[128,1],"dtype":"f16","backend":"CUDA0","t_start":200050000,"t_end":200150000}
{"type":"op","pass":1,"seq":3,"op":"MUL_MAT","name":"blk.0.ffn_gate","srcs":["blk.0.attn_norm","blk.0.ffn_gate.weight"],"shape":[256,1],"dtype":"f16","backend":"CUDA0","t_start":200150000,"t_end":200280000}
{"type":"op","pass":1,"seq":4,"op":"CPY","name":"blk.0.ffn_out","srcs":["blk.0.ffn_gate"],"shape":[128,1],"dtype":"f32","backend":"CPU","t_start":200280000,"t_end":200350000}
{"type":"pass_end","pass":1,"n_nodes":5,"ts":1700000000003}
```

- [ ] **Step 4: Install in dev mode and verify**

```bash
cd tools/trace-analyzer
pip install -e .
python -c "import trace_analyzer; print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add tools/trace-analyzer/
git commit -m "tools: scaffold trace-analyzer project with test fixture"
```

### Task 2: JSONL Parser

**Files:**
- Create: `tools/trace-analyzer/trace_analyzer/parser.py`
- Create: `tools/trace-analyzer/tests/test_parser.py`

- [ ] **Step 1: Write failing tests for parser**

```python
# tests/test_parser.py
from pathlib import Path
import polars as pl
from trace_analyzer.parser import parse_trace

FIXTURE = Path(__file__).parent / "fixtures" / "sample_trace.jsonl"

def test_parse_returns_ops_and_passes():
    ops, passes = parse_trace(FIXTURE)
    assert isinstance(ops, pl.DataFrame)
    assert isinstance(passes, pl.DataFrame)

def test_ops_columns():
    ops, _ = parse_trace(FIXTURE)
    expected = {"pass", "seq", "op", "name", "srcs", "shape", "dtype", "backend", "t_start", "t_end"}
    assert expected.issubset(set(ops.columns))

def test_ops_count():
    ops, _ = parse_trace(FIXTURE)
    assert len(ops) == 10  # 5 ops * 2 passes

def test_passes_columns():
    _, passes = parse_trace(FIXTURE)
    expected = {"pass", "n_tokens", "ts_start", "ts_end", "n_nodes"}
    assert expected.issubset(set(passes.columns))

def test_passes_count():
    _, passes = parse_trace(FIXTURE)
    assert len(passes) == 2

def test_pass_wall_ms():
    _, passes = parse_trace(FIXTURE)
    row = passes.filter(pl.col("pass") == 0).row(0, named=True)
    assert row["wall_ms"] == row["ts_end"] - row["ts_start"]

def test_malformed_line_skipped(tmp_path):
    f = tmp_path / "bad.jsonl"
    f.write_text('not json\n{"type":"pass_start","pass":0,"n_tokens":1,"ts":100}\n{"type":"pass_end","pass":0,"n_nodes":0,"ts":200}\n')
    ops, passes = parse_trace(f)
    assert len(passes) == 1
    assert len(ops) == 0

def test_truncated_trace(tmp_path):
    """pass_start without matching pass_end should still produce a pass row with ts_end=None."""
    f = tmp_path / "truncated.jsonl"
    f.write_text('{"type":"pass_start","pass":0,"n_tokens":1,"ts":100}\n{"type":"op","pass":0,"seq":0,"op":"ADD","name":"x","srcs":["a"],"shape":[1],"dtype":"f32","backend":"CPU","t_start":1000,"t_end":2000}\n')
    ops, passes = parse_trace(f)
    assert len(ops) == 1
    assert len(passes) == 1
    assert passes.row(0, named=True)["ts_end"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd tools/trace-analyzer && python -m pytest tests/test_parser.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'trace_analyzer.parser'`

- [ ] **Step 3: Implement parser.py**

```python
# trace_analyzer/parser.py
from __future__ import annotations
import json
import logging
from pathlib import Path
import polars as pl

logger = logging.getLogger(__name__)

def parse_trace(path: Path | str) -> tuple[pl.DataFrame, pl.DataFrame]:
    path = Path(path)
    ops: list[dict] = []
    pass_starts: dict[int, dict] = {}
    pass_rows: list[dict] = []

    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line %d in %s", lineno, path.name)
                continue

            t = ev.get("type")
            if t == "pass_start":
                pass_starts[ev["pass"]] = {"ts_start": ev["ts"], "n_tokens": ev["n_tokens"]}
            elif t == "pass_end":
                pid = ev["pass"]
                start = pass_starts.pop(pid, {})
                ts_start = start.get("ts_start")
                ts_end = ev["ts"]
                pass_rows.append({
                    "pass": pid,
                    "n_tokens": start.get("n_tokens"),
                    "ts_start": ts_start,
                    "ts_end": ts_end,
                    "n_nodes": ev.get("n_nodes", 0),
                    "wall_ms": (ts_end - ts_start) if ts_start is not None else None,
                })
            elif t == "op":
                ops.append({
                    "pass": ev["pass"], "seq": ev["seq"],
                    "op": ev["op"], "name": ev["name"],
                    "srcs": ev.get("srcs", []),
                    "shape": ev.get("shape", []),
                    "dtype": ev.get("dtype", ""),
                    "backend": ev.get("backend", ""),
                    "t_start": ev["t_start"], "t_end": ev["t_end"],
                })

    # Handle truncated traces: pass_start without pass_end
    for pid, start in pass_starts.items():
        pass_rows.append({
            "pass": pid,
            "n_tokens": start.get("n_tokens"),
            "ts_start": start.get("ts_start"),
            "ts_end": None,
            "n_nodes": 0,
            "wall_ms": None,
        })

    ops_df = pl.DataFrame(ops) if ops else pl.DataFrame(schema={
        "pass": pl.Int64, "seq": pl.Int64, "op": pl.Utf8, "name": pl.Utf8,
        "srcs": pl.List(pl.Utf8), "shape": pl.List(pl.Int64),
        "dtype": pl.Utf8, "backend": pl.Utf8, "t_start": pl.Int64, "t_end": pl.Int64,
    })
    passes_df = pl.DataFrame(pass_rows) if pass_rows else pl.DataFrame(schema={
        "pass": pl.Int64, "n_tokens": pl.Int64, "ts_start": pl.Int64,
        "ts_end": pl.Int64, "n_nodes": pl.Int64, "wall_ms": pl.Int64,
    })

    return ops_df, passes_df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd tools/trace-analyzer && python -m pytest tests/test_parser.py -v
```
Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/trace-analyzer/trace_analyzer/parser.py tools/trace-analyzer/tests/test_parser.py
git commit -m "tools: add JSONL trace parser with polars"
```

### Task 3: DAG Reconstruction

**Files:**
- Create: `tools/trace-analyzer/trace_analyzer/dag.py`
- Create: `tools/trace-analyzer/tests/test_dag.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dag.py
from pathlib import Path
import polars as pl
from trace_analyzer.parser import parse_trace
from trace_analyzer.dag import build_dag

FIXTURE = Path(__file__).parent / "fixtures" / "sample_trace.jsonl"

def test_build_dag_returns_nodes_and_edges():
    ops, _ = parse_trace(FIXTURE)
    dag = build_dag(ops, pass_id=1)
    assert "nodes" in dag
    assert "edges" in dag

def test_node_fields():
    ops, _ = parse_trace(FIXTURE)
    dag = build_dag(ops, pass_id=1)
    node = dag["nodes"][0]
    for key in ("id", "op", "backend", "ns", "shape", "dtype", "layer", "is_copy"):
        assert key in node, f"missing key: {key}"

def test_node_count():
    ops, _ = parse_trace(FIXTURE)
    dag = build_dag(ops, pass_id=1)
    assert len(dag["nodes"]) == 5  # 5 ops in pass 1

def test_edge_count():
    ops, _ = parse_trace(FIXTURE)
    dag = build_dag(ops, pass_id=1)
    # token_embd has 1 src, attn_norm has 1, attn_q has 2, ffn_gate has 2, ffn_out has 1 = 7
    assert len(dag["edges"]) == 7

def test_layer_grouping():
    ops, _ = parse_trace(FIXTURE)
    dag = build_dag(ops, pass_id=1)
    layers = {n["layer"] for n in dag["nodes"]}
    assert "blk.0" in layers
    assert None in layers or "" in layers  # token_embd has no blk.X prefix

def test_copy_detection():
    ops, _ = parse_trace(FIXTURE)
    dag = build_dag(ops, pass_id=1)
    copies = [n for n in dag["nodes"] if n["is_copy"]]
    assert len(copies) == 1
    assert copies[0]["id"] == "blk.0.ffn_out"

def test_edge_est_bytes():
    ops, _ = parse_trace(FIXTURE)
    dag = build_dag(ops, pass_id=1)
    edge = [e for e in dag["edges"] if e["to"] == "blk.0.attn_q" and e["from"] == "blk.0.attn_norm"][0]
    # source blk.0.attn_norm: shape=[128,1], dtype=f32 → 128*1*4 = 512 bytes
    assert edge["est_bytes"] == 512
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd tools/trace-analyzer && python -m pytest tests/test_dag.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'trace_analyzer.dag'`

- [ ] **Step 3: Implement dag.py**

```python
# trace_analyzer/dag.py
from __future__ import annotations
import re
import polars as pl

DTYPE_SIZE: dict[str, float] = {
    "f32": 4, "f16": 2, "bf16": 2,
    "q4_0": 0.5, "q4_1": 0.5, "q5_0": 0.625, "q5_1": 0.625,
    "q8_0": 1, "q8_1": 1, "i8": 1, "i16": 2, "i32": 4,
}

_LAYER_RE = re.compile(r"^(blk\.\d+)\.")

COPY_OPS = frozenset({"CPY", "DUP"})


def _estimate_bytes(shape: list[int], dtype: str) -> int:
    size = DTYPE_SIZE.get(dtype, 2)  # default f16
    total = 1
    for dim in shape:
        total *= dim
    return int(total * size)


def _extract_layer(name: str) -> str | None:
    m = _LAYER_RE.match(name)
    return m.group(1) if m else None


def build_dag(ops_df: pl.DataFrame, pass_id: int) -> dict:
    filtered = ops_df.filter(pl.col("pass") == pass_id).sort("seq")
    rows = filtered.to_dicts()

    # Build node info map keyed by name (for source lookups)
    node_info: dict[str, dict] = {}
    for row in rows:
        node_info[row["name"]] = row

    nodes = []
    edges = []

    for row in rows:
        ns = row["t_end"] - row["t_start"]
        layer = _extract_layer(row["name"])
        nodes.append({
            "id": row["name"],
            "op": row["op"],
            "backend": row["backend"],
            "ns": ns,
            "shape": row["shape"],
            "dtype": row["dtype"],
            "layer": layer,
            "is_copy": row["op"] in COPY_OPS,
        })
        for src_name in row.get("srcs", []):
            # Estimate bytes from source tensor if available, else from dest
            src = node_info.get(src_name)
            if src:
                est = _estimate_bytes(src["shape"], src["dtype"])
            else:
                est = _estimate_bytes(row["shape"], row["dtype"])
            edges.append({
                "from": src_name,
                "to": row["name"],
                "est_bytes": est,
            })

    return {"nodes": nodes, "edges": edges}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd tools/trace-analyzer && python -m pytest tests/test_dag.py -v
```
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/trace-analyzer/trace_analyzer/dag.py tools/trace-analyzer/tests/test_dag.py
git commit -m "tools: add DAG reconstruction with layer grouping"
```

### Task 4: Summary Module

**Files:**
- Create: `tools/trace-analyzer/trace_analyzer/summary.py`
- Create: `tools/trace-analyzer/tests/test_summary.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_summary.py
from pathlib import Path
from trace_analyzer.parser import parse_trace
from trace_analyzer.summary import build_summary

FIXTURE = Path(__file__).parent / "fixtures" / "sample_trace.jsonl"

def test_summary_has_all_sections():
    ops, passes = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    for key in ("meta", "timing", "op_stats", "backend_stats", "copy_stats", "layer_stats", "dag"):
        assert key in s, f"missing section: {key}"

def test_meta_fields():
    ops, passes = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    assert s["meta"]["source_file"] == "test.jsonl"
    assert s["meta"]["total_ops"] == 10
    assert s["meta"]["total_passes"] == 2

def test_timing_prefill_decode():
    ops, passes = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    t = s["timing"]
    assert t["prefill_tokens"] == 4  # pass 0 has n_tokens=4
    assert len(t["per_pass"]) == 2

def test_op_stats_sorted_by_time():
    ops, passes = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    times = [o["total_ns"] for o in s["op_stats"]]
    assert times == sorted(times, reverse=True)

def test_op_stats_pct_sums_to_100():
    ops, passes = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    total_pct = sum(o["pct_time"] for o in s["op_stats"])
    assert abs(total_pct - 100.0) < 0.1

def test_backend_stats():
    ops, passes = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    backends = {b["backend"] for b in s["backend_stats"]}
    assert "CUDA0" in backends
    assert "CPU" in backends

def test_copy_stats():
    ops, passes = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    cs = s["copy_stats"]
    assert cs["count"] == 2  # 1 CPY per pass * 2 passes
    assert len(cs["copies"]) == 2

def test_layer_stats():
    ops, passes = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    layers = {l["layer"] for l in s["layer_stats"]}
    assert "blk.0" in layers

def test_dag_present():
    ops, passes = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    assert len(s["dag"]["nodes"]) > 0
    assert len(s["dag"]["edges"]) > 0

def test_model_optional():
    ops, passes = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl", model="llama3:8b")
    assert s["meta"]["model"] == "llama3:8b"
    s2 = build_summary(ops, passes, source_file="test.jsonl")
    assert s2["meta"]["model"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd tools/trace-analyzer && python -m pytest tests/test_summary.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement summary.py**

```python
# trace_analyzer/summary.py
from __future__ import annotations
import re
import polars as pl
from .dag import build_dag, COPY_OPS, _estimate_bytes, _extract_layer


def build_summary(
    ops: pl.DataFrame,
    passes: pl.DataFrame,
    source_file: str,
    model: str | None = None,
    dag_pass: int | None = None,
) -> dict:
    total_ns = int((ops["t_end"] - ops["t_start"]).sum()) if len(ops) > 0 else 0

    # Timing
    per_pass = []
    for row in passes.sort("pass").to_dicts():
        per_pass.append({
            "pass": row["pass"],
            "n_tokens": row["n_tokens"],
            "wall_ms": row["wall_ms"],
            "n_ops": len(ops.filter(pl.col("pass") == row["pass"])),
        })

    prefill_passes = passes.filter(pl.col("n_tokens") > 1)
    decode_passes = passes.filter(pl.col("n_tokens") == 1)
    prefill_ms = float(prefill_passes["wall_ms"].sum()) if len(prefill_passes) > 0 else 0.0
    prefill_tokens = int(prefill_passes["n_tokens"].sum()) if len(prefill_passes) > 0 else 0
    decode_walls = decode_passes["wall_ms"].drop_nulls()
    decode_avg_ms = float(decode_walls.mean()) if len(decode_walls) > 0 else 0.0
    total_wall = passes["wall_ms"].drop_nulls()
    total_ms = float(total_wall.sum()) if len(total_wall) > 0 else 0.0

    # Op stats
    if len(ops) > 0:
        ops_with_ns = ops.with_columns((pl.col("t_end") - pl.col("t_start")).alias("ns"))
        op_groups = ops_with_ns.group_by("op").agg([
            pl.col("ns").sum().alias("total_ns"),
            pl.len().alias("count"),
            pl.col("ns").mean().alias("avg_ns"),
        ]).sort("total_ns", descending=True)

        op_stats = []
        for row in op_groups.to_dicts():
            entry = {
                "op": row["op"],
                "count": int(row["count"]),
                "total_ns": int(row["total_ns"]),
                "pct_time": round(row["total_ns"] / total_ns * 100, 1) if total_ns > 0 else 0.0,
                "avg_ns": int(row["avg_ns"]),
            }
            if row["op"] in COPY_OPS:
                copy_ops = ops_with_ns.filter(pl.col("op") == row["op"])
                est = sum(
                    _estimate_bytes(r["shape"], r["dtype"])
                    for r in copy_ops.to_dicts()
                )
                entry["est_bytes_total"] = est
            op_stats.append(entry)
    else:
        op_stats = []

    # Backend stats
    if len(ops) > 0:
        be_groups = ops_with_ns.group_by("backend").agg([
            pl.col("ns").sum().alias("total_ns"),
            pl.len().alias("count"),
        ]).sort("total_ns", descending=True)

        total_count = int(be_groups["count"].sum())
        backend_stats = []
        for row in be_groups.to_dicts():
            backend_stats.append({
                "backend": row["backend"],
                "count": int(row["count"]),
                "total_ns": int(row["total_ns"]),
                "pct_ops": round(row["count"] / total_count * 100, 1) if total_count > 0 else 0.0,
                "pct_time": round(row["total_ns"] / total_ns * 100, 1) if total_ns > 0 else 0.0,
            })
    else:
        backend_stats = []

    # Copy stats
    if len(ops) > 0:
        copy_df = ops_with_ns.filter(pl.col("op").is_in(list(COPY_OPS)))
        copies = []
        for row in copy_df.to_dicts():
            copies.append({
                "name": row["name"],
                "op": row["op"],
                "est_bytes": _estimate_bytes(row["shape"], row["dtype"]),
                "ns": int(row["ns"]),
                "backend": row["backend"],
            })
        copy_stats = {
            "count": len(copies),
            "total_ns": int(copy_df["ns"].sum()) if len(copy_df) > 0 else 0,
            "est_total_bytes": sum(c["est_bytes"] for c in copies),
            "copies": copies,
        }
    else:
        copy_stats = {"count": 0, "total_ns": 0, "est_total_bytes": 0, "copies": []}

    # Layer stats
    if len(ops) > 0:
        ops_layered = ops_with_ns.with_columns(
            pl.col("name").str.extract(r"^(blk\.\d+)\.", 1).fill_null("_top").alias("layer")
        )
        layer_groups = ops_layered.group_by("layer").agg([
            pl.col("ns").sum().alias("total_ns"),
            pl.len().alias("n_ops"),
        ]).sort("total_ns", descending=True)

        layer_stats = []
        for row in layer_groups.to_dicts():
            layer_ops = ops_layered.filter(pl.col("layer") == row["layer"])
            top_op_df = layer_ops.group_by("op").agg(
                pl.col("ns").sum().alias("total_ns")
            ).sort("total_ns", descending=True)
            top_op = top_op_df.row(0, named=True)["op"] if len(top_op_df) > 0 else ""
            layer_stats.append({
                "layer": row["layer"],
                "n_ops": int(row["n_ops"]),
                "total_ns": int(row["total_ns"]),
                "pct_time": round(row["total_ns"] / total_ns * 100, 1) if total_ns > 0 else 0.0,
                "top_op": top_op,
            })
    else:
        layer_stats = []

    # DAG (use first decode pass by default, or specified pass)
    if dag_pass is not None:
        dp = dag_pass
    elif len(decode_passes) > 0:
        dp = int(decode_passes["pass"].min())
    elif len(passes) > 0:
        dp = int(passes["pass"].min())
    else:
        dp = 0
    dag = build_dag(ops, dp)

    return {
        "meta": {
            "source_file": source_file,
            "model": model,
            "total_ops": len(ops),
            "total_passes": len(passes),
            "total_wall_ms": total_ms,
        },
        "timing": {
            "total_ms": total_ms,
            "prefill_ms": prefill_ms,
            "prefill_tokens": prefill_tokens,
            "decode_avg_ms": decode_avg_ms,
            "per_pass": per_pass,
        },
        "op_stats": op_stats,
        "backend_stats": backend_stats,
        "copy_stats": copy_stats,
        "layer_stats": layer_stats,
        "dag": dag,
        "timeline_ops": _build_timeline_ops(ops),
    }


def _build_timeline_ops(ops: pl.DataFrame) -> list[dict]:
    """Extract minimal per-op timing for Timeline view (all passes)."""
    if len(ops) == 0:
        return []
    return ops.select(["pass", "seq", "name", "op", "backend", "t_start", "t_end"]).to_dicts()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd tools/trace-analyzer && python -m pytest tests/test_summary.py -v
```
Expected: all 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/trace-analyzer/trace_analyzer/summary.py tools/trace-analyzer/tests/test_summary.py
git commit -m "tools: add single-trace summary module"
```

### Task 5: Compare Module

**Files:**
- Create: `tools/trace-analyzer/trace_analyzer/compare.py`
- Create: `tools/trace-analyzer/tests/test_compare.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_compare.py
from pathlib import Path
from trace_analyzer.parser import parse_trace
from trace_analyzer.summary import build_summary
from trace_analyzer.compare import build_compare

FIXTURE = Path(__file__).parent / "fixtures" / "sample_trace.jsonl"

def _make_two_summaries():
    ops, passes = parse_trace(FIXTURE)
    s1 = build_summary(ops, passes, source_file="a.jsonl")
    s2 = build_summary(ops, passes, source_file="b.jsonl")
    return s1, s2

def test_compare_has_all_sections():
    s1, s2 = _make_two_summaries()
    c = build_compare(s1, s2, labels=["A", "B"])
    for key in ("labels", "meta", "timing_diff", "op_diff", "layer_diff", "copy_diff"):
        assert key in c, f"missing section: {key}"

def test_identical_traces_zero_diff():
    s1, s2 = _make_two_summaries()
    c = build_compare(s1, s2, labels=["A", "B"])
    for op in c["op_diff"]:
        assert abs(op["diff_pct"]) < 0.01
        assert op["significant"] is False

def test_labels():
    s1, s2 = _make_two_summaries()
    c = build_compare(s1, s2, labels=["X", "Y"])
    assert c["labels"] == ["X", "Y"]
    assert c["meta"][0]["label"] == "X"
    assert c["meta"][1]["label"] == "Y"

def test_significance_threshold():
    s1, s2 = _make_two_summaries()
    # Manually bump s2 op times to create a 50% diff
    for op in s2["op_stats"]:
        op["total_ns"] = int(op["total_ns"] * 1.5)
    c = build_compare(s1, s2, labels=["A", "B"], threshold=10.0)
    sig_ops = [o for o in c["op_diff"] if o["significant"]]
    assert len(sig_ops) > 0

def test_timing_diff_arrays():
    s1, s2 = _make_two_summaries()
    c = build_compare(s1, s2, labels=["A", "B"])
    td = c["timing_diff"]
    assert len(td["prefill_ms"]) == 2
    assert len(td["total_ms"]) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd tools/trace-analyzer && python -m pytest tests/test_compare.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement compare.py**

```python
# trace_analyzer/compare.py
from __future__ import annotations


def _diff_pct(a: float, b: float) -> float:
    if a == 0:
        return 0.0 if b == 0 else 100.0
    return round((b - a) / a * 100, 1)


def build_compare(
    summary_a: dict,
    summary_b: dict,
    labels: list[str],
    threshold: float = 10.0,
) -> dict:
    la, lb = labels

    # Meta
    meta = [
        {"label": la, "source_file": summary_a["meta"]["source_file"],
         "model": summary_a["meta"].get("model"), "total_wall_ms": summary_a["meta"]["total_wall_ms"]},
        {"label": lb, "source_file": summary_b["meta"]["source_file"],
         "model": summary_b["meta"].get("model"), "total_wall_ms": summary_b["meta"]["total_wall_ms"]},
    ]

    # Timing diff
    ta, tb = summary_a["timing"], summary_b["timing"]
    timing_diff = {
        "prefill_ms": [ta["prefill_ms"], tb["prefill_ms"]],
        "decode_avg_ms": [ta["decode_avg_ms"], tb["decode_avg_ms"]],
        "total_ms": [ta["total_ms"], tb["total_ms"]],
        "diff_pct": {
            "prefill": _diff_pct(ta["prefill_ms"], tb["prefill_ms"]),
            "decode": _diff_pct(ta["decode_avg_ms"], tb["decode_avg_ms"]),
            "total": _diff_pct(ta["total_ms"], tb["total_ms"]),
        },
    }

    # Op diff — align by op name
    ops_a = {o["op"]: o for o in summary_a["op_stats"]}
    ops_b = {o["op"]: o for o in summary_b["op_stats"]}
    all_ops = sorted(set(ops_a) | set(ops_b))
    op_diff = []
    for op in all_ops:
        a_ns = ops_a.get(op, {}).get("total_ns", 0)
        b_ns = ops_b.get(op, {}).get("total_ns", 0)
        dp = _diff_pct(a_ns, b_ns)
        op_diff.append({
            "op": op,
            "values": [
                {"label": la, "total_ns": a_ns, "count": ops_a.get(op, {}).get("count", 0)},
                {"label": lb, "total_ns": b_ns, "count": ops_b.get(op, {}).get("count", 0)},
            ],
            "diff_pct": dp,
            "significant": abs(dp) > threshold,
        })
    op_diff.sort(key=lambda x: abs(x["diff_pct"]), reverse=True)

    # Layer diff
    layers_a = {l["layer"]: l for l in summary_a["layer_stats"]}
    layers_b = {l["layer"]: l for l in summary_b["layer_stats"]}
    all_layers = sorted(set(layers_a) | set(layers_b))
    layer_diff = []
    for layer in all_layers:
        a_ns = layers_a.get(layer, {}).get("total_ns", 0)
        b_ns = layers_b.get(layer, {}).get("total_ns", 0)
        dp = _diff_pct(a_ns, b_ns)
        layer_diff.append({
            "layer": layer,
            "values": [
                {"label": la, "total_ns": a_ns},
                {"label": lb, "total_ns": b_ns},
            ],
            "diff_pct": dp,
            "significant": abs(dp) > threshold,
        })
    layer_diff.sort(key=lambda x: abs(x["diff_pct"]), reverse=True)

    # Copy diff
    copies_a = {c["name"]: c for c in summary_a["copy_stats"]["copies"]}
    copies_b = {c["name"]: c for c in summary_b["copy_stats"]["copies"]}
    all_copies = sorted(set(copies_a) | set(copies_b))
    copy_diff = []
    for name in all_copies:
        a_ns = copies_a.get(name, {}).get("ns", 0)
        b_ns = copies_b.get(name, {}).get("ns", 0)
        a_bytes = copies_a.get(name, {}).get("est_bytes", 0)
        b_bytes = copies_b.get(name, {}).get("est_bytes", 0)
        dp = _diff_pct(a_ns, b_ns)
        copy_diff.append({
            "name": name,
            "values": [
                {"label": la, "ns": a_ns, "est_bytes": a_bytes},
                {"label": lb, "ns": b_ns, "est_bytes": b_bytes},
            ],
            "diff_pct": dp,
            "significant": abs(dp) > threshold,
        })

    return {
        "labels": labels,
        "meta": meta,
        "timing_diff": timing_diff,
        "op_diff": op_diff,
        "layer_diff": layer_diff,
        "copy_diff": copy_diff,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd tools/trace-analyzer && python -m pytest tests/test_compare.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/trace-analyzer/trace_analyzer/compare.py tools/trace-analyzer/tests/test_compare.py
git commit -m "tools: add two-trace comparison with significance flags"
```

### Task 6: Report Module + Jinja2 Templates

**Files:**
- Create: `tools/trace-analyzer/trace_analyzer/report.py`
- Create: `tools/trace-analyzer/trace_analyzer/templates/single_report.md.j2`
- Create: `tools/trace-analyzer/trace_analyzer/templates/compare_report.md.j2`

- [ ] **Step 1: Create single_report.md.j2**

```jinja2
# Inference Trace Report{% if meta.model %}: {{ meta.model }}{% endif %}

## Summary
- Source: `{{ meta.source_file }}`
- Total passes: {{ meta.total_passes }}, Total ops: {{ meta.total_ops }}
- Wall time: {{ "%.1f"|format(timing.total_ms) }}ms (prefill: {{ "%.1f"|format(timing.prefill_ms) }}ms/{{ timing.prefill_tokens }}tok, decode avg: {{ "%.1f"|format(timing.decode_avg_ms) }}ms/tok)

## Top Operators by Time
| Op | Count | Total (ms) | % Time | Avg (us) |
|----|-------|-----------|--------|----------|
{% for o in op_stats -%}
| {{ o.op }} | {{ o.count }} | {{ "%.1f"|format(o.total_ns / 1e6) }} | {{ o.pct_time }}% | {{ (o.avg_ns / 1000)|int }} |
{% endfor %}

## Backend Distribution
| Backend | Ops | % Ops | % Time |
|---------|-----|-------|--------|
{% for b in backend_stats -%}
| {{ b.backend }} | {{ b.count }} | {{ b.pct_ops }}% | {{ b.pct_time }}% |
{% endfor %}

{% if copy_stats.count > 0 -%}
## Data Transfers
| Tensor | Est. Size | Time (us) | Backend |
|--------|-----------|-----------|---------|
{% for c in copy_stats.copies -%}
| {{ c.name }} | {{ "%.1f"|format(c.est_bytes / 1048576) }} MB | {{ (c.ns / 1000)|int }} | {{ c.backend }} |
{% endfor %}
{% endif %}

## Per-Layer Breakdown
| Layer | Ops | Total (ms) | % Time | Top Op |
|-------|-----|-----------|--------|--------|
{% for l in layer_stats -%}
| {{ l.layer }} | {{ l.n_ops }} | {{ "%.1f"|format(l.total_ns / 1e6) }} | {{ l.pct_time }}% | {{ l.top_op }} |
{% endfor %}
```

- [ ] **Step 2: Create compare_report.md.j2**

```jinja2
# Trace Comparison: {{ labels[0] }} vs {{ labels[1] }}

## Summary
| Metric | {{ labels[0] }} | {{ labels[1] }} | Diff |
|--------|{% for _ in labels %}---------|{% endfor %}------|
| Total (ms) | {{ "%.1f"|format(timing_diff.total_ms[0]) }} | {{ "%.1f"|format(timing_diff.total_ms[1]) }} | {% if timing_diff.diff_pct.total|abs > 10 %}⚠️ {% endif %}{{ "%+.1f"|format(timing_diff.diff_pct.total) }}% |
| Prefill (ms) | {{ "%.1f"|format(timing_diff.prefill_ms[0]) }} | {{ "%.1f"|format(timing_diff.prefill_ms[1]) }} | {% if timing_diff.diff_pct.prefill|abs > 10 %}⚠️ {% endif %}{{ "%+.1f"|format(timing_diff.diff_pct.prefill) }}% |
| Decode avg (ms) | {{ "%.1f"|format(timing_diff.decode_avg_ms[0]) }} | {{ "%.1f"|format(timing_diff.decode_avg_ms[1]) }} | {% if timing_diff.diff_pct.decode|abs > 10 %}⚠️ {% endif %}{{ "%+.1f"|format(timing_diff.diff_pct.decode) }}% |

## Op Comparison
| Op | {{ labels[0] }} (ms) | {{ labels[1] }} (ms) | Diff | Significant |
|----|----------|----------|------|-------------|
{% for o in op_diff -%}
| {{ o.op }} | {{ "%.1f"|format(o.values[0].total_ns / 1e6) }} | {{ "%.1f"|format(o.values[1].total_ns / 1e6) }} | {% if o.significant %}⚠️ {% endif %}{{ "%+.1f"|format(o.diff_pct) }}% | {{ "Yes" if o.significant else "No" }} |
{% endfor %}

## Layer Comparison
| Layer | {{ labels[0] }} (ms) | {{ labels[1] }} (ms) | Diff | Significant |
|-------|----------|----------|------|-------------|
{% for l in layer_diff -%}
| {{ l.layer }} | {{ "%.1f"|format(l.values[0].total_ns / 1e6) }} | {{ "%.1f"|format(l.values[1].total_ns / 1e6) }} | {% if l.significant %}⚠️ {% endif %}{{ "%+.1f"|format(l.diff_pct) }}% | {{ "Yes" if l.significant else "No" }} |
{% endfor %}
```

- [ ] **Step 3: Implement report.py**

```python
# trace_analyzer/report.py
from __future__ import annotations
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        keep_trailing_newline=True,
    )


def render_single(summary: dict) -> str:
    tmpl = _env().get_template("single_report.md.j2")
    return tmpl.render(**summary)


def render_compare(compare_data: dict) -> str:
    tmpl = _env().get_template("compare_report.md.j2")
    return tmpl.render(**compare_data)
```

- [ ] **Step 4: Create tests/test_report.py**

```python
# tests/test_report.py
from pathlib import Path
from trace_analyzer.parser import parse_trace
from trace_analyzer.summary import build_summary
from trace_analyzer.compare import build_compare
from trace_analyzer.report import render_single, render_compare

FIXTURE = Path(__file__).parent / "fixtures" / "sample_trace.jsonl"

def test_single_report_has_sections():
    ops, passes = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    md = render_single(s)
    assert "## Top Operators" in md
    assert "MUL_MAT" in md
    assert "## Backend Distribution" in md

def test_single_report_with_model():
    ops, passes = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl", model="llama3:8b")
    md = render_single(s)
    assert "llama3:8b" in md

def test_compare_report_has_labels():
    ops, passes = parse_trace(FIXTURE)
    s1 = build_summary(ops, passes, source_file="a.jsonl")
    s2 = build_summary(ops, passes, source_file="b.jsonl")
    c = build_compare(s1, s2, labels=["X", "Y"])
    md = render_compare(c)
    assert "X" in md and "Y" in md
    assert "## Op Comparison" in md
```

Run: `cd tools/trace-analyzer && python -m pytest tests/test_report.py -v`

- [ ] **Step 5: Commit**

```bash
git add tools/trace-analyzer/trace_analyzer/report.py tools/trace-analyzer/trace_analyzer/templates/
git commit -m "tools: add Markdown report generation with Jinja2 templates"
```

### Task 7: CLI Entry Point

**Files:**
- Create: `tools/trace-analyzer/trace_analyzer/cli.py`

- [ ] **Step 1: Implement cli.py**

```python
# trace_analyzer/cli.py
from __future__ import annotations
import json
import click
from pathlib import Path


@click.group()
def main():
    """Trace Analyzer — post-process ollama inference traces."""
    pass


@main.command()
@click.argument("trace_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None)
@click.option("--model", default=None, help="Model name (optional)")
@click.option("--pass", "dag_pass", type=int, default=None, help="Pass ID for DAG (default: first decode)")
def summary(trace_file: Path, output: Path | None, model: str | None, dag_pass: int | None):
    """Generate summary.json from a single JSONL trace."""
    from .parser import parse_trace
    from .summary import build_summary

    ops, passes = parse_trace(trace_file)
    result = build_summary(ops, passes, source_file=trace_file.name, model=model, dag_pass=dag_pass)

    text = json.dumps(result, indent=2)
    if output:
        output.write_text(text)
        click.echo(f"Written to {output}")
    else:
        click.echo(text)


@main.command()
@click.argument("trace_a", type=click.Path(exists=True, path_type=Path))
@click.argument("trace_b", type=click.Path(exists=True, path_type=Path))
@click.option("--labels", required=True, help="Comma-separated labels (e.g. 'CUDA,Vulkan')")
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None)
@click.option("--model", default=None)
@click.option("--threshold", type=float, default=10.0, help="Significance threshold %")
def compare(trace_a: Path, trace_b: Path, labels: str, output: Path | None, model: str | None, threshold: float):
    """Compare two JSONL traces."""
    from .parser import parse_trace
    from .summary import build_summary
    from .compare import build_compare

    label_list = [l.strip() for l in labels.split(",")]
    if len(label_list) != 2:
        raise click.BadParameter("Exactly 2 labels required", param_hint="--labels")

    ops_a, passes_a = parse_trace(trace_a)
    ops_b, passes_b = parse_trace(trace_b)
    sa = build_summary(ops_a, passes_a, source_file=trace_a.name, model=model)
    sb = build_summary(ops_b, passes_b, source_file=trace_b.name, model=model)
    result = build_compare(sa, sb, labels=label_list, threshold=threshold)

    text = json.dumps(result, indent=2)
    if output:
        output.write_text(text)
        click.echo(f"Written to {output}")
    else:
        click.echo(text)


@main.command()
@click.argument("trace_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None)
@click.option("--model", default=None)
@click.option("--compare", "trace_b", type=click.Path(exists=True, path_type=Path), default=None, help="Second trace for comparison report")
@click.option("--labels", default=None)
def report(trace_file: Path, output: Path | None, model: str | None, trace_b: Path | None, labels: str | None):
    """Generate LLM-ready Markdown report."""
    from .parser import parse_trace
    from .summary import build_summary
    from .report import render_single, render_compare

    if trace_b:
        from .compare import build_compare
        label_list = [l.strip() for l in labels.split(",")] if labels else ["A", "B"]
        ops_a, passes_a = parse_trace(trace_file)
        ops_b, passes_b = parse_trace(trace_b)
        sa = build_summary(ops_a, passes_a, source_file=trace_file.name, model=model)
        sb = build_summary(ops_b, passes_b, source_file=trace_b.name, model=model)
        cmp = build_compare(sa, sb, labels=label_list)
        md = render_compare(cmp)
    else:
        ops, passes = parse_trace(trace_file)
        s = build_summary(ops, passes, source_file=trace_file.name, model=model)
        md = render_single(s)

    if output:
        output.write_text(md)
        click.echo(f"Written to {output}")
    else:
        click.echo(md)
```

- [ ] **Step 2: Test CLI with Click test runner**

```bash
cd tools/trace-analyzer
python -m trace_analyzer.cli summary tests/fixtures/sample_trace.jsonl | python -m json.tool > /dev/null && echo "summary OK"
python -m trace_analyzer.cli report tests/fixtures/sample_trace.jsonl | head -5
python -m trace_analyzer.cli compare tests/fixtures/sample_trace.jsonl tests/fixtures/sample_trace.jsonl --labels "A,B" | python -m json.tool > /dev/null && echo "compare OK"
```

- [ ] **Step 3: Commit**

```bash
git add tools/trace-analyzer/trace_analyzer/cli.py
git commit -m "tools: add Click CLI entry point for trace-analyzer"
```

### Task 8: Serve Command

**Files:**
- Create: `tools/trace-analyzer/trace_analyzer/serve.py`

- [ ] **Step 1: Implement serve.py**

```python
# trace_analyzer/serve.py
from __future__ import annotations
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import click


class TraceHandler(SimpleHTTPRequestHandler):
    data_dir: Path
    web_dir: Path | None

    def do_GET(self):
        if self.path == "/api/files":
            self._list_files()
        elif self.path.startswith("/data/"):
            self._serve_data_file()
        elif self.web_dir:
            # Serve React build
            self.directory = str(self.web_dir)
            super().do_GET()
        else:
            self.send_error(404)

    def _list_files(self):
        files = []
        for f in sorted(self.data_dir.glob("*.json")):
            files.append({"name": f.name, "size": f.stat().st_size})
        body = json.dumps(files).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_data_file(self):
        name = self.path[len("/data/"):]
        fpath = self.data_dir / name
        if not fpath.is_file() or not fpath.resolve().is_relative_to(self.data_dir.resolve()):
            self.send_error(404)
            return
        body = fpath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(data_dir: Path, port: int = 8765, web_dir: Path | None = None):
    TraceHandler.data_dir = data_dir
    TraceHandler.web_dir = web_dir
    server = HTTPServer(("0.0.0.0", port), TraceHandler)
    click.echo(f"Serving on http://localhost:{port}")
    click.echo(f"  Data dir: {data_dir}")
    click.echo(f"  API: http://localhost:{port}/api/files")
    server.serve_forever()
```

- [ ] **Step 2: Wire serve command into cli.py**

Add to `cli.py`:

```python
@main.command()
@click.option("--data-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--port", type=int, default=8765)
def serve(data_dir: Path, port: int):
    """Launch dev server for React frontend + JSON data."""
    from .serve import run_server
    web_dir = Path(__file__).parent.parent / "web" / "dist"
    run_server(data_dir, port, web_dir if web_dir.is_dir() else None)
```

- [ ] **Step 3: Commit**

```bash
git add tools/trace-analyzer/trace_analyzer/serve.py tools/trace-analyzer/trace_analyzer/cli.py
git commit -m "tools: add dev server with /api/files endpoint"
```

### Task 9: Run All Python Tests + Final Verification

- [ ] **Step 1: Run full test suite**

```bash
cd tools/trace-analyzer && python -m pytest tests/ -v
```
Expected: all tests PASS

- [ ] **Step 2: End-to-end CLI verification**

```bash
cd tools/trace-analyzer
python -m trace_analyzer.cli summary tests/fixtures/sample_trace.jsonl -o /tmp/summary.json
python -m trace_analyzer.cli report tests/fixtures/sample_trace.jsonl -o /tmp/report.md
python -m trace_analyzer.cli compare tests/fixtures/sample_trace.jsonl tests/fixtures/sample_trace.jsonl --labels "A,B" -o /tmp/compare.json
# Verify files exist and are valid
python -c "import json; json.load(open('/tmp/summary.json')); print('summary OK')"
python -c "import json; json.load(open('/tmp/compare.json')); print('compare OK')"
python -c "print(open('/tmp/report.md').read()[:200])"
```

- [ ] **Step 3: Commit (if any fixes needed)**

```bash
git add -A tools/trace-analyzer/
git commit -m "tools: Python CLI complete — parser, summary, compare, report, serve"
```

---

## Phase B: React SPA

### Task 10: React Project Scaffolding

**Files:**
- Create: `tools/trace-analyzer/web/package.json`
- Create: `tools/trace-analyzer/web/tsconfig.json`
- Create: `tools/trace-analyzer/web/vite.config.ts`
- Create: `tools/trace-analyzer/web/tailwind.config.js`
- Create: `tools/trace-analyzer/web/postcss.config.js`
- Create: `tools/trace-analyzer/web/index.html`
- Create: `tools/trace-analyzer/web/src/main.tsx`
- Create: `tools/trace-analyzer/web/src/index.css`

- [ ] **Step 1: Initialize with Vite**

```bash
cd tools/trace-analyzer
npm create vite@latest web -- --template react-ts
cd web
npm install
npm install cytoscape @types/cytoscape d3 @types/d3 tailwindcss @tailwindcss/vite
```

- [ ] **Step 2: Configure Tailwind**

Add Tailwind to `vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
})
```

Replace `src/index.css` content with:
```css
@import "tailwindcss";
```

- [ ] **Step 3: Verify dev server starts**

```bash
cd tools/trace-analyzer/web && npm run dev -- --port 3000 &
sleep 3 && curl -s http://localhost:3000 | head -5
kill %1
```

- [ ] **Step 4: Commit**

```bash
git add tools/trace-analyzer/web/
git commit -m "tools: scaffold React SPA with Vite + Tailwind + Cytoscape + D3"
```

### Task 11: TypeScript Types + Utilities

**Files:**
- Create: `tools/trace-analyzer/web/src/types/trace.ts`
- Create: `tools/trace-analyzer/web/src/utils/colorScale.ts`
- Create: `tools/trace-analyzer/web/src/utils/dataSize.ts`

- [ ] **Step 1: Create TypeScript interfaces matching JSON schemas**

```typescript
// src/types/trace.ts
export interface SummaryData {
  meta: {
    source_file: string;
    model: string | null;
    total_ops: number;
    total_passes: number;
    total_wall_ms: number;
  };
  timing: {
    total_ms: number;
    prefill_ms: number;
    prefill_tokens: number;
    decode_avg_ms: number;
    per_pass: Array<{ pass: number; n_tokens: number; wall_ms: number | null; n_ops: number }>;
  };
  op_stats: Array<{
    op: string; count: number; total_ns: number;
    pct_time: number; avg_ns: number; est_bytes_total?: number;
  }>;
  backend_stats: Array<{
    backend: string; count: number; total_ns: number;
    pct_ops: number; pct_time: number;
  }>;
  copy_stats: {
    count: number; total_ns: number; est_total_bytes: number;
    copies: Array<{
      name: string; op: string; est_bytes: number; ns: number; backend: string;
    }>;
  };
  layer_stats: Array<{
    layer: string; n_ops: number; total_ns: number;
    pct_time: number; top_op: string;
  }>;
  dag: DagData;
  timeline_ops: TimelineOp[];
}

export interface TimelineOp {
  pass: number; seq: number; name: string;
  op: string; backend: string; t_start: number; t_end: number;
}

export interface DagData {
  nodes: DagNode[];
  edges: DagEdge[];
}

export interface DagNode {
  id: string; op: string; backend: string;
  ns: number; shape: number[]; dtype: string;
  layer: string | null; is_copy: boolean;
}

export interface DagEdge {
  from: string; to: string; est_bytes: number;
}

export interface CompareData {
  labels: [string, string];
  meta: Array<{ label: string; source_file: string; model: string | null; total_wall_ms: number }>;
  timing_diff: {
    prefill_ms: [number, number];
    decode_avg_ms: [number, number];
    total_ms: [number, number];
    diff_pct: { prefill: number; decode: number; total: number };
  };
  op_diff: Array<{
    op: string;
    values: Array<{ label: string; total_ns: number; count?: number }>;
    diff_pct: number;
    significant: boolean;
  }>;
  layer_diff: Array<{
    layer: string;
    values: Array<{ label: string; total_ns: number }>;
    diff_pct: number;
    significant: boolean;
  }>;
  copy_diff: Array<{
    name: string;
    values: Array<{ label: string; ns: number; est_bytes: number }>;
    diff_pct: number;
    significant: boolean;
  }>;
}
```

- [ ] **Step 2: Create colorScale.ts**

```typescript
// src/utils/colorScale.ts
export const BACKEND_COLORS: Record<string, string> = {
  CPU: '#3b82f6',    // blue
  CUDA0: '#22c55e',  // green
  CUDA1: '#16a34a',
  Vulkan: '#f97316', // orange
  Metal: '#a855f7',  // purple
};

export function backendColor(backend: string): string {
  return BACKEND_COLORS[backend] ?? '#6b7280'; // gray fallback
}

export function heatmapColor(ratio: number): string {
  // 0=blue, 0.5=yellow, 1=red
  const r = ratio < 0.5 ? Math.round(ratio * 2 * 255) : 255;
  const g = ratio < 0.5 ? 255 : Math.round((1 - (ratio - 0.5) * 2) * 255);
  const b = ratio < 0.5 ? Math.round((1 - ratio * 2) * 255) : 0;
  return `rgb(${r},${g},${b})`;
}

export function diffColor(diffPct: number, threshold: number = 10): string {
  if (diffPct > threshold) return '#fee2e2';  // red-50
  if (diffPct < -threshold) return '#dcfce7'; // green-50
  return 'transparent';
}
```

- [ ] **Step 3: Create dataSize.ts**

```typescript
// src/utils/dataSize.ts
const DTYPE_SIZE: Record<string, number> = {
  f32: 4, f16: 2, bf16: 2,
  q4_0: 0.5, q4_1: 0.5, q5_0: 0.625, q5_1: 0.625,
  q8_0: 1, q8_1: 1, i8: 1, i16: 2, i32: 4,
};

export function estimateBytes(shape: number[], dtype: string): number {
  const size = DTYPE_SIZE[dtype] ?? 2;
  return shape.reduce((a, b) => a * b, 1) * size;
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1_073_741_824) return `${(bytes / 1_073_741_824).toFixed(1)} GB`;
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

export function formatNs(ns: number): string {
  if (ns >= 1_000_000_000) return `${(ns / 1_000_000_000).toFixed(2)}s`;
  if (ns >= 1_000_000) return `${(ns / 1_000_000).toFixed(1)}ms`;
  if (ns >= 1_000) return `${(ns / 1_000).toFixed(0)}us`;
  return `${ns}ns`;
}
```

- [ ] **Step 4: Verify TypeScript compiles**

```bash
cd tools/trace-analyzer/web && npx tsc --noEmit
```

- [ ] **Step 5: Commit**

```bash
git add tools/trace-analyzer/web/src/types/ tools/trace-analyzer/web/src/utils/
git commit -m "tools: add TypeScript types and utility functions"
```

### Task 12: Data Loading Hook + DAG Layout Utility

**Files:**
- Create: `tools/trace-analyzer/web/src/hooks/useTraceData.ts`
- Create: `tools/trace-analyzer/web/src/utils/dagLayout.ts`

- [ ] **Step 1: Create useTraceData.ts**

```typescript
// src/hooks/useTraceData.ts
import { useState, useEffect } from 'react';
import type { SummaryData, CompareData } from '../types/trace';

interface FileEntry { name: string; size: number; }

const BASE = import.meta.env.DEV ? 'http://localhost:8765' : '';

export function useFileList() {
  const [files, setFiles] = useState<FileEntry[]>([]);
  useEffect(() => {
    fetch(`${BASE}/api/files`).then(r => r.json()).then(setFiles).catch(() => {});
  }, []);
  return files;
}

export function useSummary(filename: string | null) {
  const [data, setData] = useState<SummaryData | null>(null);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!filename) return;
    setLoading(true);
    fetch(`${BASE}/data/${filename}`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [filename]);
  return { data, loading };
}

export function useCompare(filename: string | null) {
  const [data, setData] = useState<CompareData | null>(null);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!filename) return;
    setLoading(true);
    fetch(`${BASE}/data/${filename}`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [filename]);
  return { data, loading };
}
```

- [ ] **Step 2: Create dagLayout.ts**

Converts `DagData` into Cytoscape elements with compound nodes for layer folding.

```typescript
// src/utils/dagLayout.ts
import type { DagData, DagNode } from '../types/trace';
import type { ElementDefinition } from 'cytoscape';
import { backendColor, heatmapColor } from './colorScale';
import { formatNs } from './dataSize';

export type ColorMode = 'backend' | 'heatmap';

export function buildCytoscapeElements(
  dag: DagData,
  colorMode: ColorMode,
  collapsed: Set<string>,
): ElementDefinition[] {
  const elements: ElementDefinition[] = [];

  // Find max ns for heatmap normalization
  const maxNs = Math.max(...dag.nodes.map(n => n.ns), 1);

  // Collect layers and compute per-layer totals
  const layers = new Set<string>();
  const layerTotals = new Map<string, number>();
  for (const node of dag.nodes) {
    if (node.layer) {
      layers.add(node.layer);
      layerTotals.set(node.layer, (layerTotals.get(node.layer) ?? 0) + node.ns);
    }
  }
  const maxLayerTotal = Math.max(...layerTotals.values(), 1);

  // Add layer compound nodes
  for (const layer of layers) {
    const totalNs = layerTotals.get(layer) ?? 0;
    const bgColor = colorMode === 'heatmap'
      ? heatmapColor(totalNs / maxLayerTotal)
      : '#e5e7eb';

    elements.push({
      data: {
        id: `layer:${layer}`,
        label: `${layer} (${formatNs(totalNs)})`,
        isLayer: true,
        collapsed: collapsed.has(layer),
      },
      style: { 'background-color': bgColor },
    });
  }

  // Add op nodes
  for (const node of dag.nodes) {
    const layerId = node.layer ? `layer:${node.layer}` : undefined;
    if (layerId && collapsed.has(node.layer!)) continue; // skip if collapsed

    const bgColor = colorMode === 'heatmap'
      ? heatmapColor(node.ns / maxNs)
      : backendColor(node.backend);

    elements.push({
      data: {
        id: node.id,
        label: `${node.op}\n${node.id}`,
        parent: layerId,
        ...node,
      },
      style: {
        'background-color': bgColor,
        'border-style': node.is_copy ? 'dashed' : 'solid',
        'border-color': node.is_copy ? '#ef4444' : '#6b7280',
        'border-width': node.is_copy ? 3 : 1,
        width: Math.max(30, Math.log2(node.ns + 1) * 5),
        height: Math.max(30, Math.log2(node.ns + 1) * 5),
      },
    });
  }

  // Build node lookup map for O(1) edge filtering
  const nodeMap = new Map(dag.nodes.map(n => [n.id, n]));

  // Add edges
  for (const edge of dag.edges) {
    // Skip edges to/from nodes in collapsed layers
    const fromNode = nodeMap.get(edge.from);
    const toNode = nodeMap.get(edge.to);
    if (fromNode?.layer && collapsed.has(fromNode.layer)) continue;
    if (toNode?.layer && collapsed.has(toNode.layer)) continue;

    elements.push({
      data: {
        source: edge.from,
        target: edge.to,
        est_bytes: edge.est_bytes,
      },
      style: {
        width: Math.max(1, Math.log2(edge.est_bytes + 1) * 0.5),
      },
    });
  }

  return elements;
}
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd tools/trace-analyzer/web && npx tsc --noEmit
```

- [ ] **Step 4: Commit**

```bash
git add tools/trace-analyzer/web/src/hooks/ tools/trace-analyzer/web/src/utils/dagLayout.ts
git commit -m "tools: add data loading hooks and DAG layout utility"
```

### Task 13: HotspotPanel + NodeDetail + ColorToggle Components

**Files:**
- Create: `tools/trace-analyzer/web/src/components/HotspotPanel.tsx`
- Create: `tools/trace-analyzer/web/src/components/NodeDetail.tsx`
- Create: `tools/trace-analyzer/web/src/components/ColorToggle.tsx`

- [ ] **Step 1: Create ColorToggle.tsx**

Simple toggle button between backend and heatmap color modes.

```tsx
// src/components/ColorToggle.tsx
import type { ColorMode } from '../utils/dagLayout';

interface Props {
  mode: ColorMode;
  onChange: (mode: ColorMode) => void;
}

export function ColorToggle({ mode, onChange }: Props) {
  return (
    <div className="flex gap-1 rounded-lg bg-gray-100 p-1">
      <button
        className={`px-3 py-1 rounded text-sm ${mode === 'backend' ? 'bg-white shadow' : ''}`}
        onClick={() => onChange('backend')}
      >Backend</button>
      <button
        className={`px-3 py-1 rounded text-sm ${mode === 'heatmap' ? 'bg-white shadow' : ''}`}
        onClick={() => onChange('heatmap')}
      >Heatmap</button>
    </div>
  );
}
```

- [ ] **Step 2: Create NodeDetail.tsx**

Drawer/panel that shows details for a selected DAG node.

```tsx
// src/components/NodeDetail.tsx
import type { DagNode } from '../types/trace';
import { formatNs, formatBytes, estimateBytes } from '../utils/dataSize';

interface Props { node: DagNode | null; onClose: () => void; }

export function NodeDetail({ node, onClose }: Props) {
  if (!node) return null;
  return (
    <div className="fixed right-0 top-0 h-full w-80 bg-white shadow-lg p-4 overflow-y-auto z-50">
      <button onClick={onClose} className="float-right text-gray-400 hover:text-gray-600">X</button>
      <h3 className="font-bold text-lg mb-4">{node.id}</h3>
      <table className="w-full text-sm">
        <tbody>
          <tr><td className="text-gray-500 pr-4">Op</td><td>{node.op}</td></tr>
          <tr><td className="text-gray-500 pr-4">Backend</td><td>{node.backend}</td></tr>
          <tr><td className="text-gray-500 pr-4">Time</td><td>{formatNs(node.ns)}</td></tr>
          <tr><td className="text-gray-500 pr-4">Shape</td><td>[{node.shape.join(', ')}]</td></tr>
          <tr><td className="text-gray-500 pr-4">Dtype</td><td>{node.dtype}</td></tr>
          <tr><td className="text-gray-500 pr-4">Est. Size</td><td>{formatBytes(estimateBytes(node.shape, node.dtype))}</td></tr>
          <tr><td className="text-gray-500 pr-4">Layer</td><td>{node.layer ?? '(top-level)'}</td></tr>
          <tr><td className="text-gray-500 pr-4">Copy?</td><td>{node.is_copy ? 'Yes' : 'No'}</td></tr>
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 3: Create HotspotPanel.tsx**

Collapsible right sidebar with three ranked tabs.

```tsx
// src/components/HotspotPanel.tsx
import { useState } from 'react';
import type { SummaryData, DagNode } from '../types/trace';
import { formatNs, formatBytes } from '../utils/dataSize';

type Tab = 'ops' | 'copies_size' | 'copies_time';

interface Props {
  data: SummaryData;
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function HotspotPanel({ data, selectedId, onSelect }: Props) {
  const [tab, setTab] = useState<Tab>('ops');
  const [collapsed, setCollapsed] = useState(false);

  if (collapsed) {
    return (
      <button
        className="fixed right-0 top-1/2 bg-gray-800 text-white px-2 py-4 rounded-l z-40"
        onClick={() => setCollapsed(false)}
      >{'<'}</button>
    );
  }

  const opsRanked = [...data.dag.nodes].sort((a, b) => b.ns - a.ns);
  const copies = data.dag.nodes.filter(n => n.is_copy);
  const copiesBySize = [...copies].sort((a, b) => {
    const sa = a.shape.reduce((x, y) => x * y, 1);
    const sb = b.shape.reduce((x, y) => x * y, 1);
    return sb - sa;
  });
  const copiesByTime = [...copies].sort((a, b) => b.ns - a.ns);

  const items: DagNode[] =
    tab === 'ops' ? opsRanked :
    tab === 'copies_size' ? copiesBySize : copiesByTime;

  return (
    <div className="fixed right-0 top-0 h-full w-72 bg-white border-l shadow z-40 flex flex-col">
      <div className="flex items-center justify-between p-2 border-b">
        <span className="font-bold text-sm">Hotspots</span>
        <button onClick={() => setCollapsed(true)} className="text-gray-400">{'>'}</button>
      </div>
      <div className="flex gap-1 p-2 bg-gray-50">
        {(['ops', 'copies_size', 'copies_time'] as Tab[]).map(t => (
          <button
            key={t}
            className={`px-2 py-1 text-xs rounded ${tab === t ? 'bg-gray-800 text-white' : 'bg-gray-200'}`}
            onClick={() => setTab(t)}
          >{t === 'ops' ? 'Top Ops' : t === 'copies_size' ? 'Copies (size)' : 'Copies (time)'}</button>
        ))}
      </div>
      <div className="overflow-y-auto flex-1">
        {items.slice(0, 50).map((node, i) => (
          <div
            key={node.id + i}
            className={`px-3 py-2 cursor-pointer text-sm border-b hover:bg-blue-50 ${selectedId === node.id ? 'bg-blue-100' : ''}`}
            onClick={() => onSelect(node.id)}
          >
            <div className="flex justify-between">
              <span className="text-gray-400 mr-2">#{i + 1}</span>
              <span className="font-mono truncate flex-1">{node.id}</span>
            </div>
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>{node.op}</span>
              <span>{formatNs(node.ns)}</span>
              <span>{node.backend}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Verify TypeScript compiles**

```bash
cd tools/trace-analyzer/web && npx tsc --noEmit
```

- [ ] **Step 5: Commit**

```bash
git add tools/trace-analyzer/web/src/components/ColorToggle.tsx tools/trace-analyzer/web/src/components/NodeDetail.tsx tools/trace-analyzer/web/src/components/HotspotPanel.tsx
git commit -m "tools: add HotspotPanel, NodeDetail, ColorToggle components"
```

### Task 14: DagView Component

**Files:**
- Create: `tools/trace-analyzer/web/src/components/DagView.tsx`

- [ ] **Step 1: Create DagView.tsx**

Cytoscape.js DAG with layer folding, search, and bidirectional linking.

```tsx
// src/components/DagView.tsx
import { useRef, useEffect, useState, useCallback } from 'react';
import cytoscape, { type Core } from 'cytoscape';
import type { SummaryData, DagNode } from '../types/trace';
import { buildCytoscapeElements, type ColorMode } from '../utils/dagLayout';
import { ColorToggle } from './ColorToggle';
import { NodeDetail } from './NodeDetail';

interface Props {
  data: SummaryData;
  highlightId: string | null;
  onSelectNode: (id: string) => void;
}

export function DagView({ data, highlightId, onSelectNode }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const [colorMode, setColorMode] = useState<ColorMode>('backend');
  const [collapsed, setCollapsed] = useState<Set<string>>(() => {
    const layers = new Set<string>();
    for (const n of data.dag.nodes) { if (n.layer) layers.add(n.layer); }
    return layers; // all collapsed initially
  });
  const [selectedNode, setSelectedNode] = useState<DagNode | null>(null);
  const [search, setSearch] = useState('');

  // Initialize Cytoscape
  useEffect(() => {
    if (!containerRef.current) return;
    const elements = buildCytoscapeElements(data.dag, colorMode, collapsed);
    const cy = cytoscape({
      container: containerRef.current,
      elements,
      layout: { name: 'breadthfirst', directed: true, spacingFactor: 1.5 },
      style: [
        { selector: 'node', style: {
          label: 'data(label)', 'font-size': 10, 'text-wrap': 'wrap',
          'text-valign': 'center', 'text-halign': 'center',
        }},
        { selector: ':parent', style: {
          'text-valign': 'top', 'font-weight': 'bold',
        }},
        { selector: 'edge', style: {
          'curve-style': 'bezier', 'target-arrow-shape': 'triangle',
          'arrow-scale': 0.8, 'line-color': '#9ca3af', 'target-arrow-color': '#9ca3af',
        }},
      ],
    });

    cy.on('tap', 'node', (evt) => {
      const nodeData = evt.target.data();
      if (nodeData.isLayer) {
        // Toggle collapse
        const layer = nodeData.id.replace('layer:', '');
        setCollapsed(prev => {
          const next = new Set(prev);
          if (next.has(layer)) next.delete(layer); else next.add(layer);
          return next;
        });
      } else {
        const dagNode = data.dag.nodes.find(n => n.id === nodeData.id);
        if (dagNode) { setSelectedNode(dagNode); onSelectNode(dagNode.id); }
      }
    });

    cyRef.current = cy;
    return () => { cy.destroy(); };
  }, [data.dag, colorMode, collapsed]);

  // Highlight external selection
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !highlightId) return;
    cy.nodes().removeClass('highlighted');
    const node = cy.getElementById(highlightId);
    if (node.length) {
      node.addClass('highlighted');
      cy.animate({ center: { eles: node }, duration: 300 });
    }
  }, [highlightId]);

  // Search
  const handleSearch = useCallback((term: string) => {
    setSearch(term);
    const cy = cyRef.current;
    if (!cy || !term) return;
    const match = cy.nodes().filter(n => n.data('id')?.includes(term));
    if (match.length) {
      cy.animate({ fit: { eles: match, padding: 50 }, duration: 300 });
    }
  }, []);

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-4 p-2 border-b">
        <ColorToggle mode={colorMode} onChange={setColorMode} />
        <input
          type="text"
          placeholder="Search tensor..."
          value={search}
          onChange={e => handleSearch(e.target.value)}
          className="border rounded px-2 py-1 text-sm flex-1 max-w-xs"
        />
      </div>
      <div ref={containerRef} className="flex-1" />
      <NodeDetail node={selectedNode} onClose={() => setSelectedNode(null)} />
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd tools/trace-analyzer/web && npx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add tools/trace-analyzer/web/src/components/DagView.tsx
git commit -m "tools: add DAG view with Cytoscape.js layer folding"
```

### Task 15: TimelineView Component

**Files:**
- Create: `tools/trace-analyzer/web/src/components/TimelineView.tsx`

- [ ] **Step 1: Create TimelineView.tsx**

D3.js horizontal swimlane chart. Each pass = one row; ops = colored rectangles. Uses `timeline_ops` from summary.json which contains real `t_start`/`t_end` nanosecond timestamps for all passes.

```tsx
// src/components/TimelineView.tsx
import { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import type { SummaryData, TimelineOp } from '../types/trace';
import { backendColor } from '../utils/colorScale';
import { formatNs } from '../utils/dataSize';

interface Props {
  data: SummaryData;
  onSelectOp: (name: string) => void;
}

export function TimelineView({ data, onSelectOp }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || data.timeline_ops.length === 0) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 30, right: 20, bottom: 30, left: 80 };
    const width = svgRef.current.clientWidth - margin.left - margin.right;
    const passes = data.timing.per_pass;
    const rowHeight = 40;
    const height = passes.length * rowHeight;

    // Group ops by pass, compute per-pass time range
    const opsByPass = new Map<number, TimelineOp[]>();
    for (const op of data.timeline_ops) {
      if (!opsByPass.has(op.pass)) opsByPass.set(op.pass, []);
      opsByPass.get(op.pass)!.push(op);
    }

    // Global time range (relative: subtract each pass's min t_start)
    const globalMax = Math.max(...data.timeline_ops.map(o => o.t_end - o.t_start), 1);
    // Per-pass: make times relative to pass start
    const passMinT = new Map<number, number>();
    for (const [pid, ops] of opsByPass) {
      passMinT.set(pid, Math.min(...ops.map(o => o.t_start)));
    }
    const maxRelEnd = Math.max(...data.timeline_ops.map(o =>
      o.t_end - (passMinT.get(o.pass) ?? 0)
    ), 1);

    const x = d3.scaleLinear().domain([0, maxRelEnd]).range([0, width]);
    const y = d3.scaleBand()
      .domain(passes.map(p => `Pass ${p.pass}`))
      .range([0, height])
      .padding(0.2);

    const g = svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    g.append('g').call(d3.axisLeft(y));
    g.append('g').attr('transform', `translate(0,${height})`).call(d3.axisBottom(x).ticks(10));

    const barHeight = y.bandwidth();

    g.selectAll('.op-rect')
      .data(data.timeline_ops)
      .join('rect')
      .attr('class', 'op-rect')
      .attr('x', d => x(d.t_start - (passMinT.get(d.pass) ?? 0)))
      .attr('y', d => y(`Pass ${d.pass}`) ?? 0)
      .attr('width', d => Math.max(1, x(d.t_end - d.t_start) - x(0)))
      .attr('height', barHeight)
      .attr('fill', d => backendColor(d.backend))
      .attr('stroke', '#fff')
      .attr('stroke-width', 0.5)
      .style('cursor', 'pointer')
      .on('click', (_, d) => onSelectOp(d.name))
      .append('title')
      .text(d => `${d.name}\n${d.op} | ${formatNs(d.t_end - d.t_start)} | ${d.backend}`);

    // Zoom on X axis
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([1, 100])
      .on('zoom', (event) => {
        const newX = event.transform.rescaleX(x);
        g.selectAll<SVGRectElement, TimelineOp>('.op-rect')
          .attr('x', d => newX(d.t_start - (passMinT.get(d.pass) ?? 0)))
          .attr('width', d => Math.max(1, newX(d.t_end - d.t_start) - newX(0)));
        g.select<SVGGElement>('g:last-of-type').call(d3.axisBottom(newX).ticks(10) as any);
      });
    svg.call(zoom);
  }, [data, onSelectOp]);

  return <svg ref={svgRef} className="w-full h-full min-h-[300px]" />;
}
```

- [ ] **Step 2: Commit**

```bash
git add tools/trace-analyzer/web/src/components/TimelineView.tsx
git commit -m "tools: add Timeline swimlane view with D3"
```

### Task 16: CompareView Component

**Files:**
- Create: `tools/trace-analyzer/web/src/components/CompareView.tsx`

- [ ] **Step 1: Create CompareView.tsx**

Diff table with significance highlighting + D3 grouped bar chart.

```tsx
// src/components/CompareView.tsx
import { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import type { CompareData } from '../types/trace';
import { diffColor } from '../utils/colorScale';
import { formatNs } from '../utils/dataSize';

interface Props { data: CompareData; }

type SortKey = 'op' | 'diff';

export function CompareView({ data }: Props) {
  const chartRef = useRef<SVGSVGElement>(null);
  const [sortKey, setSortKey] = useState<SortKey>('diff');

  const sorted = [...data.op_diff].sort((a, b) =>
    sortKey === 'diff' ? Math.abs(b.diff_pct) - Math.abs(a.diff_pct) : a.op.localeCompare(b.op)
  );

  // Summary cards
  const td = data.timing_diff;
  const cards = [
    { label: 'Total', values: td.total_ms, diff: td.diff_pct.total },
    { label: 'Prefill', values: td.prefill_ms, diff: td.diff_pct.prefill },
    { label: 'Decode avg', values: td.decode_avg_ms, diff: td.diff_pct.decode },
  ];

  // Layer bar chart
  useEffect(() => {
    if (!chartRef.current) return;
    const svg = d3.select(chartRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 20, bottom: 60, left: 80 };
    const width = chartRef.current.clientWidth - margin.left - margin.right;
    const height = 250;

    const layers = data.layer_diff;
    const x0 = d3.scaleBand().domain(layers.map(l => l.layer)).range([0, width]).padding(0.3);
    const x1 = d3.scaleBand().domain(data.labels).range([0, x0.bandwidth()]).padding(0.05);
    const maxVal = d3.max(layers, l => d3.max(l.values, v => v.total_ns)) ?? 1;
    const y = d3.scaleLinear().domain([0, maxVal]).range([height, 0]);
    const color = d3.scaleOrdinal(data.labels, ['#3b82f6', '#f97316']);

    const g = svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    g.append('g').attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x0)).selectAll('text').attr('transform', 'rotate(-45)').style('text-anchor', 'end');
    g.append('g').call(d3.axisLeft(y).tickFormat(d => formatNs(+d)));

    for (const layer of layers) {
      const lg = g.append('g').attr('transform', `translate(${x0(layer.layer)},0)`);
      for (const val of layer.values) {
        lg.append('rect')
          .attr('x', x1(val.label)!)
          .attr('y', y(val.total_ns))
          .attr('width', x1.bandwidth())
          .attr('height', height - y(val.total_ns))
          .attr('fill', color(val.label))
          .attr('stroke', layer.significant ? '#000' : 'none')
          .attr('stroke-width', layer.significant ? 2 : 0);
      }
    }

    // Legend
    const legend = g.append('g').attr('transform', `translate(${width - 120}, 0)`);
    data.labels.forEach((label, i) => {
      legend.append('rect').attr('x', 0).attr('y', i * 20).attr('width', 12).attr('height', 12).attr('fill', color(label));
      legend.append('text').attr('x', 18).attr('y', i * 20 + 10).text(label).style('font-size', '12px');
    });
  }, [data]);

  return (
    <div className="p-4 space-y-6">
      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-4">
        {cards.map(c => (
          <div key={c.label} className="border rounded p-3">
            <div className="text-sm text-gray-500">{c.label}</div>
            <div className="flex justify-between mt-1">
              <span>{c.values[0].toFixed(1)}ms</span>
              <span>{c.values[1].toFixed(1)}ms</span>
            </div>
            <div className={`text-sm mt-1 ${Math.abs(c.diff) > 10 ? 'font-bold text-red-600' : 'text-gray-500'}`}>
              {c.diff > 0 ? '+' : ''}{c.diff.toFixed(1)}%
            </div>
          </div>
        ))}
      </div>

      {/* Op diff table */}
      <div>
        <div className="flex gap-2 mb-2">
          <button className={`text-sm px-2 py-1 rounded ${sortKey === 'diff' ? 'bg-gray-800 text-white' : 'bg-gray-200'}`} onClick={() => setSortKey('diff')}>Sort by diff</button>
          <button className={`text-sm px-2 py-1 rounded ${sortKey === 'op' ? 'bg-gray-800 text-white' : 'bg-gray-200'}`} onClick={() => setSortKey('op')}>Sort by name</button>
        </div>
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="border-b">
              <th className="text-left p-2">Op</th>
              <th className="text-right p-2">{data.labels[0]} (ms)</th>
              <th className="text-right p-2">{data.labels[1]} (ms)</th>
              <th className="text-right p-2">Diff</th>
              <th className="text-center p-2">Sig.</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(o => (
              <tr key={o.op} style={{ backgroundColor: diffColor(o.diff_pct) }}>
                <td className="p-2 font-mono">{o.op}</td>
                <td className="p-2 text-right">{(o.values[0].total_ns / 1e6).toFixed(1)}</td>
                <td className="p-2 text-right">{(o.values[1].total_ns / 1e6).toFixed(1)}</td>
                <td className="p-2 text-right">{o.diff_pct > 0 ? '+' : ''}{o.diff_pct.toFixed(1)}%</td>
                <td className="p-2 text-center">{o.significant ? 'Yes' : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Layer bar chart */}
      <div>
        <h3 className="font-bold mb-2">Layer Comparison</h3>
        <svg ref={chartRef} className="w-full" />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add tools/trace-analyzer/web/src/components/CompareView.tsx
git commit -m "tools: add Compare view with diff table and D3 bar chart"
```

### Task 17: App.tsx — Wire Everything Together

**Files:**
- Modify: `tools/trace-analyzer/web/src/App.tsx`

- [ ] **Step 1: Create App.tsx**

Tab layout with file selector, routing to DagView / TimelineView / CompareView + HotspotPanel.

```tsx
// src/App.tsx
import { useState } from 'react';
import { useFileList, useSummary, useCompare } from './hooks/useTraceData';
import { DagView } from './components/DagView';
import { TimelineView } from './components/TimelineView';
import { CompareView } from './components/CompareView';
import { HotspotPanel } from './components/HotspotPanel';

type View = 'dag' | 'timeline' | 'compare';

export default function App() {
  const files = useFileList();
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [compareFile, setCompareFile] = useState<string | null>(null);
  const [view, setView] = useState<View>('dag');
  const [highlightId, setHighlightId] = useState<string | null>(null);

  const summaryFiles = files.filter(f => f.name.includes('summary'));
  const compareFiles = files.filter(f => f.name.includes('compare'));

  const { data: summaryData } = useSummary(
    view !== 'compare' ? selectedFile : null
  );
  const { data: compareData } = useCompare(
    view === 'compare' ? compareFile : null
  );

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b px-4 py-2 flex items-center gap-4">
        <h1 className="font-bold text-lg">Trace Analyzer</h1>
        <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
          {(['dag', 'timeline', 'compare'] as View[]).map(v => (
            <button
              key={v}
              className={`px-3 py-1 rounded text-sm capitalize ${view === v ? 'bg-white shadow' : ''}`}
              onClick={() => setView(v)}
            >{v}</button>
          ))}
        </div>
        {view !== 'compare' ? (
          <select
            className="border rounded px-2 py-1 text-sm"
            value={selectedFile ?? ''}
            onChange={e => setSelectedFile(e.target.value || null)}
          >
            <option value="">Select summary...</option>
            {summaryFiles.map(f => <option key={f.name} value={f.name}>{f.name}</option>)}
          </select>
        ) : (
          <select
            className="border rounded px-2 py-1 text-sm"
            value={compareFile ?? ''}
            onChange={e => setCompareFile(e.target.value || null)}
          >
            <option value="">Select compare...</option>
            {compareFiles.map(f => <option key={f.name} value={f.name}>{f.name}</option>)}
          </select>
        )}
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        <div className="flex-1 overflow-auto">
          {view === 'dag' && summaryData && (
            <DagView data={summaryData} highlightId={highlightId} onSelectNode={setHighlightId} />
          )}
          {view === 'timeline' && summaryData && (
            <TimelineView data={summaryData} onSelectOp={setHighlightId} />
          )}
          {view === 'compare' && compareData && (
            <CompareView data={compareData} />
          )}
          {!summaryData && view !== 'compare' && (
            <div className="flex items-center justify-center h-full text-gray-400">Select a summary file to begin</div>
          )}
          {!compareData && view === 'compare' && (
            <div className="flex items-center justify-center h-full text-gray-400">Select a compare file to begin</div>
          )}
        </div>
        {summaryData && view !== 'compare' && (
          <HotspotPanel data={summaryData} selectedId={highlightId} onSelect={setHighlightId} />
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify build**

```bash
cd tools/trace-analyzer/web && npm run build
```
Expected: build succeeds, `dist/` directory created

- [ ] **Step 3: Commit**

```bash
git add tools/trace-analyzer/web/src/App.tsx
git commit -m "tools: wire App.tsx with tab navigation and all views"
```

### Task 18: End-to-End Integration Test

- [ ] **Step 1: Generate summary and compare JSON from fixture**

```bash
cd tools/trace-analyzer
python -m trace_analyzer.cli summary tests/fixtures/sample_trace.jsonl -o /tmp/trace-data/summary_test.json
python -m trace_analyzer.cli compare tests/fixtures/sample_trace.jsonl tests/fixtures/sample_trace.jsonl --labels "A,B" -o /tmp/trace-data/compare_test.json
```

- [ ] **Step 2: Build React SPA**

```bash
cd tools/trace-analyzer/web && npm run build
```

- [ ] **Step 3: Launch serve and verify in browser**

```bash
cd tools/trace-analyzer
python -m trace_analyzer.cli serve --data-dir /tmp/trace-data --port 8765
# Open http://localhost:8765 in browser
# Verify: file list loads, DAG renders, Timeline renders, Compare renders
```

- [ ] **Step 4: Create README**

```bash
# tools/trace-analyzer/README.md — brief usage instructions
```

Content: Install instructions, CLI usage examples, serve command, React dev workflow.

- [ ] **Step 5: Final commit**

```bash
git add tools/trace-analyzer/
git commit -m "tools: complete trace-analyzer with Python CLI + React SPA"
```
