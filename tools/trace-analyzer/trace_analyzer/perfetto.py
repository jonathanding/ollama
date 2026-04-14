"""Convert parsed trace data to Chrome Trace Event Format (Perfetto-compatible)."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def build_perfetto(
    ops: pl.DataFrame,
    passes: pl.DataFrame,
    meta: dict | None = None,
) -> dict:
    """Build a Chrome Trace Event JSON dict from parsed trace data.

    The output is compatible with chrome://tracing and Perfetto UI.
    - pid=1: all ops in one process
    - tid: one lane per backend (CPU, CUDA0, Vulkan, etc.)
    - tid=0: pass boundary markers
    """
    events: list[dict] = []
    backend_tids: dict[str, int] = {}
    tid_counter = 1  # 0 reserved for pass markers

    def _get_tid(backend: str) -> int:
        if backend not in backend_tids:
            nonlocal tid_counter
            backend_tids[backend] = tid_counter
            tid_counter += 1
        return backend_tids[backend]

    # Process metadata
    model_name = (meta or {}).get("model", "ollama")
    events.append({
        "ph": "M", "pid": 1, "tid": 0,
        "name": "process_name", "args": {"name": model_name},
    })
    events.append({
        "ph": "M", "pid": 1, "tid": 0,
        "name": "thread_name", "args": {"name": "Passes"},
    })

    # First pass: collect backend tids from ops so thread metadata is stable
    if len(ops) > 0:
        for backend in ops["backend"].unique().sort().to_list():
            if backend:
                _get_tid(backend)

    # Thread metadata for each backend
    for backend, tid in sorted(backend_tids.items(), key=lambda x: x[1]):
        events.append({
            "ph": "M", "pid": 1, "tid": tid,
            "name": "thread_name", "args": {"name": backend},
        })

    # Find the earliest op timestamp to use as time base (ns)
    time_base_ns = 0
    if len(ops) > 0:
        time_base_ns = ops["t_start"].min()

    # Pass boundary events (begin/end) on tid=0
    for row in passes.iter_rows(named=True):
        pid = row["pass"]
        n_tokens = row.get("n_tokens")
        label = f"Pass {pid}"
        if n_tokens is not None:
            label += f" ({'prefill' if n_tokens > 1 else 'decode'}, {n_tokens} tok)"

        # Use the first/last op timestamps in this pass for accurate timing
        pass_ops = ops.filter(pl.col("pass") == pid) if len(ops) > 0 else pl.DataFrame()
        if len(pass_ops) > 0:
            ts_start_ns = pass_ops["t_start"].min()
            ts_end_ns = pass_ops["t_end"].max()
        else:
            continue

        events.append({
            "ph": "B", "pid": 1, "tid": 0,
            "name": label, "ts": (ts_start_ns - time_base_ns) / 1000,
        })
        events.append({
            "ph": "E", "pid": 1, "tid": 0,
            "name": label, "ts": (ts_end_ns - time_base_ns) / 1000,
        })

    # Op events — complete ("X") events
    if len(ops) > 0:
        for row in ops.iter_rows(named=True):
            backend = row.get("backend", "")
            tid = _get_tid(backend) if backend else 1
            t_start = row["t_start"]
            t_end = row["t_end"]
            dur_ns = t_end - t_start

            args: dict = {"tensor": row["name"]}
            if row.get("dtype"):
                args["dtype"] = row["dtype"]
            shape = row.get("shape")
            if shape:
                args["shape"] = shape
            args["pass"] = row["pass"]
            args["seq"] = row["seq"]

            events.append({
                "ph": "X", "pid": 1,
                "tid": tid,
                "name": row["op"],
                "cat": backend,
                "ts": (t_start - time_base_ns) / 1000,  # ns -> us
                "dur": dur_ns / 1000,
                "args": args,
            })

    result: dict = {"traceEvents": events, "displayTimeUnit": "ms"}
    if meta:
        result["metadata"] = {
            k: v for k, v in meta.items() if k != "type"
        }

    return result


def export_perfetto(
    ops: pl.DataFrame,
    passes: pl.DataFrame,
    output: Path,
    meta: dict | None = None,
) -> None:
    """Export trace data to a Perfetto-compatible JSON file."""
    data = build_perfetto(ops, passes, meta=meta)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(data, f)
