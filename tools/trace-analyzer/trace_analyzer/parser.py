from __future__ import annotations
import json
import logging
from pathlib import Path
import polars as pl

logger = logging.getLogger(__name__)

def parse_trace(path: Path | str) -> tuple[pl.DataFrame, pl.DataFrame, dict | None]:
    """Parse a JSONL trace file.

    Returns (ops_df, passes_df, meta) where meta is the header dict or None.
    """
    path = Path(path)
    ops: list[dict] = []
    pass_starts: dict[int, dict] = {}
    pass_rows: list[dict] = []
    meta: dict | None = None

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
            if t == "meta":
                meta = ev
                continue
            elif t == "pass_start":
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

    for pid, start in pass_starts.items():
        pass_rows.append({
            "pass": pid, "n_tokens": start.get("n_tokens"),
            "ts_start": start.get("ts_start"), "ts_end": None,
            "n_nodes": 0, "wall_ms": None,
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

    return ops_df, passes_df, meta
