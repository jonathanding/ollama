from __future__ import annotations
from datetime import datetime, timezone
import polars as pl
from .dag import build_dag, assign_layers, COPY_OPS, _estimate_bytes


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
                est = sum(_estimate_bytes(r["shape"], r["dtype"]) for r in copy_ops.to_dicts())
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
        copies_raw = []
        for row in copy_df.to_dicts():
            shape = row["shape"]
            dtype = row["dtype"]
            est = _estimate_bytes(shape, dtype)
            label = row["name"].strip()
            if not label or label == "(copy)":
                label = f"{dtype} {'x'.join(str(s) for s in shape)}"
            copies_raw.append({
                "name": label, "op": row["op"],
                "shape": shape, "dtype": dtype,
                "est_bytes": est, "ns": int(row["ns"]),
                "backend": row["backend"],
            })
        # Aggregate by (name, shape_key, dtype, backend) to deduplicate per-pass copies
        agg: dict[tuple, dict] = {}
        for c in copies_raw:
            key = (c["name"], str(c["shape"]), c["dtype"], c["backend"])
            if key not in agg:
                agg[key] = {
                    "name": c["name"], "op": c["op"],
                    "shape": c["shape"], "dtype": c["dtype"],
                    "est_bytes": c["est_bytes"],
                    "total_ns": 0, "count": 0,
                    "backend": c["backend"],
                }
            agg[key]["total_ns"] += c["ns"]
            agg[key]["count"] += 1
        copies = sorted(agg.values(), key=lambda x: x["total_ns"], reverse=True)
        copy_stats = {
            "count": len(copies_raw),
            "total_ns": int(copy_df["ns"].sum()) if len(copy_df) > 0 else 0,
            "est_total_bytes": sum(c["est_bytes"] for c in copies_raw),
            "copies": copies,
        }
    else:
        copy_stats = {"count": 0, "total_ns": 0, "est_total_bytes": 0, "copies": []}

    # Layer stats
    if len(ops) > 0:
        ops_layered = assign_layers(ops_with_ns)
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

    # DAG
    if dag_pass is not None:
        dp = dag_pass
    elif len(decode_passes) > 0:
        dp = int(decode_passes["pass"].min())
    elif len(passes) > 0:
        dp = int(passes["pass"].min())
    else:
        dp = 0
    ops_with_layers = assign_layers(ops) if len(ops) > 0 else ops
    dag = build_dag(ops_with_layers, dp)

    # Extract timestamp from first pass
    timestamp = None
    if len(passes) > 0:
        ts_start_col = passes.sort("pass")["ts_start"].drop_nulls()
        if len(ts_start_col) > 0:
            ts_ms = int(ts_start_col[0])
            timestamp = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()

    return {
        "meta": {
            "source_file": source_file, "model": model,
            "timestamp": timestamp,
            "total_ops": len(ops), "total_passes": len(passes),
            "total_wall_ms": total_ms,
        },
        "timing": {
            "total_ms": total_ms, "prefill_ms": prefill_ms,
            "prefill_tokens": prefill_tokens, "decode_avg_ms": decode_avg_ms,
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
    if len(ops) == 0:
        return []
    return ops.select(["pass", "seq", "name", "op", "backend", "t_start", "t_end"]).to_dicts()
