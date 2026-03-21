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
