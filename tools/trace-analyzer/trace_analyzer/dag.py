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
    size = DTYPE_SIZE.get(dtype, 2)
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
            src = node_info.get(src_name)
            if not src:
                continue  # skip weight/leaf tensors not in compute graph
            est = _estimate_bytes(src["shape"], src["dtype"])
            edges.append({
                "from": src_name,
                "to": row["name"],
                "est_bytes": est,
            })

    return {"nodes": nodes, "edges": edges}
