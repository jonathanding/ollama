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


def _detect_layers_by_add_pairs(ops: list[str]) -> list[str]:
    """Detect transformer layers by finding pairs of ADD ops.

    In a typical transformer, each layer ends with two ADD ops
    (attn residual + FFN residual). We detect these pairs and
    assign layer numbers based on the repeating pattern.
    """
    # Find ADD positions
    add_positions = [i for i, op in enumerate(ops) if op == "ADD"]
    if len(add_positions) < 4:
        return ["_top"] * len(ops)

    # Check if ADDs come in pairs with consistent spacing
    # Pair them: (add_positions[0], add_positions[1]), (add_positions[2], add_positions[3]), ...
    gaps = []
    for i in range(0, len(add_positions) - 1, 2):
        if i + 2 < len(add_positions):
            gaps.append(add_positions[i + 2] - add_positions[i])

    if not gaps:
        return ["_top"] * len(ops)

    # Check consistency: all gaps should be roughly equal
    median_gap = sorted(gaps)[len(gaps) // 2]
    consistent = all(abs(g - median_gap) <= 2 for g in gaps)
    if not consistent:
        return ["_top"] * len(ops)

    # Assign layers: everything before first ADD pair = _pre, then blk.0, blk.1, etc.
    layers = ["_top"] * len(ops)

    # Find preamble end: ops before the first block pattern starts
    # The first block typically starts a few ops before the first ADD pair
    # We use the gap size to estimate where block 0 starts
    block_size = median_gap
    first_block_end = add_positions[1]  # end of first block = second ADD of first pair
    first_block_start = max(0, first_block_end - block_size + 1)

    # Assign preamble
    for i in range(first_block_start):
        layers[i] = "_pre"

    # Assign blocks
    n_pairs = len(add_positions) // 2
    for block_idx in range(n_pairs):
        pair_start = block_idx * 2
        if pair_start + 1 >= len(add_positions):
            break
        block_end = add_positions[pair_start + 1]

        if block_idx == 0:
            block_start = first_block_start
        else:
            prev_block_end = add_positions[pair_start - 1]
            block_start = prev_block_end + 1

        layer_name = f"blk.{block_idx}"
        for i in range(block_start, min(block_end + 1, len(ops))):
            layers[i] = layer_name

    # Anything after the last block = _post
    last_block_end = add_positions[n_pairs * 2 - 1] if n_pairs > 0 else 0
    for i in range(last_block_end + 1, len(ops)):
        layers[i] = "_post"

    return layers


def assign_layers(ops_df: pl.DataFrame) -> pl.DataFrame:
    """Add a 'layer' column to ops DataFrame.

    First tries blk.N. prefix extraction. If that yields no layers,
    falls back to automatic detection via ADD-pair pattern matching.
    """
    # Try blk.N. prefix first
    result = ops_df.with_columns(
        pl.col("name").str.extract(r"^(blk\.\d+)\.", 1).alias("_blk_layer")
    )
    n_matched = result.filter(pl.col("_blk_layer").is_not_null()).height

    if n_matched > 0:
        return result.with_columns(
            pl.col("_blk_layer").fill_null("_top").alias("layer")
        ).drop("_blk_layer")

    # Fallback: detect by ADD pairs, per-pass
    all_layers = []
    for pass_id in ops_df["pass"].unique().sort().to_list():
        pass_ops = ops_df.filter(pl.col("pass") == pass_id).sort("seq")
        op_list = pass_ops["op"].to_list()
        layers = _detect_layers_by_add_pairs(op_list)
        all_layers.extend(layers)

    # Build in original order
    ordered = ops_df.sort(["pass", "seq"])
    ordered = ordered.with_columns(pl.Series("layer", all_layers))
    # Join back by pass+seq to preserve original row order
    return ops_df.join(
        ordered.select(["pass", "seq", "layer"]),
        on=["pass", "seq"],
        how="left",
    )


def build_dag(ops_df: pl.DataFrame, pass_id: int) -> dict:
    filtered = ops_df.filter(pl.col("pass") == pass_id).sort("seq")
    rows = filtered.to_dicts()

    node_info: dict[str, dict] = {}
    for row in rows:
        node_info[row["name"]] = row

    nodes = []
    edges = []

    # Use pre-assigned layer if available, otherwise extract from name
    has_layer_col = "layer" in filtered.columns

    for row in rows:
        ns = row["t_end"] - row["t_start"]
        if has_layer_col:
            layer = row["layer"]
            # Convert internal markers to display names; _top stays ungrouped
            if layer == "_top":
                layer = None
            elif layer == "_pre":
                layer = "pre"
            elif layer == "_post":
                layer = "post"
        else:
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
        for src_name in (row.get("srcs") or []):
            src = node_info.get(src_name)
            if not src:
                continue
            est = _estimate_bytes(src["shape"], src["dtype"])
            edges.append({
                "from": src_name,
                "to": row["name"],
                "est_bytes": est,
            })

    return {"nodes": nodes, "edges": edges}
