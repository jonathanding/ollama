from pathlib import Path
import polars as pl
from trace_analyzer.parser import parse_trace
from trace_analyzer.dag import build_dag

FIXTURE = Path(__file__).parent / "fixtures" / "sample_trace.jsonl"

def test_build_dag_returns_nodes_and_edges():
    ops, _, _ = parse_trace(FIXTURE)
    dag = build_dag(ops, pass_id=1)
    assert "nodes" in dag
    assert "edges" in dag

def test_node_fields():
    ops, _, _ = parse_trace(FIXTURE)
    dag = build_dag(ops, pass_id=1)
    node = dag["nodes"][0]
    for key in ("id", "op", "backend", "ns", "shape", "dtype", "layer", "is_copy"):
        assert key in node, f"missing key: {key}"

def test_node_count():
    ops, _, _ = parse_trace(FIXTURE)
    dag = build_dag(ops, pass_id=1)
    assert len(dag["nodes"]) == 5

def test_edge_count():
    ops, _, _ = parse_trace(FIXTURE)
    dag = build_dag(ops, pass_id=1)
    # Only edges between compute nodes (weight/leaf tensors filtered out)
    # token_embd->attn_norm, attn_norm->attn_q, attn_norm->ffn_gate, ffn_gate->ffn_out = 4
    assert len(dag["edges"]) == 4

def test_layer_grouping():
    ops, _, _ = parse_trace(FIXTURE)
    dag = build_dag(ops, pass_id=1)
    layers = {n["layer"] for n in dag["nodes"]}
    assert "blk.0" in layers
    assert None in layers or "" in layers

def test_copy_detection():
    ops, _, _ = parse_trace(FIXTURE)
    dag = build_dag(ops, pass_id=1)
    copies = [n for n in dag["nodes"] if n["is_copy"]]
    assert len(copies) == 1
    assert copies[0]["id"] == "blk.0.ffn_out"

def test_edge_est_bytes():
    ops, _, _ = parse_trace(FIXTURE)
    dag = build_dag(ops, pass_id=1)
    edge = [e for e in dag["edges"] if e["to"] == "blk.0.attn_q" and e["from"] == "blk.0.attn_norm"][0]
    # source blk.0.attn_norm: shape=[128,1], dtype=f32 -> 128*1*4 = 512
    assert edge["est_bytes"] == 512
