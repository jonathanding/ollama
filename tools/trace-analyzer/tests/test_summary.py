from pathlib import Path
from trace_analyzer.parser import parse_trace
from trace_analyzer.summary import build_summary

FIXTURE = Path(__file__).parent / "fixtures" / "sample_trace.jsonl"

def test_summary_has_all_sections():
    ops, passes, _ = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    for key in ("meta", "timing", "op_stats", "backend_stats", "copy_stats", "layer_stats", "dag"):
        assert key in s, f"missing section: {key}"

def test_meta_fields():
    ops, passes, _ = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    assert s["meta"]["source_file"] == "test.jsonl"
    assert s["meta"]["total_ops"] == 10
    assert s["meta"]["total_passes"] == 2

def test_timing_prefill_decode():
    ops, passes, _ = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    t = s["timing"]
    assert t["prefill_tokens"] == 4
    assert len(t["per_pass"]) == 2

def test_op_stats_sorted_by_time():
    ops, passes, _ = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    times = [o["total_ns"] for o in s["op_stats"]]
    assert times == sorted(times, reverse=True)

def test_op_stats_pct_sums_to_100():
    ops, passes, _ = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    total_pct = sum(o["pct_time"] for o in s["op_stats"])
    assert abs(total_pct - 100.0) < 0.1

def test_backend_stats():
    ops, passes, _ = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    backends = {b["backend"] for b in s["backend_stats"]}
    assert "CUDA0" in backends
    assert "CPU" in backends

def test_copy_stats():
    ops, passes, _ = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    cs = s["copy_stats"]
    assert cs["count"] == 2
    assert len(cs["copies"]) == 2

def test_layer_stats():
    ops, passes, _ = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    layers = {l["layer"] for l in s["layer_stats"]}
    assert "blk.0" in layers

def test_dag_present():
    ops, passes, _ = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    assert len(s["dag"]["nodes"]) > 0
    assert len(s["dag"]["edges"]) > 0

def test_model_optional():
    ops, passes, _ = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl", model="llama3:8b")
    assert s["meta"]["model"] == "llama3:8b"
    s2 = build_summary(ops, passes, source_file="test.jsonl")
    assert s2["meta"]["model"] is None
