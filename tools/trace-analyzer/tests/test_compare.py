# tests/test_compare.py
from pathlib import Path
from trace_analyzer.parser import parse_trace
from trace_analyzer.summary import build_summary
from trace_analyzer.compare import build_compare

FIXTURE = Path(__file__).parent / "fixtures" / "sample_trace.jsonl"

def _make_two_summaries():
    ops, passes, _ = parse_trace(FIXTURE)
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
