# tests/test_report.py
from pathlib import Path
from trace_analyzer.parser import parse_trace
from trace_analyzer.summary import build_summary
from trace_analyzer.compare import build_compare
from trace_analyzer.report import render_single, render_compare

FIXTURE = Path(__file__).parent / "fixtures" / "sample_trace.jsonl"

def test_single_report_has_sections():
    ops, passes, _ = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl")
    md = render_single(s)
    assert "## Top Operators" in md
    assert "MUL_MAT" in md
    assert "## Backend Distribution" in md

def test_single_report_with_model():
    ops, passes, _ = parse_trace(FIXTURE)
    s = build_summary(ops, passes, source_file="test.jsonl", model="llama3:8b")
    md = render_single(s)
    assert "llama3:8b" in md

def test_compare_report_has_labels():
    ops, passes, _ = parse_trace(FIXTURE)
    s1 = build_summary(ops, passes, source_file="a.jsonl")
    s2 = build_summary(ops, passes, source_file="b.jsonl")
    c = build_compare(s1, s2, labels=["X", "Y"])
    md = render_compare(c)
    assert "X" in md and "Y" in md
    assert "## Op Comparison" in md
