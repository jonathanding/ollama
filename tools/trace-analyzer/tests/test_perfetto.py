from pathlib import Path
from trace_analyzer.parser import parse_trace
from trace_analyzer.perfetto import build_perfetto, export_perfetto

FIXTURE = Path(__file__).parent / "fixtures" / "sample_trace.jsonl"


def test_build_perfetto_has_trace_events():
    ops, passes, _ = parse_trace(FIXTURE)
    result = build_perfetto(ops, passes)
    assert "traceEvents" in result
    assert len(result["traceEvents"]) > 0


def test_op_events_are_complete_type():
    ops, passes, _ = parse_trace(FIXTURE)
    result = build_perfetto(ops, passes)
    op_events = [e for e in result["traceEvents"] if e.get("ph") == "X"]
    assert len(op_events) == 10  # same as ops count in fixture
    for ev in op_events:
        assert "ts" in ev
        assert "dur" in ev
        assert ev["dur"] > 0
        assert "name" in ev
        assert "cat" in ev  # backend
        assert "args" in ev


def test_pass_boundaries_present():
    ops, passes, _ = parse_trace(FIXTURE)
    result = build_perfetto(ops, passes)
    begins = [e for e in result["traceEvents"] if e.get("ph") == "B"]
    ends = [e for e in result["traceEvents"] if e.get("ph") == "E"]
    assert len(begins) == 2  # 2 passes in fixture
    assert len(ends) == 2


def test_backend_thread_separation():
    ops, passes, _ = parse_trace(FIXTURE)
    result = build_perfetto(ops, passes)
    op_events = [e for e in result["traceEvents"] if e.get("ph") == "X"]
    tids = {e["tid"] for e in op_events}
    # fixture has CUDA0 and CPU backends -> 2 different tids
    assert len(tids) == 2


def test_thread_metadata_present():
    ops, passes, _ = parse_trace(FIXTURE)
    result = build_perfetto(ops, passes)
    meta_events = [e for e in result["traceEvents"]
                   if e.get("ph") == "M" and e.get("name") == "thread_name"]
    names = {e["args"]["name"] for e in meta_events}
    assert "CUDA0" in names
    assert "CPU" in names
    assert "Passes" in names


def test_timestamps_are_microseconds():
    ops, passes, _ = parse_trace(FIXTURE)
    result = build_perfetto(ops, passes)
    op_events = [e for e in result["traceEvents"] if e.get("ph") == "X"]
    first = op_events[0]
    # fixture t_start=100000000 ns, time_base=100000000 -> ts=0 us
    assert first["ts"] == 0.0
    # fixture first op dur = 50000 ns = 50 us
    assert first["dur"] == 50.0


def test_meta_passed_through():
    ops, passes, _ = parse_trace(FIXTURE)
    meta = {"type": "meta", "model": "qwen3:0.6b", "request_id": "abc"}
    result = build_perfetto(ops, passes, meta=meta)
    assert result["metadata"]["model"] == "qwen3:0.6b"
    assert result["metadata"]["request_id"] == "abc"
    assert "type" not in result["metadata"]


def test_export_creates_file(tmp_path):
    ops, passes, _ = parse_trace(FIXTURE)
    out = tmp_path / "trace.perfetto.json"
    export_perfetto(ops, passes, out)
    assert out.exists()
    assert out.stat().st_size > 0

    import json
    data = json.loads(out.read_text())
    assert "traceEvents" in data


def test_model_name_in_process_metadata():
    ops, passes, _ = parse_trace(FIXTURE)
    meta = {"type": "meta", "model": "llama3:8b"}
    result = build_perfetto(ops, passes, meta=meta)
    proc_meta = [e for e in result["traceEvents"]
                 if e.get("ph") == "M" and e.get("name") == "process_name"]
    assert len(proc_meta) == 1
    assert proc_meta[0]["args"]["name"] == "llama3:8b"
