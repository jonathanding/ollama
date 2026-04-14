from pathlib import Path
import polars as pl
from trace_analyzer.parser import parse_trace

FIXTURE = Path(__file__).parent / "fixtures" / "sample_trace.jsonl"

def test_parse_returns_ops_and_passes():
    ops, passes, _ = parse_trace(FIXTURE)
    assert isinstance(ops, pl.DataFrame)
    assert isinstance(passes, pl.DataFrame)

def test_ops_columns():
    ops, _, _ = parse_trace(FIXTURE)
    expected = {"pass", "seq", "op", "name", "srcs", "shape", "dtype", "backend", "t_start", "t_end"}
    assert expected.issubset(set(ops.columns))

def test_ops_count():
    ops, _, _ = parse_trace(FIXTURE)
    assert len(ops) == 10

def test_passes_columns():
    _, passes, _ = parse_trace(FIXTURE)
    expected = {"pass", "n_tokens", "ts_start", "ts_end", "n_nodes"}
    assert expected.issubset(set(passes.columns))

def test_passes_count():
    _, passes, _ = parse_trace(FIXTURE)
    assert len(passes) == 2

def test_pass_wall_ms():
    _, passes, _ = parse_trace(FIXTURE)
    row = passes.filter(pl.col("pass") == 0).row(0, named=True)
    assert row["wall_ms"] == row["ts_end"] - row["ts_start"]

def test_malformed_line_skipped(tmp_path):
    f = tmp_path / "bad.jsonl"
    f.write_text('not json\n{"type":"pass_start","pass":0,"n_tokens":1,"ts":100}\n{"type":"pass_end","pass":0,"n_nodes":0,"ts":200}\n')
    ops, passes, _ = parse_trace(f)
    assert len(passes) == 1
    assert len(ops) == 0

def test_truncated_trace(tmp_path):
    f = tmp_path / "truncated.jsonl"
    f.write_text('{"type":"pass_start","pass":0,"n_tokens":1,"ts":100}\n{"type":"op","pass":0,"seq":0,"op":"ADD","name":"x","srcs":["a"],"shape":[1],"dtype":"f32","backend":"CPU","t_start":1000,"t_end":2000}\n')
    ops, passes, _ = parse_trace(f)
    assert len(ops) == 1
    assert len(passes) == 1
    assert passes.row(0, named=True)["ts_end"] is None
