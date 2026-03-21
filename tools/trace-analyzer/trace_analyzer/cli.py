# trace_analyzer/cli.py
from __future__ import annotations
import json
import click
from pathlib import Path


@click.group()
def main():
    """Ollama Trace Analyzer — post-process and visualize inference traces."""
    pass


@main.command()
@click.argument("trace_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None)
@click.option("--model", default=None, help="Model name (optional)")
@click.option("--pass", "dag_pass", type=int, default=None, help="Pass ID for DAG (default: first decode)")
def summary(trace_file: Path, output: Path | None, model: str | None, dag_pass: int | None):
    """Generate summary.json from a single JSONL trace."""
    from .parser import parse_trace
    from .summary import build_summary

    click.echo(f"Parsing {trace_file.name}...")
    ops, passes = parse_trace(trace_file)
    click.echo(f"  {len(ops)} ops across {len(passes)} passes")

    result = build_summary(ops, passes, source_file=trace_file.name, model=model, dag_pass=dag_pass)

    n_layers = len(result["layer_stats"])
    top_op = result["op_stats"][0]["op"] if result["op_stats"] else "N/A"
    click.echo(f"  {n_layers} layers, top op: {top_op}, wall time: {result['timing']['total_ms']:.1f}ms")

    text = json.dumps(result, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        click.echo(click.style(f"  -> {output} ({len(text)} bytes)", fg="green"))
    else:
        click.echo(text)


@main.command()
@click.argument("trace_a", type=click.Path(exists=True, path_type=Path))
@click.argument("trace_b", type=click.Path(exists=True, path_type=Path))
@click.option("--labels", required=True, help="Comma-separated labels (e.g. 'CUDA,Vulkan')")
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None)
@click.option("--model", default=None)
@click.option("--threshold", type=float, default=10.0, help="Significance threshold %")
def compare(trace_a: Path, trace_b: Path, labels: str, output: Path | None, model: str | None, threshold: float):
    """Compare two JSONL traces."""
    from .parser import parse_trace
    from .summary import build_summary
    from .compare import build_compare

    label_list = [l.strip() for l in labels.split(",")]
    if len(label_list) != 2:
        raise click.BadParameter("Exactly 2 labels required", param_hint="--labels")

    click.echo(f"Parsing {trace_a.name} ({label_list[0]})...")
    ops_a, passes_a = parse_trace(trace_a)
    click.echo(f"  {len(ops_a)} ops across {len(passes_a)} passes")

    click.echo(f"Parsing {trace_b.name} ({label_list[1]})...")
    ops_b, passes_b = parse_trace(trace_b)
    click.echo(f"  {len(ops_b)} ops across {len(passes_b)} passes")

    sa = build_summary(ops_a, passes_a, source_file=trace_a.name, model=model)
    sb = build_summary(ops_b, passes_b, source_file=trace_b.name, model=model)
    result = build_compare(sa, sb, labels=label_list, threshold=threshold)

    sig_ops = sum(1 for o in result["op_diff"] if o["significant"])
    click.echo(f"Compared: {len(result['op_diff'])} ops, {sig_ops} significant (>{threshold}%)")

    text = json.dumps(result, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        click.echo(click.style(f"  -> {output} ({len(text)} bytes)", fg="green"))
    else:
        click.echo(text)


@main.command()
@click.argument("trace_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None)
@click.option("--model", default=None)
@click.option("--compare", "trace_b", type=click.Path(exists=True, path_type=Path), default=None, help="Second trace for comparison report")
@click.option("--labels", default=None)
def report(trace_file: Path, output: Path | None, model: str | None, trace_b: Path | None, labels: str | None):
    """Generate LLM-ready Markdown report."""
    from .parser import parse_trace
    from .summary import build_summary
    from .report import render_single, render_compare

    if trace_b:
        from .compare import build_compare
        label_list = [l.strip() for l in labels.split(",")] if labels else ["A", "B"]
        click.echo(f"Generating comparison report: {trace_file.name} vs {trace_b.name}...")
        ops_a, passes_a = parse_trace(trace_file)
        ops_b, passes_b = parse_trace(trace_b)
        sa = build_summary(ops_a, passes_a, source_file=trace_file.name, model=model)
        sb = build_summary(ops_b, passes_b, source_file=trace_b.name, model=model)
        cmp = build_compare(sa, sb, labels=label_list)
        md = render_compare(cmp)
    else:
        click.echo(f"Generating report for {trace_file.name}...")
        ops, passes = parse_trace(trace_file)
        s = build_summary(ops, passes, source_file=trace_file.name, model=model)
        md = render_single(s)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(md)
        click.echo(click.style(f"  -> {output} ({len(md)} bytes)", fg="green"))
    else:
        click.echo(md)


@main.command()
@click.option("--data-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--port", type=int, default=8765)
def serve(data_dir: Path, port: int):
    """Launch dev server for React frontend + JSON data."""
    from .serve import run_server
    web_dir = Path(__file__).parent.parent / "web" / "dist"
    run_server(data_dir, port, web_dir if web_dir.is_dir() else None)


if __name__ == "__main__":
    main()
