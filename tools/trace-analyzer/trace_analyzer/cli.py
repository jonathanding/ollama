# trace_analyzer/cli.py
from __future__ import annotations
import json
import click
from pathlib import Path


@click.group()
def main():
    """Trace Analyzer — post-process ollama inference traces."""
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

    ops, passes = parse_trace(trace_file)
    result = build_summary(ops, passes, source_file=trace_file.name, model=model, dag_pass=dag_pass)

    text = json.dumps(result, indent=2)
    if output:
        output.write_text(text)
        click.echo(f"Written to {output}")
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

    ops_a, passes_a = parse_trace(trace_a)
    ops_b, passes_b = parse_trace(trace_b)
    sa = build_summary(ops_a, passes_a, source_file=trace_a.name, model=model)
    sb = build_summary(ops_b, passes_b, source_file=trace_b.name, model=model)
    result = build_compare(sa, sb, labels=label_list, threshold=threshold)

    text = json.dumps(result, indent=2)
    if output:
        output.write_text(text)
        click.echo(f"Written to {output}")
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
        ops_a, passes_a = parse_trace(trace_file)
        ops_b, passes_b = parse_trace(trace_b)
        sa = build_summary(ops_a, passes_a, source_file=trace_file.name, model=model)
        sb = build_summary(ops_b, passes_b, source_file=trace_b.name, model=model)
        cmp = build_compare(sa, sb, labels=label_list)
        md = render_compare(cmp)
    else:
        ops, passes = parse_trace(trace_file)
        s = build_summary(ops, passes, source_file=trace_file.name, model=model)
        md = render_single(s)

    if output:
        output.write_text(md)
        click.echo(f"Written to {output}")
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
