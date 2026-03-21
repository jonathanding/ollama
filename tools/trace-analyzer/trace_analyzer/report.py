# trace_analyzer/report.py
from __future__ import annotations
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        keep_trailing_newline=True,
    )


def render_single(summary: dict) -> str:
    tmpl = _env().get_template("single_report.md.j2")
    return tmpl.render(**summary)


def render_compare(compare_data: dict) -> str:
    tmpl = _env().get_template("compare_report.md.j2")
    return tmpl.render(**compare_data)
