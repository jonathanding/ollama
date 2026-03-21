# trace_analyzer/serve.py
from __future__ import annotations
import json
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import click


# Ensure common web MIME types are registered
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")


def _detect_json_type(path: Path) -> str | None:
    """Peek at top-level keys to classify a JSON file as 'summary' or 'compare'."""
    try:
        with open(path, encoding="utf-8") as f:
            # Read just enough to parse top-level keys
            obj = json.load(f)
        if not isinstance(obj, dict):
            return None
        if "labels" in obj and "op_diff" in obj:
            return "compare"
        if "dag" in obj or "op_stats" in obj:
            return "summary"
        return None
    except Exception:
        return None


class TraceHandler(SimpleHTTPRequestHandler):
    data_dir: Path
    web_dir: Path | None

    def do_GET(self):
        if self.path == "/api/files":
            self._list_files()
        elif self.path.startswith("/data/"):
            self._serve_data_file()
        elif self.web_dir:
            self._serve_static()
        else:
            self.send_error(404)

    def _list_files(self):
        files = []
        for f in sorted(self.data_dir.glob("*.json")):
            ftype = _detect_json_type(f)
            if ftype:
                files.append({"name": f.name, "size": f.stat().st_size, "type": ftype})
        self._json_response(json.dumps(files))

    def _serve_data_file(self):
        name = self.path[len("/data/"):]
        fpath = self.data_dir / name
        if not fpath.is_file() or not fpath.resolve().is_relative_to(self.data_dir.resolve()):
            self.send_error(404)
            return
        self._json_response(fpath.read_text(encoding="utf-8"))

    def _serve_static(self):
        """Serve files from web_dir; fall back to index.html for SPA routing."""
        assert self.web_dir is not None
        # Strip query string and leading /
        url_path = self.path.split("?")[0].lstrip("/")
        fpath = self.web_dir / url_path if url_path else self.web_dir / "index.html"

        # SPA fallback: if the path doesn't match a file, serve index.html
        if not fpath.is_file():
            fpath = self.web_dir / "index.html"

        if not fpath.is_file():
            self.send_error(404)
            return

        content_type, _ = mimetypes.guess_type(str(fpath))
        body = fpath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_response(self, text: str):
        body = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Suppress noisy Chrome DevTools requests
        if ".well-known/" in str(args[0] if args else ""):
            return
        super().log_message(format, *args)


def find_web_dist() -> Path | None:
    """Locate the bundled web/dist directory relative to this package."""
    # Try: package_dir/../web/dist (development & release layout)
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir.parent / "web" / "dist",       # tools/trace-analyzer/web/dist
        pkg_dir / "web_dist",                   # trace_analyzer/web_dist (future alt)
    ]
    for c in candidates:
        if c.is_dir() and (c / "index.html").is_file():
            return c
    return None


def run_server(data_dir: Path, port: int = 8765, web_dir: Path | None = None):
    if web_dir is None:
        web_dir = find_web_dist()
    TraceHandler.data_dir = data_dir
    TraceHandler.web_dir = web_dir
    server = HTTPServer(("0.0.0.0", port), TraceHandler)
    if web_dir:
        click.echo(f"Trace Analyzer server running at http://localhost:{port}")
    else:
        click.echo(f"API-only server at http://localhost:{port} (no web UI found)")
    click.echo(f"Data directory: {data_dir.resolve()}")
    server.serve_forever()
