# trace_analyzer/serve.py
from __future__ import annotations
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import click


class TraceHandler(SimpleHTTPRequestHandler):
    data_dir: Path
    web_dir: Path | None

    def do_GET(self):
        if self.path == "/api/files":
            self._list_files()
        elif self.path.startswith("/data/"):
            self._serve_data_file()
        elif self.web_dir:
            # Serve React build
            self.directory = str(self.web_dir)
            super().do_GET()
        else:
            self.send_error(404)

    def _list_files(self):
        files = []
        for f in sorted(self.data_dir.glob("*.json")):
            files.append({"name": f.name, "size": f.stat().st_size})
        body = json.dumps(files).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_data_file(self):
        name = self.path[len("/data/"):]
        fpath = self.data_dir / name
        if not fpath.is_file() or not fpath.resolve().is_relative_to(self.data_dir.resolve()):
            self.send_error(404)
            return
        body = fpath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(data_dir: Path, port: int = 8765, web_dir: Path | None = None):
    TraceHandler.data_dir = data_dir
    TraceHandler.web_dir = web_dir
    server = HTTPServer(("0.0.0.0", port), TraceHandler)
    click.echo(f"Serving on http://localhost:{port}  (data: {data_dir})")
    server.serve_forever()
