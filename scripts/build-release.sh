#!/usr/bin/env bash
#
# Build a release package for Ollama + Trace Analyzer.
#
# Usage:
#   ./scripts/build-release.sh
#
# Output:
#   release/ollama-<VERSION>/
#     ├── ollama.exe (or ollama)         # main binary
#     ├── tools/trace-analyzer/
#     │   ├── trace_analyzer/            # Python CLI
#     │   ├── web/dist/                  # pre-built React SPA
#     │   ├── pyproject.toml
#     │   ├── README.md
#     │   └── tests/
#     └── VERSION
#
# After running, just zip the release/ollama-<VERSION>/ directory.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="$(date +%Y%m%d-%H%M%S)"
RELEASE_DIR="$ROOT/release/ollama-$VERSION"

echo "=== Ollama Release Build ==="
echo "Version: $VERSION"
echo "Output:  $RELEASE_DIR"
echo ""

# ─── Clean ───
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"

# ─── 1. Build Ollama binary ───
echo ">>> Building Ollama..."
(cd "$ROOT" && go build -o "$RELEASE_DIR/ollama$(go env GOEXE)" .)
echo "    Binary: $RELEASE_DIR/ollama$(go env GOEXE)"

# ─── 2. Build web frontend ───
echo ">>> Building web frontend..."
WEB_DIR="$ROOT/tools/trace-analyzer/web"

# Install npm deps if needed
if [ ! -d "$WEB_DIR/node_modules" ]; then
    echo "    Installing npm dependencies..."
    npm --prefix "$WEB_DIR" install
fi

npm --prefix "$WEB_DIR" run build
echo "    Dist built at $WEB_DIR/dist/"

# ─── 3. Assemble trace-analyzer ───
echo ">>> Assembling trace-analyzer..."
TA_SRC="$ROOT/tools/trace-analyzer"
TA_DST="$RELEASE_DIR/tools/trace-analyzer"

mkdir -p "$TA_DST"

# Python package
cp -r "$TA_SRC/trace_analyzer" "$TA_DST/trace_analyzer"
# Remove __pycache__
find "$TA_DST/trace_analyzer" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Pre-built web dist
mkdir -p "$TA_DST/web"
cp -r "$WEB_DIR/dist" "$TA_DST/web/dist"

# Project files
cp "$TA_SRC/pyproject.toml" "$TA_DST/"
cp "$TA_SRC/README.md" "$TA_DST/"

# Tests (optional, for verification)
if [ -d "$TA_SRC/tests" ]; then
    cp -r "$TA_SRC/tests" "$TA_DST/tests"
    find "$TA_DST/tests" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
fi

# ─── 4. Version file ───
echo "$VERSION" > "$RELEASE_DIR/VERSION"

# ─── Summary ───
echo ""
echo "=== Release built successfully ==="
echo "Directory: $RELEASE_DIR"
echo ""
echo "Contents:"
find "$RELEASE_DIR" -maxdepth 3 -type f | sort | while read -r f; do
    size=$(wc -c < "$f" | tr -d ' ')
    rel="${f#$RELEASE_DIR/}"
    printf "  %-60s %s\n" "$rel" "$(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B")"
done
echo ""
echo "Next steps:"
echo "  1. cd $RELEASE_DIR && pip install -e tools/trace-analyzer/"
echo "  2. Zip the directory: zip -r ollama-$VERSION.zip ollama-$VERSION/"
echo "  3. Upload the zip"
