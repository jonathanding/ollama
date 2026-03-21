#
# Ollama Trace Analyzer — Release Packaging Script
#
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File tools\trace-analyzer\build-release.ps1
#
# Output:
#   release\ollama-trace-<VERSION>\
#     ├── ollama.exe
#     ├── tools\trace-analyzer\
#     │   ├── trace_analyzer\       (Python CLI)
#     │   ├── web\dist\             (pre-built React SPA)
#     │   ├── pyproject.toml
#     │   ├── README.md
#     │   └── tests\
#     └── VERSION
#
# Then just zip release\ollama-trace-<VERSION>\ and upload.
#

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
$Version = Get-Date -Format "yyyyMMdd-HHmmss"
$ReleaseDir = Join-Path $Root "release\ollama-trace-$Version"

Write-Host "=== Ollama Trace Analyzer Release ===" -ForegroundColor Cyan
Write-Host "Version: $Version"
Write-Host "Output:  $ReleaseDir"
Write-Host ""

# --- Clean ---
if (Test-Path $ReleaseDir) { Remove-Item $ReleaseDir -Recurse -Force }
New-Item -ItemType Directory -Path $ReleaseDir -Force | Out-Null

# --- 1. Build Ollama binary ---
Write-Host ">>> Building Ollama..." -ForegroundColor Yellow
Push-Location $Root
try {
    $GoExe = & go env GOEXE
    go build -o (Join-Path $ReleaseDir "ollama$GoExe") .
    if ($LASTEXITCODE -ne 0) { throw "go build failed" }
    Write-Host "    Binary: ollama$GoExe" -ForegroundColor Green
} finally {
    Pop-Location
}

# --- 2. Build web frontend ---
Write-Host ">>> Building web frontend..." -ForegroundColor Yellow
$WebDir = Join-Path $Root "tools\trace-analyzer\web"

if (-not (Test-Path (Join-Path $WebDir "node_modules"))) {
    Write-Host "    Installing npm dependencies..."
    npm --prefix $WebDir install
    if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
}

npm --prefix $WebDir run build
if ($LASTEXITCODE -ne 0) { throw "npm run build failed" }
Write-Host "    Dist built" -ForegroundColor Green

# --- 3. Assemble trace-analyzer ---
Write-Host ">>> Assembling trace-analyzer..." -ForegroundColor Yellow
$TaSrc = Join-Path $Root "tools\trace-analyzer"
$TaDst = Join-Path $ReleaseDir "tools\trace-analyzer"
New-Item -ItemType Directory -Path $TaDst -Force | Out-Null

# Python package (exclude __pycache__)
Copy-Item (Join-Path $TaSrc "trace_analyzer") (Join-Path $TaDst "trace_analyzer") -Recurse
Get-ChildItem (Join-Path $TaDst "trace_analyzer") -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

# Pre-built web dist (no source, no node_modules)
$DstWebDir = Join-Path $TaDst "web"
New-Item -ItemType Directory -Path $DstWebDir -Force | Out-Null
Copy-Item (Join-Path $WebDir "dist") (Join-Path $DstWebDir "dist") -Recurse

# Project files
Copy-Item (Join-Path $TaSrc "pyproject.toml") $TaDst
Copy-Item (Join-Path $TaSrc "README.md") $TaDst

# Tests
$TestsSrc = Join-Path $TaSrc "tests"
if (Test-Path $TestsSrc) {
    Copy-Item $TestsSrc (Join-Path $TaDst "tests") -Recurse
    Get-ChildItem (Join-Path $TaDst "tests") -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

# --- 4. Version file ---
$Version | Out-File (Join-Path $ReleaseDir "VERSION") -Encoding utf8 -NoNewline

# --- Summary ---
Write-Host ""
Write-Host "=== Release built successfully ===" -ForegroundColor Green
Write-Host "Directory: $ReleaseDir"
Write-Host ""
Write-Host "Contents:"
Get-ChildItem $ReleaseDir -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Substring($ReleaseDir.Length + 1)
    $size = if ($_.Length -gt 1MB) { "{0:N1} MB" -f ($_.Length / 1MB) }
            elseif ($_.Length -gt 1KB) { "{0:N1} KB" -f ($_.Length / 1KB) }
            else { "$($_.Length) B" }
    Write-Host ("  {0,-55} {1}" -f $rel, $size)
}
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. pip install -e $TaDst"
Write-Host "  2. Compress-Archive -Path '$ReleaseDir' -DestinationPath '$ReleaseDir.zip'"
Write-Host "  3. Upload the zip"
