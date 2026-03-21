#
# Ollama with Tracing - Release Packaging Script
#
# Assembles a release from pre-built artifacts. Does NOT build anything.
# All components must be built beforehand.
#
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\package-ollama-with-tracing.ps1
#
# Output:
#   release\ollama-with-tracing-windows-x64-<VERSION>\
#

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path "$PSScriptRoot\..").Path
$Version = Get-Date -Format "yyyyMMdd-HHmmss"
$ReleaseDir = Join-Path $Root "release\ollama-with-tracing-windows-x64-$Version"

Write-Host "=== Ollama with Tracing - Release ===" -ForegroundColor Cyan
Write-Host "Version: $Version"
Write-Host ""

# ============================================================
# Validate all required artifacts exist
# ============================================================

$errors = @()

# --- ollama.exe ---
# Prefer dist\windows-amd64\ollama.exe (official build), fall back to repo root
$OllamaExe = $null
$distExe = Join-Path $Root "dist\windows-amd64\ollama.exe"
$rootExe = Join-Path $Root "ollama.exe"
if (Test-Path $distExe) { $OllamaExe = $distExe }
elseif (Test-Path $rootExe) { $OllamaExe = $rootExe }
else { $errors += "ollama.exe not found (checked dist\windows-amd64\ and repo root). Run: go build ." }

# --- Native libraries (DLLs) ---
# Prefer dist\windows-amd64\lib\ollama\ (official build), fall back to build\lib\ollama\
$LibDir = $null
$distLib = Join-Path $Root "dist\windows-amd64\lib\ollama"
$buildLib = Join-Path $Root "build\lib\ollama"
if (Test-Path $distLib) { $LibDir = $distLib }
elseif (Test-Path $buildLib) { $LibDir = $buildLib }
else { $errors += "Native libraries not found (checked dist\windows-amd64\lib\ollama\ and build\lib\ollama\). Run: cmake --build build --config Release" }

if ($LibDir) {
    $dlls = Get-ChildItem $LibDir -Filter "*.dll" -ErrorAction SilentlyContinue
    if ($dlls.Count -eq 0) { $errors += "No .dll files found in $LibDir" }
}

# --- Web frontend dist ---
$WebDist = Join-Path $Root "tools\trace-analyzer\web\dist"
if (-not (Test-Path (Join-Path $WebDist "index.html"))) {
    $errors += "Web frontend not built (tools\trace-analyzer\web\dist\index.html missing). Run: npm --prefix tools\trace-analyzer\web run build"
}

# --- Python package ---
$TaSrc = Join-Path $Root "tools\trace-analyzer"
if (-not (Test-Path (Join-Path $TaSrc "trace_analyzer\cli.py"))) {
    $errors += "trace_analyzer package not found at tools\trace-analyzer\trace_analyzer\"
}
if (-not (Test-Path (Join-Path $TaSrc "pyproject.toml"))) {
    $errors += "pyproject.toml not found at tools\trace-analyzer\"
}

# --- README ---
$ReadmeSrc = Join-Path $TaSrc "RELEASE-README.md"
if (-not (Test-Path $ReadmeSrc)) {
    $errors += "RELEASE-README.md not found at tools\trace-analyzer\"
}

# --- Report errors and exit ---
if ($errors.Count -gt 0) {
    Write-Host "ERROR: Missing required artifacts:" -ForegroundColor Red
    foreach ($e in $errors) {
        Write-Host ("  - " + $e) -ForegroundColor Red
    }
    exit 1
}

# ============================================================
# Assemble release
# ============================================================

Write-Host ("Output:  " + $ReleaseDir)
Write-Host ""

if (Test-Path $ReleaseDir) { Remove-Item $ReleaseDir -Recurse -Force }
New-Item -ItemType Directory -Path $ReleaseDir -Force | Out-Null

# --- 1. ollama.exe ---
Write-Host ">>> ollama.exe" -ForegroundColor Yellow
Copy-Item $OllamaExe (Join-Path $ReleaseDir "ollama.exe")
Write-Host ("    from " + $OllamaExe) -ForegroundColor Green

# --- 2. lib\ollama\*.dll ---
Write-Host ">>> lib\ollama\" -ForegroundColor Yellow
$LibDst = Join-Path $ReleaseDir "lib\ollama"
New-Item -ItemType Directory -Path $LibDst -Force | Out-Null
$dlls = Get-ChildItem $LibDir -Filter "*.dll"
foreach ($dll in $dlls) {
    Copy-Item $dll.FullName $LibDst
}
Write-Host ("    " + $dlls.Count + " DLLs from " + $LibDir) -ForegroundColor Green

# --- 3. tools\trace-analyzer\ ---
Write-Host ">>> tools\trace-analyzer\" -ForegroundColor Yellow
$TaDst = Join-Path $ReleaseDir "tools\trace-analyzer"
New-Item -ItemType Directory -Path $TaDst -Force | Out-Null

# Python package (exclude __pycache__)
Copy-Item (Join-Path $TaSrc "trace_analyzer") (Join-Path $TaDst "trace_analyzer") -Recurse
Get-ChildItem (Join-Path $TaDst "trace_analyzer") -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Pre-built web dist
$DstWebDir = Join-Path $TaDst "web"
New-Item -ItemType Directory -Path $DstWebDir -Force | Out-Null
Copy-Item $WebDist (Join-Path $DstWebDir "dist") -Recurse

# pyproject.toml
Copy-Item (Join-Path $TaSrc "pyproject.toml") $TaDst
Write-Host "    trace_analyzer + web/dist + pyproject.toml" -ForegroundColor Green

# --- 4. README + VERSION ---
Copy-Item $ReadmeSrc (Join-Path $ReleaseDir "README.md")
$Version | Out-File (Join-Path $ReleaseDir "VERSION") -Encoding utf8 -NoNewline

# ============================================================
# Summary
# ============================================================

Write-Host ""
Write-Host "=== Release assembled ===" -ForegroundColor Green
Write-Host ""
Get-ChildItem $ReleaseDir -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Substring($ReleaseDir.Length + 1)
    $sz = $_.Length
    if ($sz -gt 1MB) { $disp = "{0:N1} MB" -f ($sz / 1MB) }
    elseif ($sz -gt 1KB) { $disp = "{0:N1} KB" -f ($sz / 1KB) }
    else { $disp = "$sz B" }
    Write-Host ("  {0,-55} {1}" -f $rel, $disp)
}
Write-Host ""
Write-Host "To create a zip:" -ForegroundColor Cyan
Write-Host ("  powershell Compress-Archive -Path '" + $ReleaseDir + "' -DestinationPath '" + $ReleaseDir + ".zip'")
