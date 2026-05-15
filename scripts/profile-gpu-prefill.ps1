<#
.SYNOPSIS
    GPU utilization profiler for prefill benchmark (ollama vs llama runner).

.DESCRIPTION
    Runs nvidia-smi sampling concurrently with bench-sweep so we can see
    the GPU activity pattern (utilization / power / VRAM / PCIe traffic)
    across the prefill phase. Outputs per-sample CSV files and an
    optional matplotlib PNG.

    Requires: nvidia-smi on PATH, bench-sweep.exe on PATH, and Python
    (for plotting; falls back to "csv only" if matplotlib/pandas missing).

    YOU manage ollama serve manually -- this script only spawns the
    sampler and the bench-sweep client, and writes everything into a
    timestamped output directory.

.PARAMETER RunnerName
    Free-form label for this run (goes into output folder + plot title).
    Examples: "ollama-runner", "llama-runner", "ollama-with-flag-X".

.PARAMETER BatchSize
    bench-sweep --batch-size value. Defaults to 1024.

.PARAMETER Sizes
    bench-sweep --sizes value. Defaults to 1024 (single workload).

.PARAMETER Epochs
    bench-sweep --epochs value. Defaults to 6.

.PARAMETER Warmup
    bench-sweep --warmup value. Defaults to 4.

.PARAMETER MaxTokens
    bench-sweep --max-tokens value. Defaults to 16.

.PARAMETER Model
    bench-sweep --model value. Defaults to qwen3-coder-next.

.PARAMETER SampleIntervalMs
    nvidia-smi sampling interval in milliseconds. Defaults to 50ms.
    Note: nvidia-smi on Windows may throttle to ~50-100ms in practice.

.PARAMETER OutputRoot
    Directory under which a timestamped subfolder is created. Defaults
    to .\test\gpu-profile.

.PARAMETER GpuIndex
    Which GPU to sample (for multi-GPU systems). Defaults to 0. The
    RTX 3090 in the test rig is GPU 0.

.PARAMETER NoPlot
    Skip the Python plotting step (CSV only).

.EXAMPLE
    .\scripts\profile-gpu-prefill.ps1 -RunnerName ollama-runner

.EXAMPLE
    .\scripts\profile-gpu-prefill.ps1 -RunnerName llama-runner -SampleIntervalMs 100
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$RunnerName,

    [int]$BatchSize = 1024,
    [string]$Sizes = "1024",
    [int]$Epochs = 6,
    [int]$Warmup = 4,
    [int]$MaxTokens = 16,
    [string]$Model = "qwen3-coder-next",

    [int]$SampleIntervalMs = 50,
    [int]$GpuIndex = 0,

    [string]$OutputRoot = ".\test\gpu-profile",

    [switch]$NoPlot
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

function Require-Command {
    param([string]$Name, [string]$Hint)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        Write-Error "Missing required command: $Name. $Hint"
        exit 1
    }
    return $cmd.Source
}

$nvidiaSmi = Require-Command "nvidia-smi" "Install NVIDIA driver and ensure nvidia-smi.exe is on PATH."

# Hardcoded test-machine location for bench-sweep.exe. Override by setting
# $env:BENCH_SWEEP_EXE if you ever move it.
$benchSweep = $env:BENCH_SWEEP_EXE
if (-not $benchSweep) {
    $benchSweep = "C:\workspace\bench-sweep\bench-sweep.exe"
}
if (-not (Test-Path $benchSweep)) {
    Write-Error "bench-sweep.exe not found at $benchSweep. Set `$env:BENCH_SWEEP_EXE to override."
    exit 1
}

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runDir = Join-Path $OutputRoot "$RunnerName-$timestamp"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
Write-Host "Output directory: $runDir" -ForegroundColor Cyan

$queryCsv  = Join-Path $runDir "gpu-query.csv"      # nvidia-smi --query-gpu output
$dmonCsv   = Join-Path $runDir "gpu-dmon.txt"       # nvidia-smi dmon output (whitespace-formatted)
$benchOut  = Join-Path $runDir "bench-sweep.txt"
$metaJson  = Join-Path $runDir "meta.json"
$plotPng   = Join-Path $runDir "gpu-trace.png"

# ---------------------------------------------------------------------------
# Track the absolute start time so the Python plotter can align timelines.
# nvidia-smi --query-gpu emits its own timestamp column; we still record our
# wall-clock anchor for cross-correlation with the bench-sweep stdout times.
# ---------------------------------------------------------------------------

$tStartUtc = (Get-Date).ToUniversalTime()

# Save run metadata up-front (some fields are filled in later).
$meta = @{
    runner_name        = $RunnerName
    timestamp          = $timestamp
    t_start_utc        = $tStartUtc.ToString("o")
    bench_args         = "--model $Model --max-tokens $MaxTokens --sizes $Sizes --batch-size $BatchSize --epochs $Epochs --warmup $Warmup"
    gpu_index          = $GpuIndex
    sample_interval_ms = $SampleIntervalMs
}
$meta | ConvertTo-Json | Set-Content -Path $metaJson

# ---------------------------------------------------------------------------
# Start the samplers in background.
#
# Two parallel samplers:
#   1. nvidia-smi --query-gpu  -> CSV with utilization, power, memory,
#      clocks, etc. Includes its own UTC timestamp column for alignment.
#   2. nvidia-smi dmon         -> includes PCIe rx/tx (rxpci/txpci, MB/s)
#      which --query-gpu does not expose. We capture the raw whitespace
#      output and parse later.
#
# Both are started with Start-Process to a fresh process tree so they
# survive powershell errors; we register a finally block to kill them.
# ---------------------------------------------------------------------------

# --- Sampler 1: --query-gpu ------------------------------------------------
$queryFields = @(
    "timestamp",                # ISO with millisecond precision
    "index",
    "utilization.gpu",          # %
    "utilization.memory",       # %
    "memory.used",              # MiB
    "memory.free",              # MiB
    "power.draw",               # W
    "clocks.current.sm",        # MHz
    "clocks.current.memory",    # MHz
    "pstate"                    # P0..P15
) -join ","

$queryArgs = @(
    "--query-gpu=$queryFields",
    "--format=csv,nounits",
    "-lms", $SampleIntervalMs,
    "-i", $GpuIndex
)

# Header line goes into the file naturally; subsequent samples appended by nvidia-smi itself.
Write-Host "Starting nvidia-smi --query-gpu sampler -> $queryCsv" -ForegroundColor DarkGray
$queryProc = Start-Process -FilePath $nvidiaSmi `
    -ArgumentList $queryArgs `
    -RedirectStandardOutput $queryCsv `
    -PassThru `
    -WindowStyle Hidden

# --- Sampler 2: dmon -------------------------------------------------------
# dmon -s pucvmet flags:
#   p = power, u = utilization, c = clocks, v = volatile errors,
#   m = memory, e = ecc, t = throttle reasons.
# We mainly care about u (gpu/mem util) and m (mem usage). dmon doesn't
# directly print PCIe TX/RX in modern drivers, but pcie tx/rx columns are
# available with `-s u` on Hopper+. For Ampere we'll rely on memory.util
# and memory bandwidth from --query-gpu instead.
#
# Using dmon as a backup signal: its sample interval flag is integer seconds.
# We pass -d 1 (one second) so we get a low-rate but consistent log
# alongside the high-rate --query-gpu data.
$dmonArgs = @(
    "dmon",
    "-s", "pucvm",
    "-d", "1",
    "-i", $GpuIndex
)

Write-Host "Starting nvidia-smi dmon sampler -> $dmonCsv" -ForegroundColor DarkGray
$dmonProc = Start-Process -FilePath $nvidiaSmi `
    -ArgumentList $dmonArgs `
    -RedirectStandardOutput $dmonCsv `
    -PassThru `
    -WindowStyle Hidden

# Give samplers a moment to print their headers and stabilize.
Start-Sleep -Milliseconds 800

# ---------------------------------------------------------------------------
# Run bench-sweep in foreground so this script blocks until it finishes.
# We wrap it so that any exit code or exception still triggers the cleanup
# block at the end (samplers must be killed in all cases).
# ---------------------------------------------------------------------------

try {
    $benchArgs = @(
        "run",
        "--model", $Model,
        "--max-tokens", $MaxTokens,
        "--sizes", $Sizes,
        "--batch-size", $BatchSize,
        "--epochs", $Epochs,
        "--warmup", $Warmup,
        "--name", "$RunnerName-gpu-profile-$timestamp"
    )

    $benchStartUtc = (Get-Date).ToUniversalTime()
    "BENCH_START_UTC=$($benchStartUtc.ToString('o'))" | Set-Content -Path $benchOut

    Write-Host "Running bench-sweep..." -ForegroundColor Cyan
    Write-Host "  $benchSweep $($benchArgs -join ' ')" -ForegroundColor DarkGray

    # Use Start-Process so we can capture both stdout/stderr and reuse the
    # same wall clock that nvidia-smi sees. Append, since we already wrote
    # the BENCH_START_UTC line above.
    $benchOutTmp = "$benchOut.stdout"
    $benchErrTmp = "$benchOut.stderr"
    $benchProc = Start-Process -FilePath $benchSweep `
        -ArgumentList $benchArgs `
        -RedirectStandardOutput $benchOutTmp `
        -RedirectStandardError  $benchErrTmp `
        -PassThru `
        -WindowStyle Hidden -Wait

    $benchEndUtc = (Get-Date).ToUniversalTime()

    # Concatenate stdout+stderr into one file with markers for easier diffing.
    Add-Content -Path $benchOut -Value "BENCH_END_UTC=$($benchEndUtc.ToString('o'))"
    Add-Content -Path $benchOut -Value "BENCH_EXIT_CODE=$($benchProc.ExitCode)"
    Add-Content -Path $benchOut -Value "----- STDOUT -----"
    Add-Content -Path $benchOut -Value (Get-Content -Path $benchOutTmp -Raw)
    Add-Content -Path $benchOut -Value "----- STDERR -----"
    Add-Content -Path $benchOut -Value (Get-Content -Path $benchErrTmp -Raw)
    Remove-Item $benchOutTmp,$benchErrTmp -ErrorAction SilentlyContinue

    Write-Host "bench-sweep finished, exit=$($benchProc.ExitCode)" -ForegroundColor Cyan
}
finally {
    # Always shut down samplers, even if bench-sweep crashed.
    # We give them ~500ms to capture the GPU returning to idle so the tail
    # of the trace is meaningful. nvidia-smi flushes its CSV output line
    # by line, so killing it after this delay preserves all already-written
    # samples.
    Start-Sleep -Milliseconds 500
    foreach ($p in @($queryProc, $dmonProc)) {
        if ($p -and -not $p.HasExited) {
            try {
                # CloseMainWindow first (graceful) - usually a no-op for
                # console processes spawned hidden, but harmless.
                $null = $p.CloseMainWindow()
                Start-Sleep -Milliseconds 100
                if (-not $p.HasExited) {
                    Stop-Process -Id $p.Id -Force -ErrorAction Stop
                }
            } catch { }
        }
    }
}

# ---------------------------------------------------------------------------
# Update meta with bench timing for the plotter.
# ---------------------------------------------------------------------------

$meta.bench_start_utc = $benchStartUtc.ToString("o")
$meta.bench_end_utc   = $benchEndUtc.ToString("o")
$meta.bench_exit_code = $benchProc.ExitCode
$meta | ConvertTo-Json | Set-Content -Path $metaJson

Write-Host ""
Write-Host "Files written:" -ForegroundColor Green
Write-Host "  $queryCsv"
Write-Host "  $dmonCsv"
Write-Host "  $benchOut"
Write-Host "  $metaJson"

# ---------------------------------------------------------------------------
# Optionally plot.
# ---------------------------------------------------------------------------

if (-not $NoPlot) {
    $plotter = Join-Path $PSScriptRoot "profile-gpu-prefill-plot.py"
    if (-not (Test-Path $plotter)) {
        Write-Warning "Plot script not found at $plotter - skipping plot."
    } else {
        $py = $null
        foreach ($candidate in @("py","python3","python")) {
            $resolved = Get-Command $candidate -ErrorAction SilentlyContinue
            if ($resolved) { $py = $resolved.Source; break }
        }
        if (-not $py) {
            Write-Warning "Python not found on PATH - skipping plot. CSV files are still saved."
        } else {
            Write-Host "Plotting to $plotPng ..." -ForegroundColor Cyan
            & $py $plotter --run-dir $runDir --output $plotPng
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Plot script returned $LASTEXITCODE. Inspect CSVs manually."
            } else {
                Write-Host "Plot saved: $plotPng" -ForegroundColor Green
            }
        }
    }
}

Write-Host ""
Write-Host "Done. Run dir: $runDir" -ForegroundColor Green
