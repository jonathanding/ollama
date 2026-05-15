# GPU prefill profiler

Quick how-to for comparing GPU activity between ollama runner and llama runner
during a `bench-sweep` prefill workload.

## Files

- `profile-gpu-prefill.ps1` — main driver. Spawns two `nvidia-smi` samplers
  in the background, runs `bench-sweep`, and produces a CSV + PNG.
- `profile-gpu-prefill-plot.py` — Python plotter (matplotlib). Called
  automatically by the .ps1 unless `-NoPlot` is passed.

## Prerequisites on the test machine

- `nvidia-smi.exe` on PATH (comes with the NVIDIA driver)
- `bench-sweep.exe` at `C:\workspace\bench-sweep\bench-sweep.exe`
  (hardcoded in the script). To use a different location:
  ```powershell
  $env:BENCH_SWEEP_EXE = "D:\elsewhere\bench-sweep.exe"
  ```
- Python 3 + `matplotlib` (only needed if you want the PNG)
  ```powershell
  py -m pip install matplotlib
  ```

## Usage

You manage `ollama serve` yourself. The script never touches the server
process.

### 1. ollama runner mode (default)

```powershell
# Terminal A — ollama serve
$env:OLLAMA_PREFILL_PROFILE = "1"   # optional, for cross-correlation
ollama serve

# Terminal B — sampling + bench
cd C:\path\to\ollama
.\scripts\profile-gpu-prefill.ps1 -RunnerName ollama-runner
```

### 2. llama runner mode

```powershell
# Terminal A — restart ollama serve with the force flag you've been using
ollama serve

# Terminal B
.\scripts\profile-gpu-prefill.ps1 -RunnerName llama-runner
```

### 3. Compare

Each run produces a folder under `.\test\gpu-profile\<RunnerName>-<ts>\`:

```
gpu-query.csv      ← high-rate (50ms) nvidia-smi --query-gpu samples
gpu-dmon.txt       ← 1Hz nvidia-smi dmon output (backup signal)
bench-sweep.txt    ← bench-sweep stdout/stderr + start/end UTC markers
meta.json          ← run metadata (args, timestamps)
gpu-trace.png      ← matplotlib plot (if Python+matplotlib available)
```

Open the two `gpu-trace.png` side by side. Things to look for:

- **GPU utilization shape** — ollama tends to show a "sawtooth" pattern
  (short GPU bursts separated by gaps where the CPU layers run); llama
  tends to show a flatter, near-saturated curve.
- **Time axis duration** — if the bench window for ollama is ~2s and
  llama is ~1.5s on the same workload, that's the prefill gap visible
  on the timeline.
- **Power draw** — sustained high power draw correlates with sustained
  GPU work.
- **Memory ctrl utilization** — high memory util with low GPU util
  suggests PCIe/DMA-bound workload (e.g. weight streaming via shared
  GPU memory).

## Common knobs

```powershell
# different sample rate (50ms is the default)
.\scripts\profile-gpu-prefill.ps1 -RunnerName ollama -SampleIntervalMs 100

# different workload size
.\scripts\profile-gpu-prefill.ps1 -RunnerName ollama -BatchSize 512 -Sizes 4096

# CSV only, skip plotting
.\scripts\profile-gpu-prefill.ps1 -RunnerName ollama -NoPlot

# multi-GPU machine — pick GPU index
.\scripts\profile-gpu-prefill.ps1 -RunnerName ollama -GpuIndex 1
```

## Notes on the sampler

- `nvidia-smi --query-gpu -lms 50` is what we use for the high-rate
  trace. On Windows the driver may throttle this to ~50–100 ms in
  practice; this is fine for a 1.5–2 s prefill (still 30+ samples).
- `nvidia-smi dmon -d 1` runs at 1 Hz and is our backup low-rate
  sanity-check. PCIe TX/RX columns are not exposed on Ampere
  consumer cards, so we mostly read it for power and utilization.
- Both samplers are killed in a `finally` block. If you Ctrl+C the
  PowerShell mid-run, that cleanup still runs.

## Caveats

- Sampler startup overhead: the script sleeps ~800 ms before launching
  bench-sweep so the samplers have time to print their headers. The
  plotter shades the actual bench window (from `BENCH_START_UTC`
  to `BENCH_END_UTC`) so you can ignore the head/tail noise.
- Because Windows time precision is coarse and `nvidia-smi`'s timestamp
  is local time while our markers are UTC, expect ±100 ms alignment
  jitter between the bench window and the sampled trace. This is fine
  for visual comparison.
- `bench-sweep` runs `--epochs N` + `--warmup M` rounds, so the bench
  window contains ~N+M prefill phases plus the decode phases between
  them. The first 1–2 prefills are warmup and may include model-load
  spikes if the model is just loaded; treat the steady-state region
  (epochs after warmup) as the comparison.
