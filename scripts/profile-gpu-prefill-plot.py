"""Plot GPU utilization trace produced by profile-gpu-prefill.ps1.

Inputs (in --run-dir):
    gpu-query.csv      — nvidia-smi --query-gpu output (high-rate)
    gpu-dmon.txt       — nvidia-smi dmon output (low-rate, optional)
    bench-sweep.txt    — bench-sweep stdout/stderr with BENCH_START_UTC and BENCH_END_UTC
    meta.json          — run metadata

Output:
    PNG with two stacked plots:
      - GPU utilization (%) and memory-controller utilization (%) over time
      - GPU power (W) and SM clock (MHz) over time
    The bench-sweep wall-clock window is shaded to mark the prefill region.

Falls back to a "csv only" notice if matplotlib / pandas are not available.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_iso_utc(s: str) -> datetime:
    # PowerShell's "o" round-trip format is ISO 8601 with offset.
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def parse_query_csv(path: Path) -> tuple[list[datetime], dict[str, list[float]]]:
    """Parse nvidia-smi --query-gpu CSV output.

    Sample first lines (with --format=csv,nounits):
        timestamp, index, utilization.gpu [%], utilization.memory [%], ...
        2026/05/15 12:34:56.789, 0, 0, 0, ...
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return [], {}

    # Header is the first non-empty line.
    header = None
    data_lines = []
    for ln in lines:
        if not ln.strip():
            continue
        if header is None:
            header = [c.strip() for c in ln.split(",")]
        else:
            data_lines.append(ln)

    if header is None:
        return [], {}

    # Strip "[unit]" suffixes from header names so the keys match what we want.
    norm = [re.sub(r"\s*\[.*\]\s*$", "", c) for c in header]

    times: list[datetime] = []
    cols: dict[str, list[float]] = {k: [] for k in norm if k != "timestamp" and k != "index"}

    for ln in data_lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(norm):
            continue
        # nvidia-smi timestamp format: "yyyy/MM/dd HH:mm:ss.fff"
        try:
            ts = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f")
        except ValueError:
            try:
                ts = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S")
            except ValueError:
                continue
        # nvidia-smi prints in local time; the bench script wrote UTC. Treat
        # nvidia-smi's stamp as local and convert. We can't 100% trust the
        # local TZ here (depends on driver), but it's good enough for relative
        # alignment within a few seconds. We'll align via meta.json's UTC
        # range below.
        ts = ts.astimezone() if ts.tzinfo else ts.replace(tzinfo=datetime.now().astimezone().tzinfo)
        ts_utc = ts.astimezone(timezone.utc)
        times.append(ts_utc)

        for k, v in zip(norm[1:], parts[1:]):
            if k == "index":
                continue
            try:
                cols[k].append(float(v))
            except ValueError:
                # nvidia-smi prints "[Not Supported]" sometimes.
                cols[k].append(float("nan"))

    return times, cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    query_csv = run_dir / "gpu-query.csv"
    meta_json = run_dir / "meta.json"

    if not query_csv.exists():
        print(f"[plot] gpu-query.csv missing — nothing to plot.", file=sys.stderr)
        return 1

    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.dates as mdates  # type: ignore
    except ImportError:
        print(
            "[plot] matplotlib not installed. CSV files are still saved at "
            f"{run_dir}. To install: py -m pip install matplotlib",
            file=sys.stderr,
        )
        return 0  # not an error — just skip

    meta: dict = {}
    if meta_json.exists():
        try:
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    times, cols = parse_query_csv(query_csv)
    if not times:
        print("[plot] no samples parsed from gpu-query.csv.", file=sys.stderr)
        return 1

    # Convert to seconds since the first sample.
    t0 = times[0]
    secs = [(t - t0).total_seconds() for t in times]

    # Bench window relative to t0.
    bench_start = bench_end = None
    if "bench_start_utc" in meta and "bench_end_utc" in meta:
        try:
            bs = parse_iso_utc(meta["bench_start_utc"])
            be = parse_iso_utc(meta["bench_end_utc"])
            bench_start = (bs - t0).total_seconds()
            bench_end = (be - t0).total_seconds()
        except Exception:
            pass

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # --- Top: utilization ---
    ax = axes[0]
    if "utilization.gpu" in cols:
        ax.plot(secs, cols["utilization.gpu"], label="GPU util (%)", linewidth=1.0, color="tab:blue")
    if "utilization.memory" in cols:
        ax.plot(secs, cols["utilization.memory"], label="Memory ctrl util (%)", linewidth=1.0, alpha=0.7, color="tab:orange")
    ax.set_ylabel("Utilization (%)")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # --- Bottom: power and SM clock ---
    ax2 = axes[1]
    if "power.draw" in cols:
        ax2.plot(secs, cols["power.draw"], label="Power (W)", linewidth=1.0, color="tab:red")
    ax2.set_ylabel("Power (W)")
    ax2.set_xlabel("Time since first sample (s)")
    ax2.grid(True, alpha=0.3)

    if "clocks.current.sm" in cols:
        ax3 = ax2.twinx()
        ax3.plot(secs, cols["clocks.current.sm"], label="SM clock (MHz)", linewidth=0.8, alpha=0.7, color="tab:green")
        ax3.set_ylabel("Clock (MHz)")
        # Combine legends.
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax3.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, loc="upper right")
    else:
        ax2.legend(loc="upper right")

    # Shade the bench window so prefill is obvious.
    if bench_start is not None and bench_end is not None and bench_end > bench_start:
        for ax_ in axes:
            ax_.axvspan(bench_start, bench_end, alpha=0.10, color="gray", label="bench-sweep window")

    title_bits = []
    if "runner_name" in meta:
        title_bits.append(meta["runner_name"])
    if "bench_args" in meta:
        title_bits.append(meta["bench_args"])
    fig.suptitle(" — ".join(title_bits) if title_bits else "GPU profile", fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.output, dpi=130)
    print(f"[plot] saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
