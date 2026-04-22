"""Generate the Stage 1-5 prefill-ms trend chart used in
docs/perf/2026-04-22-moe-split-prefill-full-experiment-report.md.

Run: python docs/perf/assets/2026-04-22-prefill-trend.py
Output: docs/perf/assets/2026-04-22-prefill-trend.png
"""
from pathlib import Path

import matplotlib.pyplot as plt

STAGES = [
    ("Baseline",       2096.3),
    ("MoE split",      2060.4),
    ("Pinned",         1391.7),
    ("DSB full",       1647.6),
    ("DSB selective",  1249.3),
]

xs = list(range(1, len(STAGES) + 1))
labels = [name for name, _ in STAGES]
ys = [ms for _, ms in STAGES]

baseline = ys[0]
final    = ys[-1]

fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=140)

ax.plot(xs, ys, marker="o", color="#2f6fed", linewidth=2.2,
        markersize=8, markerfacecolor="white", markeredgewidth=2.2,
        markeredgecolor="#2f6fed", zorder=3)

LABEL_OFFSETS = {
    # name: (dx, dy, va)
    "Baseline":      (0,  14, "bottom"),
    "MoE split":     (0,  14, "bottom"),
    "Pinned":        (0, -14, "top"),
    "DSB full":      (0,  14, "bottom"),
    "DSB selective": (10,  14, "bottom"),
}
for x, (name, ms) in zip(xs, STAGES):
    dx, dy, va = LABEL_OFFSETS[name]
    ax.annotate(f"{name}\n{ms:.1f} ms",
                xy=(x, ms),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center", va=va,
                fontsize=10,
                color="#1b3a7a")

# Baseline reference line
ax.axhline(baseline, color="#999999", linestyle="--", linewidth=1, zorder=1)
ax.text(xs[-1] + 0.03, baseline, f" baseline {baseline:.1f} ms",
        va="center", ha="left", fontsize=9, color="#666666")

delta_ms = baseline - final
delta_pct = delta_ms / baseline * 100

ax.set_xticks(xs, labels, fontsize=10)
ax.set_xlabel("Stage", fontsize=11)
ax.set_ylabel("prefill_mean (ms, lower is better)", fontsize=11)
ax.set_ylim(1100, 2250)
ax.set_xlim(0.7, len(STAGES) + 0.5)
ax.grid(True, axis="y", linestyle=":", color="#cccccc", zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title(
    "Qwen3-Coder-Next 80B Q4_K_M — 1024-token prefill across 5 stages\n"
    f"End-to-end: baseline {baseline:.1f} ms  →  final {final:.1f} ms   "
    f"(−{delta_ms:.1f} ms, −{delta_pct:.1f}%)",
    fontsize=12, pad=12,
)

out = Path(__file__).with_suffix(".png")
fig.tight_layout()
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"Saved: {out}")
