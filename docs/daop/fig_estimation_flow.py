"""
daop-estimate: High-level estimation flow diagram.
Generates fig_estimation_flow.png.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse

fig, ax = plt.subplots(1, 1, figsize=(11, 13))
ax.set_xlim(0, 11)
ax.set_ylim(0, 13)
ax.axis('off')
ax.set_aspect('equal')

# Colors
c_input = '#FFF3E0'
c_phase1 = '#E8F5E9'
c_phase2 = '#E3F2FD'
c_output = '#F3E5F5'
c_profile = '#FFF9C4'
c_border_input = '#E65100'
c_border_p1 = '#2E7D32'
c_border_p2 = '#1565C0'
c_border_output = '#6A1B9A'
c_border_profile = '#F9A825'
c_ann = '#B71C1C'


def draw_box(x, y, w, h, text, fc, ec, fontsize=9.5, bold=False):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.1",
                         facecolor=fc, edgecolor=ec, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            weight='bold' if bold else 'normal')


def draw_oval(x, y, w, h, text, fc, ec, fontsize=10):
    oval = Ellipse((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=2.5)
    ax.add_patch(oval)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, weight='bold')


def arrow(x1, y1, x2, y2, color='#424242', style='->', ls='solid'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.8,
                                linestyle=ls))


def ann_left(x, y, text):
    ax.text(x, y, text, ha='right', va='center', fontsize=8,
            color=c_ann, style='italic')


def ann_right(x, y, text):
    pass  # disabled


# ─── Layout constants ───
main_x = 6.5        # main flow center x
p1_x = 2.2          # phase 1 center x
spacing = 1.4        # vertical spacing between boxes

# ─── Title ───
ax.text(5.5, 12.7, 'Two-Phase Static Model-Based Latency Prediction',
        ha='center', va='center', fontsize=14, weight='bold')

# ─── Input oval ───
y = 11.8
draw_oval(main_x, y, 4.5, 1.0, 'New Model\n(e.g. llama3:8b-q4_0)', c_input, c_border_input, fontsize=10)

# ─── Phase 2 main flow ───
y1 = y - spacing
draw_box(main_x, y1, 4.2, 0.9, 'Load Model Metadata\n(GGUF header only, no weights)', c_phase2, c_border_p2, fontsize=10.5)
arrow(main_x, y - 0.5, main_x, y1 + 0.45)

y2 = y1 - spacing
draw_box(main_x, y2, 4.2, 0.9, 'Capture Compute Graph\n(~300 ops per phase)', c_phase2, c_border_p2, fontsize=10.5)
arrow(main_x, y1 - 0.45, main_x, y2 + 0.45)

y3 = y2 - spacing
draw_box(main_x, y3, 4.2, 0.9, 'Backend Assignment\n+ Op Fusion', c_phase2, c_border_p2, fontsize=10.5)
arrow(main_x, y2 - 0.45, main_x, y3 + 0.45)

y4 = y3 - spacing
draw_box(main_x, y4, 4.2, 1.0, 'Piecewise Log-space\nInterpolation per Op', c_phase2, c_border_p2, fontsize=10.5, bold=True)
arrow(main_x, y3 - 0.45, main_x, y4 + 0.5)

y5 = y4 - spacing
draw_box(main_x, y5, 4.2, 0.75, 'Sum per-op latencies', c_phase2, c_border_p2, fontsize=10.5)
arrow(main_x, y4 - 0.5, main_x, y5 + 0.38)

y_out = y5 - spacing
draw_oval(main_x, y_out, 3.6, 0.9, 'Predict tok/s', c_output, c_border_output, fontsize=11)
arrow(main_x, y5 - 0.35, main_x, y_out + 0.45)

# ─── Phase 1: Benchmark (left side) ───
phase1_bg = FancyBboxPatch((0.3, y4 - 0.3), 3.8, 4.8,
                            boxstyle="round,pad=0.15",
                            facecolor=c_phase1, edgecolor=c_border_p1,
                            linewidth=1.5, linestyle='--', alpha=0.4)
ax.add_patch(phase1_bg)
ax.text(p1_x, y4 + 4.25, 'Phase 1: Benchmark', ha='center',
        fontsize=9.5, weight='bold', color=c_border_p1)
ax.text(p1_x, y4 + 3.85, '(once per GPU, ~90s)', ha='center',
        fontsize=8.5, color=c_border_p1)

# HW Char
hw_y = y4 + 3.1
draw_box(p1_x, hw_y, 3.2, 0.8, 'Hardware Characterization\n(Peak TOPS, Bandwidth)',
         c_phase1, c_border_p1, fontsize=9.5)

# Op Bench
ob_y = hw_y - 1.2
draw_box(p1_x, ob_y, 3.2, 0.8, 'Operator Benchmarking\n81 curves × ~12 adaptive pts',
         c_phase1, c_border_p1, fontsize=9.5)
arrow(p1_x, hw_y - 0.4, p1_x, ob_y + 0.4)

# Profile JSON
pj_y = ob_y - 1.1
draw_box(p1_x, pj_y, 2.8, 0.65, 'Profile JSON (~1000 pts)',
         c_profile, c_border_profile, fontsize=9.5, bold=True)
arrow(p1_x, ob_y - 0.4, p1_x, pj_y + 0.33)

# Profile -> Interpolation (dashed arrow)
arrow(p1_x + 1.4, pj_y, main_x - 2.0, y4, color=c_border_profile, ls='dashed')
ax.text(3.8, y4 + 0.55, 'calibration data', ha='center', fontsize=8,
        color='#F57F17', style='italic')

# ─── Legend at bottom ───
leg_y = y_out - 1.2
ax.plot([1, 10], [leg_y + 0.4, leg_y + 0.4], color='#BDBDBD', lw=0.5)
ax.text(5.5, leg_y - 0.1, 'Total estimation latency:  ~ms  (graph capture + interpolation + sum)',
        ha='center', fontsize=10, weight='bold', color='#1B5E20')
ax.text(5.5, leg_y - 0.55, 'No model weights download or GPU compute required',
        ha='center', fontsize=9, color='#424242')

plt.tight_layout()
plt.savefig('C:/workspace/daop-ollama/docs/daop/fig_estimation_flow.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("Saved: docs/daop/fig_estimation_flow.png")
