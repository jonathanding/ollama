package perf

import (
	"fmt"
	"io"
	"sort"
	"strings"
)

// PrintProfile prints a human-readable v2 profile summary to w.
func PrintProfile(w io.Writer, p *Profile, detail bool) {
	fmt.Fprintln(w, "Hardware Profile (v2)")
	fmt.Fprintln(w, strings.Repeat("-", 60))

	for _, bi := range p.Hardware.Backends {
		fmt.Fprintf(w, "  Backend: %s (%s)\n", bi.Name, bi.Device)
		if bi.VRAMBytes > 0 {
			fmt.Fprintf(w, "  VRAM: %s\n", formatSI(float64(bi.VRAMBytes), "B"))
		}
	}

	for dtype, tops := range p.Hardware.PeakTOPS {
		bp := p.Hardware.BalancePoints[dtype]
		fmt.Fprintf(w, "  %s: %s peak, %.1f FLOP/byte balance\n",
			dtype, formatSI(tops, "OPS"), bp)
	}
	fmt.Fprintf(w, "  Bandwidth: %s\n", formatSI(p.Hardware.PeakBandwidthBytesPerSec, "B/s"))

	fmt.Fprintf(w, "\nOperator Curves: %d\n", len(p.Operators))
	fmt.Fprintln(w, strings.Repeat("-", 60))

	if detail {
		sorted := make([]OperatorCurve, len(p.Operators))
		copy(sorted, p.Operators)
		sort.Slice(sorted, func(i, j int) bool {
			if sorted[i].Op != sorted[j].Op {
				return sorted[i].Op < sorted[j].Op
			}
			return sorted[i].ComputeDtype < sorted[j].ComputeDtype
		})

		for _, c := range sorted {
			label := c.Op
			if c.WeightDtype != "" {
				label += fmt.Sprintf(" (%s->%s)", c.WeightDtype, c.ComputeDtype)
			} else {
				label += fmt.Sprintf(" (%s)", c.ComputeDtype)
			}
			if len(c.FixedDims) > 0 {
				label += fmt.Sprintf(" fixed=%v", c.FixedDims)
			}
			fmt.Fprintf(w, "  %-50s %d points\n", label, len(c.Points))
		}
	} else {
		opCounts := make(map[string]int)
		for _, c := range p.Operators {
			opCounts[c.Op]++
		}
		for op, count := range opCounts {
			fmt.Fprintf(w, "  %-20s %d curves\n", op, count)
		}
	}
}

// PrintEstimateResult prints a human-readable v2 estimation result to w.
func PrintEstimateResult(w io.Writer, r *EstimateResult, detail bool) {
	fmt.Fprintf(w, "Model: %s\n\n", r.Model)

	fmt.Fprintln(w, "Prefill")
	fmt.Fprintln(w, strings.Repeat("-", 50))
	fmt.Fprintf(w, "  Latency: %.1fms (%.0f tok/s)\n",
		r.Prefill.TotalLatencyMs, r.Prefill.TokensPerSec)
	printTopOps(w, r.Prefill.TopOps, detail)

	fmt.Fprintln(w)

	fmt.Fprintln(w, "Decode (per token)")
	fmt.Fprintln(w, strings.Repeat("-", 50))
	fmt.Fprintf(w, "  Latency: %.3fms/tok (%.0f tok/s)\n",
		r.Decode.TotalLatencyMs, r.Decode.TokensPerSec)
	printTopOps(w, r.Decode.TopOps, detail)

	if len(r.Warnings) > 0 {
		fmt.Fprintln(w)
		fmt.Fprintln(w, "Warnings:")
		for _, w2 := range r.Warnings {
			fmt.Fprintf(w, "  ! %s\n", w2)
		}
	}
}

func printTopOps(w io.Writer, ops []OpBreakdown, detail bool) {
	if len(ops) == 0 {
		return
	}
	fmt.Fprintln(w, "  Top ops:")
	limit := 5
	if detail {
		limit = 10
	}
	for i, op := range ops {
		if i >= limit {
			break
		}
		dtype := op.ComputeDtype
		if op.WeightDtype != "" {
			dtype = op.WeightDtype
		}
		fmt.Fprintf(w, "    %-16s %-8s %4dx  %8.1fus  %5.1f%%\n",
			op.Op, dtype, op.Count, op.TotalUs, op.Percentage*100)
	}
}

func formatSI(val float64, unit string) string {
	switch {
	case val >= 1e12:
		return fmt.Sprintf("%.1f T%s", val/1e12, unit)
	case val >= 1e9:
		return fmt.Sprintf("%.1f G%s", val/1e9, unit)
	case val >= 1e6:
		return fmt.Sprintf("%.1f M%s", val/1e6, unit)
	case val >= 1e3:
		return fmt.Sprintf("%.1f K%s", val/1e3, unit)
	default:
		return fmt.Sprintf("%.1f %s", val, unit)
	}
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
