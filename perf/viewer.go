package perf

import (
	"fmt"
	"io"
	"sort"
	"strings"
)

// PrintProfile prints a human-readable profile summary to w.
func PrintProfile(w io.Writer, p *Profile, detail bool) {
	fmt.Fprintln(w, "Hardware Profile")
	fmt.Fprintln(w, strings.Repeat("-", 60))
	fmt.Fprintf(w, "%-10s %-16s %-6s %-14s %-12s %s\n",
		"Backend", "Device", "Dtype", "Peak FLOPS", "Peak BW", "Balance Point")

	for _, bp := range p.Hardware.Backends {
		for dtype, flops := range bp.PeakFLOPS {
			balPt := bp.BalancePoints[dtype]
			fmt.Fprintf(w, "%-10s %-16s %-6s %-14s %-12s %.1f FLOP/byte\n",
				bp.Name, truncate(bp.Device, 16), dtype,
				formatSI(flops, "FLOPS"), formatSI(bp.PeakBandwidth, "B/s"), balPt)
		}
	}

	if len(p.Interconnects) > 0 {
		fmt.Fprintln(w)
		fmt.Fprintln(w, "Interconnects")
		fmt.Fprintln(w, strings.Repeat("-", 40))
		for _, ic := range p.Interconnects {
			fmt.Fprintf(w, "  %s -> %s: %s\n", ic.From, ic.To, formatSI(ic.Bandwidth, "B/s"))
		}
	}

	if detail {
		fmt.Fprintln(w)
		fmt.Fprintln(w, "Operator Calibration (eta)")
		fmt.Fprintln(w, strings.Repeat("-", 80))
		fmt.Fprintf(w, "%-16s %-10s %-8s %-8s %8s %10s %6s\n",
			"Op", "Backend", "Compute", "Weight", "eta", "Variance", "Pts")

		sorted := make([]OperatorProfile, len(p.Operators))
		copy(sorted, p.Operators)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i].Op < sorted[j].Op })

		for _, op := range sorted {
			fmt.Fprintf(w, "%-16s %-10s %-8s %-8s %8.4f %10.6f %6d\n",
				op.Op, op.Backend, op.ComputeDtype, op.WeightDtype,
				op.Eta, op.EtaVariance, op.NumPoints)
		}
	}
}

// PrintEstimateResult prints a human-readable estimate result to w.
func PrintEstimateResult(w io.Writer, r *EstimateResult, detail bool) {
	backends := make([]string, len(r.Backends))
	for i, b := range r.Backends {
		backends[i] = fmt.Sprintf("%s (%s)", b.Name, b.Device)
	}
	fmt.Fprintf(w, "Model: %s | Backend: %s\n", r.Model, strings.Join(backends, " + "))
	fmt.Fprintf(w, "Input: %d tokens | Output: %d tokens | Max batch: %d\n\n",
		r.InputLength, r.OutputLength, r.MaxBatchSize)

	fmt.Fprintf(w, "Prefill (%d tokens", r.InputLength)
	if r.Prefill.NumBatches > 1 {
		fmt.Fprintf(w, ", %d batches of %d", r.Prefill.NumBatches, r.MaxBatchSize)
	}
	fmt.Fprintln(w, ")")
	fmt.Fprintln(w, strings.Repeat("-", 50))
	fmt.Fprintf(w, "  Estimated: %.0fms total, %.0f tok/s",
		r.Prefill.TotalLatencyMs, r.Prefill.TokensPerSec)
	if r.Prefill.TTFTMs > 0 {
		fmt.Fprintf(w, ", TTFT ~ %.0fms", r.Prefill.TTFTMs)
	}
	fmt.Fprintln(w)
	fmt.Fprintf(w, "  Bottleneck: %s-bound\n", r.Prefill.Bottleneck)
	printTopOps(w, r.Prefill.TopOps, detail)

	fmt.Fprintln(w)

	fmt.Fprintf(w, "Decode (avg over %d positions)\n", r.OutputLength)
	fmt.Fprintln(w, strings.Repeat("-", 50))
	fmt.Fprintf(w, "  Estimated: %.1fms/tok, %.0f tok/s\n",
		r.Decode.TotalLatencyMs, r.Decode.TokensPerSec)
	fmt.Fprintf(w, "  Bottleneck: %s-bound\n", r.Decode.Bottleneck)
	printTopOps(w, r.Decode.TopOps, detail)

	if len(r.Warnings) > 0 {
		fmt.Fprintln(w)
		fmt.Fprintln(w, "Warnings:")
		for _, w2 := range r.Warnings {
			fmt.Fprintf(w, "  ! %s\n", w2)
		}
	}

	fmt.Fprintln(w)
	fmt.Fprintf(w, "Summary: %s\n", r.Summary)
}

func printTopOps(w io.Writer, ops []OpBreakdown, detail bool) {
	if len(ops) == 0 {
		return
	}
	fmt.Fprintln(w, "  Top ops:")
	fmt.Fprintf(w, "    %-16s %-8s %6s %10s %8s  %s\n",
		"Op", "Dtype", "Count", "Total ms", "%", "Bound breakdown")
	for _, op := range ops {
		dtype := op.ComputeDtype
		if op.WeightDtype != "" {
			dtype = op.WeightDtype
		}
		fmt.Fprintf(w, "    %-16s %-8s %6d %10.1fms %7.1f%%  %s\n",
			op.Op, dtype, op.Count, op.TotalMs, op.Percentage, op.BoundBreakdown)
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
	return s[:maxLen-1] + "..."
}
