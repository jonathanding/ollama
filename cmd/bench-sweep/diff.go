package main

import (
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

func runDiff(args []string) error {
	if len(args) < 2 {
		return fmt.Errorf("usage: bench-sweep diff <run-a> <run-b>")
	}
	a, err := loadRun(args[0])
	if err != nil {
		return fmt.Errorf("load %q: %w", args[0], err)
	}
	b, err := loadRun(args[1])
	if err != nil {
		return fmt.Errorf("load %q: %w", args[1], err)
	}
	printDiff(os.Stdout, a, b)
	return nil
}

func printDiff(w io.Writer, a, b RunRecord) {
	fmt.Fprintf(w, "Diff: %s → %s  |  Model: %s\n", a.Name, b.Name, b.Model)
	fmt.Fprintf(w, "Note: Δ%% negative = improvement for TTFT (lower is better); positive = improvement for prefill_tps (higher is better)\n\n")

	header := fmt.Sprintf("%-13s │ %-30s │ %8s │ %-24s │ %-24s │ %8s │ %s",
		"prompt_tokens",
		"prefill_tps baseline→new",
		"Δ%",
		"TTFT mean baseline→new",
		"TTFT p99 baseline→new",
		"Δ%",
		"note",
	)
	fmt.Fprintln(w, header)
	fmt.Fprintln(w, strings.Repeat("─", len(header)+10))

	bBySize := make(map[int]SizeResult)
	for _, r := range b.Results {
		bBySize[r.PromptTokens] = r
	}
	aBySize := make(map[int]SizeResult)
	for _, r := range a.Results {
		aBySize[r.PromptTokens] = r
	}

	for _, ra := range a.Results {
		rb, ok := bBySize[ra.PromptTokens]
		if !ok {
			fmt.Fprintf(w, "%-13d │ (only in %s, skipping)\n", ra.PromptTokens, a.Name)
			continue
		}

		prefillDelta := 0.0
		if ra.Stats.PrefillTPS.Mean > 0 {
			prefillDelta = (rb.Stats.PrefillTPS.Mean - ra.Stats.PrefillTPS.Mean) / ra.Stats.PrefillTPS.Mean * 100
		}
		ttftDelta := 0.0
		if ra.Stats.TTFTMs.Mean > 0 {
			ttftDelta = (rb.Stats.TTFTMs.Mean - ra.Stats.TTFTMs.Mean) / ra.Stats.TTFTMs.Mean * 100
		}

		note := ""
		if !ra.Stable {
			note += fmt.Sprintf("⚠ %s CV=%.1f%%", a.Name, math.Max(ra.Stats.PrefillTPS.CVPct, ra.Stats.TTFTMs.CVPct))
		}
		if !rb.Stable {
			if note != "" {
				note += " "
			}
			note += fmt.Sprintf("⚠ %s CV=%.1f%%", b.Name, math.Max(rb.Stats.PrefillTPS.CVPct, rb.Stats.TTFTMs.CVPct))
		}

		fmt.Fprintf(w, "%-13d │ %13.0f → %-13.0f t/s │ %+7.1f%% │ %9.0f → %-12.0f ms │ %9.0f → %-12.0f ms │ %+7.1f%% │ %s\n",
			ra.PromptTokens,
			ra.Stats.PrefillTPS.Mean, rb.Stats.PrefillTPS.Mean, prefillDelta,
			ra.Stats.TTFTMs.Mean, rb.Stats.TTFTMs.Mean,
			ra.Stats.TTFTMs.P99, rb.Stats.TTFTMs.P99, ttftDelta,
			note,
		)
	}

	for _, rb := range b.Results {
		if _, ok := aBySize[rb.PromptTokens]; !ok {
			fmt.Fprintf(w, "%-13d │ (only in %s, skipping)\n", rb.PromptTokens, b.Name)
		}
	}
}
