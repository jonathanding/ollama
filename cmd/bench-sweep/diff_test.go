package main

import (
	"strings"
	"testing"
	"time"
)

func makeRunRecord(name string, sizes []int, stable bool, prefillMean, ttftMean, ttftP99 float64) RunRecord {
	var results []SizeResult
	for _, s := range sizes {
		results = append(results, SizeResult{
			PromptTokens: s,
			Stable:       stable,
			Stats: SizeStats{
				PrefillTPS: MetricStats{Mean: prefillMean, P99: prefillMean * 0.95, CVPct: 1.5},
				TTFTMs:     MetricStats{Mean: ttftMean, P99: ttftP99, CVPct: 2.0},
				GenTPS:     MetricStats{Mean: 37},
			},
		})
	}
	return RunRecord{
		Name:      name,
		Model:     "qwen3",
		Timestamp: time.Now(),
		Config:    RunConfig{Sizes: sizes},
		Results:   results,
	}
}

func TestPrintDiff_ShowsImprovement(t *testing.T) {
	a := makeRunRecord("baseline", []int{512, 1024}, true, 4000, 50, 60)
	b := makeRunRecord("optimized", []int{512, 1024}, true, 4400, 44, 53) // +10% prefill, -12% TTFT

	var sb strings.Builder
	printDiff(&sb, a, b)
	out := sb.String()

	if !strings.Contains(out, "baseline") || !strings.Contains(out, "optimized") {
		t.Errorf("expected both run names in diff output:\n%s", out)
	}
	if !strings.Contains(out, "4000") || !strings.Contains(out, "4400") {
		t.Errorf("expected prefill values in diff output:\n%s", out)
	}
	if !strings.Contains(out, "+10.0") && !strings.Contains(out, "+10") {
		t.Errorf("expected +10%% prefill delta in diff output:\n%s", out)
	}
}

func TestPrintDiff_TTFTNegativeMeansImprovement(t *testing.T) {
	a := makeRunRecord("a", []int{512}, true, 4000, 50, 60)
	b := makeRunRecord("b", []int{512}, true, 4000, 40, 48) // TTFT -20%

	var sb strings.Builder
	printDiff(&sb, a, b)
	out := sb.String()

	if !strings.Contains(out, "-20.0") && !strings.Contains(out, "-20") {
		t.Errorf("expected -20%% TTFT delta (improvement), got:\n%s", out)
	}
}

func TestPrintDiff_UnstableWarning(t *testing.T) {
	a := makeRunRecord("a", []int{512}, false, 4000, 50, 60) // unstable
	b := makeRunRecord("b", []int{512}, true, 4400, 44, 53)

	var sb strings.Builder
	printDiff(&sb, a, b)
	out := sb.String()

	if !strings.Contains(out, "⚠") {
		t.Errorf("expected ⚠ for unstable run in diff:\n%s", out)
	}
}

func TestPrintDiff_MissingSizeSkipped(t *testing.T) {
	a := makeRunRecord("a", []int{512, 1024}, true, 4000, 50, 60)
	b := makeRunRecord("b", []int{512}, true, 4400, 44, 53) // missing size 1024

	var sb strings.Builder
	printDiff(&sb, a, b)
	out := sb.String()

	if !strings.Contains(out, "only in") {
		t.Errorf("expected 'only in' note for mismatched size:\n%s", out)
	}
}
