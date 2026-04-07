package main

import (
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
)

// sendRequest sends one /api/generate request and returns the epoch metrics.
// TTFT is measured as wall-clock time from request start to first response token.
func sendRequest(ctx context.Context, client *api.Client, model, prompt string, maxTokens, numCtx, batchSize int) (EpochResult, error) {
	options := map[string]interface{}{
		"temperature": 0,
		"num_predict": maxTokens,
	}
	if numCtx > 0 {
		options["num_ctx"] = numCtx
	}
	if batchSize > 0 {
		options["num_batch"] = batchSize
	}
	req := &api.GenerateRequest{
		Model:   model,
		Prompt:  prompt,
		Raw:     true,
		Options: options,
	}

	var result EpochResult
	var ttftOnce sync.Once
	start := time.Now()

	err := client.Generate(ctx, req, func(resp api.GenerateResponse) error {
		if resp.Response != "" || resp.Thinking != "" {
			ttftOnce.Do(func() {
				result.TTFTMs = float64(time.Since(start).Nanoseconds()) / 1e6
			})
		}
		if resp.Done {
			m := resp.Metrics
			result.PromptEvalCount = m.PromptEvalCount
			result.EvalCount = m.EvalCount
			result.PrefillMs = float64(m.PromptEvalDuration.Nanoseconds()) / 1e6
			result.GenMs = float64(m.EvalDuration.Nanoseconds()) / 1e6
			if result.PrefillMs > 0 {
				result.PrefillTPS = float64(result.PromptEvalCount) / (result.PrefillMs / 1000)
			}
			if result.GenMs > 0 {
				result.GenTPS = float64(result.EvalCount) / (result.GenMs / 1000)
			}
		}
		return nil
	})
	return result, err
}

// fetchVRAM reads VRAM usage from /api/ps after model is loaded.
func fetchVRAM(ctx context.Context, client *api.Client, model string) Hardware {
	hw := Hardware{OS: currentOS()}
	psCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	resp, err := client.ListRunning(psCtx)
	if err != nil {
		return hw
	}
	for _, m := range resp.Models {
		if strings.HasPrefix(m.Name, model) || strings.HasPrefix(m.Model, model) {
			hw.VRAMUsedBytes = m.SizeVRAM
			// VRAMTotalBytes intentionally not set — api.ProcessModelResponse.Size is
			// model weight size, not device VRAM capacity. No API field exposes GPU total VRAM.
			return hw
		}
	}
	return hw
}

// runBenchmark executes the full sweep and returns per-size results and hardware info.
func runBenchmark(ctx context.Context, client *api.Client, model string, cfg RunConfig) ([]SizeResult, Hardware, error) {
	var hw Hardware
	hwFetched := false

	var allResults []SizeResult
	firstRow := true
	for sizeIdx, size := range cfg.Sizes {
		chars := initialChars(size)

		// Warmup phase: calibrate on first request, track tps for adequacy check
		var warmupTPS []float64
		for i := 0; i < cfg.Warmup; i++ {
			wCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
			result, err := sendRequest(wCtx, client, model, promptText(chars, (sizeIdx+1)*1000+i), cfg.MaxTokens, cfg.NumCtx, cfg.BatchSize)
			cancel()
			if err != nil {
				fmt.Fprintf(os.Stderr, "Warning: warmup %d/%d for size=%d failed: %v\n", i+1, cfg.Warmup, size, err)
				continue
			}
			warmupTPS = append(warmupTPS, result.PrefillTPS)
			if i == 0 && result.PromptEvalCount > 0 {
				chars = calibrateChars(chars, size, result.PromptEvalCount)
			}
		}

		// Fetch hardware once after first warmup (model is loaded and stable)
		if !hwFetched {
			hw = fetchVRAM(ctx, client, model)
			hwFetched = true
		}

		// Warmup adequacy check (requires warmup >= 2)
		if cfg.Warmup >= 2 && len(warmupTPS) >= 2 {
			first := warmupTPS[0]
			last := warmupTPS[len(warmupTPS)-1]
			if first > 0 {
				change := math.Abs(last-first) / first * 100
				if change > 15 {
					fmt.Fprintf(os.Stderr, "Warning: warmup may be insufficient — prefill_tps changed %.0f%% between warmup iterations\n", change)
					fmt.Fprintf(os.Stderr, "  hint: increase -warmup (current: %d)\n", cfg.Warmup)
				}
			}
		}

		// Timed epochs
		var epochResults []EpochResult
		for epoch := 0; epoch < cfg.Epochs; epoch++ {
			eCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
			result, err := sendRequest(eCtx, client, model, promptText(chars, sizeIdx*10000+epoch), cfg.MaxTokens, cfg.NumCtx, cfg.BatchSize)
			cancel()
			if err != nil {
				fmt.Fprintf(os.Stderr, "Warning: epoch %d/%d for size=%d failed: %v\n", epoch+1, cfg.Epochs, size, err)
				continue
			}
			epochResults = append(epochResults, result)
		}
		if len(epochResults) == 0 {
			return nil, hw, fmt.Errorf("all timed epochs failed for size=%d", size)
		}

		// Filter degenerate epochs where the model hit EOS well before max_tokens.
		// These epochs have inflated gen_tps and may have ttft_ms=0, both of which
		// corrupt CV% statistics. Degenerate epochs are still saved to JSON for inspection.
		statsEpochs := make([]EpochResult, 0, len(epochResults))
		for _, r := range epochResults {
			if r.EvalCount*2 < cfg.MaxTokens {
				continue
			}
			statsEpochs = append(statsEpochs, r)
		}
		if skipped := len(epochResults) - len(statsEpochs); skipped > 0 {
			fmt.Fprintf(os.Stderr, "  note: %d epoch(s) excluded from stats for size=%d (early EOS)\n", skipped, size)
		}
		if len(statsEpochs) == 0 {
			// All epochs hit early EOS; fall back to using all epochs rather than failing.
			statsEpochs = epochResults
			fmt.Fprintf(os.Stderr, "Warning: all timed epochs hit early EOS for size=%d, stats may be unreliable\n", size)
		}

		// Compute stats from valid epochs only.
		prefillMsVals  := make([]float64, len(statsEpochs))
		prefillTPSVals := make([]float64, len(statsEpochs))
		ttftVals       := make([]float64, len(statsEpochs))
		genMsVals      := make([]float64, len(statsEpochs))
		genTPSVals     := make([]float64, len(statsEpochs))
		for i, r := range statsEpochs {
			prefillMsVals[i]  = r.PrefillMs
			prefillTPSVals[i] = r.PrefillTPS
			ttftVals[i]       = r.TTFTMs
			genMsVals[i]      = r.GenMs
			genTPSVals[i]     = r.GenTPS
		}
		stats := SizeStats{
			PrefillMs:  computeStats(prefillMsVals),
			PrefillTPS: computeStats(prefillTPSVals),
			TTFTMs:     computeStats(ttftVals),
			GenMs:      computeStats(genMsVals),
			GenTPS:     computeStats(genTPSVals),
		}
		stable := stats.PrefillTPS.CVPct <= cfg.CVThreshPct && stats.TTFTMs.CVPct <= cfg.CVThreshPct

		allResults = append(allResults, SizeResult{
			PromptTokens: size,
			Stable:       stable,
			Epochs:       epochResults,
			Stats:        stats,
		})

		// Print row immediately so the user sees progress
		printRunTable(os.Stdout, model, allResults[len(allResults)-1:], cfg, firstRow)
		firstRow = false

		// Warn if unstable
		if !stable {
			if stats.PrefillTPS.CVPct > cfg.CVThreshPct {
				fmt.Fprintf(os.Stderr, "⚠ [size=%d] prefill_tps CV=%.1f%% exceeds threshold %.1f%%\n", size, stats.PrefillTPS.CVPct, cfg.CVThreshPct)
				fmt.Fprintf(os.Stderr, "  hint: consider increasing -warmup (current: %d) or closing background processes\n", cfg.Warmup)
			}
			if stats.TTFTMs.CVPct > cfg.CVThreshPct {
				fmt.Fprintf(os.Stderr, "⚠ [size=%d] ttft_ms CV=%.1f%% exceeds threshold %.1f%%\n", size, stats.TTFTMs.CVPct, cfg.CVThreshPct)
			}
		}
	}
	return allResults, hw, nil
}

// printRunTable writes a formatted result row. Pass a single-element slice to
// print one row at a time; pass first=true to print the header above the first row.
func printRunTable(w io.Writer, model string, results []SizeResult, cfg RunConfig, first bool) {
	if len(results) == 0 {
		return
	}
	if first {
		fmt.Fprintf(w, "\nModel: %s  |  Epochs: %d  |  Warmup: %d\n", model, cfg.Epochs, cfg.Warmup)
		fmt.Fprintf(w, "note: prefill_ms is the server-side prompt processing time (Ollama internal metric); ttft_ms is the wall-clock time to first token (= prefill_ms + HTTP round-trip + scheduling, typically 80–200 ms longer on localhost).\n\n")
		// Column widths: prompt_tokens=13, each metric value column matches its header label width,
		// CV% columns are 5 chars (" 1.8%"), status is variable.
		fmt.Fprintf(w, "%-13s │ %-10s │ %5s │ %-11s │ %5s │ %-10s │ %5s │ %-9s │ %5s │ %-9s │ %5s │ %s\n",
			"prompt_tokens",
			"prefill_ms", "CV%",
			"prefill_tps", "CV%",
			"ttft_ms", "CV%",
			"gen_ms", "CV%",
			"gen_tps", "CV%",
			"status",
		)
		fmt.Fprintln(w, strings.Repeat("─", 122))
	}
	r := results[len(results)-1]
	status := "✓"
	if !r.Stable {
		status = "⚠"
	}
	// Data widths: %7.0f ms=10, %7.0f t/s=11, %7.0f ms=10, %6.0f ms=9, %5.0f t/s=9 — match header labels.
	fmt.Fprintf(w, "%-13d │ %7.0f ms │ %4.1f%% │ %7.0f t/s │ %4.1f%% │ %7.0f ms │ %4.1f%% │ %6.0f ms │ %4.1f%% │ %5.0f t/s │ %4.1f%% │ %s\n",
		r.PromptTokens,
		r.Stats.PrefillMs.Mean,  r.Stats.PrefillMs.CVPct,
		r.Stats.PrefillTPS.Mean, r.Stats.PrefillTPS.CVPct,
		r.Stats.TTFTMs.Mean,     r.Stats.TTFTMs.CVPct,
		r.Stats.GenMs.Mean,      r.Stats.GenMs.CVPct,
		r.Stats.GenTPS.Mean,     r.Stats.GenTPS.CVPct,
		status,
	)
}
