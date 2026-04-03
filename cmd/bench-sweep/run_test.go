package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// mockGenServer creates a test server that returns the given responses for /api/generate
// and stubs /api/ps with the given VRAM values.
func mockGenServer(t *testing.T, responses []api.GenerateResponse, vramUsed, vramTotal int64) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/generate":
			for _, resp := range responses {
				data, _ := json.Marshal(resp)
				w.Write(data)
				w.Write([]byte("\n"))
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			}
		case "/api/ps":
			json.NewEncoder(w).Encode(api.ProcessResponse{
				Models: []api.ProcessModelResponse{
					{Name: "test-model", Model: "test-model", Size: vramTotal, SizeVRAM: vramUsed},
				},
			})
		default:
			http.Error(w, "not found", 404)
		}
	}))
}

func goodResponses(promptTokens, evalTokens int) []api.GenerateResponse {
	return []api.GenerateResponse{
		{Response: "first", Done: false},
		{
			Response: "last",
			Done:     true,
			Metrics: api.Metrics{
				PromptEvalCount:    promptTokens,
				PromptEvalDuration: 200 * time.Millisecond,
				EvalCount:          evalTokens,
				EvalDuration:       400 * time.Millisecond,
			},
		},
	}
}

func TestSendRequest_CapsTTFT(t *testing.T) {
	srv := mockGenServer(t, goodResponses(512, 16), 0, 0)
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	client, _ := api.ClientFromEnvironment()
	result, err := sendRequest(context.Background(), client, "test-model", "hello", 16)
	if err != nil {
		t.Fatalf("sendRequest: %v", err)
	}
	if result.TTFTMs <= 0 {
		t.Error("TTFT should be positive")
	}
	if result.PromptEvalCount != 512 {
		t.Errorf("PromptEvalCount: got %d, want 512", result.PromptEvalCount)
	}
	if result.PrefillTPS <= 0 {
		t.Errorf("PrefillTPS should be positive, got %.2f", result.PrefillTPS)
	}
}

func TestSendRequest_ComputesTPS(t *testing.T) {
	// PromptEvalDuration=200ms, PromptEvalCount=512 → PrefillTPS = 512/0.2 = 2560
	srv := mockGenServer(t, goodResponses(512, 16), 0, 0)
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	client, _ := api.ClientFromEnvironment()
	result, err := sendRequest(context.Background(), client, "test-model", "hello", 16)
	if err != nil {
		t.Fatalf("sendRequest: %v", err)
	}
	// Allow ±10% tolerance
	if result.PrefillTPS < 2304 || result.PrefillTPS > 2816 {
		t.Errorf("PrefillTPS: got %.2f, want ~2560 (±10%%)", result.PrefillTPS)
	}
}

func TestRunBenchmark_ReturnsSizeResults(t *testing.T) {
	srv := mockGenServer(t, goodResponses(512, 16), 20e9, 24e9)
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	client, _ := api.ClientFromEnvironment()
	cfg := RunConfig{
		Epochs:      3,
		Warmup:      1,
		MaxTokens:   16,
		CVThreshPct: 5.0,
		Sizes:       []int{512},
	}
	results, hw, err := runBenchmark(context.Background(), client, "test-model", cfg)
	if err != nil {
		t.Fatalf("runBenchmark: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 SizeResult, got %d", len(results))
	}
	if results[0].PromptTokens != 512 {
		t.Errorf("PromptTokens: got %d, want 512", results[0].PromptTokens)
	}
	if len(results[0].Epochs) != 3 {
		t.Errorf("expected 3 epoch results, got %d", len(results[0].Epochs))
	}
	if results[0].Stats.PrefillTPS.Mean <= 0 {
		t.Error("PrefillTPS mean should be positive")
	}
	if hw.VRAMUsedBytes != 20e9 {
		t.Errorf("VRAMUsedBytes: got %d, want 20e9", hw.VRAMUsedBytes)
	}
}

func TestRunBenchmark_MultipleSizes(t *testing.T) {
	srv := mockGenServer(t, goodResponses(512, 16), 0, 0)
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	client, _ := api.ClientFromEnvironment()
	cfg := RunConfig{Epochs: 2, Warmup: 1, MaxTokens: 16, CVThreshPct: 5.0, Sizes: []int{512, 1024}}
	results, _, err := runBenchmark(context.Background(), client, "test-model", cfg)
	if err != nil {
		t.Fatalf("runBenchmark: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("expected 2 SizeResults, got %d", len(results))
	}
}

func TestPrintRunTable_ContainsExpectedColumns(t *testing.T) {
	results := []SizeResult{
		{
			PromptTokens: 512,
			Stable:       true,
			Stats: SizeStats{
				PrefillTPS: MetricStats{Mean: 4850, P99: 4720, CVPct: 1.8},
				TTFTMs:     MetricStats{Mean: 28, P99: 35, CVPct: 2.3},
				GenTPS:     MetricStats{Mean: 37},
			},
		},
	}
	var sb strings.Builder
	printRunTable(&sb, "test-model", results, RunConfig{Epochs: 6, Warmup: 2}, true)
	out := sb.String()
	for _, want := range []string{"512", "4850", "28", "37", "1.8", "2.3", "✓"} {
		if !strings.Contains(out, want) {
			t.Errorf("expected %q in run table output:\n%s", want, out)
		}
	}
}

func TestPrintRunTable_UnstableFlag(t *testing.T) {
	results := []SizeResult{
		{
			PromptTokens: 4096,
			Stable:       false,
			Stats: SizeStats{
				PrefillTPS: MetricStats{Mean: 3890, P99: 3200, CVPct: 8.7},
				TTFTMs:     MetricStats{Mean: 198, P99: 240, CVPct: 9.2},
				GenTPS:     MetricStats{Mean: 36},
			},
		},
	}
	var sb strings.Builder
	printRunTable(&sb, "test-model", results, RunConfig{Epochs: 6, Warmup: 2}, true)
	out := sb.String()
	if !strings.Contains(out, "⚠") {
		t.Errorf("expected ⚠ for unstable result, got:\n%s", out)
	}
	if !strings.Contains(out, "8.7") {
		t.Errorf("expected CV%% 8.7 in output, got:\n%s", out)
	}
}

func TestFetchVRAM_ReturnsValues(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(api.ProcessResponse{
			Models: []api.ProcessModelResponse{
				{Name: "qwen3", Model: "qwen3", Size: 24e9, SizeVRAM: 22e9},
			},
		})
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	client, _ := api.ClientFromEnvironment()
	hw := fetchVRAM(context.Background(), client, "qwen3")
	if hw.VRAMUsedBytes != 22e9 {
		t.Errorf("VRAMUsedBytes: got %d, want 22e9", hw.VRAMUsedBytes)
	}
	if hw.VRAMTotalBytes != 0 {
		t.Errorf("VRAMTotalBytes should be 0 (m.Size is model weight size, not GPU capacity), got %d", hw.VRAMTotalBytes)
	}
}

func TestFetchVRAM_PrefixMatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(api.ProcessResponse{
			Models: []api.ProcessModelResponse{
				{Name: "qwen3:latest", Model: "qwen3:latest", Size: 24e9, SizeVRAM: 22e9},
			},
		})
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	client, _ := api.ClientFromEnvironment()
	hw := fetchVRAM(context.Background(), client, "qwen3")
	if hw.VRAMUsedBytes != 22e9 {
		t.Errorf("VRAMUsedBytes via prefix match: got %d, want 22e9", hw.VRAMUsedBytes)
	}
}

func TestRunBenchmark_FiltersEarlyEOS(t *testing.T) {
	// Return eval_count=1 (early EOS) for every request — should fall back to all epochs.
	eosResp := []api.GenerateResponse{
		{
			Done: true,
			Metrics: api.Metrics{
				PromptEvalCount:    512,
				PromptEvalDuration: 200 * time.Millisecond,
				EvalCount:          1,
				EvalDuration:       10 * time.Millisecond,
			},
		},
	}
	srv := mockGenServer(t, eosResp, 0, 0)
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	client, _ := api.ClientFromEnvironment()
	cfg := RunConfig{Epochs: 3, Warmup: 1, MaxTokens: 16, CVThreshPct: 5.0, Sizes: []int{512}}
	results, _, err := runBenchmark(context.Background(), client, "test-model", cfg)
	if err != nil {
		t.Fatalf("runBenchmark: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	// All 3 epochs should still be stored in JSON even though all hit early EOS.
	if len(results[0].Epochs) != 3 {
		t.Errorf("expected 3 epochs stored in JSON, got %d", len(results[0].Epochs))
	}
}

// Suppress unused import warnings from mock helpers
var _ = fmt.Sprintf
var _ = io.Discard
