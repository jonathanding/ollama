# bench-sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `cmd/bench-sweep`, a repeatable CLI benchmark tool that sweeps multiple prompt sizes, measures prefill TPS and TTFT, stores named run history, and can diff two runs.

**Architecture:** New standalone Go binary in `cmd/bench-sweep/`. All files share `package main`. Uses `github.com/ollama/ollama/api` for Ollama HTTP calls; no changes to any existing package. Tests follow the same `httptest.NewServer` + `t.Setenv("OLLAMA_HOST", ...)` pattern as `cmd/bench/bench_test.go`.

**Tech Stack:** Go 1.24, `github.com/ollama/ollama/api`, `net/http/httptest` for tests, `encoding/json`, `go:embed` for corpus.

**Spec:** [`docs/superpowers/specs/2026-04-02-bench-sweep-design.md`](../specs/2026-04-02-bench-sweep-design.md)

---

## File Map

| File | Responsibility |
|---|---|
| `cmd/bench-sweep/corpus.txt` | Embedded ~50 KB public-domain English prose |
| `cmd/bench-sweep/stats.go` | `computeStats([]float64) MetricStats` — mean/median/p99/stddev/CV% |
| `cmd/bench-sweep/stats_test.go` | Unit tests for stats |
| `cmd/bench-sweep/history.go` | JSON types + `saveRun` / `loadRun` / `listRuns` / `resolveRunName` |
| `cmd/bench-sweep/history_test.go` | Unit tests for history, using `t.TempDir()` override |
| `cmd/bench-sweep/prompt.go` | `promptText(chars, epoch)` corpus slicer + `calibrateChars` |
| `cmd/bench-sweep/prompt_test.go` | Unit tests for prompt generation |
| `cmd/bench-sweep/run.go` | `runBenchmark` sweep loop, `sendRequest`, `fetchVRAM`, `printRunTable` |
| `cmd/bench-sweep/run_test.go` | Tests using mock Ollama server |
| `cmd/bench-sweep/diff.go` | `runDiff` subcommand + `printDiff` |
| `cmd/bench-sweep/diff_test.go` | Tests for diff table rendering |
| `cmd/bench-sweep/list.go` | `runList` subcommand + `printList` |
| `cmd/bench-sweep/list_test.go` | Tests for list table rendering |
| `cmd/bench-sweep/main.go` | CLI entry point, subcommand dispatch |
| `cmd/bench-sweep/README.md` | Build instructions, usage, flag reference, design rationale |

---

## Task 1: Fetch corpus

**Files:**
- Create: `cmd/bench-sweep/corpus.txt`

- [ ] **Step 1: Download Project Gutenberg text and trim to ~50 KB**

Run this PowerShell command from the repo root:

```powershell
$raw = (Invoke-WebRequest -Uri "https://www.gutenberg.org/files/1342/1342-0.txt").Content
# Find start of actual novel text
$start = $raw.IndexOf("It is a truth universally acknowledged")
$text = $raw.Substring($start, [Math]::Min(51200, $raw.Length - $start))
# Strip Windows line endings, write UTF-8
[System.IO.File]::WriteAllText("cmd\bench-sweep\corpus.txt", $text, [System.Text.Encoding]::UTF8)
Write-Host "corpus.txt written: $((Get-Item cmd\bench-sweep\corpus.txt).Length) bytes"
```

Expected output: `corpus.txt written: 51200 bytes` (or close to it).

- [ ] **Step 2: Verify the file**

```powershell
Get-Content cmd\bench-sweep\corpus.txt | Select-Object -First 3
```

Expected: first line starts with `"It is a truth universally acknowledged"`.

- [ ] **Step 3: Commit**

```bash
git add cmd/bench-sweep/corpus.txt
git commit -m "bench-sweep: add embedded corpus for prompt generation"
```

---

## Task 2: stats.go

**Files:**
- Create: `cmd/bench-sweep/stats.go`
- Create: `cmd/bench-sweep/stats_test.go`

- [ ] **Step 1: Write the failing test**

Create `cmd/bench-sweep/stats_test.go`:

```go
package main

import (
	"testing"
)

func TestComputeStats_OddSlice(t *testing.T) {
	s := computeStats([]float64{100, 200, 300, 400, 500})
	if s.Mean != 300 {
		t.Errorf("mean: got %.2f, want 300", s.Mean)
	}
	if s.Median != 300 {
		t.Errorf("median: got %.2f, want 300", s.Median)
	}
	// p99: ceil(5*0.99)-1 = 5-1 = 4, sorted[4] = 500
	if s.P99 != 500 {
		t.Errorf("p99: got %.2f, want 500", s.P99)
	}
}

func TestComputeStats_EvenSlice(t *testing.T) {
	// median of [1,3,5,7] = (3+5)/2 = 4
	s := computeStats([]float64{1, 3, 5, 7})
	if s.Median != 4 {
		t.Errorf("median: got %.2f, want 4", s.Median)
	}
}

func TestComputeStats_ConstantCV(t *testing.T) {
	s := computeStats([]float64{100, 100, 100, 100})
	if s.CVPct != 0 {
		t.Errorf("CV for constant slice: got %.4f, want 0", s.CVPct)
	}
	if s.Stddev != 0 {
		t.Errorf("stddev for constant slice: got %.4f, want 0", s.Stddev)
	}
}

func TestComputeStats_Empty(t *testing.T) {
	s := computeStats(nil)
	// Should not panic; zero values acceptable
	_ = s
}

func TestComputeStats_Single(t *testing.T) {
	s := computeStats([]float64{42})
	if s.Mean != 42 {
		t.Errorf("mean: got %.2f, want 42", s.Mean)
	}
	if s.CVPct != 0 {
		t.Errorf("CV for single value: got %.4f, want 0", s.CVPct)
	}
}

func TestComputeStats_KnownCV(t *testing.T) {
	// mean=100, stddev≈50, CV≈50%
	s := computeStats([]float64{50, 100, 150})
	if s.Mean != 100 {
		t.Errorf("mean: got %.2f, want 100", s.Mean)
	}
	if s.CVPct < 40 || s.CVPct > 60 {
		t.Errorf("CV: got %.2f, want ~50", s.CVPct)
	}
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
go test ./cmd/bench-sweep/... 2>&1 | head -20
```

Expected: compile error `undefined: computeStats`.

- [ ] **Step 3: Implement stats.go**

Create `cmd/bench-sweep/stats.go`:

```go
package main

import (
	"math"
	"sort"
)

// MetricStats holds summary statistics for a single metric across epochs.
type MetricStats struct {
	Mean   float64 `json:"mean"`
	Median float64 `json:"median"`
	P99    float64 `json:"p99"`
	Stddev float64 `json:"stddev"`
	CVPct  float64 `json:"cv_pct"`
}

// computeStats returns summary statistics for values.
// With N < 100 values, P99 equals the maximum value.
// Returns zero MetricStats for an empty slice.
func computeStats(values []float64) MetricStats {
	n := len(values)
	if n == 0 {
		return MetricStats{}
	}

	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(n)

	variance := 0.0
	for _, v := range values {
		d := v - mean
		variance += d * d
	}
	if n > 1 {
		variance /= float64(n - 1)
	}
	stddev := math.Sqrt(variance)

	sorted := make([]float64, n)
	copy(sorted, values)
	sort.Float64s(sorted)

	var median float64
	if n%2 == 0 {
		median = (sorted[n/2-1] + sorted[n/2]) / 2
	} else {
		median = sorted[n/2]
	}

	idx := int(math.Ceil(float64(n)*0.99)) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= n {
		idx = n - 1
	}
	p99 := sorted[idx]

	cvPct := 0.0
	if mean > 0 {
		cvPct = stddev / mean * 100
	}

	return MetricStats{
		Mean:   mean,
		Median: median,
		P99:    p99,
		Stddev: stddev,
		CVPct:  cvPct,
	}
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
go test ./cmd/bench-sweep/... -run TestComputeStats -v
```

Expected: all `TestComputeStats_*` tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cmd/bench-sweep/stats.go cmd/bench-sweep/stats_test.go
git commit -m "bench-sweep: add stats package (mean/median/p99/stddev/CV%)"
```

---

## Task 3: history.go

**Files:**
- Create: `cmd/bench-sweep/history.go`
- Create: `cmd/bench-sweep/history_test.go`

- [ ] **Step 1: Write the failing tests**

Create `cmd/bench-sweep/history_test.go`:

```go
package main

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func withTempHistoryDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	historyDirOverride = filepath.Join(dir, ".ollama", "bench")
	t.Cleanup(func() { historyDirOverride = "" })
	return historyDirOverride
}

func TestResolveRunName_NoConflict(t *testing.T) {
	dir := t.TempDir()
	chosen, renamed := resolveRunName(dir, "baseline")
	if chosen != "baseline" {
		t.Errorf("got %q, want %q", chosen, "baseline")
	}
	if renamed {
		t.Error("should not be renamed when no conflict exists")
	}
}

func TestResolveRunName_OneConflict(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "baseline.json"), []byte("{}"), 0644)
	chosen, renamed := resolveRunName(dir, "baseline")
	if chosen != "baseline_1" {
		t.Errorf("got %q, want baseline_1", chosen)
	}
	if !renamed {
		t.Error("should be renamed")
	}
}

func TestResolveRunName_MultipleConflicts(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "baseline.json"), []byte("{}"), 0644)
	os.WriteFile(filepath.Join(dir, "baseline_1.json"), []byte("{}"), 0644)
	chosen, _ := resolveRunName(dir, "baseline")
	if chosen != "baseline_2" {
		t.Errorf("got %q, want baseline_2", chosen)
	}
}

func TestSaveAndLoadRun(t *testing.T) {
	withTempHistoryDir(t)
	rec := RunRecord{
		Name:      "test-run",
		Model:     "qwen3",
		Timestamp: time.Date(2026, 4, 2, 10, 0, 0, 0, time.UTC),
		Hardware:  Hardware{OS: runtime.GOOS, VRAMUsedBytes: 1000},
		Config:    RunConfig{Epochs: 6, Warmup: 2, MaxTokens: 16, CVThreshPct: 5.0, Sizes: []int{512, 1024}},
		Results: []SizeResult{
			{PromptTokens: 512, Stable: true},
			{PromptTokens: 1024, Stable: false},
		},
	}

	if err := saveRun(rec); err != nil {
		t.Fatalf("saveRun: %v", err)
	}

	loaded, err := loadRun("test-run")
	if err != nil {
		t.Fatalf("loadRun: %v", err)
	}
	if loaded.Name != "test-run" {
		t.Errorf("name: got %q, want test-run", loaded.Name)
	}
	if len(loaded.Results) != 2 {
		t.Fatalf("results len: got %d, want 2", len(loaded.Results))
	}
	if loaded.Results[0].PromptTokens != 512 || !loaded.Results[0].Stable {
		t.Errorf("result[0] mismatch: %+v", loaded.Results[0])
	}
}

func TestLoadRun_NotFound(t *testing.T) {
	withTempHistoryDir(t)
	_, err := loadRun("nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent run, got nil")
	}
}

func TestListRuns_Empty(t *testing.T) {
	withTempHistoryDir(t)
	records, err := listRuns()
	if err != nil {
		t.Fatalf("listRuns: %v", err)
	}
	if len(records) != 0 {
		t.Errorf("expected 0 records, got %d", len(records))
	}
}

func TestListRuns_Multiple(t *testing.T) {
	withTempHistoryDir(t)
	for i, name := range []string{"run-a", "run-b", "run-c"} {
		rec := RunRecord{
			Name:      name,
			Model:     "m",
			Timestamp: time.Date(2026, 4, 2, i, 0, 0, 0, time.UTC),
			Config:    RunConfig{Sizes: []int{512}},
		}
		if err := saveRun(rec); err != nil {
			t.Fatalf("saveRun %s: %v", name, err)
		}
	}
	records, err := listRuns()
	if err != nil {
		t.Fatalf("listRuns: %v", err)
	}
	if len(records) != 3 {
		t.Errorf("expected 3 records, got %d", len(records))
	}
	// Should be sorted newest first: run-c (hour=2) > run-b (hour=1) > run-a (hour=0)
	if records[0].Name != "run-c" {
		t.Errorf("first record should be run-c (newest), got %s", records[0].Name)
	}
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
go test ./cmd/bench-sweep/... -run TestResolveRunName -v 2>&1 | head -10
```

Expected: compile error `undefined: resolveRunName` (and other undefined symbols).

- [ ] **Step 3: Implement history.go**

Create `cmd/bench-sweep/history.go`:

```go
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

// historyDirOverride is set in tests to redirect history storage to a temp dir.
var historyDirOverride string

// RunConfig holds the parameters used for a benchmark run.
type RunConfig struct {
	Epochs      int     `json:"epochs"`
	Warmup      int     `json:"warmup"`
	MaxTokens   int     `json:"max_tokens"`
	CVThreshPct float64 `json:"cv_threshold_pct"`
	Sizes       []int   `json:"sizes"`
}

// Hardware captures the hardware state at benchmark time.
type Hardware struct {
	OS             string `json:"os"`
	VRAMUsedBytes  int64  `json:"vram_used_bytes"`
	VRAMTotalBytes int64  `json:"vram_total_bytes,omitempty"`
}

// EpochResult holds per-request raw metrics for one timed epoch.
type EpochResult struct {
	PromptEvalCount int     `json:"prompt_eval_count"`
	EvalCount       int     `json:"eval_count"`
	PrefillMs       float64 `json:"prefill_ms"`
	GenMs           float64 `json:"gen_ms"`
	TTFTMs          float64 `json:"ttft_ms"`
	PrefillTPS      float64 `json:"prefill_tps"`
	GenTPS          float64 `json:"gen_tps"`
}

// SizeStats holds aggregate statistics for all three metrics.
type SizeStats struct {
	PrefillTPS MetricStats `json:"prefill_tps"`
	TTFTMs     MetricStats `json:"ttft_ms"`
	GenTPS     MetricStats `json:"gen_tps"`
}

// SizeResult holds all data for one prompt size in a run.
type SizeResult struct {
	PromptTokens int           `json:"prompt_tokens"`
	Stable       bool          `json:"stable"`
	Epochs       []EpochResult `json:"epochs"`
	Stats        SizeStats     `json:"stats"`
}

// RunRecord is the top-level structure stored as JSON for each named run.
type RunRecord struct {
	Name      string       `json:"name"`
	Model     string       `json:"model"`
	Timestamp time.Time    `json:"timestamp"`
	Hardware  Hardware     `json:"hardware"`
	Config    RunConfig    `json:"config"`
	Results   []SizeResult `json:"results"`
}

// historyDir returns the path to the bench history directory, creating it if needed.
func historyDir() (string, error) {
	if historyDirOverride != "" {
		if err := os.MkdirAll(historyDirOverride, 0o755); err != nil {
			return "", err
		}
		return historyDirOverride, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("cannot find home directory: %w", err)
	}
	dir := filepath.Join(home, ".ollama", "bench")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("cannot create history directory %s: %w", dir, err)
	}
	return dir, nil
}

// resolveRunName returns name if <dir>/<name>.json does not exist.
// Otherwise returns name_1, name_2, etc. and renamed=true.
func resolveRunName(dir, name string) (chosen string, renamed bool) {
	candidate := name
	for i := 1; ; i++ {
		if _, err := os.Stat(filepath.Join(dir, candidate+".json")); os.IsNotExist(err) {
			return candidate, candidate != name
		}
		candidate = fmt.Sprintf("%s_%d", name, i)
	}
}

// saveRun writes rec to <historyDir>/<rec.Name>.json.
func saveRun(rec RunRecord) error {
	dir, err := historyDir()
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal run record: %w", err)
	}
	return os.WriteFile(filepath.Join(dir, rec.Name+".json"), data, 0o644)
}

// loadRun reads and parses <historyDir>/<name>.json.
func loadRun(name string) (RunRecord, error) {
	dir, err := historyDir()
	if err != nil {
		return RunRecord{}, err
	}
	path := filepath.Join(dir, name+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return RunRecord{}, fmt.Errorf("run %q not found", name)
		}
		return RunRecord{}, err
	}
	var rec RunRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return RunRecord{}, fmt.Errorf("parse %s: %w", path, err)
	}
	return rec, nil
}

// listRuns returns all run records from historyDir, sorted newest first.
func listRuns() ([]RunRecord, error) {
	dir, err := historyDir()
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var records []RunRecord
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		name := strings.TrimSuffix(e.Name(), ".json")
		rec, err := loadRun(name)
		if err != nil {
			continue // skip corrupt files silently
		}
		records = append(records, rec)
	}
	sort.Slice(records, func(i, j int) bool {
		return records[i].Timestamp.After(records[j].Timestamp)
	})
	return records, nil
}

func currentOS() string {
	return runtime.GOOS
}
```

- [ ] **Step 4: Run tests**

```bash
go test ./cmd/bench-sweep/... -run "TestResolveRunName|TestSaveAndLoad|TestLoadRun|TestListRuns" -v
```

Expected: all 8 history tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cmd/bench-sweep/history.go cmd/bench-sweep/history_test.go
git commit -m "bench-sweep: add history storage (JSON, run naming, list)"
```

---

## Task 4: prompt.go

**Files:**
- Create: `cmd/bench-sweep/prompt.go`
- Create: `cmd/bench-sweep/prompt_test.go`

Prerequisite: `cmd/bench-sweep/corpus.txt` must exist (Task 1).

- [ ] **Step 1: Write the failing tests**

Create `cmd/bench-sweep/prompt_test.go`:

```go
package main

import (
	"testing"
)

func TestPromptText_ExactLength(t *testing.T) {
	for _, n := range []int{100, 500, 1000, 4000} {
		got := promptText(n, 0)
		if len(got) != n {
			t.Errorf("promptText(%d, 0): got len %d, want %d", n, len(got), n)
		}
	}
}

func TestPromptText_VariesByEpoch(t *testing.T) {
	t0 := promptText(500, 0)
	t1 := promptText(500, 1)
	t2 := promptText(500, 2)
	if t0 == t1 || t1 == t2 || t0 == t2 {
		t.Error("expected different text for different epochs")
	}
}

func TestPromptText_DifferentPrefix(t *testing.T) {
	// The first 100 chars must differ across epochs to defeat KV cache matching
	t0 := promptText(500, 0)
	t1 := promptText(500, 1)
	if len(t0) >= 100 && len(t1) >= 100 && t0[:100] == t1[:100] {
		t.Error("epoch 0 and 1 share first 100 chars — KV cache defeat may fail")
	}
}

func TestPromptText_LargerThanCorpus(t *testing.T) {
	big := len(corpus)*2 + 100
	got := promptText(big, 0)
	if len(got) != big {
		t.Errorf("expected %d chars, got %d", big, len(got))
	}
}

func TestPromptText_Zero(t *testing.T) {
	if promptText(0, 0) != "" {
		t.Error("expected empty string for zero length")
	}
}

func TestCalibrateChars_ScalesUp(t *testing.T) {
	// asked 4000 chars → got 800 tokens, want 1000 → scale to 5000
	result := calibrateChars(4000, 1000, 800)
	if result != 5000 {
		t.Errorf("got %d, want 5000", result)
	}
}

func TestCalibrateChars_ScalesDown(t *testing.T) {
	// asked 4000 chars → got 1200 tokens, want 1000 → scale to 3333
	result := calibrateChars(4000, 1000, 1200)
	if result != 3333 {
		t.Errorf("got %d, want 3333", result)
	}
}

func TestCalibrateChars_ZeroActual(t *testing.T) {
	// Should return unchanged when actualTokens=0
	result := calibrateChars(4000, 1000, 0)
	if result != 4000 {
		t.Errorf("got %d, want 4000", result)
	}
}

func TestInitialChars(t *testing.T) {
	if initialChars(512) != 2048 {
		t.Errorf("expected 2048 for 512 tokens, got %d", initialChars(512))
	}
	if initialChars(4096) != 16384 {
		t.Errorf("expected 16384 for 4096 tokens, got %d", initialChars(4096))
	}
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
go test ./cmd/bench-sweep/... -run TestPromptText -v 2>&1 | head -10
```

Expected: compile error `undefined: promptText`.

- [ ] **Step 3: Implement prompt.go**

Create `cmd/bench-sweep/prompt.go`:

```go
package main

import _ "embed"

//go:embed corpus.txt
var corpus string

// promptText returns exactly targetChars characters from the corpus,
// starting at an offset derived from epoch (using prime 7919 for distribution).
// Wraps around the corpus when targetChars > len(corpus).
func promptText(targetChars int, epoch int) string {
	n := len(corpus)
	if n == 0 || targetChars <= 0 {
		return ""
	}
	offset := (epoch * 7919) % n
	buf := make([]byte, 0, targetChars)
	pos := offset
	for len(buf) < targetChars {
		remaining := targetChars - len(buf)
		available := n - pos
		take := min(remaining, available)
		buf = append(buf, corpus[pos:pos+take]...)
		pos = (pos + take) % n
	}
	return string(buf)
}

// calibrateChars scales charCount so that the next prompt will hit targetTokens,
// given that the current charCount produced actualTokens.
// Returns charCount unchanged if actualTokens is zero.
func calibrateChars(charCount, targetTokens, actualTokens int) int {
	if actualTokens == 0 {
		return charCount
	}
	return int(float64(charCount) * float64(targetTokens) / float64(actualTokens))
}

// initialChars returns the starting character-count estimate for targetTokens.
// Uses 4 chars/token as a conservative estimate for English prose.
func initialChars(targetTokens int) int {
	return targetTokens * 4
}
```

- [ ] **Step 4: Run tests**

```bash
go test ./cmd/bench-sweep/... -run "TestPromptText|TestCalibrateChars|TestInitialChars" -v
```

Expected: all prompt tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cmd/bench-sweep/prompt.go cmd/bench-sweep/prompt_test.go
git commit -m "bench-sweep: add corpus-based prompt generator with epoch variation"
```

---

## Task 5: run.go

**Files:**
- Create: `cmd/bench-sweep/run.go`
- Create: `cmd/bench-sweep/run_test.go`

- [ ] **Step 1: Write the failing tests**

Create `cmd/bench-sweep/run_test.go`:

```go
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
	printRunTable(&sb, "test-model", results, RunConfig{Epochs: 6, Warmup: 2})
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
	printRunTable(&sb, "test-model", results, RunConfig{Epochs: 6, Warmup: 2})
	out := sb.String()
	if !strings.Contains(out, "⚠") {
		t.Errorf("expected ⚠ for unstable result, got:\n%s", out)
	}
	if !strings.Contains(out, "8.7") {
		t.Errorf("expected CV% 8.7 in output, got:\n%s", out)
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
	// Model name without tag should still match "qwen3:latest" via prefix
	hw := fetchVRAM(context.Background(), client, "qwen3")
	if hw.VRAMUsedBytes != 22e9 {
		t.Errorf("VRAMUsedBytes via prefix match: got %d, want 22e9", hw.VRAMUsedBytes)
	}
}

// Helpers used by the server mock
var _ = fmt.Sprintf
var _ = io.Discard
```

- [ ] **Step 2: Run to verify it fails**

```bash
go test ./cmd/bench-sweep/... -run "TestSendRequest|TestRunBenchmark|TestPrintRunTable|TestFetchVRAM" -v 2>&1 | head -15
```

Expected: compile error `undefined: sendRequest` (and others).

- [ ] **Step 3: Implement run.go**

Create `cmd/bench-sweep/run.go`:

```go
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
func sendRequest(ctx context.Context, client *api.Client, model, prompt string, maxTokens int) (EpochResult, error) {
	req := &api.GenerateRequest{
		Model:  model,
		Prompt: prompt,
		Raw:    true,
		Options: map[string]interface{}{
			"temperature": 0,
			"num_predict": maxTokens,
		},
	}

	var result EpochResult
	var ttftOnce sync.Once
	start := time.Now()

	err := client.Generate(ctx, req, func(resp api.GenerateResponse) error {
		ttftOnce.Do(func() {
			if resp.Response != "" || resp.Thinking != "" {
				result.TTFTMs = float64(time.Since(start).Nanoseconds()) / 1e6
			}
		})
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
			hw.VRAMTotalBytes = m.Size
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
	for _, size := range cfg.Sizes {
		chars := initialChars(size)

		// Warmup phase: run cfg.Warmup requests, calibrate on first, track tps for adequacy check
		var warmupTPS []float64
		for i := 0; i < cfg.Warmup; i++ {
			wCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
			result, err := sendRequest(wCtx, client, model, promptText(chars, -(i+1)), cfg.MaxTokens)
			cancel()
			if err != nil {
				fmt.Fprintf(os.Stderr, "Warning: warmup %d/%d for size=%d failed: %v\n", i+1, cfg.Warmup, size, err)
				continue
			}
			warmupTPS = append(warmupTPS, result.PrefillTPS)
			// Calibrate on first warmup
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
			result, err := sendRequest(eCtx, client, model, promptText(chars, epoch), cfg.MaxTokens)
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

		// Compute stats
		prefillVals := make([]float64, len(epochResults))
		ttftVals := make([]float64, len(epochResults))
		genVals := make([]float64, len(epochResults))
		for i, r := range epochResults {
			prefillVals[i] = r.PrefillTPS
			ttftVals[i] = r.TTFTMs
			genVals[i] = r.GenTPS
		}
		stats := SizeStats{
			PrefillTPS: computeStats(prefillVals),
			TTFTMs:     computeStats(ttftVals),
			GenTPS:     computeStats(genVals),
		}
		stable := stats.PrefillTPS.CVPct <= cfg.CVThreshPct && stats.TTFTMs.CVPct <= cfg.CVThreshPct

		allResults = append(allResults, SizeResult{
			PromptTokens: size,
			Stable:       stable,
			Epochs:       epochResults,
			Stats:        stats,
		})

		// Print row immediately so the user sees progress
		printRunTable(os.Stdout, model, allResults[len(allResults)-1:], cfg)

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

// printRunTable writes a formatted result row (or header if it is the first call).
// Pass a slice of length 1 to print a single row after each size completes.
func printRunTable(w io.Writer, model string, results []SizeResult, cfg RunConfig) {
	if len(results) == 0 {
		return
	}
	// Print header only for first row
	if len(results) == 1 {
		fmt.Fprintf(w, "\nModel: %s  |  Epochs: %d  |  Warmup: %d\n\n", model, cfg.Epochs, cfg.Warmup)
		fmt.Fprintf(w, "%-13s │ %-18s │ %-18s │ %6s │ %-10s │ %-10s │ %6s │ %-8s │ %s\n",
			"prompt_tokens",
			"prefill_tps (mean)",
			"prefill_tps (p99)",
			"CV%",
			"TTFT mean",
			"TTFT p99",
			"CV%",
			"gen_tps",
			"status",
		)
		fmt.Fprintln(w, strings.Repeat("─", 110))
	}
	r := results[len(results)-1]
	status := "✓"
	if !r.Stable {
		status = "⚠"
	}
	fmt.Fprintf(w, "%-13d │ %15.0f t/s    │ %15.0f t/s    │ %5.1f%% │ %7.0f ms │ %7.0f ms │ %5.1f%% │ %5.0f t/s │ %s\n",
		r.PromptTokens,
		r.Stats.PrefillTPS.Mean,
		r.Stats.PrefillTPS.P99,
		r.Stats.PrefillTPS.CVPct,
		r.Stats.TTFTMs.Mean,
		r.Stats.TTFTMs.P99,
		r.Stats.TTFTMs.CVPct,
		r.Stats.GenTPS.Mean,
		status,
	)
}
```

- [ ] **Step 4: Run tests**

```bash
go test ./cmd/bench-sweep/... -run "TestSendRequest|TestRunBenchmark|TestPrintRunTable|TestFetchVRAM" -v
```

Expected: all run tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cmd/bench-sweep/run.go cmd/bench-sweep/run_test.go
git commit -m "bench-sweep: add sweep loop, sendRequest, fetchVRAM, run table"
```

---

## Task 6: diff.go

**Files:**
- Create: `cmd/bench-sweep/diff.go`
- Create: `cmd/bench-sweep/diff_test.go`

- [ ] **Step 1: Write the failing tests**

Create `cmd/bench-sweep/diff_test.go`:

```go
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

	// Should contain both run names
	if !strings.Contains(out, "baseline") || !strings.Contains(out, "optimized") {
		t.Errorf("expected both run names in diff output:\n%s", out)
	}
	// Should show prefill values
	if !strings.Contains(out, "4000") || !strings.Contains(out, "4400") {
		t.Errorf("expected prefill values in diff output:\n%s", out)
	}
	// Should show positive delta for prefill (improvement)
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
```

- [ ] **Step 2: Run to verify it fails**

```bash
go test ./cmd/bench-sweep/... -run TestPrintDiff -v 2>&1 | head -10
```

Expected: compile error `undefined: printDiff`.

- [ ] **Step 3: Implement diff.go**

Create `cmd/bench-sweep/diff.go`:

```go
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
```

- [ ] **Step 4: Run tests**

```bash
go test ./cmd/bench-sweep/... -run TestPrintDiff -v
```

Expected: all diff tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cmd/bench-sweep/diff.go cmd/bench-sweep/diff_test.go
git commit -m "bench-sweep: add diff subcommand"
```

---

## Task 7: list.go

**Files:**
- Create: `cmd/bench-sweep/list.go`
- Create: `cmd/bench-sweep/list_test.go`

- [ ] **Step 1: Write the failing tests**

Create `cmd/bench-sweep/list_test.go`:

```go
package main

import (
	"strings"
	"testing"
	"time"
)

func TestPrintList_Headers(t *testing.T) {
	var sb strings.Builder
	printList(&sb, []RunRecord{})
	// Empty list prints "No benchmark runs found."
	if !strings.Contains(sb.String(), "No benchmark runs found") {
		t.Errorf("expected empty message, got: %s", sb.String())
	}
}

func TestPrintList_ShowsRuns(t *testing.T) {
	records := []RunRecord{
		{
			Name:      "baseline",
			Model:     "qwen3",
			Timestamp: time.Date(2026, 4, 2, 10, 0, 0, 0, time.UTC),
			Config:    RunConfig{Sizes: []int{512, 1024, 2048}},
			Results: []SizeResult{
				{Stable: true},
				{Stable: true},
				{Stable: false},
			},
		},
	}
	var sb strings.Builder
	printList(&sb, records)
	out := sb.String()

	for _, want := range []string{"baseline", "qwen3", "2026-04-02", "512,1024,2048", "2/3"} {
		if !strings.Contains(out, want) {
			t.Errorf("expected %q in list output:\n%s", want, out)
		}
	}
}

func TestPrintList_ColumnHeaders(t *testing.T) {
	records := []RunRecord{{
		Name: "r", Model: "m", Timestamp: time.Now(),
		Config: RunConfig{Sizes: []int{512}},
	}}
	var sb strings.Builder
	printList(&sb, records)
	out := sb.String()
	for _, col := range []string{"NAME", "MODEL", "DATE", "SIZES", "STABLE"} {
		if !strings.Contains(out, col) {
			t.Errorf("expected column header %q:\n%s", col, out)
		}
	}
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
go test ./cmd/bench-sweep/... -run TestPrintList -v 2>&1 | head -10
```

Expected: compile error `undefined: printList`.

- [ ] **Step 3: Implement list.go**

Create `cmd/bench-sweep/list.go`:

```go
package main

import (
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

func runList(_ []string) error {
	records, err := listRuns()
	if err != nil {
		return err
	}
	printList(os.Stdout, records)
	return nil
}

func printList(w io.Writer, records []RunRecord) {
	if len(records) == 0 {
		fmt.Fprintln(w, "No benchmark runs found.")
		return
	}
	fmt.Fprintf(w, "%-24s %-28s %-12s %-20s %s\n", "NAME", "MODEL", "DATE", "SIZES", "STABLE")
	fmt.Fprintln(w, strings.Repeat("─", 95))
	for _, rec := range records {
		sizes := make([]string, len(rec.Config.Sizes))
		for i, s := range rec.Config.Sizes {
			sizes[i] = strconv.Itoa(s)
		}
		stableCount := 0
		for _, r := range rec.Results {
			if r.Stable {
				stableCount++
			}
		}
		fmt.Fprintf(w, "%-24s %-28s %-12s %-20s %d/%d\n",
			rec.Name,
			rec.Model,
			rec.Timestamp.Format("2006-01-02"),
			strings.Join(sizes, ","),
			stableCount,
			len(rec.Results),
		)
	}
}
```

- [ ] **Step 4: Run tests**

```bash
go test ./cmd/bench-sweep/... -run TestPrintList -v
```

Expected: all list tests PASS.

- [ ] **Step 5: Run all tests to confirm nothing regressed**

```bash
go test ./cmd/bench-sweep/... -v 2>&1 | tail -20
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add cmd/bench-sweep/list.go cmd/bench-sweep/list_test.go
git commit -m "bench-sweep: add list subcommand"
```

---

## Task 8: main.go

**Files:**
- Create: `cmd/bench-sweep/main.go`

No tests needed for main (it is pure dispatch with no logic).

- [ ] **Step 1: Implement main.go**

Create `cmd/bench-sweep/main.go`:

```go
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}
	var err error
	switch os.Args[1] {
	case "run":
		err = runRun(os.Args[2:])
	case "diff":
		err = runDiff(os.Args[2:])
	case "list":
		err = runList(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "Unknown subcommand %q\n\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Fprintln(os.Stderr, "bench-sweep — Ollama inference benchmark with multi-size sweep and run history")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Usage:")
	fmt.Fprintln(os.Stderr, "  bench-sweep run  -model <model> -name <run-name> [options]")
	fmt.Fprintln(os.Stderr, "  bench-sweep diff <run-a> <run-b>")
	fmt.Fprintln(os.Stderr, "  bench-sweep list")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Run 'bench-sweep run -help' for run options.")
}

func runRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	model     := fs.String("model", "", "Ollama model name (required)")
	name      := fs.String("name", "", "Run name for history (required)")
	sizesStr  := fs.String("sizes", "512,1024,2048,4096", "Comma-separated prompt token sizes to sweep")
	epochs    := fs.Int("epochs", 6, "Number of timed iterations per size")
	warmup    := fs.Int("warmup", 2, "Warmup iterations before timing (>=2 recommended)")
	maxTokens := fs.Int("max-tokens", 16, "Max output tokens per request")
	cvThresh  := fs.Float64("cv-threshold", 5.0, "CV% threshold above which a result is flagged unstable")
	host      := fs.String("host", "", "Ollama host URL (default: $OLLAMA_HOST or http://localhost:11434)")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if *model == "" {
		return fmt.Errorf("-model is required")
	}
	if *name == "" {
		return fmt.Errorf("-name is required")
	}

	var sizes []int
	for _, s := range strings.Split(*sizesStr, ",") {
		s = strings.TrimSpace(s)
		n, err := strconv.Atoi(s)
		if err != nil || n <= 0 {
			return fmt.Errorf("invalid size %q: must be a positive integer", s)
		}
		sizes = append(sizes, n)
	}

	if *host != "" {
		os.Setenv("OLLAMA_HOST", *host)
	}
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return fmt.Errorf("cannot create Ollama client: %w", err)
	}

	cfg := RunConfig{
		Epochs:      *epochs,
		Warmup:      *warmup,
		MaxTokens:   *maxTokens,
		CVThreshPct: *cvThresh,
		Sizes:       sizes,
	}

	dir, err := historyDir()
	if err != nil {
		return err
	}
	chosen, renamed := resolveRunName(dir, *name)
	if renamed {
		fmt.Fprintf(os.Stderr, "Warning: run name %q already exists, renamed to %q\n", *name, chosen)
	}

	fmt.Printf("Starting benchmark: model=%s  sizes=%s  epochs=%d  warmup=%d\n",
		*model, *sizesStr, *epochs, *warmup)

	sizeResults, hw, err := runBenchmark(context.Background(), client, *model, cfg)
	if err != nil {
		return err
	}

	rec := RunRecord{
		Name:      chosen,
		Model:     *model,
		Timestamp: time.Now().UTC(),
		Hardware:  hw,
		Config:    cfg,
		Results:   sizeResults,
	}
	if err := saveRun(rec); err != nil {
		return fmt.Errorf("save run: %w", err)
	}
	fmt.Printf("\nRun %q saved to %s\n", chosen, filepath.Join(dir, chosen+".json"))
	return nil
}
```

- [ ] **Step 2: Build to verify it compiles**

```bash
go build ./cmd/bench-sweep/
```

Expected: `bench-sweep.exe` (or `bench-sweep` on Linux/Mac) is created in the current directory with no errors.

- [ ] **Step 3: Run all tests one final time**

```bash
go test ./cmd/bench-sweep/... -v 2>&1 | tail -30
```

Expected: all tests PASS, no failures.

- [ ] **Step 4: Quick smoke test (requires ollama serve running)**

```bash
./bench-sweep run -model qwen3-coder-next -name smoke-test -sizes 128 -epochs 2 -warmup 1
```

Expected: prints a table with one row (size=128) and saves a JSON file.

- [ ] **Step 5: Commit**

```bash
git add cmd/bench-sweep/main.go
git commit -m "bench-sweep: add main entry point and run subcommand"
```

---

## Task 9: README.md

**Files:**
- Create: `cmd/bench-sweep/README.md`

- [ ] **Step 1: Create README.md**

Create `cmd/bench-sweep/README.md`:

````markdown
# bench-sweep

Repeatable inference benchmark for Ollama. Sweeps multiple prompt sizes in one run, measures prefill throughput (tokens/s) and TTFT, flags unstable results via CV%, and stores named run history for cross-run comparison.

---

## Build from Source

Prerequisites: Go 1.24+, this repo checked out.

**Windows (PowerShell):**
```powershell
go build -o bench-sweep.exe ./cmd/bench-sweep/
```

**Linux / macOS:**
```bash
go build -o bench-sweep ./cmd/bench-sweep/
```

The binary has no runtime dependencies beyond `ollama serve` running.

---

## Prerequisites

Start Ollama with the model you want to benchmark:
```bash
ollama serve
ollama pull qwen3-coder-next
```

To benchmark the Go-native OllamaRunner engine (if the model supports it):
```powershell
$env:OLLAMA_NEW_ENGINE = "1"
ollama serve
```

---

## Subcommands

### `run` — Execute a benchmark sweep

```bash
bench-sweep run -model <model> -name <run-name> [options]
```

| Flag | Default | Description |
|---|---|---|
| `-model` | (required) | Ollama model name |
| `-name` | (required) | Name for this run; auto-renamed `_1`, `_2`… on conflict |
| `-sizes` | `512,1024,2048,4096` | Comma-separated prompt token sizes to sweep |
| `-epochs` | `6` | Timed iterations per size |
| `-warmup` | `2` | Warmup iterations before timing (>=2 recommended) |
| `-max-tokens` | `16` | Max output tokens per request (keep small to isolate prefill) |
| `-cv-threshold` | `5.0` | CV% above which a result is flagged ⚠ unstable |
| `-host` | `$OLLAMA_HOST` | Ollama server URL |

**Example:**
```bash
bench-sweep run -model qwen3-coder-next -name baseline -sizes 512,1024,2048,4096
```

**Output:**
```
Starting benchmark: model=qwen3-coder-next  sizes=512,1024,2048,4096  epochs=6  warmup=2

Model: qwen3-coder-next  |  Epochs: 6  |  Warmup: 2

prompt_tokens │ prefill_tps (mean) │ prefill_tps (p99) │   CV% │ TTFT mean │ TTFT p99 │   CV% │ gen_tps │ status
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
          512 │         4,850 t/s  │         4,720 t/s │  1.8% │     28 ms │    35 ms │  2.3% │  37 t/s │ ✓
        1,024 │         4,266 t/s  │         4,100 t/s │  2.1% │     52 ms │    64 ms │  2.8% │  37 t/s │ ✓
        2,048 │         4,180 t/s  │         4,050 t/s │  2.4% │    103 ms │   121 ms │  3.1% │  37 t/s │ ✓
        4,096 │         3,890 t/s  │         3,200 t/s │  8.7% │    198 ms │   240 ms │  9.2% │  36 t/s │ ⚠

⚠ [size=4096] prefill_tps CV=8.7% exceeds threshold 5.0%
  hint: consider increasing -warmup (current: 2) or closing background processes

Run "baseline" saved to C:\Users\you\.ollama\bench\baseline.json
```

### `diff` — Compare two runs

```bash
bench-sweep diff <run-a> <run-b>
```

**Example:**
```bash
bench-sweep diff baseline cpu-affinity
```

**Output:**
```
Diff: baseline → cpu-affinity  |  Model: qwen3-coder-next
Note: Δ% negative = improvement for TTFT (lower is better); positive = improvement for prefill_tps (higher is better)

prompt_tokens │ prefill_tps baseline→new      │    Δ%   │ TTFT mean baseline→new  │ TTFT p99 baseline→new  │    Δ%   │ note
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
          512 │    4,850 → 5,120 t/s           │  +5.6%  │      28 → 25 ms         │      35 → 30 ms        │ -14.3%  │
        1,024 │    4,266 → 4,580 t/s           │  +7.4%  │      52 → 47 ms         │      64 → 55 ms        │ -14.1%  │
        4,096 │    3,890 → 4,200 t/s           │  +8.0%  │     198 → 181 ms        │     240 → 215 ms       │ -10.4%  │ ⚠ baseline CV=8.7%
```

If a run name conflicts, rename it before diffing:
```bash
# If you need to diff two runs named the same, use the auto-renamed version
bench-sweep list   # see actual stored names
bench-sweep diff baseline_1 baseline_2
```

### `list` — Show all stored runs

```bash
bench-sweep list
```

**Output:**
```
NAME                     MODEL                        DATE         SIZES                STABLE
────────────────────────────────────────────────────────────────────────────────────────────────
cpu-affinity             qwen3-coder-next             2026-04-02   512,1024,2048,4096   4/4
baseline                 qwen3-coder-next             2026-04-02   512,1024,2048,4096   3/4
```

Run history is stored in `~/.ollama/bench/` as JSON files.

---

## Understanding Results

### prefill_tps
Tokens per second during the prefill phase, derived from Ollama's `prompt_eval_duration` metric. Higher is better. Measures how fast the model processes your input prompt.

### TTFT (Time to First Token)
Wall-clock milliseconds from request start to first output token. Combines prefill time and any scheduling/overhead. Lower is better.

### CV% (Coefficient of Variation)
`stddev / mean × 100`. Measures run-to-run stability. A result is flagged ⚠ if CV% > threshold (default 5%). High CV% means the measurement is noisy — results may not reflect real optimization differences.

### p99
With the default 6 epochs, p99 equals the worst observed value (maximum). It is most useful with `-epochs 20+` where it captures tail latency.

### Stability warnings
If CV% exceeds `-cv-threshold` for either prefill_tps or TTFT at a given size, that row is marked ⚠ and the run's entry in `diff` output will note the instability.

**If you see instability warnings:**
1. Increase `-warmup` (try `-warmup 4`)
2. Close browser, background services, and other GPU workloads
3. On Windows, check Task Manager for CPU/GPU spikes

---

## Design Notes

**Why a separate tool from `cmd/bench`?**  
`cmd/bench` is upstream Ollama code kept in sync with the main project. `bench-sweep` adds opinionated features (multi-size sweep, history, CV% checks) without polluting the upstream tool.

**Why use Ollama's `prompt_eval_duration` rather than wall-clock prefill time?**  
`prompt_eval_duration` is measured inside the Ollama server, excluding HTTP and scheduling overhead. It is more stable and accurate for comparing different model configurations. TTFT (wall-clock) is still measured separately for user-experience context.

**Why only 16 output tokens by default?**  
The goal is to stress prefill. Generating fewer output tokens means less time spent in the decode phase, making the benchmark faster and keeping prefill as the dominant cost.

**Why vary the prompt per epoch?**  
Ollama caches KV state for prompts with matching prefixes. Without variation, every epoch after the first would get a cache hit and show unrealistically fast prefill. The corpus-based generator ensures each epoch has a different starting offset.

**How is the prompt token count calibrated?**  
The first warmup request is sent with a heuristic prompt length (4 chars/token). The actual `prompt_eval_count` from the response is used to compute a scaling factor, which is applied to all subsequent requests for that size. Accuracy is typically ±5 tokens.
````

- [ ] **Step 2: Build one final time to confirm everything compiles**

```bash
go build ./cmd/bench-sweep/
go test ./cmd/bench-sweep/... 2>&1 | tail -5
```

Expected: build succeeds, all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add cmd/bench-sweep/README.md
git commit -m "bench-sweep: add README with build instructions and usage guide"
```

---

## Self-Review

**Spec coverage:**
- ✅ `run` subcommand with multi-size sweep — Task 5 + 8
- ✅ Named runs, auto-rename on conflict — Task 3 + 8
- ✅ `diff` subcommand with prefill_tps and TTFT columns — Task 6
- ✅ `list` subcommand — Task 7
- ✅ CV% stability check + warnings — Task 5
- ✅ Warmup adequacy check — Task 5
- ✅ Corpus-based exact-length prompt generator with calibration — Task 4
- ✅ JSON history file with full schema — Task 3
- ✅ VRAM fetched from `/api/ps` — Task 5
- ✅ README with build instructions — Task 9
- ✅ Runner path selection note — in README

**Type consistency check:**
- `MetricStats` defined in `stats.go`, referenced in `history.go` (`SizeStats`) — ✅ same package
- `EpochResult`, `SizeResult`, `RunRecord`, `RunConfig`, `Hardware` all defined in `history.go` — ✅
- `sendRequest` returns `EpochResult` — used in `run.go` — ✅
- `runBenchmark` returns `([]SizeResult, Hardware, error)` — consumed in `main.go:runRun` — ✅
- `printRunTable(w io.Writer, model string, results []SizeResult, cfg RunConfig)` — called in `run.go` and tested in `run_test.go` — ✅
- `printDiff(w io.Writer, a, b RunRecord)` — called in `diff.go` and tested in `diff_test.go` — ✅
- `printList(w io.Writer, records []RunRecord)` — called in `list.go` and tested — ✅
