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
