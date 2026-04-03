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

func TestSanitizeModelName(t *testing.T) {
	cases := []struct{ in, want string }{
		{"qwen3", "qwen3"},
		{"qwen3:latest", "qwen3"},
		{"qwen3-coder:7b", "qwen3-coder_7b"},
		{"namespace/model:tag", "namespace_model_tag"},
		{"my model", "my_model"},
	}
	for _, c := range cases {
		got := sanitizeModelName(c.in)
		if got != c.want {
			t.Errorf("sanitizeModelName(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestResolveRunName_NoConflict(t *testing.T) {
	dir := t.TempDir()
	chosen, renamed := resolveRunName(dir, "qwen3", "baseline")
	if chosen != "baseline" {
		t.Errorf("got %q, want %q", chosen, "baseline")
	}
	if renamed {
		t.Error("should not be renamed when no conflict exists")
	}
}

func TestResolveRunName_OneConflict(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "qwen3_baseline.json"), []byte("{}"), 0644)
	chosen, renamed := resolveRunName(dir, "qwen3", "baseline")
	if chosen != "baseline_1" {
		t.Errorf("got %q, want baseline_1", chosen)
	}
	if !renamed {
		t.Error("should be renamed")
	}
}

func TestResolveRunName_MultipleConflicts(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "qwen3_baseline.json"), []byte("{}"), 0644)
	os.WriteFile(filepath.Join(dir, "qwen3_baseline_1.json"), []byte("{}"), 0644)
	chosen, _ := resolveRunName(dir, "qwen3", "baseline")
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

	loaded, err := loadRun("qwen3_test-run")
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
