package daop

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadConfig_NotExists(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("USERPROFILE", tmpDir)
	t.Setenv("HOME", tmpDir)

	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg != nil {
		t.Fatal("expected nil config when file doesn't exist")
	}
}

func TestLoadConfig_Valid(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("USERPROFILE", tmpDir)
	t.Setenv("HOME", tmpDir)

	ollamaDir := filepath.Join(tmpDir, ".ollama")
	os.MkdirAll(ollamaDir, 0755)

	configJSON := `{
		"enabled": true,
		"probe_model": "/models/qwen3-0.6b-q8_0.gguf",
		"probe_layer": 14,
		"mf_weights": "/weights/mf_weights.bin",
		"gate_stats": "/weights/gate_stats.json",
		"gate_threshold": 0.75,
		"accuracy_threshold": 0.85,
		"supported_models": ["qwen3-8b-q4", "qwen3-4b-q4"]
	}`
	os.WriteFile(filepath.Join(ollamaDir, "daop.json"), []byte(configJSON), 0644)

	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg == nil {
		t.Fatal("expected non-nil config")
	}
	if !cfg.Enabled {
		t.Error("expected enabled=true")
	}
	if cfg.GateThreshold != 0.75 {
		t.Errorf("gate_threshold: got %f, want 0.75", cfg.GateThreshold)
	}
	if cfg.AccuracyThreshold != 0.85 {
		t.Errorf("accuracy_threshold: got %f, want 0.85", cfg.AccuracyThreshold)
	}
	if len(cfg.SupportedModels) != 2 {
		t.Errorf("supported_models: got %d, want 2", len(cfg.SupportedModels))
	}
	if !cfg.IsModelSupported("qwen3-8b-q4") {
		t.Error("qwen3-8b-q4 should be supported")
	}
	if cfg.IsModelSupported("unknown-model") {
		t.Error("unknown-model should not be supported")
	}
}

func TestLoadConfig_Defaults(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("USERPROFILE", tmpDir)
	t.Setenv("HOME", tmpDir)

	ollamaDir := filepath.Join(tmpDir, ".ollama")
	os.MkdirAll(ollamaDir, 0755)

	configJSON := `{"enabled": true, "probe_model": "test.gguf", "mf_weights": "w.bin", "gate_stats": "g.json", "supported_models": ["m1"]}`
	os.WriteFile(filepath.Join(ollamaDir, "daop.json"), []byte(configJSON), 0644)

	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.GateThreshold != 0.80 {
		t.Errorf("default gate_threshold: got %f, want 0.80", cfg.GateThreshold)
	}
	if cfg.AccuracyThreshold != 0.80 {
		t.Errorf("default accuracy_threshold: got %f, want 0.80", cfg.AccuracyThreshold)
	}
	if cfg.ProbeLayer != 14 {
		t.Errorf("default probe_layer: got %d, want 14", cfg.ProbeLayer)
	}
}
