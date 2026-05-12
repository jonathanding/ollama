package daop

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type Config struct {
	Enabled             bool     `json:"enabled"`
	ProbeModel          string   `json:"probe_model"`
	ProbeLayer          int      `json:"probe_layer"`
	MFWeights           string   `json:"mf_weights"`
	GateStats           string   `json:"gate_stats"`
	SubtaskClassifier   string   `json:"subtask_classifier"`
	GateThreshold       float64  `json:"gate_threshold"`
	AccuracyThreshold   float64  `json:"accuracy_threshold"`
	Temperature         float64  `json:"temperature"`
	SupportedModels     []string `json:"supported_models"`
	PrefillMsPerByte    float64  `json:"prefill_ms_per_byte"`
	PrefillBaseMs       float64  `json:"prefill_base_ms"`
}

func (c *Config) IsModelSupported(model string) bool {
	for _, m := range c.SupportedModels {
		if m == model {
			return true
		}
	}
	return false
}

// LoadConfig loads daop.json from:
//  1. <executable_dir>/daop_demo/data/daop.json (self-contained deployment)
//  2. ~/.ollama/daop.json (development fallback)
//
// Relative paths in the config are resolved against the config file's directory.
func LoadConfig() (*Config, error) {
	configPath := findConfig()
	if configPath == "" {
		return nil, nil
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("read daop.json: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse daop.json: %w", err)
	}

	if cfg.GateThreshold == 0 {
		cfg.GateThreshold = 0.80
	}
	if cfg.AccuracyThreshold == 0 {
		cfg.AccuracyThreshold = 0.80
	}
	if cfg.ProbeLayer == 0 {
		cfg.ProbeLayer = 14
	}
	if cfg.Temperature == 0 {
		cfg.Temperature = 1.21
	}

	// Resolve relative paths against config file directory
	configDir := filepath.Dir(configPath)
	cfg.ProbeModel = resolvePath(configDir, cfg.ProbeModel)
	cfg.MFWeights = resolvePath(configDir, cfg.MFWeights)
	cfg.GateStats = resolvePath(configDir, cfg.GateStats)
	cfg.SubtaskClassifier = resolvePath(configDir, cfg.SubtaskClassifier)

	return &cfg, nil
}

func findConfig() string {
	// 1. Next to executable: <exe_dir>/daop_demo/data/daop.json
	if exe, err := os.Executable(); err == nil {
		p := filepath.Join(filepath.Dir(exe), "daop_demo", "data", "daop.json")
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	// 2. ~/.ollama/daop.json
	if home, err := os.UserHomeDir(); err == nil {
		p := filepath.Join(home, ".ollama", "daop.json")
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	return ""
}

func resolvePath(base, path string) string {
	if path == "" || filepath.IsAbs(path) {
		return path
	}
	return filepath.Join(base, path)
}
