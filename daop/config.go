package daop

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type Config struct {
	Enabled           bool     `json:"enabled"`
	ProbeModel        string   `json:"probe_model"`
	ProbeLayer        int      `json:"probe_layer"`
	MFWeights         string   `json:"mf_weights"`
	GateStats         string   `json:"gate_stats"`
	GateThreshold     float64  `json:"gate_threshold"`
	AccuracyThreshold float64  `json:"accuracy_threshold"`
	SupportedModels   []string `json:"supported_models"`
}

func (c *Config) IsModelSupported(model string) bool {
	for _, m := range c.SupportedModels {
		if m == model {
			return true
		}
	}
	return false
}

func LoadConfig() (*Config, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("cannot find home dir: %w", err)
	}

	path := filepath.Join(home, ".ollama", "daop.json")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil // No config = DAOP disabled
		}
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

	return &cfg, nil
}
