package daop

import (
	"encoding/json"
	"fmt"
	"os"
)

// SubtaskGate checks if a model's historical pass rate on a subtask exceeds the threshold.
type SubtaskGate struct {
	// stats maps model_name → subtask → pass_rate
	stats     map[string]map[string]float64
	threshold float64
}

func NewSubtaskGate(statsPath string, threshold float64) (*SubtaskGate, error) {
	data, err := os.ReadFile(statsPath)
	if err != nil {
		return nil, fmt.Errorf("read gate_stats: %w", err)
	}

	var stats map[string]map[string]float64
	if err := json.Unmarshal(data, &stats); err != nil {
		return nil, fmt.Errorf("parse gate_stats: %w", err)
	}

	return &SubtaskGate{stats: stats, threshold: threshold}, nil
}

// Check returns (pass, passRate). If subtask or model not found, returns (true, -1)
// indicating the gate cannot make a decision (defaults to pass).
func (g *SubtaskGate) Check(model, subtask string) (pass bool, passRate float64) {
	modelStats, ok := g.stats[model]
	if !ok {
		return true, -1
	}
	rate, ok := modelStats[subtask]
	if !ok {
		return true, -1
	}
	return rate >= g.threshold, rate
}
