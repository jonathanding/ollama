package daop

import (
	"os"
	"path/filepath"
	"testing"
)

func writeGateStats(t *testing.T, dir string) string {
	t.Helper()
	path := filepath.Join(dir, "gate_stats.json")
	data := `{
		"qwen3-8b-q4": {
			"temporal_sequences": 0.95,
			"causal_judgement": 0.42,
			"algebra": 0.78
		},
		"qwen3-4b-q4": {
			"temporal_sequences": 0.85,
			"gpqa_diamond": 0.35
		}
	}`
	os.WriteFile(path, []byte(data), 0644)
	return path
}

func TestSubtaskGate_Pass(t *testing.T) {
	path := writeGateStats(t, t.TempDir())
	gate, err := NewSubtaskGate(path, 0.80)
	if err != nil {
		t.Fatal(err)
	}

	pass, rate := gate.Check("qwen3-8b-q4", "temporal_sequences")
	if !pass {
		t.Error("expected pass for temporal_sequences (0.95 >= 0.80)")
	}
	if rate != 0.95 {
		t.Errorf("rate: got %f, want 0.95", rate)
	}
}

func TestSubtaskGate_Fail(t *testing.T) {
	path := writeGateStats(t, t.TempDir())
	gate, err := NewSubtaskGate(path, 0.80)
	if err != nil {
		t.Fatal(err)
	}

	pass, rate := gate.Check("qwen3-8b-q4", "causal_judgement")
	if pass {
		t.Error("expected fail for causal_judgement (0.42 < 0.80)")
	}
	if rate != 0.42 {
		t.Errorf("rate: got %f, want 0.42", rate)
	}
}

func TestSubtaskGate_UnknownModel(t *testing.T) {
	path := writeGateStats(t, t.TempDir())
	gate, err := NewSubtaskGate(path, 0.80)
	if err != nil {
		t.Fatal(err)
	}

	pass, rate := gate.Check("unknown-model", "algebra")
	if !pass {
		t.Error("expected pass for unknown model (default behavior)")
	}
	if rate != -1 {
		t.Errorf("rate: got %f, want -1", rate)
	}
}

func TestSubtaskGate_UnknownSubtask(t *testing.T) {
	path := writeGateStats(t, t.TempDir())
	gate, err := NewSubtaskGate(path, 0.80)
	if err != nil {
		t.Fatal(err)
	}

	pass, rate := gate.Check("qwen3-8b-q4", "unknown_subtask")
	if !pass {
		t.Error("expected pass for unknown subtask (default behavior)")
	}
	if rate != -1 {
		t.Errorf("rate: got %f, want -1", rate)
	}
}
