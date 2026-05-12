package daop

import (
	"os"
	"path/filepath"
	"testing"
)

func setupRouter(t *testing.T) (*Router, string) {
	t.Helper()
	dir := t.TempDir()

	// Gate stats uses the same model names as test weights
	gatePath := filepath.Join(dir, "gate_stats.json")
	gateData := `{
		"model-a": {"temporal_sequences": 0.95, "causal_judgement": 0.42},
		"model-b": {"temporal_sequences": 0.85, "gpqa_diamond": 0.35}
	}`
	os.WriteFile(gatePath, []byte(gateData), 0644)

	// Create weights with models "model-a" and "model-b" (textDim=8)
	weightsPath := createTestWeights(t, dir)

	cfg := &Config{
		Enabled:           true,
		AccuracyThreshold: 0.80,
		GateThreshold:     0.80,
		SupportedModels:   []string{"model-a", "model-b"},
	}

	gate, err := NewSubtaskGate(gatePath, cfg.GateThreshold)
	if err != nil {
		t.Fatal(err)
	}

	scorer, err := NewMFScorer(weightsPath, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	// Mock probe returns embedding with textDim=8 matching test weights
	mockProbe := func(text string) ([]float32, error) {
		emb := make([]float32, 8)
		emb[0] = 3.0 // Projects to [3,0,0,0], model-a dot = 3.0, sigmoid(3)≈0.95 > 0.80
		return emb, nil
	}

	router := NewRouter(cfg, gate, nil, scorer, mockProbe)
	return router, dir
}

func TestRouter_UnsupportedModel(t *testing.T) {
	router, _ := setupRouter(t)
	result := router.Route("unknown-model", "test prompt", nil)
	if result != nil {
		t.Error("expected nil for unsupported model")
	}
}

func TestRouter_GateBlocks(t *testing.T) {
	router, _ := setupRouter(t)
	ctx := &DaopContext{Subtask: "causal_judgement"}
	result := router.Route("model-a", "test prompt", ctx)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if result.Decision != "fallback" {
		t.Errorf("decision: got %q, want fallback", result.Decision)
	}
	if result.FallbackReason != "gate" {
		t.Errorf("reason: got %q, want gate", result.FallbackReason)
	}
	if result.Confidence != nil {
		t.Error("confidence should be nil when gate blocks")
	}
}

func TestRouter_GatePassesOffload(t *testing.T) {
	router, _ := setupRouter(t)
	ctx := &DaopContext{Subtask: "temporal_sequences"}
	result := router.Route("model-a", "test prompt", ctx)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	// Gate passes (0.95 >= 0.80), probe runs, score = sigmoid(3) ≈ 0.95 > 0.80
	if result.Confidence == nil {
		t.Fatal("confidence should not be nil when probe ran")
	}
	if result.Decision != "offload" {
		t.Errorf("decision: got %q, want offload (sigmoid(3)=0.95 > 0.80)", result.Decision)
	}
}

func TestRouter_NoSubtask(t *testing.T) {
	router, _ := setupRouter(t)
	// No daop_context — gate is skipped, goes straight to probe
	result := router.Route("model-a", "test prompt", nil)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if result.Confidence == nil {
		t.Fatal("expected confidence when no subtask (gate skipped)")
	}
	if result.Decision != "offload" {
		t.Errorf("decision: got %q, want offload", result.Decision)
	}
}
