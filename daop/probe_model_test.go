package daop

import (
	"os"
	"testing"
)

const testGGUFPath = `C:\jonathan_workspace\models\Qwen3-0.6B-Q8_0.gguf`

func TestHiddenStateProbe_Load(t *testing.T) {
	if _, err := os.Stat(testGGUFPath); err != nil {
		t.Skipf("GGUF file not available: %v", err)
	}

	probe, err := NewHiddenStateProbe(testGGUFPath, 14)
	if err != nil {
		t.Fatalf("NewHiddenStateProbe: %v", err)
	}
	defer probe.Close()

	if probe.dim != 1024 {
		t.Errorf("dim = %d, want 1024", probe.dim)
	}
	if probe.maxLayer != 14 {
		t.Errorf("maxLayer = %d, want 14", probe.maxLayer)
	}
}

func TestHiddenStateProbe_Extract(t *testing.T) {
	if _, err := os.Stat(testGGUFPath); err != nil {
		t.Skipf("GGUF file not available: %v", err)
	}

	probe, err := NewHiddenStateProbe(testGGUFPath, 14)
	if err != nil {
		t.Fatalf("NewHiddenStateProbe: %v", err)
	}
	defer probe.Close()

	embedding, err := probe.Extract("What is 2+2?")
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}

	if len(embedding) != 1024 {
		t.Errorf("len(embedding) = %d, want 1024", len(embedding))
	}

	// Sanity check: embedding should not be all zeros
	allZero := true
	for _, v := range embedding {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("embedding is all zeros")
	}

	// Determinism: same input should give same output
	embedding2, err := probe.Extract("What is 2+2?")
	if err != nil {
		t.Fatalf("Extract (2nd call): %v", err)
	}
	for i := range embedding {
		if embedding[i] != embedding2[i] {
			t.Errorf("non-deterministic at index %d: %f vs %f", i, embedding[i], embedding2[i])
			break
		}
	}
}

func TestHiddenStateProbe_DifferentInputs(t *testing.T) {
	if _, err := os.Stat(testGGUFPath); err != nil {
		t.Skipf("GGUF file not available: %v", err)
	}

	probe, err := NewHiddenStateProbe(testGGUFPath, 14)
	if err != nil {
		t.Fatalf("NewHiddenStateProbe: %v", err)
	}
	defer probe.Close()

	emb1, err := probe.Extract("What is 2+2?")
	if err != nil {
		t.Fatalf("Extract 1: %v", err)
	}

	emb2, err := probe.Extract("Explain the theory of relativity in simple terms.")
	if err != nil {
		t.Fatalf("Extract 2: %v", err)
	}

	// Different inputs should produce different embeddings
	same := true
	for i := range emb1 {
		if emb1[i] != emb2[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("different inputs produced identical embeddings")
	}
}
