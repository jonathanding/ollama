package daop

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func createTestWeights(t *testing.T, dir string) string {
	t.Helper()
	path := filepath.Join(dir, "test_weights.bin")

	dim := 4
	textDim := 8
	numModels := 2

	header := weightsHeader{
		Dim:       dim,
		TextDim:   textDim,
		NumModels: numModels,
		Models:    []string{"model-a", "model-b"},
	}
	headerJSON, _ := json.Marshal(header)
	headerBytes := append(headerJSON, 0) // null terminated

	f, _ := os.Create(path)
	defer f.Close()

	// Header length
	binary.Write(f, binary.LittleEndian, uint32(len(headerBytes)))
	f.Write(headerBytes)

	// text_proj: [textDim][dim] = 8x4 = 32 floats (identity-like for testing)
	for i := 0; i < textDim*dim; i++ {
		var val float32
		if i/dim == i%dim { // diagonal-ish
			val = 1.0
		}
		binary.Write(f, binary.LittleEndian, val)
	}

	// classifier: [dim] = 4 floats, all 1.0
	for i := 0; i < dim; i++ {
		binary.Write(f, binary.LittleEndian, float32(1.0))
	}

	// model embeddings: [2][4]
	// model-a: [1, 0, 0, 0]
	// model-b: [0, 1, 0, 0]
	modelEmbeds := []float32{1, 0, 0, 0, 0, 1, 0, 0}
	for _, v := range modelEmbeds {
		binary.Write(f, binary.LittleEndian, v)
	}

	return path
}

func TestMFScorer_Load(t *testing.T) {
	path := createTestWeights(t, t.TempDir())
	scorer, err := NewMFScorer(path, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	if scorer.dim != 4 {
		t.Errorf("dim: got %d, want 4", scorer.dim)
	}
	if scorer.textDim != 8 {
		t.Errorf("textDim: got %d, want 8", scorer.textDim)
	}
	if !scorer.HasModel("model-a") {
		t.Error("model-a should exist")
	}
	if scorer.HasModel("model-c") {
		t.Error("model-c should not exist")
	}
}

func TestMFScorer_Score(t *testing.T) {
	path := createTestWeights(t, t.TempDir())
	scorer, err := NewMFScorer(path, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	// With the identity-like text_proj and unit model embeddings,
	// we can predict the output deterministically
	embedding := make([]float32, 8)
	embedding[0] = 2.0 // This should project to [2, 0, 0, 0, ...]

	score, err := scorer.Score("model-a", embedding)
	if err != nil {
		t.Fatal(err)
	}

	// model-a embed = [1,0,0,0], normalized = [1,0,0,0]
	// projected = [2, 0, 0, 0] (first 4 dims of embedding via identity proj)
	// hadamard = [2, 0, 0, 0]
	// classifier dot = 2.0
	// sigmoid(2.0) ~ 0.8808
	expected := 1.0 / (1.0 + math.Exp(-2.0))
	if math.Abs(score-expected) > 0.001 {
		t.Errorf("score: got %f, want %f", score, expected)
	}
}

func TestMFScorer_Temperature(t *testing.T) {
	path := createTestWeights(t, t.TempDir())

	// With T=1.0: logit=2.0 → sigmoid(2.0) ≈ 0.8808
	s1, _ := NewMFScorer(path, 1.0)
	embedding := make([]float32, 8)
	embedding[0] = 2.0
	score1, _ := s1.Score("model-a", embedding)

	// With T=2.0: logit=2.0 → sigmoid(2.0/2.0) = sigmoid(1.0) ≈ 0.7311
	s2, _ := NewMFScorer(path, 2.0)
	score2, _ := s2.Score("model-a", embedding)

	expected1 := 1.0 / (1.0 + math.Exp(-2.0))
	expected2 := 1.0 / (1.0 + math.Exp(-1.0))

	if math.Abs(score1-expected1) > 0.001 {
		t.Errorf("T=1.0: got %f, want %f", score1, expected1)
	}
	if math.Abs(score2-expected2) > 0.001 {
		t.Errorf("T=2.0: got %f, want %f", score2, expected2)
	}
	if score2 >= score1 {
		t.Errorf("higher temperature should compress scores: T=1 score=%f, T=2 score=%f", score1, score2)
	}
}

func TestMFScorer_UnknownModel(t *testing.T) {
	path := createTestWeights(t, t.TempDir())
	scorer, err := NewMFScorer(path, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	embedding := make([]float32, 8)
	_, err = scorer.Score("unknown", embedding)
	if err == nil {
		t.Error("expected error for unknown model")
	}
}

func TestMFScorer_WrongDim(t *testing.T) {
	path := createTestWeights(t, t.TempDir())
	scorer, err := NewMFScorer(path, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	embedding := make([]float32, 5) // wrong dim
	_, err = scorer.Score("model-a", embedding)
	if err == nil {
		t.Error("expected error for wrong embedding dimension")
	}
}
