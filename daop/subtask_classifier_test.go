package daop

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func createTestClassifier(t *testing.T, dir string) string {
	t.Helper()
	path := filepath.Join(dir, "classifier.bin")

	header := classifierHeader{
		NumClasses:          3,
		Dim:                 4,
		Classes:             []string{"math", "logic", "language"},
		ConfidenceThreshold: 0.7,
	}
	headerBytes, _ := json.Marshal(header)
	headerBytes = append(headerBytes, 0)

	// Weights: 3 classes × 4 dim
	// Class 0 (math): [1, 0, 0, 0] — responds to dim 0
	// Class 1 (logic): [0, 1, 0, 0] — responds to dim 1
	// Class 2 (language): [0, 0, 1, 0] — responds to dim 2
	weights := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}
	bias := []float32{0, 0, 0}

	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	binary.Write(f, binary.LittleEndian, uint32(len(headerBytes)))
	f.Write(headerBytes)
	for _, v := range weights {
		binary.Write(f, binary.LittleEndian, v)
	}
	for _, v := range bias {
		binary.Write(f, binary.LittleEndian, v)
	}

	return path
}

func TestSubtaskClassifier_Load(t *testing.T) {
	dir := t.TempDir()
	path := createTestClassifier(t, dir)

	c, err := NewSubtaskClassifier(path)
	if err != nil {
		t.Fatal(err)
	}
	if c.numClasses != 3 {
		t.Errorf("numClasses: got %d, want 3", c.numClasses)
	}
	if c.dim != 4 {
		t.Errorf("dim: got %d, want 4", c.dim)
	}
}

func TestSubtaskClassifier_Predict(t *testing.T) {
	dir := t.TempDir()
	path := createTestClassifier(t, dir)

	c, err := NewSubtaskClassifier(path)
	if err != nil {
		t.Fatal(err)
	}

	// Strong signal for "math" (dim 0 = 5.0)
	subtask, conf := c.Predict([]float32{5.0, 0, 0, 0})
	if subtask != "math" {
		t.Errorf("subtask: got %q, want math", subtask)
	}
	if conf < 0.9 {
		t.Errorf("confidence: got %.4f, want >= 0.9", conf)
	}

	// Strong signal for "logic" (dim 1 = 5.0)
	subtask, conf = c.Predict([]float32{0, 5.0, 0, 0})
	if subtask != "logic" {
		t.Errorf("subtask: got %q, want logic", subtask)
	}

	// Weak signal — all similar, confidence should be below threshold
	subtask, conf = c.Predict([]float32{0.1, 0.1, 0.1, 0})
	if subtask != "" {
		t.Errorf("expected empty subtask for low confidence, got %q (conf=%.4f)", subtask, conf)
	}
	expectedConf := 1.0 / 3.0
	if math.Abs(conf-expectedConf) > 0.01 {
		t.Errorf("confidence: got %.4f, want ~%.4f", conf, expectedConf)
	}
}

func TestSubtaskClassifier_RealBinary(t *testing.T) {
	path := filepath.Join("..", "daop_demo", "data", "subtask_classifier.bin")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("subtask_classifier.bin not found (run train_subtask_classifier.py first)")
	}

	c, err := NewSubtaskClassifier(path)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Loaded: %d classes, %d dim, threshold=%.2f",
		c.numClasses, c.dim, c.confidenceThreshold)

	if c.numClasses != 94 {
		t.Errorf("numClasses: got %d, want 94", c.numClasses)
	}
	if c.dim != 1024 {
		t.Errorf("dim: got %d, want 1024", c.dim)
	}

	// Zero embedding should produce low confidence
	zeroEmb := make([]float32, 1024)
	subtask, conf := c.Predict(zeroEmb)
	t.Logf("Zero embedding: subtask=%q, conf=%.4f", subtask, conf)
}
