package daop

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// SubtaskClassifier predicts which subtask a prompt belongs to
// using a linear classifier on the probe hidden state.
type SubtaskClassifier struct {
	numClasses          int
	dim                 int
	classes             []string
	confidenceThreshold float64
	weights             []float32 // [numClasses × dim] row-major
	bias                []float32 // [numClasses]
}

type classifierHeader struct {
	NumClasses          int      `json:"num_classes"`
	Dim                 int      `json:"dim"`
	Classes             []string `json:"classes"`
	ConfidenceThreshold float64  `json:"confidence_threshold"`
}

func NewSubtaskClassifier(path string) (*SubtaskClassifier, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read subtask classifier: %w", err)
	}

	if len(data) < 4 {
		return nil, fmt.Errorf("classifier file too small")
	}

	headerLen := binary.LittleEndian.Uint32(data[:4])
	if int(headerLen)+4 > len(data) {
		return nil, fmt.Errorf("invalid header length: %d", headerLen)
	}

	headerBytes := data[4 : 4+headerLen]
	if headerBytes[len(headerBytes)-1] == 0 {
		headerBytes = headerBytes[:len(headerBytes)-1]
	}

	var header classifierHeader
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, fmt.Errorf("parse classifier header: %w", err)
	}

	body := data[4+headerLen:]
	expectedFloats := header.NumClasses*header.Dim + header.NumClasses
	expectedBytes := expectedFloats * 4
	if len(body) < expectedBytes {
		return nil, fmt.Errorf("classifier body too small: got %d, want %d", len(body), expectedBytes)
	}

	c := &SubtaskClassifier{
		numClasses:          header.NumClasses,
		dim:                 header.Dim,
		classes:             header.Classes,
		confidenceThreshold: header.ConfidenceThreshold,
	}

	offset := 0
	c.weights = bytesToFloat32(body[offset : offset+header.NumClasses*header.Dim*4])
	offset += header.NumClasses * header.Dim * 4
	c.bias = bytesToFloat32(body[offset : offset+header.NumClasses*4])

	return c, nil
}

// Predict returns (subtask, confidence). If confidence < threshold, subtask is "".
func (c *SubtaskClassifier) Predict(embedding []float32) (subtask string, confidence float64) {
	if len(embedding) < c.dim {
		return "", 0
	}

	// Compute logits: W @ x + b
	logits := make([]float64, c.numClasses)
	maxLogit := math.Inf(-1)
	for i := 0; i < c.numClasses; i++ {
		var sum float64
		rowOffset := i * c.dim
		for j := 0; j < c.dim; j++ {
			sum += float64(c.weights[rowOffset+j]) * float64(embedding[j])
		}
		logits[i] = sum + float64(c.bias[i])
		if logits[i] > maxLogit {
			maxLogit = logits[i]
		}
	}

	// Softmax
	var sumExp float64
	for i := range logits {
		logits[i] = math.Exp(logits[i] - maxLogit)
		sumExp += logits[i]
	}

	// Find argmax probability
	bestIdx := 0
	bestProb := 0.0
	for i := range logits {
		p := logits[i] / sumExp
		if p > bestProb {
			bestProb = p
			bestIdx = i
		}
	}

	if bestProb < c.confidenceThreshold {
		return "", bestProb
	}
	return c.classes[bestIdx], bestProb
}
