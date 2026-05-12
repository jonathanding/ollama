package daop

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// MFScorer implements the pointwise MF model inference in pure Go.
// Architecture: score = sigmoid(sum(classifier * (normalize(P[model]) ⊙ text_proj(embedding))) / T)
type MFScorer struct {
	dim         int
	textDim     int
	temperature float64
	models      []string
	modelIdx    map[string]int

	// Weights (row-major float32)
	textProj   []float32 // [textDim][dim]
	classifier []float32 // [dim]
	modelEmb   []float32 // [numModels][dim]
}

type weightsHeader struct {
	Dim       int      `json:"dim"`
	TextDim   int      `json:"text_dim"`
	NumModels int      `json:"num_models"`
	Models    []string `json:"models"`
}

func NewMFScorer(weightsPath string, temperature float64) (*MFScorer, error) {
	data, err := os.ReadFile(weightsPath)
	if err != nil {
		return nil, fmt.Errorf("read weights: %w", err)
	}

	if len(data) < 4 {
		return nil, fmt.Errorf("weights file too small")
	}

	// Read header length
	headerLen := binary.LittleEndian.Uint32(data[:4])
	if int(headerLen)+4 > len(data) {
		return nil, fmt.Errorf("invalid header length: %d", headerLen)
	}

	// Parse header (null-terminated JSON)
	headerBytes := data[4 : 4+headerLen]
	// Remove null terminator if present
	if headerBytes[len(headerBytes)-1] == 0 {
		headerBytes = headerBytes[:len(headerBytes)-1]
	}
	var header weightsHeader
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, fmt.Errorf("parse weights header: %w", err)
	}

	// Read body
	body := data[4+headerLen:]
	expectedFloats := header.TextDim*header.Dim + header.Dim + header.NumModels*header.Dim
	expectedBytes := expectedFloats * 4
	if len(body) < expectedBytes {
		return nil, fmt.Errorf("weights body too small: got %d, want %d bytes", len(body), expectedBytes)
	}

	if temperature <= 0 {
		temperature = 1.0
	}

	s := &MFScorer{
		dim:         header.Dim,
		textDim:     header.TextDim,
		temperature: temperature,
		models:      header.Models,
		modelIdx:    make(map[string]int),
	}
	for i, m := range header.Models {
		s.modelIdx[m] = i
	}

	offset := 0
	s.textProj = bytesToFloat32(body[offset : offset+header.TextDim*header.Dim*4])
	offset += header.TextDim * header.Dim * 4

	s.classifier = bytesToFloat32(body[offset : offset+header.Dim*4])
	offset += header.Dim * 4

	s.modelEmb = bytesToFloat32(body[offset : offset+header.NumModels*header.Dim*4])

	return s, nil
}

// Score computes P(correct) for a given model and prompt embedding.
func (s *MFScorer) Score(model string, embedding []float32) (float64, error) {
	modelI, ok := s.modelIdx[model]
	if !ok {
		return 0, fmt.Errorf("model %q not in scorer", model)
	}
	if len(embedding) != s.textDim {
		return 0, fmt.Errorf("embedding dim %d != expected %d", len(embedding), s.textDim)
	}

	// Step 1: text_proj(embedding) -> projected [dim]
	projected := make([]float32, s.dim)
	for j := 0; j < s.dim; j++ {
		var sum float32
		for i := 0; i < s.textDim; i++ {
			sum += embedding[i] * s.textProj[i*s.dim+j]
		}
		projected[j] = sum
	}

	// Step 2: Get model embedding and normalize
	modelE := s.modelEmb[modelI*s.dim : (modelI+1)*s.dim]
	var norm float32
	for j := 0; j < s.dim; j++ {
		norm += modelE[j] * modelE[j]
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm < 1e-8 {
		norm = 1e-8
	}

	// Step 3: Hadamard product of normalized model emb and projected prompt
	// Step 4: classifier dot product + sigmoid
	var logit float32
	for j := 0; j < s.dim; j++ {
		normalizedM := modelE[j] / norm
		logit += s.classifier[j] * (normalizedM * projected[j])
	}

	return sigmoid(float64(logit) / s.temperature), nil
}

func (s *MFScorer) HasModel(model string) bool {
	_, ok := s.modelIdx[model]
	return ok
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func bytesToFloat32(b []byte) []float32 {
	n := len(b) / 4
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(b[i*4 : (i+1)*4])
		result[i] = math.Float32frombits(bits)
	}
	return result
}
