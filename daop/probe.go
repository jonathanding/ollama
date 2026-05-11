package daop

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
)

// Probe extracts prompt embeddings for the MF scorer.
// POC implementation: loads pre-computed embeddings and looks up by prompt hash.
type Probe struct {
	dim        int
	embeddings []float32
	hashIndex  map[string]int
}

type probeIndex struct {
	Dim    int            `json:"dim"`
	Count  int            `json:"count"`
	Hashes map[string]int `json:"hashes"`
}

func NewProbe(cfg *Config) (*Probe, error) {
	indexPath := cfg.ProbeModel + ".index.json"
	embPath := cfg.ProbeModel + ".embeddings.bin"

	indexData, err := os.ReadFile(indexPath)
	if err != nil {
		return nil, fmt.Errorf("read probe index: %w", err)
	}
	var idx probeIndex
	if err := json.Unmarshal(indexData, &idx); err != nil {
		return nil, fmt.Errorf("parse probe index: %w", err)
	}

	embData, err := os.ReadFile(embPath)
	if err != nil {
		return nil, fmt.Errorf("read probe embeddings: %w", err)
	}

	expectedBytes := idx.Count * idx.Dim * 4
	if len(embData) < expectedBytes {
		return nil, fmt.Errorf("embeddings file too small: got %d, want %d", len(embData), expectedBytes)
	}

	embeddings := make([]float32, idx.Count*idx.Dim)
	for i := range embeddings {
		bits := binary.LittleEndian.Uint32(embData[i*4 : (i+1)*4])
		embeddings[i] = math.Float32frombits(bits)
	}

	slog.Info("daop: probe initialized (embedding lookup mode)",
		"prompts", idx.Count, "dim", idx.Dim)

	return &Probe{
		dim:        idx.Dim,
		embeddings: embeddings,
		hashIndex:  idx.Hashes,
	}, nil
}

// Extract returns the prompt embedding by looking up the pre-computed table.
func (p *Probe) Extract(promptText string) ([]float32, error) {
	if promptText == "" {
		return nil, fmt.Errorf("empty prompt text")
	}

	hash := sha256Hash(promptText)
	rowIdx, ok := p.hashIndex[hash]
	if !ok {
		return nil, fmt.Errorf("prompt not in embedding lookup (hash=%s...)", hash[:12])
	}

	start := rowIdx * p.dim
	end := start + p.dim
	result := make([]float32, p.dim)
	copy(result, p.embeddings[start:end])
	return result, nil
}

func sha256Hash(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}
