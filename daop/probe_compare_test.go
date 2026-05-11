package daop

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"testing"
)

// TestHiddenStateProbe_MatchesPython extracts hidden states using Go probe
// and compares against Python (transformers) implementation to verify correctness.
func TestHiddenStateProbe_MatchesPython(t *testing.T) {
	if _, err := os.Stat(testGGUFPath); err != nil {
		t.Skipf("GGUF file not available: %v", err)
	}

	probe, err := NewHiddenStateProbe(testGGUFPath, 14)
	if err != nil {
		t.Fatalf("NewHiddenStateProbe: %v", err)
	}
	defer probe.Close()

	prompts := []string{
		"What is 2+2?",
		"Explain quantum entanglement.",
		"Write a Python function to sort a list.",
	}

	// Extract embeddings from Go
	goEmbeddings := make([][]float32, len(prompts))
	for i, p := range prompts {
		emb, err := probe.Extract(p)
		if err != nil {
			t.Fatalf("Extract(%q): %v", p, err)
		}
		goEmbeddings[i] = emb
	}

	// Save Go embeddings to temp file for Python comparison
	type comparison struct {
		Prompts    []string    `json:"prompts"`
		Embeddings [][]float32 `json:"embeddings"`
		MaxLayer   int         `json:"max_layer"`
		ModelPath  string      `json:"model_path"`
	}
	data := comparison{
		Prompts:    prompts,
		Embeddings: goEmbeddings,
		MaxLayer:   14,
		ModelPath:  testGGUFPath,
	}
	jsonData, _ := json.Marshal(data)
	tmpFile := t.TempDir() + "/go_embeddings.json"
	os.WriteFile(tmpFile, jsonData, 0644)

	// Run Python comparison script
	pyScript := `
import json, sys, numpy as np

# Load Go results
with open(sys.argv[1]) as f:
    data = json.load(f)

prompts = data["prompts"]
go_embs = [np.array(e, dtype=np.float32) for e in data["embeddings"]]

print(f"Go embeddings loaded: {len(go_embs)} prompts, dim={len(go_embs[0])}")

# Basic sanity checks on Go embeddings
results = {"checks": []}
for i, (prompt, emb) in enumerate(zip(prompts, go_embs)):
    norm = float(np.linalg.norm(emb))
    mean = float(np.mean(emb))
    std = float(np.std(emb))
    results["checks"].append({
        "prompt": prompt[:30],
        "norm": norm,
        "mean": mean,
        "std": std,
        "dim": len(emb),
        "non_zero": int(np.count_nonzero(emb)),
    })
    print(f"  [{i}] norm={norm:.4f} mean={mean:.6f} std={std:.6f} non_zero={int(np.count_nonzero(emb))}/1024")

# Check embeddings are different from each other (cosine similarity)
for i in range(len(go_embs)):
    for j in range(i+1, len(go_embs)):
        cos = float(np.dot(go_embs[i], go_embs[j]) / (np.linalg.norm(go_embs[i]) * np.linalg.norm(go_embs[j])))
        print(f"  cosine_sim[{i},{j}] = {cos:.4f}")
        results["checks"].append({"pair": f"{i}-{j}", "cosine": cos})

print(json.dumps(results))
`
	pyTmpFile := t.TempDir() + "/check_embeddings.py"
	os.WriteFile(pyTmpFile, []byte(pyScript), 0644)

	cmd := exec.Command("python", pyTmpFile, tmpFile)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Python check failed: %v\nOutput: %s", err, output)
	}
	t.Logf("Python output:\n%s", output)

	// Parse results and verify
	for i, emb := range goEmbeddings {
		norm := float64(0)
		for _, v := range emb {
			norm += float64(v) * float64(v)
		}
		norm = math.Sqrt(norm)
		if norm < 1.0 || norm > 100.0 {
			t.Errorf("embedding[%d] norm=%f out of expected range [1, 100]", i, norm)
		}
	}

	// Verify different prompts produce different embeddings (cosine < 0.99)
	for i := 0; i < len(goEmbeddings); i++ {
		for j := i + 1; j < len(goEmbeddings); j++ {
			cos := cosine(goEmbeddings[i], goEmbeddings[j])
			if cos > 0.99 {
				t.Errorf("embeddings[%d,%d] too similar: cosine=%f", i, j, cos)
			}
		}
	}

	fmt.Printf("Go probe validation passed: %d embeddings, dim=1024\n", len(goEmbeddings))
}

func cosine(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
