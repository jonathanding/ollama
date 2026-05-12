package daop

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"testing"
	"time"
)

type promptEntry struct {
	Idx        int    `json:"idx"`
	SampleID   string `json:"sample_id"`
	PromptText string `json:"prompt_text"`
}

func TestBatchExtractEmbeddings(t *testing.T) {
	promptsPath := `C:\jonathan_workspace\daop-accuracy\data\go_extract\all_prompts.jsonl`
	outputPath := `C:\jonathan_workspace\daop-accuracy\data\go_extract\go_hidden_states_layer14.bin`
	indexPath := `C:\jonathan_workspace\daop-accuracy\data\go_extract\go_hidden_states_index.json`
	modelPath := `C:\jonathan_workspace\models\Qwen3-0.6B-Q8_0.gguf`

	// Load prompts
	f, err := os.Open(promptsPath)
	if err != nil {
		t.Fatalf("open prompts: %v", err)
	}
	defer f.Close()

	var prompts []promptEntry
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		var p promptEntry
		if err := json.Unmarshal(scanner.Bytes(), &p); err != nil {
			t.Fatalf("parse prompt line: %v", err)
		}
		prompts = append(prompts, p)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan: %v", err)
	}
	t.Logf("Loaded %d prompts", len(prompts))

	// Initialize probe
	probe, err := NewHiddenStateProbe(modelPath, 14)
	if err != nil {
		t.Fatalf("init probe: %v", err)
	}
	defer probe.Close()

	dim := probe.dim
	t.Logf("Probe dim: %d", dim)

	// Open output file
	out, err := os.Create(outputPath)
	if err != nil {
		t.Fatalf("create output: %v", err)
	}
	defer out.Close()

	writer := bufio.NewWriter(out)
	defer writer.Flush()

	// Extract embeddings
	start := time.Now()
	var sampleIDs []string
	failed := 0

	for i, p := range prompts {
		if p.PromptText == "" {
			// Write zeros for empty prompts
			zeros := make([]byte, dim*4)
			writer.Write(zeros)
			sampleIDs = append(sampleIDs, p.SampleID)
			failed++
			continue
		}

		emb, err := probe.Extract(p.PromptText)
		if err != nil {
			t.Logf("WARN: prompt %d (%s) failed: %v", i, p.SampleID, err)
			zeros := make([]byte, dim*4)
			writer.Write(zeros)
			sampleIDs = append(sampleIDs, p.SampleID)
			failed++
			continue
		}

		// Write float32 binary (little-endian)
		for _, v := range emb {
			binary.Write(writer, binary.LittleEndian, v)
		}
		sampleIDs = append(sampleIDs, p.SampleID)

		if (i+1)%500 == 0 {
			elapsed := time.Since(start)
			speed := float64(i+1) / elapsed.Seconds()
			eta := float64(len(prompts)-i-1) / speed
			t.Logf("[%d/%d] %.1f prompts/s, ETA %.0fs, failed=%d",
				i+1, len(prompts), speed, eta, failed)
		}
	}

	writer.Flush()
	elapsed := time.Since(start)
	t.Logf("Done: %d prompts in %s (%.1f prompts/s), failed=%d",
		len(prompts), elapsed.Round(time.Second), float64(len(prompts))/elapsed.Seconds(), failed)

	// Verify output file size
	info, _ := out.Stat()
	expectedSize := int64(len(prompts)) * int64(dim) * 4
	if info.Size() != expectedSize {
		t.Errorf("output size: got %d, want %d", info.Size(), expectedSize)
	}

	// Write index
	indexData := struct {
		Dim       int      `json:"dim"`
		Count     int      `json:"count"`
		SampleIDs []string `json:"sample_ids"`
	}{
		Dim:       dim,
		Count:     len(sampleIDs),
		SampleIDs: sampleIDs,
	}
	indexJSON, _ := json.Marshal(indexData)
	os.WriteFile(indexPath, indexJSON, 0644)
	t.Logf("Index written to %s", indexPath)

	// Sanity check: verify a few embeddings have non-zero norm
	out.Seek(0, 0)
	buf := make([]byte, dim*4)
	nonZeroCount := 0
	for i := 0; i < min(100, len(prompts)); i++ {
		out.Read(buf)
		var norm float64
		for j := 0; j < dim; j++ {
			v := math.Float32frombits(binary.LittleEndian.Uint32(buf[j*4 : (j+1)*4]))
			norm += float64(v) * float64(v)
		}
		if norm > 0 {
			nonZeroCount++
		}
	}
	t.Logf("Sanity: %d/%d of first 100 embeddings have non-zero norm", nonZeroCount, min(100, len(prompts)))
}
