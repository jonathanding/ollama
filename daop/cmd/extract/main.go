package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/ollama/ollama/daop"
)

type promptEntry struct {
	Idx        int    `json:"idx"`
	SampleID   string `json:"sample_id"`
	PromptText string `json:"prompt_text"`
}

func main() {
	promptsPath := flag.String("prompts", "", "Path to all_prompts.jsonl")
	outputPath := flag.String("output", "", "Path to output .bin file")
	indexPath := flag.String("index", "", "Path to output index .json file")
	modelPath := flag.String("model", "", "Path to GGUF model file")
	maxLayer := flag.Int("layer", 14, "Max layer for hidden state extraction")
	flag.Parse()

	if *promptsPath == "" || *outputPath == "" || *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Usage: extract -prompts <jsonl> -output <bin> -model <gguf> [-index <json>] [-layer 14]\n")
		os.Exit(1)
	}

	// Load prompts
	f, err := os.Open(*promptsPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open prompts: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	var prompts []promptEntry
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		var p promptEntry
		if err := json.Unmarshal(scanner.Bytes(), &p); err != nil {
			fmt.Fprintf(os.Stderr, "parse prompt: %v\n", err)
			os.Exit(1)
		}
		prompts = append(prompts, p)
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "scan: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Loaded %d prompts\n", len(prompts))

	// Initialize probe
	probe, err := daop.NewHiddenStateProbe(*modelPath, *maxLayer)
	if err != nil {
		fmt.Fprintf(os.Stderr, "init probe: %v\n", err)
		os.Exit(1)
	}
	defer probe.Close()

	dim := probe.Dim()
	fmt.Printf("Probe dim: %d, maxLayer: %d\n", dim, *maxLayer)

	// Open output file
	out, err := os.Create(*outputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "create output: %v\n", err)
		os.Exit(1)
	}
	defer out.Close()

	writer := bufio.NewWriter(out)

	// Extract embeddings
	start := time.Now()
	var sampleIDs []string
	failed := 0

	for i, p := range prompts {
		if p.PromptText == "" {
			zeros := make([]byte, dim*4)
			writer.Write(zeros)
			sampleIDs = append(sampleIDs, p.SampleID)
			failed++
			continue
		}

		emb, err := probe.Extract(p.PromptText)
		if err != nil {
			fmt.Fprintf(os.Stderr, "WARN: prompt %d (%s) failed: %v\n", i, p.SampleID, err)
			zeros := make([]byte, dim*4)
			writer.Write(zeros)
			sampleIDs = append(sampleIDs, p.SampleID)
			failed++
			continue
		}

		for _, v := range emb {
			binary.Write(writer, binary.LittleEndian, v)
		}
		sampleIDs = append(sampleIDs, p.SampleID)

		if (i+1)%500 == 0 {
			elapsed := time.Since(start)
			speed := float64(i+1) / elapsed.Seconds()
			eta := float64(len(prompts)-i-1) / speed
			fmt.Printf("[%d/%d] %.1f prompts/s, ETA %.0fs, failed=%d\n",
				i+1, len(prompts), speed, eta, failed)
		}
	}

	writer.Flush()
	elapsed := time.Since(start)
	fmt.Printf("Done: %d prompts in %s (%.1f prompts/s), failed=%d\n",
		len(prompts), elapsed.Round(time.Second), float64(len(prompts))/elapsed.Seconds(), failed)

	// Verify output file size
	info, _ := out.Stat()
	expectedSize := int64(len(prompts)) * int64(dim) * 4
	if info.Size() != expectedSize {
		fmt.Fprintf(os.Stderr, "WARNING: output size %d != expected %d\n", info.Size(), expectedSize)
	}

	// Write index
	if *indexPath != "" {
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
		os.WriteFile(*indexPath, indexJSON, 0644)
		fmt.Printf("Index written: %s\n", *indexPath)
	}

	// Sanity check
	out.Seek(0, 0)
	buf := make([]byte, dim*4)
	nonZero := 0
	check := min(100, len(prompts))
	for i := 0; i < check; i++ {
		out.Read(buf)
		var norm float64
		for j := 0; j < dim; j++ {
			v := math.Float32frombits(binary.LittleEndian.Uint32(buf[j*4 : (j+1)*4]))
			norm += float64(v) * float64(v)
		}
		if norm > 0 {
			nonZero++
		}
	}
	fmt.Printf("Sanity: %d/%d of first %d embeddings have non-zero norm\n", nonZero, check, check)
}
