package daop

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"runtime"
	"sync"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/model/models/qwen3"
	"github.com/ollama/ollama/tokenizer"
)

var ensureLibraryPath = sync.OnceFunc(func() {
	if _, ok := os.LookupEnv("OLLAMA_LIBRARY_PATH"); !ok {
		os.Setenv("OLLAMA_LIBRARY_PATH", ml.LibOllamaPath)
	}
	if runtime.GOOS == "windows" {
		os.Setenv("PATH", ml.LibOllamaPath+string(os.PathListSeparator)+os.Getenv("PATH"))
	}
})

const probeMaxSeqLen = 512

// HiddenStateProbe loads a Qwen3 GGUF model and extracts
// layer N hidden states via partial forward pass + mean pooling.
type HiddenStateProbe struct {
	model    *qwen3.Model
	tok      tokenizer.Tokenizer
	backend  ml.Backend
	maxLayer int
	dim      int
	mu       sync.Mutex
}

func NewHiddenStateProbe(modelPath string, maxLayer int) (*HiddenStateProbe, error) {
	ensureLibraryPath()

	numBlocks := maxLayer + 1
	layers := make([]int, numBlocks)
	for i := range layers {
		layers[i] = i
	}

	// First load to discover GPU device IDs (AllocMemory=false is fine for discovery)
	discoveryParams := ml.BackendParams{
		AllocMemory: false,
		NumThreads:  8,
		GPULayers:   ml.GPULayersList{{}},
	}
	dm, err := model.New(modelPath, discoveryParams)
	if err != nil {
		return nil, fmt.Errorf("load probe model (discovery): %w", err)
	}
	mem := dm.Backend().BackendMemory()
	dm.Backend().Close()

	params := ml.BackendParams{
		AllocMemory: true,
		NumThreads:  8,
	}

	if len(mem.GPUs) > 0 {
		params.GPULayers = ml.GPULayersList{{
			DeviceID: mem.GPUs[0].DeviceID,
			Layers:   layers,
		}}
	}

	m, err := model.New(modelPath, params)
	if err != nil && len(mem.GPUs) > 0 {
		slog.Warn("daop: GPU offload failed, falling back to CPU", "error", err)
		params.GPULayers = nil
		m, err = model.New(modelPath, params)
	}
	if err != nil {
		return nil, fmt.Errorf("load probe model: %w", err)
	}

	if err := m.Backend().Load(context.Background(), nil); err != nil {
		return nil, fmt.Errorf("load probe weights: %w", err)
	}

	qm, ok := m.(*qwen3.Model)
	if !ok {
		m.Backend().Close()
		return nil, fmt.Errorf("probe model is not qwen3 (got %T)", m)
	}

	tok, ok := m.(tokenizer.Tokenizer)
	if !ok {
		m.Backend().Close()
		return nil, fmt.Errorf("probe model does not implement tokenizer")
	}

	numLayers := len(qm.Layers)
	if maxLayer > numLayers {
		maxLayer = numLayers
	}

	// We don't need KV cache for single-pass embedding extraction
	qm.Cache = nil

	dim := int(m.Backend().Config().Uint("embedding_length"))

	slog.Info("daop: hidden state probe loaded",
		"model", modelPath,
		"layers", numLayers,
		"maxLayer", maxLayer,
		"dim", dim)

	return &HiddenStateProbe{
		model:    qm,
		tok:      tok,
		backend:  m.Backend(),
		maxLayer: maxLayer,
		dim:      dim,
	}, nil
}

// Extract tokenizes the prompt, runs partial forward (first maxLayer layers),
// and returns the mean-pooled hidden state.
func (p *HiddenStateProbe) Extract(promptText string) ([]float32, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	tokens, err := p.tok.Encode(promptText, true)
	if err != nil {
		return nil, fmt.Errorf("tokenize: %w", err)
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty token sequence")
	}
	if len(tokens) > probeMaxSeqLen {
		tokens = tokens[:probeMaxSeqLen]
	}

	ctx := p.backend.NewContext()
	defer ctx.Close()

	seqLen := len(tokens)

	positions := make([]int32, seqLen)
	for i := range positions {
		positions[i] = int32(i)
	}

	sequences := make([]int, seqLen)

	// All positions are outputs (we need the full sequence for mean pooling)
	outputIndices := make([]int32, seqLen)
	for i := range outputIndices {
		outputIndices[i] = int32(i)
	}

	batch := input.Batch{
		Inputs:    ctx.Input().Empty(ml.DTypeI32, seqLen),
		Outputs:   ctx.Input().FromInts(outputIndices, seqLen),
		Positions: positions,
		Sequences: sequences,
	}

	ctx.SetBatchSize(seqLen)

	// Build partial forward graph: embedding + first maxLayer layers
	hiddenState, err := p.partialForward(ctx, batch)
	if err != nil {
		return nil, fmt.Errorf("forward: %w", err)
	}

	// Mean pooling across sequence dimension
	pooled := hiddenState.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx).Mean(ctx)
	pooled = pooled.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	ctx.Forward(pooled)

	// Fill in token values and compute
	tokenInts := make([]int32, seqLen)
	for i, t := range tokens {
		tokenInts[i] = t
	}
	batch.Inputs.FromInts(tokenInts)

	ctx.Compute(pooled)

	result := pooled.Floats()
	if len(result) < p.dim {
		return nil, fmt.Errorf("output too short: got %d, want %d", len(result), p.dim)
	}

	return result[:p.dim], nil
}

// partialForward runs token embedding + first maxLayer transformer layers.
// Returns the hidden state tensor (before output norm/projection).
// We pass nil for cache since we don't need KV caching for single-pass embedding extraction.
func (p *HiddenStateProbe) partialForward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positionsTensor := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	hiddenStates := p.model.TokenEmbedding.Forward(ctx, batch.Inputs)

	for i := 0; i < p.maxLayer; i++ {
		layer := p.model.Layers[i]

		var outputs ml.Tensor
		if i == p.maxLayer-1 {
			outputs = batch.Outputs
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positionsTensor, outputs, nil, p.model.Options)
	}

	return hiddenStates, nil
}

func (p *HiddenStateProbe) Dim() int {
	return p.dim
}

func (p *HiddenStateProbe) Close() {
	if p.backend != nil {
		p.backend.Close()
	}
}
