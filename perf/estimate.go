package perf

import (
	"fmt"
	"log/slog"
	"os"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

// nodeToQueryShape extracts the performance-relevant dimensions from a GraphNode.
func nodeToQueryShape(node ml.GraphNode) (op string, shape []int64, computeDtype, weightDtype string) {
	op = node.Op
	computeDtype = node.ComputeDtype
	weightDtype = node.WeightDtype

	switch op {
	case "MUL_MAT", "MUL_MAT_ADD":
		if len(node.InputShapes) >= 2 && len(node.InputShapes[0]) >= 2 && len(node.InputShapes[1]) >= 2 {
			// GGML weight tensor: ne[0]=K (inner dim), ne[1]=M (output dim)
			// Activation tensor:  ne[0]=K (inner dim), ne[1]=N (batch/seq)
			K := node.InputShapes[0][0]
			M := node.InputShapes[0][1]
			N := node.InputShapes[1][1]
			shape = []int64{M, K, N}
			return
		}
		shape = []int64{totalElements(node.Shape)}
		return

	case "FLASH_ATTN_EXT":
		if len(node.InputShapes) >= 2 && len(node.InputShapes[0]) >= 2 && len(node.InputShapes[1]) >= 2 {
			// GGML layout: Q ne=[head_dim, seqQ, num_heads, batch]
			//              K ne=[head_dim, seqKV, num_kv_heads, batch]
			seqQ := node.InputShapes[0][1]
			seqKV := node.InputShapes[1][1]
			shape = []int64{seqQ, seqKV}
			return
		}
		shape = []int64{totalElements(node.Shape)}
		return

	default:
		shape = []int64{totalElements(node.Shape)}
		return
	}
}

// totalElements computes the product of non-zero dimensions in a GraphNode shape.
func totalElements(shape [4]int64) int64 {
	total := int64(1)
	for _, d := range shape {
		if d > 0 {
			total *= d
		}
	}
	return total
}

// fullOffloadSchedule constructs a GPULayers list that offloads all model layers
// (including the output layer) to the primary GPU device.
// Returns nil if no GPU devices are available (CPU-only system).
func fullOffloadSchedule(backend ml.Backend, numLayers int) ml.GPULayersList {
	devices := backend.BackendDevices()
	if len(devices) == 0 {
		return nil
	}
	// numLayers repeating layers + 1 output layer
	layers := make([]int, numLayers+1)
	for i := range layers {
		layers[i] = i
	}
	return ml.GPULayersList{{
		DeviceID: devices[0].DeviceID,
		Layers:   layers,
	}}
}

var blkLayerRe = regexp.MustCompile(`\bblk\.(\d+)\.`)

// parseLayerIndex extracts a layer index from a tensor name like "blk.5.attn_q.weight".
// Returns -1 if no layer index is found.
func parseLayerIndex(name string) int {
	m := blkLayerRe.FindStringSubmatch(name)
	if m == nil {
		return -1
	}
	idx, _ := strconv.Atoi(m[1])
	return idx
}

// assignBackends assigns a backend name to each graph node based on the schedule.
// It parses layer indices from input tensor names (weight tensors) and maps them
// to backends via the schedule. Nodes without identifiable layers use adjacency
// expansion (same backend as neighboring compute ops).
func assignBackends(nodes []ml.GraphNode, schedule ml.GPULayersList, blockCount int, cpuBackendName string) {
	// Build layer → backend name map from schedule
	layerBackend := make(map[int]string)
	for _, gpu := range schedule {
		for _, layer := range gpu.Layers {
			layerBackend[layer] = gpu.DeviceID.Library
		}
	}

	// Pass 1: assign from input tensor names
	for i := range nodes {
		layer := -1
		// Check input tensor names for layer index (e.g., "blk.5.attn_q.weight")
		for _, name := range nodes[i].InputNames {
			if l := parseLayerIndex(name); l >= 0 {
				layer = l
				break
			}
		}
		// Also check output tensor name
		if layer < 0 {
			layer = parseLayerIndex(nodes[i].Name)
		}
		// Handle output/embedding tensors
		if layer < 0 {
			for _, name := range nodes[i].InputNames {
				if strings.HasPrefix(name, "output.") || strings.HasPrefix(name, "output_norm.") {
					layer = blockCount // output layer
					break
				}
				if strings.HasPrefix(name, "token_embd.") {
					layer = -2 // input, stays on CPU for now
					break
				}
			}
		}

		if layer >= 0 {
			if backend, ok := layerBackend[layer]; ok {
				nodes[i].Backend = backend
			} else {
				nodes[i].Backend = cpuBackendName
			}
		}
		// layer == -2: explicit CPU
		if layer == -2 {
			nodes[i].Backend = cpuBackendName
		}
	}

	// Pass 2: expand GPU backends down (same as split_graph pass 2)
	curBackend := ""
	for i := range nodes {
		if nodes[i].Backend != "" && nodes[i].Backend != cpuBackendName {
			curBackend = nodes[i].Backend
		} else if nodes[i].Backend == "" && curBackend != "" {
			nodes[i].Backend = curBackend
		}
	}

	// Pass 3: expand GPU backends up
	curBackend = ""
	for i := len(nodes) - 1; i >= 0; i-- {
		if nodes[i].Backend != "" && nodes[i].Backend != cpuBackendName {
			curBackend = nodes[i].Backend
		} else if nodes[i].Backend == "" && curBackend != "" {
			nodes[i].Backend = curBackend
		}
	}

	// Pass 4: remaining unassigned → CPU
	for i := range nodes {
		if nodes[i].Backend == "" {
			nodes[i].Backend = cpuBackendName
		}
	}
}

// ensureLibraryPath sets OLLAMA_LIBRARY_PATH and PATH so that GGML can discover
// backend DLLs (e.g., ggml-vulkan.dll) in the development build directory.
// Same logic as NewForBench — needed here because model.New also calls initDevices.
func ensureLibraryPath() {
	if _, ok := os.LookupEnv("OLLAMA_LIBRARY_PATH"); !ok {
		os.Setenv("OLLAMA_LIBRARY_PATH", ml.LibOllamaPath)
	}
	if runtime.GOOS == "windows" {
		os.Setenv("PATH", ml.LibOllamaPath+string(os.PathListSeparator)+os.Getenv("PATH"))
	}
}

// discoverModelSchedule loads the model once (lightweight, no weights) to discover
// GPU devices and layer count, then constructs the schedule (GPULayers).
func discoverModelSchedule(modelPath string) (ml.GPULayersList, error) {
	ensureLibraryPath()

	// First pass: load with no GPU offload to discover metadata
	m, err := model.New(modelPath, ml.BackendParams{AllocMemory: false})
	if err != nil {
		return nil, fmt.Errorf("load model for discovery: %w", err)
	}
	defer m.Backend().Close()

	numLayers := int(m.Backend().Config().Uint("block_count"))
	schedule := fullOffloadSchedule(m.Backend(), numLayers)
	if schedule != nil {
		slog.Info("estimate schedule", "strategy", "full_offload",
			"device", schedule[0].DeviceID.Library, "layers", len(schedule[0].Layers))
	} else {
		slog.Info("estimate schedule", "strategy", "cpu_only")
	}
	return schedule, nil
}

// buildModelGraphNodes loads a model with the given schedule and captures
// prefill+decode computation graphs. The schedule determines which layers
// run on which backend (GPU/CPU), affecting backend assignment in the graph.
func buildModelGraphNodes(modelPath string, schedule ml.GPULayersList) (prefill, decode []ml.GraphNode, err error) {
	m, err := model.New(modelPath, ml.BackendParams{
		AllocMemory:    false,
		GPULayers:      schedule,
		FlashAttention: ml.FlashAttentionEnabled,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("load model: %w", err)
	}
	defer m.Backend().Close()

	// Initialize the KV cache before graph capture. The cache is created by model.New()
	// but Init() (which sets config, allocates cells, etc.) is normally called by the runner.
	// We need it initialized so that StartForward/buildMask don't hit nil config.
	if cache := m.Config().Cache; cache != nil {
		cache.Init(m.Backend(), ml.DTypeF16, 1, 2048, 512)
	}

	captureGraph := func(batchSize int) ([]ml.GraphNode, error) {
		ctx := m.Backend().NewContext()
		defer ctx.Close()

		batchInputs := make([]int32, batchSize)
		positions := make([]int32, batchSize)
		sequences := make([]int, batchSize)
		for i := 0; i < batchSize; i++ {
			positions[i] = int32(i)
		}
		batch := input.Batch{
			Inputs:    ctx.Input().FromInts(batchInputs, batchSize),
			Outputs:   ctx.Input().Empty(ml.DTypeI32, 1),
			Positions: positions,
			Sequences: sequences,
		}

		if cache := m.Config().Cache; cache != nil {
			if err := cache.StartForward(ctx, batch, true); err != nil {
				return nil, fmt.Errorf("cache start: %w", err)
			}
		}

		t, err := m.Forward(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("forward: %w", err)
		}

		// PlanGraph: split_graph (backend assignment + graph_optimize) + capture + reset.
		// Unlike Reserve(), does not allocate memory — only captures graph structure.
		ctx.SetBatchSize(batchSize)
		ctx.Forward(t)
		ctx.PlanGraph()

		return ctx.GraphNodes(), nil
	}

	prefill, err = captureGraph(512)
	if err != nil {
		return nil, nil, fmt.Errorf("prefill graph: %w", err)
	}
	decode, err = captureGraph(1)
	if err != nil {
		return nil, nil, fmt.Errorf("decode graph: %w", err)
	}

	return prefill, decode, nil
}

// lookupLatency finds the estimated latency for one graph node operation.
func lookupLatency(profile *Profile, op string, shape []int64,
	computeDtype, weightDtype, backend string) (float64, error) {

	switch op {
	case "MUL_MAT":
		if len(shape) < 3 {
			return 0, fmt.Errorf("MUL_MAT requires 3 shape dims, got %d", len(shape))
		}
		mappedWdt := mapWeightDtype(weightDtype)
		lat := PredictMulMatLatency(&profile.Hardware, shape[0], shape[1], shape[2], mappedWdt)
		if lat == 0 {
			return 0, fmt.Errorf("no efficiency constants for MUL_MAT — run daop-bench first")
		}
		return lat, nil

	case "FLASH_ATTN_EXT":
		if len(shape) < 2 {
			return 0, fmt.Errorf("FLASH_ATTN_EXT requires 2 shape dims, got %d", len(shape))
		}
		// FLASH_ATTN_EXT output tensor is f32 but the op runs on f16 Q/K/V inputs.
		// Match by op + backend only (benchmark only produces one dtype config).
		for i := range profile.Operators {
			c := &profile.Operators[i]
			if c.Op == op && c.Backend == backend {
				return InterpolateFlashAttn(c, shape[0], shape[1]), nil
			}
		}
		return 0, fmt.Errorf("uncalibrated: %s(%s on %s)", op, computeDtype, backend)

	case "GET_ROWS":
		// GET_ROWS is a pure memory copy (embedding lookup), called once per forward pass.
		// Its cost is negligible (<10μs) relative to compute ops, so we use a fixed estimate
		// rather than benchmarking it (which would require impractically large output tensors).
		return 10.0, nil

	default:
		if len(shape) < 1 {
			return 0, fmt.Errorf("op %s requires at least 1 shape dim", op)
		}
		for _, c := range profile.Operators {
			if c.Op == op && c.ComputeDtype == computeDtype && c.Backend == backend {
				return Interpolate1D(c.Points, shape[0]), nil
			}
		}
		return 0, fmt.Errorf("uncalibrated: %s(%s on %s)", op, computeDtype, backend)
	}
}

// lookupLatencyV3 extends lookupLatency with MUL_MAT direct interpolation and roofline fallback.
func lookupLatencyV3(profile *Profile, op string, shape []int64,
	computeDtype, weightDtype, backend string, caps *BackendCapabilities) (float64, error) {

	switch op {
	case "MUL_MAT", "MUL_MAT_ADD":
		if len(shape) < 3 {
			return 0, fmt.Errorf("MUL_MAT requires 3 shape dims, got %d", len(shape))
		}
		M, K, N := shape[0], shape[1], shape[2]
		mappedWdt := mapWeightDtype(weightDtype)

		// Primary: direct interpolation from reference curves (Phase 2 design)
		lat := PredictMulMatDirect(profile, M, K, N, mappedWdt)
		if lat > 0 {
			return lat, nil
		}
		// Fallback: roofline (for backward compatibility with v2 profiles
		// that lack multi-(M,K) reference curves)
		lat = PredictMulMatLatency(&profile.Hardware, M, K, N, mappedWdt)
		if lat > 0 {
			return lat, nil
		}
		return 0, fmt.Errorf("no MUL_MAT calibration data for dtype %s — run daop-bench first", mappedWdt)

	default:
		// Delegate to existing lookupLatency for all other ops
		return lookupLatency(profile, op, shape, computeDtype, weightDtype, backend)
	}
}

// lookupOrchestrationOverhead queries the CPU overhead curve for a given graph size.
func lookupOrchestrationOverhead(profile *Profile, numNodes int, backend string) float64 {
	for _, c := range profile.Operators {
		if c.Op == "ORCHESTRATION_OVERHEAD" && c.Backend == backend {
			return Interpolate1D(c.Points, int64(numNodes))
		}
	}
	return 0
}

// estimatePhaseV3 computes total latency with fusion simulation and orchestration overhead.
func estimatePhaseV3(profile *Profile, nodes []ml.GraphNode, caps *BackendCapabilities, warnings *[]string) PhaseEstimation {
	// 1. Apply fusion
	var fusedNodes []ml.GraphNode
	if caps != nil {
		fusedNodes = ApplyFusion(nodes, caps.FusionRules)
	} else {
		fusedNodes = nodes
	}

	// 2. Sum per-op GPU time
	opStats := make(map[OpKey]*OpBreakdown)
	var totalGPUUs float64

	for _, fnode := range fusedNodes {
		if IsZeroCostOp(fnode.Op) {
			continue
		}
		op, shape, cdt, wdt := nodeToQueryShape(fnode)
		lat, err := lookupLatencyV3(profile, op, shape, cdt, wdt, fnode.Backend, caps)
		if err != nil {
			*warnings = append(*warnings, err.Error())
			continue
		}
		totalGPUUs += lat

		key := OpKey{op, fnode.Backend, cdt, wdt}
		if s, ok := opStats[key]; ok {
			s.Count++
			s.TotalUs += lat
		} else {
			opStats[key] = &OpBreakdown{
				Op: op, Backend: fnode.Backend,
				ComputeDtype: cdt, WeightDtype: wdt,
				Count: 1, TotalUs: lat,
			}
		}
	}

	// 3. Add orchestration overhead
	var overheadUs float64
	if caps != nil && caps.HasGPUTimestamp {
		overheadUs = lookupOrchestrationOverhead(profile, len(nodes), caps.Name)
	}
	totalUs := totalGPUUs + overheadUs

	// 4. Build top-ops breakdown
	var topOps []OpBreakdown
	for _, s := range opStats {
		if totalUs > 0 {
			s.Percentage = s.TotalUs / totalUs
		}
		topOps = append(topOps, *s)
	}
	sort.Slice(topOps, func(i, j int) bool { return topOps[i].TotalUs > topOps[j].TotalUs })
	if len(topOps) > 10 {
		topOps = topOps[:10]
	}

	tokPerSec := 0.0
	if totalUs > 0 {
		tokPerSec = 1e6 / totalUs
	}

	return PhaseEstimation{
		TotalLatencyMs: totalUs / 1000,
		TokensPerSec:   tokPerSec,
		TopOps:         topOps,
	}
}

// estimatePhase computes total latency for a set of graph nodes with per-op breakdown.
func estimatePhase(profile *Profile, nodes []ml.GraphNode, warnings *[]string) PhaseEstimation {
	opStats := make(map[OpKey]*OpBreakdown)
	var totalUs float64

	for _, node := range nodes {
		if IsZeroCostOp(node.Op) {
			continue
		}
		op, shape, cdt, wdt := nodeToQueryShape(node)
		lat, err := lookupLatency(profile, op, shape, cdt, wdt, node.Backend)
		if err != nil {
			*warnings = append(*warnings, err.Error())
			continue
		}
		totalUs += lat

		key := OpKey{op, node.Backend, cdt, wdt}
		if s, ok := opStats[key]; ok {
			s.Count++
			s.TotalUs += lat
		} else {
			opStats[key] = &OpBreakdown{
				Op: op, Backend: node.Backend,
				ComputeDtype: cdt, WeightDtype: wdt,
				Count: 1, TotalUs: lat,
			}
		}
	}

	var topOps []OpBreakdown
	for _, s := range opStats {
		if totalUs > 0 {
			s.Percentage = s.TotalUs / totalUs
		}
		topOps = append(topOps, *s)
	}
	sort.Slice(topOps, func(i, j int) bool { return topOps[i].TotalUs > topOps[j].TotalUs })
	if len(topOps) > 10 {
		topOps = topOps[:10]
	}

	tokPerSec := 0.0
	if totalUs > 0 {
		tokPerSec = 1e6 / totalUs
	}

	return PhaseEstimation{
		TotalLatencyMs: totalUs / 1000,
		TokensPerSec:   tokPerSec,
		TopOps:         topOps,
	}
}

// EstimateModel estimates inference performance for a model using a calibrated profile.
func EstimateModel(profile *Profile, modelPath string) (*EstimateResult, error) {
	ensureLibraryPath()

	// Single model load: skip weight buffer allocation, capture raw graph
	m, err := model.New(modelPath, ml.BackendParams{
		AllocMemory:     false,
		SkipWeightAlloc: true,
		FlashAttention:  ml.FlashAttentionEnabled,
	})
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	defer m.Backend().Close()

	// Build schedule from metadata + available devices
	blockCount := int(m.Backend().Config().Uint("block_count"))
	schedule := fullOffloadSchedule(m.Backend(), blockCount)
	if schedule != nil {
		slog.Info("estimate schedule", "strategy", "full_offload",
			"device", schedule[0].DeviceID.Library, "layers", len(schedule[0].Layers))
	} else {
		slog.Info("estimate schedule", "strategy", "cpu_only")
	}

	// Initialize KV cache for graph capture
	if cache := m.Config().Cache; cache != nil {
		cache.Init(m.Backend(), ml.DTypeF16, 1, 2048, 512)
	}

	// Capture raw graph nodes (no split_graph, backend="" initially)
	captureGraph := func(batchSize int) ([]ml.GraphNode, error) {
		ctx := m.Backend().NewContext()
		defer ctx.Close()

		batchInputs := make([]int32, batchSize)
		positions := make([]int32, batchSize)
		sequences := make([]int, batchSize)
		for i := 0; i < batchSize; i++ {
			positions[i] = int32(i)
		}
		batch := input.Batch{
			Inputs:    ctx.Input().FromInts(batchInputs, batchSize),
			Outputs:   ctx.Input().Empty(ml.DTypeI32, 1),
			Positions: positions,
			Sequences: sequences,
		}

		if cache := m.Config().Cache; cache != nil {
			if err := cache.StartForward(ctx, batch, true); err != nil {
				return nil, fmt.Errorf("cache start: %w", err)
			}
		}

		t, err := m.Forward(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("forward: %w", err)
		}

		ctx.SetBatchSize(batchSize)
		ctx.Forward(t)
		ctx.CaptureGraphRaw() // no split_graph needed

		return ctx.GraphNodes(), nil
	}

	prefillNodes, err := captureGraph(512)
	if err != nil {
		return nil, fmt.Errorf("prefill graph: %w", err)
	}
	decodeNodes, err := captureGraph(1)
	if err != nil {
		return nil, fmt.Errorf("decode graph: %w", err)
	}

	// Go-level backend assignment (replaces split_graph)
	assignBackends(prefillNodes, schedule, blockCount, "CPU")
	assignBackends(decodeNodes, schedule, blockCount, "CPU")

	result := &EstimateResult{Model: modelPath}

	// Use v3 path if backend caps are available, otherwise fall back to v2
	if profile.Version >= 3 && len(profile.BackendCaps) > 0 {
		var caps *BackendCapabilities
		for name := range profile.BackendCaps {
			c := GetBackendCapabilities(name)
			caps = &c
			break
		}
		result.Prefill = estimatePhaseV3(profile, prefillNodes, caps, &result.Warnings)
		result.Decode = estimatePhaseV3(profile, decodeNodes, caps, &result.Warnings)
	} else {
		result.Prefill = estimatePhase(profile, prefillNodes, &result.Warnings)
		result.Decode = estimatePhase(profile, decodeNodes, &result.Warnings)
	}

	result.PrefillLatencyUs = result.Prefill.TotalLatencyMs * 1000
	result.PrefillMs = result.Prefill.TotalLatencyMs
	if result.Decode.TotalLatencyMs > 0 {
		result.DecodeLatencyUsPerToken = result.Decode.TotalLatencyMs * 1000
		result.DecodeTokensPerSec = 1e6 / result.DecodeLatencyUsPerToken
	}

	return result, nil
}

// RunEstimate is the CLI entry point for estimation.
func RunEstimate(modelRef string, profilePath string) (*EstimateResult, error) {
	if profilePath == "" {
		profilePath = ProfilePath()
	}
	profile, err := LoadProfile(profilePath)
	if err != nil {
		return nil, fmt.Errorf("load profile: %w (have you run 'ollama daop-bench'?)", err)
	}

	ggufPath, err := ResolveModelPath(modelRef)
	if err != nil {
		return nil, err
	}

	return EstimateModel(profile, ggufPath)
}
