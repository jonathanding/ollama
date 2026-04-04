package perf

import (
	"fmt"
	"sort"

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
	case "MUL_MAT":
		if len(node.InputShapes) >= 2 && len(node.InputShapes[0]) >= 2 && len(node.InputShapes[1]) >= 2 {
			M := node.InputShapes[0][0]
			K := node.InputShapes[0][1]
			N := node.InputShapes[1][1]
			shape = []int64{M, K, N}
			return
		}
		shape = []int64{totalElements(node.Shape)}
		return

	case "FLASH_ATTN_EXT":
		if len(node.InputShapes) >= 2 && len(node.InputShapes[0]) >= 3 && len(node.InputShapes[1]) >= 3 {
			seqQ := node.InputShapes[0][2]
			seqKV := node.InputShapes[1][2]
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

// buildModelGraphNodes loads a model and captures prefill+decode computation graphs.
func buildModelGraphNodes(modelPath string) (prefill, decode []ml.GraphNode, err error) {
	m, err := model.New(modelPath, ml.BackendParams{AllocMemory: false})
	if err != nil {
		return nil, nil, fmt.Errorf("load model: %w", err)
	}
	defer m.Backend().Close()

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
		ctx.Forward(t).Reserve()

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
		for i := range profile.Operators {
			c := &profile.Operators[i]
			if c.Op == op && c.ComputeDtype == computeDtype && c.Backend == backend {
				return InterpolateFlashAttn(c, shape[0], shape[1]), nil
			}
		}
		return 0, fmt.Errorf("uncalibrated: %s(%s on %s)", op, computeDtype, backend)

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
	prefillNodes, decodeNodes, err := buildModelGraphNodes(modelPath)
	if err != nil {
		return nil, err
	}

	result := &EstimateResult{Model: modelPath}
	result.Prefill = estimatePhase(profile, prefillNodes, &result.Warnings)
	result.Decode = estimatePhase(profile, decodeNodes, &result.Warnings)

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
