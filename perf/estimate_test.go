package perf

import (
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNodeToQueryShape_SILU(t *testing.T) {
	node := ml.GraphNode{
		Op:           "SILU",
		Backend:      "cuda",
		Shape:        [4]int64{4096, 1, 1, 1},
		ComputeDtype: "f32",
		InputShapes:  [][]int64{{4096, 1, 1, 1}},
	}
	op, shape, cdt, wdt := nodeToQueryShape(node)
	assert.Equal(t, "SILU", op)
	assert.Equal(t, []int64{4096}, shape) // total elements
	assert.Equal(t, "f32", cdt)
	assert.Equal(t, "", wdt)
}

func TestNodeToQueryShape_SILU_MultiDim(t *testing.T) {
	node := ml.GraphNode{
		Op:           "SILU",
		Backend:      "cuda",
		Shape:        [4]int64{4096, 32, 1, 1},
		ComputeDtype: "f32",
		InputShapes:  [][]int64{{4096, 32, 1, 1}},
	}
	op, shape, _, _ := nodeToQueryShape(node)
	assert.Equal(t, "SILU", op)
	assert.Equal(t, []int64{131072}, shape)
}

func TestNodeToQueryShape_MulMat(t *testing.T) {
	node := ml.GraphNode{
		Op:           "MUL_MAT",
		Backend:      "cuda",
		Shape:        [4]int64{4096, 32, 1, 1},
		ComputeDtype: "f16",
		WeightDtype:  "q4_0",
		InputShapes:  [][]int64{{4096, 4096}, {4096, 32}},
	}
	op, shape, cdt, wdt := nodeToQueryShape(node)
	assert.Equal(t, "MUL_MAT", op)
	require.Len(t, shape, 3)
	assert.Equal(t, int64(4096), shape[0]) // M
	assert.Equal(t, int64(4096), shape[1]) // K
	assert.Equal(t, int64(32), shape[2])   // N
	assert.Equal(t, "f16", cdt)
	assert.Equal(t, "q4_0", wdt)
}

func TestNodeToQueryShape_MulMat_LargeN(t *testing.T) {
	node := ml.GraphNode{
		Op:           "MUL_MAT",
		Backend:      "cuda",
		ComputeDtype: "f16",
		WeightDtype:  "q4_0",
		InputShapes:  [][]int64{{14336, 4096}, {4096, 512}},
	}
	_, shape, _, _ := nodeToQueryShape(node)
	assert.Equal(t, int64(14336), shape[0]) // M
	assert.Equal(t, int64(4096), shape[1])  // K
	assert.Equal(t, int64(512), shape[2])   // N
}

func TestNodeToQueryShape_FlashAttn(t *testing.T) {
	node := ml.GraphNode{
		Op:           "FLASH_ATTN_EXT",
		Backend:      "cuda",
		ComputeDtype: "f16",
		InputShapes: [][]int64{
			{128, 32, 1, 1},    // Q
			{128, 32, 2048, 1}, // K
		},
	}
	op, shape, _, _ := nodeToQueryShape(node)
	assert.Equal(t, "FLASH_ATTN_EXT", op)
	require.Len(t, shape, 2)
	assert.Equal(t, int64(1), shape[0])    // seq_q
	assert.Equal(t, int64(2048), shape[1]) // seq_kv
}

func TestNodeToQueryShape_FlashAttn_Prefill(t *testing.T) {
	node := ml.GraphNode{
		Op:           "FLASH_ATTN_EXT",
		Backend:      "cuda",
		ComputeDtype: "f16",
		InputShapes: [][]int64{
			{128, 32, 512, 1},
			{128, 32, 512, 1},
		},
	}
	_, shape, _, _ := nodeToQueryShape(node)
	assert.Equal(t, int64(512), shape[0])
	assert.Equal(t, int64(512), shape[1])
}

func TestNodeToQueryShape_UnknownOp(t *testing.T) {
	node := ml.GraphNode{
		Op:           "CUSTOM_OP",
		Backend:      "cuda",
		Shape:        [4]int64{256, 32, 1, 1},
		ComputeDtype: "f32",
	}
	_, shape, _, _ := nodeToQueryShape(node)
	assert.Equal(t, []int64{8192}, shape)
}

func TestNodeToQueryShape_MulMat_InsufficientInputShapes(t *testing.T) {
	node := ml.GraphNode{
		Op:           "MUL_MAT",
		Backend:      "cuda",
		ComputeDtype: "f16",
		InputShapes:  [][]int64{{4096, 4096}},
	}
	_, shape, _, _ := nodeToQueryShape(node)
	assert.NotEmpty(t, shape)
}

// --- lookupLatency tests ---

func makeTestProfileForEstimation() *Profile {
	return &Profile{
		Version: 2,
		Hardware: HardwareProfile{
			Backends:                 []BackendInfo{{Name: "cuda", Device: "RTX 4090"}},
			PeakTOPS:                 map[string]float64{"f16": 330e12},
			PeakBandwidthBytesPerSec: 1008e9,
			BalancePoints:            map[string]float64{"f16": 327.38},
		},
		Operators: []OperatorCurve{
			// SILU 1D curve
			{
				Op: "SILU", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 2.5},
					{Shape: []int64{65536}, LatencyUs: 15.0},
					{Shape: []int64{1048576}, LatencyUs: 200.0},
					{Shape: []int64{16777216}, LatencyUs: 3000.0},
				},
			},
			// MUL_MAT curve 1: (M=4096, K=4096), points store [N] only
			{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Dimensions: []string{"N"},
				FixedDims:  map[string]int64{"M": 4096, "K": 4096},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 10.0},
					{Shape: []int64{32}, LatencyUs: 50.0},
					{Shape: []int64{256}, LatencyUs: 200.0},
					{Shape: []int64{4096}, LatencyUs: 3000.0},
				},
			},
			// MUL_MAT curve 2: (M=14336, K=4096), points store [N] only
			{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Dimensions: []string{"N"},
				FixedDims:  map[string]int64{"M": 14336, "K": 4096},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 25.0},
					{Shape: []int64{32}, LatencyUs: 120.0},
					{Shape: []int64{4096}, LatencyUs: 8000.0},
				},
			},
			// FLASH_ATTN_EXT curve
			{
				Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
				Dimensions: []string{"seq_q", "seq_kv"},
				FixedDims:  map[string]int64{"num_heads": 32, "head_dim": 128},
				Points: []LatencyPoint{
					// Decode points (seq_q=1)
					{Shape: []int64{1, 128}, LatencyUs: 5.0},
					{Shape: []int64{1, 512}, LatencyUs: 15.0},
					{Shape: []int64{1, 2048}, LatencyUs: 55.0},
					// Prefill points (seq_q=seq_kv)
					{Shape: []int64{128, 128}, LatencyUs: 20.0},
					{Shape: []int64{512, 512}, LatencyUs: 100.0},
					{Shape: []int64{2048, 2048}, LatencyUs: 500.0},
				},
			},
		},
	}
}

func TestLookupLatency_SILU(t *testing.T) {
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "SILU", []int64{65536}, "f32", "", "cuda")
	require.NoError(t, err)
	assert.InDelta(t, 15.0, lat, 0.001, "exact match at measured point")
}

func TestLookupLatency_SILU_Interpolated(t *testing.T) {
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "SILU", []int64{500000}, "f32", "", "cuda")
	require.NoError(t, err)
	assert.Greater(t, lat, 15.0)
	assert.Less(t, lat, 200.0)
}

func TestLookupLatency_MulMat_ExactMK(t *testing.T) {
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "MUL_MAT", []int64{4096, 4096, 1}, "f16", "q4_0", "cuda")
	require.NoError(t, err)
	assert.InDelta(t, 10.0, lat, 0.001, "exact M,K,N match")
}

func TestLookupLatency_MulMat_InterpolatedN(t *testing.T) {
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "MUL_MAT", []int64{4096, 4096, 128}, "f16", "q4_0", "cuda")
	require.NoError(t, err)
	assert.Greater(t, lat, 50.0)
	assert.Less(t, lat, 200.0)
}

func TestLookupLatency_FlashAttn_Decode(t *testing.T) {
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "FLASH_ATTN_EXT", []int64{1, 512}, "f16", "", "cuda")
	require.NoError(t, err)
	assert.InDelta(t, 15.0, lat, 0.5)
}

func TestLookupLatency_FlashAttn_Prefill(t *testing.T) {
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "FLASH_ATTN_EXT", []int64{512, 512}, "f16", "", "cuda")
	require.NoError(t, err)
	assert.InDelta(t, 100.0, lat, 1.0)
}

func TestLookupLatency_Uncalibrated(t *testing.T) {
	p := makeTestProfileForEstimation()
	_, err := lookupLatency(p, "GELU", []int64{4096}, "f32", "", "cuda")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "uncalibrated")
}

func TestLookupLatency_WrongBackend(t *testing.T) {
	p := makeTestProfileForEstimation()
	_, err := lookupLatency(p, "SILU", []int64{4096}, "f32", "", "cpu")
	assert.Error(t, err)
}

// --- estimatePhase tests ---

func TestEstimatePhase_SimpleGraph(t *testing.T) {
	p := makeTestProfileForEstimation()
	nodes := []ml.GraphNode{
		{Op: "SILU", Backend: "cuda", Shape: [4]int64{65536, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "SILU", Backend: "cuda", Shape: [4]int64{65536, 1, 1, 1}, ComputeDtype: "f32"},
	}
	var warnings []string
	result := estimatePhase(p, nodes, &warnings)
	assert.Empty(t, warnings)
	assert.InDelta(t, 0.03, result.TotalLatencyMs, 0.001)
	assert.Len(t, result.TopOps, 1)
	assert.Equal(t, "SILU", result.TopOps[0].Op)
	assert.Equal(t, 2, result.TopOps[0].Count)
}

func TestEstimatePhase_MixedOps(t *testing.T) {
	p := makeTestProfileForEstimation()
	nodes := []ml.GraphNode{
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		{Op: "SILU", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{14336, 4096}, {4096, 1}}},
	}
	var warnings []string
	result := estimatePhase(p, nodes, &warnings)
	assert.Empty(t, warnings)
	assert.Greater(t, result.TotalLatencyMs, 0.0)
	assert.Equal(t, "MUL_MAT", result.TopOps[0].Op)
	assert.Greater(t, result.TopOps[0].Percentage, 0.5)
}

func TestEstimatePhase_SkipsZeroCostOps(t *testing.T) {
	p := makeTestProfileForEstimation()
	nodes := []ml.GraphNode{
		{Op: "VIEW", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "RESHAPE", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "SILU", Backend: "cuda", Shape: [4]int64{65536, 1, 1, 1}, ComputeDtype: "f32"},
	}
	var warnings []string
	result := estimatePhase(p, nodes, &warnings)
	assert.Len(t, result.TopOps, 1)
	assert.Equal(t, "SILU", result.TopOps[0].Op)
}

func TestEstimatePhase_UncalibratedWarning(t *testing.T) {
	p := makeTestProfileForEstimation()
	nodes := []ml.GraphNode{
		{Op: "GELU", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
	}
	var warnings []string
	result := estimatePhase(p, nodes, &warnings)
	assert.NotEmpty(t, warnings)
	assert.Contains(t, warnings[0], "uncalibrated")
	assert.InDelta(t, 0.0, result.TotalLatencyMs, 0.001)
}

func TestEstimatePhase_EmptyGraph(t *testing.T) {
	p := makeTestProfileForEstimation()
	var warnings []string
	result := estimatePhase(p, nil, &warnings)
	assert.InDelta(t, 0.0, result.TotalLatencyMs, 0.001)
	assert.Empty(t, result.TopOps)
}

func TestEstimatePhase_LlamaLikeDecodeLayer(t *testing.T) {
	p := makeTestProfileForEstimation()
	nodes := []ml.GraphNode{
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		{Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
			InputShapes: [][]int64{{128, 32, 1, 1}, {128, 32, 2048, 1}}},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{14336, 4096}, {4096, 1}}},
		{Op: "SILU", Backend: "cuda", Shape: [4]int64{14336, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 14336}, {14336, 1}}},
		{Op: "VIEW", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
	}

	var warnings []string
	result := estimatePhase(p, nodes, &warnings)

	assert.Greater(t, result.TotalLatencyMs, 0.0)
	assert.Equal(t, "MUL_MAT", result.TopOps[0].Op)
	assert.Greater(t, result.TopOps[0].Percentage, 0.5,
		"MUL_MAT should be >50%% of decode latency")
}

func TestLookupLatency_MulMat_InsufficientShape(t *testing.T) {
	p := makeTestProfileForEstimation()
	_, err := lookupLatency(p, "MUL_MAT", []int64{4096, 4096}, "f16", "q4_0", "cuda")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "requires 3 shape dims")
}

func TestLookupLatency_FlashAttn_InsufficientShape(t *testing.T) {
	p := makeTestProfileForEstimation()
	_, err := lookupLatency(p, "FLASH_ATTN_EXT", []int64{1}, "f16", "", "cuda")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "requires 2 shape dims")
}

func TestEstimatePhase_AllNodesUncalibrated(t *testing.T) {
	p := makeTestProfileForEstimation()
	nodes := []ml.GraphNode{
		{Op: "GELU", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "TANH", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "RELU", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
	}
	var warnings []string
	result := estimatePhase(p, nodes, &warnings)
	assert.Len(t, warnings, 3, "should warn for each uncalibrated op")
	assert.InDelta(t, 0.0, result.TotalLatencyMs, 0.001)
	assert.Empty(t, result.TopOps)
}

func TestEstimatePhase_PartialUncalibrated(t *testing.T) {
	p := makeTestProfileForEstimation()
	nodes := []ml.GraphNode{
		{Op: "SILU", Backend: "cuda", Shape: [4]int64{65536, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "GELU", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
	}
	var warnings []string
	result := estimatePhase(p, nodes, &warnings)
	assert.Len(t, warnings, 1, "only GELU should warn")
	assert.Contains(t, warnings[0], "GELU")
	assert.Greater(t, result.TotalLatencyMs, 0.0)
	require.Len(t, result.TopOps, 1)
	assert.Equal(t, "SILU", result.TopOps[0].Op)
}
