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
			PeakTOPS:                 map[string]float64{"f16": 330e12, "f32": 82.6e12},
			PeakBandwidthBytesPerSec: 1008e9,
			BalancePoints:            map[string]float64{"f16": 327.38},
			EfficiencyConstants: map[string]OpEfficiency{
				"MUL_MAT_f16":  {ComputeEff: 0.95, BWEff: 0.80, OverheadUs: 5},
				"MUL_MAT_q4_0": {ComputeEff: 0.90, BWEff: 0.70, OverheadUs: 10},
			},
		},
		Operators: []OperatorCurve{
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
			{
				Op: "ADD", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 2.0},
					{Shape: []int64{65536}, LatencyUs: 12.0},
					{Shape: []int64{1048576}, LatencyUs: 180.0},
					{Shape: []int64{16777216}, LatencyUs: 2800.0},
				},
			},
			{
				Op: "MUL", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 2.0},
					{Shape: []int64{65536}, LatencyUs: 12.0},
					{Shape: []int64{1048576}, LatencyUs: 180.0},
					{Shape: []int64{16777216}, LatencyUs: 2800.0},
				},
			},
			{
				Op: "GELU", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 2.5},
					{Shape: []int64{65536}, LatencyUs: 15.0},
					{Shape: []int64{1048576}, LatencyUs: 200.0},
					{Shape: []int64{16777216}, LatencyUs: 3000.0},
				},
			},
			{
				Op: "SOFT_MAX", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 3.0},
					{Shape: []int64{65536}, LatencyUs: 16.0},
					{Shape: []int64{1048576}, LatencyUs: 210.0},
					{Shape: []int64{16777216}, LatencyUs: 3100.0},
				},
			},
			{
				Op: "CONT", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 1.5},
					{Shape: []int64{65536}, LatencyUs: 8.0},
					{Shape: []int64{1048576}, LatencyUs: 120.0},
					{Shape: []int64{16777216}, LatencyUs: 1800.0},
				},
			},
			{
				Op: "RMS_NORM", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 3.0},
					{Shape: []int64{65536}, LatencyUs: 18.0},
					{Shape: []int64{1048576}, LatencyUs: 220.0},
					{Shape: []int64{16777216}, LatencyUs: 3200.0},
				},
			},
			{
				Op: "GET_ROWS", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 5.0},
					{Shape: []int64{32}, LatencyUs: 8.0},
					{Shape: []int64{512}, LatencyUs: 30.0},
					{Shape: []int64{4096}, LatencyUs: 200.0},
				},
			},
			{
				Op: "ROPE", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 3.5},
					{Shape: []int64{65536}, LatencyUs: 20.0},
					{Shape: []int64{1048576}, LatencyUs: 250.0},
					{Shape: []int64{16777216}, LatencyUs: 3500.0},
				},
			},
			{
				Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
				Dimensions: []string{"seq_q", "seq_kv"},
				FixedDims:  map[string]int64{"num_heads": 32, "head_dim": 128},
				Points: []LatencyPoint{
					{Shape: []int64{1, 128}, LatencyUs: 5.0},
					{Shape: []int64{1, 512}, LatencyUs: 15.0},
					{Shape: []int64{1, 2048}, LatencyUs: 55.0},
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

func TestLookupLatency_MulMat_Roofline(t *testing.T) {
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "MUL_MAT", []int64{4096, 4096, 1}, "f16", "q4_0", "cuda")
	require.NoError(t, err)
	assert.Greater(t, lat, 0.0, "roofline prediction should return positive latency")
}

func TestLookupLatency_MulMat_ScalesWithN(t *testing.T) {
	p := makeTestProfileForEstimation()
	lat1, _ := lookupLatency(p, "MUL_MAT", []int64{4096, 4096, 1}, "f16", "q4_0", "cuda")
	lat4096, _ := lookupLatency(p, "MUL_MAT", []int64{4096, 4096, 4096}, "f16", "q4_0", "cuda")
	assert.Greater(t, lat4096, lat1, "latency should increase with N")
}

func TestLookupLatency_MulMat_DtypeMapping(t *testing.T) {
	p := makeTestProfileForEstimation()
	// q4_K should map to q4_0
	lat, err := lookupLatency(p, "MUL_MAT", []int64{4096, 4096, 1}, "f16", "q4_K", "cuda")
	require.NoError(t, err)
	assert.Greater(t, lat, 0.0, "q4_K should map to q4_0 and succeed")
}

func TestLookupLatency_MulMat_NoEfficiencyConstants(t *testing.T) {
	p := &Profile{
		Version: 2,
		Hardware: HardwareProfile{
			Backends:                 []BackendInfo{{Name: "cuda", Device: "RTX 4090"}},
			PeakTOPS:                 map[string]float64{"f32": 50e9},
			PeakBandwidthBytesPerSec: 27e9,
		},
	}
	_, err := lookupLatency(p, "MUL_MAT", []int64{4096, 4096, 1}, "f16", "q4_0", "cuda")
	assert.Error(t, err, "should error when no efficiency constants available")
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
	_, err := lookupLatency(p, "TANH", []int64{4096}, "f32", "", "cuda")
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
		{Op: "TANH", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
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
		{Op: "TANH", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "RELU", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "SWIGLU", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
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
		{Op: "TANH", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
	}
	var warnings []string
	result := estimatePhase(p, nodes, &warnings)
	assert.Len(t, warnings, 1, "only TANH should warn")
	assert.Contains(t, warnings[0], "TANH")
	assert.Greater(t, result.TotalLatencyMs, 0.0)
	require.Len(t, result.TopOps, 1)
	assert.Equal(t, "SILU", result.TopOps[0].Op)
}

func TestLookupLatency_NewOps(t *testing.T) {
	p := makeTestProfileForEstimation()
	tests := []struct {
		op    string
		shape []int64
	}{
		{"ADD", []int64{65536}},
		{"MUL", []int64{65536}},
		{"GELU", []int64{65536}},
		{"SOFT_MAX", []int64{65536}},
		{"CONT", []int64{65536}},
		{"RMS_NORM", []int64{65536}},
		{"ROPE", []int64{4096}},
		{"GET_ROWS", []int64{32}},
	}
	for _, tt := range tests {
		t.Run(tt.op, func(t *testing.T) {
			lat, err := lookupLatency(p, tt.op, tt.shape, "f32", "", "cuda")
			require.NoError(t, err, "op %s should not be uncalibrated", tt.op)
			assert.Greater(t, lat, 0.0, "op %s should return positive latency", tt.op)
		})
	}
}

func TestLookupLatency_NewOps_Interpolated(t *testing.T) {
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "ADD", []int64{500000}, "f32", "", "cuda")
	require.NoError(t, err)
	assert.Greater(t, lat, 12.0, "should be > latency at 65536")
	assert.Less(t, lat, 180.0, "should be < latency at 1048576")
}

func TestEstimatePhase_LlamaDecodeLayerNoWarnings(t *testing.T) {
	p := makeTestProfileForEstimation()
	nodes := []ml.GraphNode{
		{Op: "RMS_NORM", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "MUL", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		{Op: "ROPE", Backend: "cuda", Shape: [4]int64{128, 32, 1, 1}, ComputeDtype: "f32"},
		{Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
			InputShapes: [][]int64{{128, 32, 1, 1}, {128, 32, 2048, 1}}},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		{Op: "ADD", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "RMS_NORM", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "MUL", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{14336, 4096}, {4096, 1}}},
		{Op: "SILU", Backend: "cuda", Shape: [4]int64{14336, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 14336}, {14336, 1}}},
		{Op: "ADD", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "VIEW", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "TRANSPOSE", Backend: "cuda", Shape: [4]int64{4096, 32, 1, 1}, ComputeDtype: "f32"},
	}

	var warnings []string
	result := estimatePhase(p, nodes, &warnings)

	assert.Empty(t, warnings, "should have no uncalibrated warnings for standard Llama ops")
	assert.Greater(t, result.TotalLatencyMs, 0.0)
	assert.Equal(t, "MUL_MAT", result.TopOps[0].Op, "MUL_MAT should dominate")
}

func TestEstimatePhase_GemmaDecodeLayerNoWarnings(t *testing.T) {
	p := makeTestProfileForEstimation()
	nodes := []ml.GraphNode{
		{Op: "RMS_NORM", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		{Op: "ROPE", Backend: "cuda", Shape: [4]int64{128, 32, 1, 1}, ComputeDtype: "f32"},
		{Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
			InputShapes: [][]int64{{128, 32, 1, 1}, {128, 32, 2048, 1}}},
		{Op: "SOFT_MAX", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "ADD", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{14336, 4096}, {4096, 1}}},
		{Op: "GELU", Backend: "cuda", Shape: [4]int64{14336, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 14336}, {14336, 1}}},
		{Op: "ADD", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "CONT", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
	}

	var warnings []string
	result := estimatePhase(p, nodes, &warnings)

	assert.Empty(t, warnings, "should have no uncalibrated warnings for Gemma-like ops")
	assert.Greater(t, result.TotalLatencyMs, 0.0)
}
