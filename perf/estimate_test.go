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
	// InputShapes[0] = {4096, 4096}: ne[0]=K=4096, ne[1]=M=4096
	assert.Equal(t, int64(4096), shape[0]) // M (weight ne[1])
	assert.Equal(t, int64(4096), shape[1]) // K (weight ne[0])
	assert.Equal(t, int64(32), shape[2])   // N (activation ne[1])
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
	// InputShapes[0] = {14336, 4096}: ne[0]=K=14336, ne[1]=M=4096
	assert.Equal(t, int64(4096), shape[0])  // M (weight ne[1])
	assert.Equal(t, int64(14336), shape[1]) // K (weight ne[0])
	assert.Equal(t, int64(512), shape[2])   // N (activation ne[1])
}

func TestNodeToQueryShape_FlashAttn(t *testing.T) {
	node := ml.GraphNode{
		Op:           "FLASH_ATTN_EXT",
		Backend:      "cuda",
		ComputeDtype: "f16",
		InputShapes: [][]int64{
			{128, 1, 32, 1},    // Q: ne=[head_dim, seqQ=1, num_heads, batch]
			{128, 2048, 32, 1}, // K: ne=[head_dim, seqKV=2048, num_kv_heads, batch]
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
			{128, 512, 32, 1}, // Q: ne=[head_dim, seqQ=512, num_heads, batch]
			{128, 512, 32, 1}, // K: ne=[head_dim, seqKV=512, num_kv_heads, batch]
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
				Op: "RELU", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 2.0},
					{Shape: []int64{65536}, LatencyUs: 12.0},
					{Shape: []int64{1048576}, LatencyUs: 170.0},
					{Shape: []int64{16777216}, LatencyUs: 2600.0},
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
	// q4_K now passes through as identity (not mapped to q4_0).
	// Without calibration data for q4_K, it should fail until fallback chain is added (Task 5).
	_, err := lookupLatency(p, "MUL_MAT", []int64{4096, 4096, 1}, "f16", "q4_K", "cuda")
	assert.Error(t, err, "q4_K should fail without calibration data (before Task 5 fallback)")
	assert.Contains(t, err.Error(), "no efficiency constants")
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
			InputShapes: [][]int64{{128, 1, 32, 1}, {128, 2048, 32, 1}}},
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
		{Op: "SWIGLU", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "NORM", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
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
	}
	for _, tt := range tests {
		t.Run(tt.op, func(t *testing.T) {
			lat, err := lookupLatency(p, tt.op, tt.shape, "f32", "", "cuda")
			require.NoError(t, err, "op %s should not be uncalibrated", tt.op)
			assert.Greater(t, lat, 0.0, "op %s should return positive latency", tt.op)
		})
	}
}

func TestLookupLatency_GET_ROWS_FixedConstant(t *testing.T) {
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "GET_ROWS", []int64{32}, "f32", "", "cuda")
	require.NoError(t, err, "GET_ROWS should use fixed constant, not require a curve")
	assert.InDelta(t, 10.0, lat, 0.001, "GET_ROWS fixed constant should be 10μs")
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
			InputShapes: [][]int64{{128, 1, 32, 1}, {128, 2048, 32, 1}}},
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
			InputShapes: [][]int64{{128, 1, 32, 1}, {128, 2048, 32, 1}}},
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

// --- v3 estimation tests ---

func TestLookupLatencyV3_MulMatRooflineFallback(t *testing.T) {
	// Profile with roofline efficiency constants but no reference curves.
	// All N values fall through PredictMulMatDirect (returns 0, no curves)
	// and use PredictMulMatLatency (roofline) as fallback.
	profile := &Profile{
		Version: 3,
		Hardware: HardwareProfile{
			PeakTOPS:                 map[string]float64{"f16": 55e9, "f32": 59e9},
			PeakBandwidthBytesPerSec: 37e9,
			EfficiencyConstants: map[string]OpEfficiency{
				"MUL_MAT_f16":     {ComputeEff: 0.8, BWEff: 0.5, OverheadUs: 100},
				"MUL_MAT_VEC_f16": {ComputeEff: 0.5, BWEff: 0.7, OverheadUs: 10},
			},
		},
	}
	caps := GetBackendCapabilities("Vulkan")

	// N=1 — routes through direct (no curves) → roofline fallback
	latN1, err := lookupLatencyV3(profile, "MUL_MAT", []int64{4096, 4096, 1}, "f32", "f16", "Vulkan", &caps)
	require.NoError(t, err)
	assert.Greater(t, latN1, 0.0)

	// N=8
	latN8, err := lookupLatencyV3(profile, "MUL_MAT", []int64{4096, 4096, 8}, "f32", "f16", "Vulkan", &caps)
	require.NoError(t, err)
	assert.Greater(t, latN8, 0.0)

	// N=9
	latN9, err := lookupLatencyV3(profile, "MUL_MAT", []int64{4096, 4096, 9}, "f32", "f16", "Vulkan", &caps)
	require.NoError(t, err)
	assert.Greater(t, latN9, 0.0)

	// N=512
	latN512, err := lookupLatencyV3(profile, "MUL_MAT", []int64{4096, 4096, 512}, "f32", "f16", "Vulkan", &caps)
	require.NoError(t, err)
	assert.Greater(t, latN512, 0.0)

	// N=1 should be faster than N=512 (roofline: both BW and compute scale with N)
	assert.Less(t, latN1, latN512, "N=1 should be faster than N=512")
}

func TestLookupLatencyV3_MulMatRooflineOnly(t *testing.T) {
	// Profile with MUL_MAT roofline constants only (no reference curves).
	// Direct interpolation returns 0, roofline provides the estimate.
	profile := &Profile{
		Version: 3,
		Hardware: HardwareProfile{
			PeakTOPS:                 map[string]float64{"f16": 55e9},
			PeakBandwidthBytesPerSec: 37e9,
			EfficiencyConstants: map[string]OpEfficiency{
				"MUL_MAT_f16": {ComputeEff: 0.8, BWEff: 0.5, OverheadUs: 100},
			},
		},
	}
	caps := GetBackendCapabilities("Vulkan")

	// No curves → direct returns 0 → roofline fallback
	lat, err := lookupLatencyV3(profile, "MUL_MAT", []int64{4096, 4096, 1}, "f32", "f16", "Vulkan", &caps)
	require.NoError(t, err)
	assert.Greater(t, lat, 0.0, "roofline fallback should produce positive latency")
}

func TestLookupLatencyV3_MulMatNoCapsNoVec(t *testing.T) {
	profile := &Profile{
		Version: 3,
		Hardware: HardwareProfile{
			PeakTOPS:                 map[string]float64{"f16": 55e9},
			PeakBandwidthBytesPerSec: 37e9,
			EfficiencyConstants: map[string]OpEfficiency{
				"MUL_MAT_f16": {ComputeEff: 0.8, BWEff: 0.5, OverheadUs: 100},
			},
		},
	}
	cpuCaps := GetBackendCapabilities("CPU") // HasMulMatVec=false

	lat, err := lookupLatencyV3(profile, "MUL_MAT", []int64{4096, 4096, 1}, "f32", "f16", "CPU", &cpuCaps)
	require.NoError(t, err)
	assert.Greater(t, lat, 0.0, "should use regular MUL_MAT path for CPU")
}

func TestLookupLatencyV3_MulMatAdd(t *testing.T) {
	// MUL_MAT_ADD shares the same routing path as MUL_MAT.
	// With no reference curves, falls through to roofline.
	profile := &Profile{
		Version: 3,
		Hardware: HardwareProfile{
			PeakTOPS:                 map[string]float64{"f16": 55e9},
			PeakBandwidthBytesPerSec: 37e9,
			EfficiencyConstants: map[string]OpEfficiency{
				"MUL_MAT_f16":     {ComputeEff: 0.8, BWEff: 0.5, OverheadUs: 100},
				"MUL_MAT_VEC_f16": {ComputeEff: 0.5, BWEff: 0.7, OverheadUs: 10},
			},
		},
	}
	caps := GetBackendCapabilities("Vulkan")

	// MUL_MAT_ADD routes through same path as MUL_MAT
	lat, err := lookupLatencyV3(profile, "MUL_MAT_ADD", []int64{4096, 4096, 1}, "f32", "f16", "Vulkan", &caps)
	require.NoError(t, err)
	assert.Greater(t, lat, 0.0)
}

func TestLookupLatencyV3_DelegatesNonMulMat(t *testing.T) {
	profile := newTestProfile() // already has SILU curve
	caps := GetBackendCapabilities("Vulkan")

	// SILU should delegate to lookupLatency (v2 path)
	lat, err := lookupLatencyV3(profile, "SILU", []int64{4096}, "f32", "", "cuda", &caps)
	require.NoError(t, err)
	assert.Greater(t, lat, 0.0)
}

func TestEstimatePhaseV3_Fusion(t *testing.T) {
	profile := &Profile{
		Version: 3,
		Hardware: HardwareProfile{
			PeakTOPS:                 map[string]float64{"f32": 59e9},
			PeakBandwidthBytesPerSec: 37e9,
		},
		Operators: []OperatorCurve{
			{
				Op: "RMS_NORM_MUL", Backend: "Vulkan", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 12},
					{Shape: []int64{4096}, LatencyUs: 15},
				},
			},
		},
	}
	caps := GetBackendCapabilities("Vulkan")

	// Graph: RMS_NORM + MUL → should fuse to RMS_NORM_MUL
	nodes := []ml.GraphNode{
		{Op: "RMS_NORM", Backend: "Vulkan", ComputeDtype: "f32", Shape: [4]int64{2048, 1, 1, 1}},
		{Op: "MUL", Backend: "Vulkan", ComputeDtype: "f32", Shape: [4]int64{2048, 1, 1, 1}},
	}
	var warnings []string
	result := estimatePhaseV3(profile, nodes, &caps, &warnings)

	assert.Greater(t, result.TotalLatencyMs, 0.0)
	// Should have 1 fused op in breakdown, not 2 separate ops
	assert.Len(t, result.TopOps, 1)
	assert.Equal(t, "RMS_NORM_MUL", result.TopOps[0].Op)
	assert.Equal(t, 1, result.TopOps[0].Count)
}

func TestEstimatePhaseV3_NilCaps(t *testing.T) {
	profile := newTestProfile()

	nodes := []ml.GraphNode{
		{Op: "SILU", Backend: "cuda", ComputeDtype: "f32", Shape: [4]int64{4096, 1, 1, 1}},
	}
	var warnings []string
	result := estimatePhaseV3(profile, nodes, nil, &warnings)

	assert.Greater(t, result.TotalLatencyMs, 0.0)
	assert.Len(t, result.TopOps, 1)
	assert.Equal(t, "SILU", result.TopOps[0].Op)
}

func TestEstimatePhaseV3_OrchestrationOverhead(t *testing.T) {
	profile := &Profile{
		Version: 3,
		Hardware: HardwareProfile{
			PeakTOPS:                 map[string]float64{"f32": 59e9},
			PeakBandwidthBytesPerSec: 37e9,
		},
		Operators: []OperatorCurve{
			{
				Op: "ADD", Backend: "Vulkan", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 5},
					{Shape: []int64{4096}, LatencyUs: 8},
				},
			},
			{
				Op: "ORCHESTRATION_OVERHEAD", Backend: "Vulkan", ComputeDtype: "f32",
				Dimensions: []string{"num_nodes"},
				Points: []LatencyPoint{
					{Shape: []int64{50}, LatencyUs: 3000},
					{Shape: []int64{100}, LatencyUs: 5500},
					{Shape: []int64{300}, LatencyUs: 15000},
				},
			},
		},
	}
	caps := GetBackendCapabilities("Vulkan")

	// 10 ADD nodes
	nodes := make([]ml.GraphNode, 10)
	for i := range nodes {
		nodes[i] = ml.GraphNode{Op: "ADD", Backend: "Vulkan", ComputeDtype: "f32",
			Shape: [4]int64{2048, 1, 1, 1}}
	}

	var warnings []string
	result := estimatePhaseV3(profile, nodes, &caps, &warnings)

	// Total includes GPU time + orchestration overhead
	assert.Greater(t, result.TotalLatencyMs, 0.0)
	// Verify overhead is non-zero (interpolated from overhead curve)
	gpuOnlyUs := float64(0)
	for _, op := range result.TopOps {
		gpuOnlyUs += op.TotalUs
	}
	totalUs := result.TotalLatencyMs * 1000
	assert.Greater(t, totalUs, gpuOnlyUs, "total should include overhead beyond GPU time")
}

func TestLookupOrchestrationOverhead(t *testing.T) {
	profile := &Profile{
		Operators: []OperatorCurve{
			{
				Op: "ORCHESTRATION_OVERHEAD", Backend: "Vulkan", ComputeDtype: "f32",
				Dimensions: []string{"num_nodes"},
				Points: []LatencyPoint{
					{Shape: []int64{50}, LatencyUs: 3000},
					{Shape: []int64{100}, LatencyUs: 5500},
				},
			},
		},
	}

	// Exact match
	oh50 := lookupOrchestrationOverhead(profile, 50, "Vulkan")
	assert.InDelta(t, 3000.0, oh50, 1.0)

	// Interpolated
	oh75 := lookupOrchestrationOverhead(profile, 75, "Vulkan")
	assert.Greater(t, oh75, 3000.0)
	assert.Less(t, oh75, 5500.0)

	// Wrong backend → 0
	ohCPU := lookupOrchestrationOverhead(profile, 50, "CPU")
	assert.Equal(t, 0.0, ohCPU)

	// No overhead curve → 0
	emptyProfile := &Profile{}
	oh := lookupOrchestrationOverhead(emptyProfile, 50, "Vulkan")
	assert.Equal(t, 0.0, oh)
}

func TestNodeToQueryShape_MulMatAdd(t *testing.T) {
	node := ml.GraphNode{
		Op:           "MUL_MAT_ADD",
		Backend:      "Vulkan",
		ComputeDtype: "f16",
		WeightDtype:  "q4_0",
		InputShapes:  [][]int64{{4096, 4096}, {4096, 1}},
	}
	op, shape, cdt, wdt := nodeToQueryShape(node)
	assert.Equal(t, "MUL_MAT_ADD", op)
	require.Len(t, shape, 3)
	assert.Equal(t, int64(4096), shape[0]) // M
	assert.Equal(t, int64(4096), shape[1]) // K
	assert.Equal(t, int64(1), shape[2])    // N
	assert.Equal(t, "f16", cdt)
	assert.Equal(t, "q4_0", wdt)
}

func TestEstimatePhaseV3_BackwardCompatV2(t *testing.T) {
	// With a v2-style profile (no BackendCaps), estimatePhaseV3 should still work
	// (delegation to lookupLatency v2 path)
	profile := newTestProfile() // has SILU curve

	nodes := []ml.GraphNode{
		{Op: "SILU", Backend: "cuda", ComputeDtype: "f32", Shape: [4]int64{4096, 1, 1, 1}},
	}
	caps := BackendCapabilities{Name: "cuda", HasGPUTimestamp: false}
	var warnings []string
	result := estimatePhaseV3(profile, nodes, &caps, &warnings)

	assert.Greater(t, result.TotalLatencyMs, 0.0)
	// No overhead (HasGPUTimestamp=false)
	assert.Empty(t, warnings)
}
