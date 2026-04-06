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
	require.Len(t, shape, 3)
	assert.Equal(t, int64(1), shape[0])    // seq_q
	assert.Equal(t, int64(2048), shape[1]) // seq_kv
	assert.Equal(t, int64(32), shape[2])   // num_heads
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
	assert.Equal(t, int64(32), shape[2]) // num_heads
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
	lat, err := lookupLatency(p, "FLASH_ATTN_EXT", []int64{1, 512, 32}, "f16", "", "cuda")
	require.NoError(t, err)
	assert.InDelta(t, 15.0, lat, 0.5)
}

func TestLookupLatency_FlashAttn_Prefill(t *testing.T) {
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "FLASH_ATTN_EXT", []int64{512, 512, 32}, "f16", "", "cuda")
	require.NoError(t, err)
	assert.InDelta(t, 100.0, lat, 1.0)
}

func TestLookupLatency_Uncalibrated(t *testing.T) {
	// With the new fallback chain, uncalibrated ops fall back to bandwidth roofline
	// rather than erroring (as long as PeakBandwidthBytesPerSec is set).
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "TANH", []int64{4096}, "f32", "", "cuda")
	require.NoError(t, err)
	assert.Greater(t, lat, 0.0)
	// Should use bandwidth roofline: 4096 * 4 * 2 / bandwidth * 1e6
	expectedUs := float64(4096) * 4 * 2 / p.Hardware.PeakBandwidthBytesPerSec * 1e6
	assert.InDelta(t, expectedUs, lat, 0.001)
}

func TestLookupLatency_WrongBackend(t *testing.T) {
	// With the new fallback chain, even wrong backend queries fall back to bandwidth roofline
	// rather than erroring (as long as PeakBandwidthBytesPerSec is set).
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "SILU", []int64{4096}, "f32", "", "cpu")
	require.NoError(t, err)
	assert.Greater(t, lat, 0.0)
	// Should use bandwidth roofline
	expectedUs := float64(4096) * 4 * 2 / p.Hardware.PeakBandwidthBytesPerSec * 1e6
	assert.InDelta(t, expectedUs, lat, 0.001)
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
	// With the new fallback chain, uncalibrated ops fall back to bandwidth roofline
	// and succeed. The warnings are logged via slog.Warn but not captured in the warnings slice.
	// The estimate should now succeed with non-zero latency from the roofline model.
	p := makeTestProfileForEstimation()
	nodes := []ml.GraphNode{
		{Op: "TANH", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
	}
	var warnings []string
	result := estimatePhase(p, nodes, &warnings)
	assert.Empty(t, warnings, "no errors should be returned - falls back to roofline")
	assert.Greater(t, result.TotalLatencyMs, 0.0, "should have non-zero estimate from roofline")
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
	assert.Contains(t, err.Error(), "requires 3 shape dims")
}

func TestEstimatePhase_AllNodesUncalibrated(t *testing.T) {
	// With the new fallback chain, uncalibrated ops fall back to bandwidth roofline
	// and succeed. The warnings are logged via slog.Warn but not captured in the warnings slice.
	p := makeTestProfileForEstimation()
	nodes := []ml.GraphNode{
		{Op: "TANH", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "SWIGLU", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "NORM", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
	}
	var warnings []string
	result := estimatePhase(p, nodes, &warnings)
	assert.Empty(t, warnings, "no errors should be returned - falls back to roofline")
	assert.Greater(t, result.TotalLatencyMs, 0.0, "should have non-zero estimate from roofline")
	assert.Len(t, result.TopOps, 3, "all ops should be estimated via roofline")
}

func TestEstimatePhase_PartialUncalibrated(t *testing.T) {
	// With the new fallback chain, TANH falls back to bandwidth roofline and succeeds.
	// Both ops should be estimated successfully.
	p := makeTestProfileForEstimation()
	nodes := []ml.GraphNode{
		{Op: "SILU", Backend: "cuda", Shape: [4]int64{65536, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "TANH", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
	}
	var warnings []string
	result := estimatePhase(p, nodes, &warnings)
	assert.Empty(t, warnings, "no errors should be returned")
	assert.Greater(t, result.TotalLatencyMs, 0.0)
	require.Len(t, result.TopOps, 2, "both ops should be estimated")
	assert.Equal(t, "SILU", result.TopOps[0].Op, "SILU should be first (larger latency)")
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

func TestLookupLatencyV3_MulMat_DirectQ4K(t *testing.T) {
	// Profile has q4_K curves — should use them directly, no fallback
	p := makeTestProfileForEstimation()
	p.Operators = append(p.Operators, OperatorCurve{
		Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "q4_K", WeightDtype: "q4_K",
		Dimensions: []string{"N"},
		FixedDims:  map[string]int64{"M": 4096, "K": 4096},
		Points:     []LatencyPoint{{Shape: []int64{1}, LatencyUs: 200.0}, {Shape: []int64{512}, LatencyUs: 5000.0}},
	})
	caps := &BackendCapabilities{Name: "cuda"}
	lat, err := lookupLatencyV3(p, "MUL_MAT", []int64{4096, 4096, 1}, "f16", "q4_K", "cuda", caps)
	require.NoError(t, err)
	assert.InDelta(t, 200.0, lat, 10.0)
}

func TestLookupLatencyV3_MulMat_FallbackQ4KtoQ40(t *testing.T) {
	// Profile has q4_0 reference curves but NOT q4_K.
	// Should fall back from q4_K -> q4_0 via dtypeFallback with warning.
	p := makeTestProfileForEstimation()
	// Add q4_0 reference curves with FixedDims (needed for PredictMulMatDirect)
	p.Operators = append(p.Operators, OperatorCurve{
		Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
		Dimensions: []string{"N"},
		FixedDims:  map[string]int64{"M": 4096, "K": 4096},
		Points:     []LatencyPoint{{Shape: []int64{1}, LatencyUs: 100.0}, {Shape: []int64{512}, LatencyUs: 2500.0}},
	})
	caps := &BackendCapabilities{Name: "cuda"}
	lat, err := lookupLatencyV3(p, "MUL_MAT", []int64{4096, 4096, 1}, "f16", "q4_K", "cuda", caps)
	require.NoError(t, err)
	assert.InDelta(t, 100.0, lat, 10.0, "should use q4_0 fallback curves")
}

func TestLookupLatencyV3_MulMat_FallbackQ6KtoQ80(t *testing.T) {
	// Profile has q8_0 reference curves but NOT q6_K.
	// Should fall back from q6_K -> q8_0 via dtypeFallback with warning.
	p := makeTestProfileForEstimation()
	p.Operators = append(p.Operators, OperatorCurve{
		Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q8_0",
		Dimensions: []string{"N"},
		FixedDims:  map[string]int64{"M": 4096, "K": 4096},
		Points:     []LatencyPoint{{Shape: []int64{1}, LatencyUs: 150.0}, {Shape: []int64{512}, LatencyUs: 3500.0}},
	})
	caps := &BackendCapabilities{Name: "cuda"}
	lat, err := lookupLatencyV3(p, "MUL_MAT", []int64{4096, 4096, 1}, "f16", "q6_K", "cuda", caps)
	require.NoError(t, err)
	assert.InDelta(t, 150.0, lat, 10.0, "should use q8_0 fallback curves")
}

func TestLookupLatency_1DOp_FallbackQ4KDtype(t *testing.T) {
	// Profile has SILU calibrated for f32 but not q4_K.
	// dtypeFallback(q4_K) = q4_0, but q4_0 is also not calibrated for SILU.
	// Should fall through to bandwidth roofline.
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "SILU", []int64{4096}, "q4_K", "", "cuda")
	require.NoError(t, err)
	assert.Greater(t, lat, 0.0)
	// Bandwidth roofline: 4096 * 0.5625 * 2 / bandwidth * 1e6
	expectedUs := float64(4096) * (18.0 / 32.0) * 2 / p.Hardware.PeakBandwidthBytesPerSec * 1e6
	assert.InDelta(t, expectedUs, lat, 0.001)
}

func TestLookupLatency_1DOp_DtypeFallback(t *testing.T) {
	// Profile has SILU calibrated for q4_0.
	// When asked for SILU q4_K, dtypeFallback returns q4_0 — should use it with warning.
	p := makeTestProfileForEstimation()
	p.Operators = append(p.Operators, OperatorCurve{
		Op: "SILU", Backend: "cuda", ComputeDtype: "q4_0",
		Dimensions: []string{"N"},
		Points:     []LatencyPoint{{Shape: []int64{4096}, LatencyUs: 5.0}},
	})
	lat, err := lookupLatency(p, "SILU", []int64{4096}, "q4_K", "", "cuda")
	require.NoError(t, err)
	assert.InDelta(t, 5.0, lat, 0.1)
}

func TestEstimatePhase_FlashAttnScalesWithSeqLen(t *testing.T) {
	// Core test: FLASH_ATTN_EXT latency should scale with seqQ×seqKV.
	p := makeTestProfileForEstimation()

	makeFlashAttnNode := func(seqQ, seqKV int64) ml.GraphNode {
		return ml.GraphNode{
			Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
			InputShapes: [][]int64{
				{128, seqQ, 32, 1},
				{128, seqKV, 32, 1},
			},
		}
	}

	nodes130 := []ml.GraphNode{makeFlashAttnNode(130, 130)}
	nodes512 := []ml.GraphNode{makeFlashAttnNode(512, 512)}

	var w1, w2 []string
	result130 := estimatePhase(p, nodes130, &w1)
	result512 := estimatePhase(p, nodes512, &w2)

	require.NotEmpty(t, result130.TopOps)
	require.NotEmpty(t, result512.TopOps)
	assert.Equal(t, "FLASH_ATTN_EXT", result130.TopOps[0].Op)
	assert.Equal(t, "FLASH_ATTN_EXT", result512.TopOps[0].Op)

	lat130 := result130.TopOps[0].TotalUs
	lat512 := result512.TopOps[0].TotalUs

	assert.Greater(t, lat512, lat130,
		"FLASH_ATTN at seqlen=512 (%.1fus) should be greater than seqlen=130 (%.1fus)",
		lat512, lat130)
	// Log-space interpolation gives: seqlen=130 → ~20.37us, seqlen=512 → 100.0us, ratio ≈ 4.91x
	ratio := lat512 / lat130
	assert.Greater(t, ratio, 4.0,
		"latency ratio should reflect quadratic scaling, got %.1fx", ratio)
}

func TestEstimatePhase_FlashAttnDecodeScalesWithKVLen(t *testing.T) {
	p := makeTestProfileForEstimation()

	makeDecodeFlashAttn := func(seqKV int64) ml.GraphNode {
		return ml.GraphNode{
			Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
			InputShapes: [][]int64{
				{128, 1, 32, 1},
				{128, seqKV, 32, 1},
			},
		}
	}

	var w1, w2 []string
	result130 := estimatePhase(p, []ml.GraphNode{makeDecodeFlashAttn(130)}, &w1)
	result2048 := estimatePhase(p, []ml.GraphNode{makeDecodeFlashAttn(2048)}, &w2)

	require.NotEmpty(t, result130.TopOps)
	require.NotEmpty(t, result2048.TopOps)
	assert.Equal(t, "FLASH_ATTN_EXT", result130.TopOps[0].Op)
	assert.Equal(t, "FLASH_ATTN_EXT", result2048.TopOps[0].Op)

	lat130 := result130.TopOps[0].TotalUs
	lat2048 := result2048.TopOps[0].TotalUs

	assert.Greater(t, lat2048, lat130,
		"decode FLASH_ATTN at seqKV=2048 (%.1fus) should be greater than seqKV=130 (%.1fus)",
		lat2048, lat130)
	ratio := lat2048 / lat130
	assert.Greater(t, ratio, 3.0,
		"decode latency ratio should reflect linear KV scaling, got %.1fx", ratio)
}

func TestEstimatePhase_LlamaDecodeFlashAttnPercentageIncreasesWithKVLen(t *testing.T) {
	// MUL_MAT always dominates by absolute value (4 MUL_MATs vs 1 FLASH_ATTN 5~55us).
	// But FLASH_ATTN's percentage should increase with longer KV.
	p := makeTestProfileForEstimation()

	makeLlamaDecodeLayer := func(seqKV int64) []ml.GraphNode {
		return []ml.GraphNode{
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
			{Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
				InputShapes: [][]int64{{128, 1, 32, 1}, {128, seqKV, 32, 1}}},
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				InputShapes: [][]int64{{14336, 4096}, {4096, 1}}},
			{Op: "SILU", Backend: "cuda", Shape: [4]int64{14336, 1, 1, 1}, ComputeDtype: "f32"},
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				InputShapes: [][]int64{{4096, 14336}, {14336, 1}}},
		}
	}

	var w1, w2 []string
	resultShort := estimatePhase(p, makeLlamaDecodeLayer(128), &w1)
	resultLong := estimatePhase(p, makeLlamaDecodeLayer(2048), &w2)

	// Find FLASH_ATTN percentage in each result
	flashPctShort := 0.0
	flashPctLong := 0.0
	totalShortUs := resultShort.TotalLatencyMs * 1000
	totalLongUs := resultLong.TotalLatencyMs * 1000
	for _, op := range resultShort.TopOps {
		if op.Op == "FLASH_ATTN_EXT" {
			flashPctShort = op.TotalUs / totalShortUs * 100
		}
	}
	for _, op := range resultLong.TopOps {
		if op.Op == "FLASH_ATTN_EXT" {
			flashPctLong = op.TotalUs / totalLongUs * 100
		}
	}

	assert.Greater(t, flashPctLong, flashPctShort,
		"FLASH_ATTN percentage should increase with longer KV: short=%.1f%%, long=%.1f%%",
		flashPctShort, flashPctLong)
	assert.Greater(t, flashPctLong, 15.0,
		"with seqKV=2048, FLASH_ATTN should be >15%% of total, got %.1f%%", flashPctLong)
	assert.Greater(t, resultLong.TotalLatencyMs, resultShort.TotalLatencyMs,
		"longer KV cache should increase total decode latency")
}

func TestEstimatePhase_PrefillMulMatScalesWithInputLength(t *testing.T) {
	// captureGraph(inputLength) changes MUL_MAT activation's N dimension.
	p := makeTestProfileForEstimation()

	makePrefillMulMatNodes := func(seqLen int64) []ml.GraphNode {
		return []ml.GraphNode{
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				InputShapes: [][]int64{{4096, 4096}, {4096, seqLen}}},
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				InputShapes: [][]int64{{4096, 4096}, {4096, seqLen}}},
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				InputShapes: [][]int64{{14336, 4096}, {4096, seqLen}}},
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				InputShapes: [][]int64{{4096, 14336}, {14336, seqLen}}},
		}
	}

	var w1, w2 []string
	result130 := estimatePhase(p, makePrefillMulMatNodes(130), &w1)
	result512 := estimatePhase(p, makePrefillMulMatNodes(512), &w2)

	assert.Greater(t, result512.TotalLatencyMs, result130.TotalLatencyMs,
		"prefill MUL_MAT at N=512 (%.3fms) should be greater than N=130 (%.3fms)",
		result512.TotalLatencyMs, result130.TotalLatencyMs)
	ratio := result512.TotalLatencyMs / result130.TotalLatencyMs
	assert.Greater(t, ratio, 2.0,
		"prefill MUL_MAT latency ratio should reflect N scaling, got %.1fx", ratio)
}

func TestNodeToQueryShape_FlashAttn_GQA(t *testing.T) {
	node := ml.GraphNode{
		Op:           "FLASH_ATTN_EXT",
		Backend:      "cuda",
		ComputeDtype: "f16",
		InputShapes: [][]int64{
			{128, 130, 32, 1},
			{128, 256, 8, 1},
		},
	}
	op, shape, _, _ := nodeToQueryShape(node)
	assert.Equal(t, "FLASH_ATTN_EXT", op)
	require.Len(t, shape, 3)
	assert.Equal(t, int64(130), shape[0], "seqQ should come from Q ne[1], not ne[2]")
	assert.Equal(t, int64(256), shape[1], "seqKV should come from K ne[1], not ne[2]")
	assert.Equal(t, int64(32), shape[2], "numHeads from Q tensor ne[2]")
}

func TestEstimatePhaseV3_FlashAttnScalesWithSeqLen(t *testing.T) {
	p := makeTestProfileForEstimation()
	p.Version = 3

	makeFlashAttnNode := func(seqQ, seqKV int64) ml.GraphNode {
		return ml.GraphNode{
			Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
			InputShapes: [][]int64{
				{128, seqQ, 32, 1},
				{128, seqKV, 32, 1},
			},
		}
	}

	caps := &BackendCapabilities{Name: "cuda", HasGPUTimestamp: false}
	var w1, w2 []string
	result130 := estimatePhaseV3(p, []ml.GraphNode{makeFlashAttnNode(130, 130)}, caps, &w1)
	result512 := estimatePhaseV3(p, []ml.GraphNode{makeFlashAttnNode(512, 512)}, caps, &w2)

	require.NotEmpty(t, result130.TopOps)
	require.NotEmpty(t, result512.TopOps)
	assert.Equal(t, "FLASH_ATTN_EXT", result130.TopOps[0].Op)
	assert.Equal(t, "FLASH_ATTN_EXT", result512.TopOps[0].Op)

	lat130 := result130.TopOps[0].TotalUs
	lat512 := result512.TopOps[0].TotalUs

	assert.Greater(t, lat512, lat130,
		"v3: FLASH_ATTN at seqlen=512 (%.1fus) should be greater than seqlen=130 (%.1fus)",
		lat512, lat130)
	ratio := lat512 / lat130
	assert.Greater(t, ratio, 4.0,
		"v3: latency ratio should reflect quadratic scaling, got %.1fx", ratio)
}

func TestEstimatePhase_EdgeCase_InputLengthOne(t *testing.T) {
	// inputLength=1: both FLASH_ATTN seqQ=seqKV=1 and MUL_MAT N=1.
	// Must not panic and should produce valid estimates.
	p := makeTestProfileForEstimation()

	nodes := []ml.GraphNode{
		{Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
			InputShapes: [][]int64{{128, 1, 32, 1}, {128, 1, 32, 1}}},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
	}

	var warnings []string
	result := estimatePhase(p, nodes, &warnings)

	assert.Greater(t, result.TotalLatencyMs, 0.0, "inputLength=1 should have non-zero latency")
	assert.NotEmpty(t, result.TopOps, "should have op breakdown")
}

func TestNodeToQueryShape_FlashAttn_NumHeads(t *testing.T) {
	node := ml.GraphNode{
		Op: "FLASH_ATTN_EXT",
		InputShapes: [][]int64{
			{128, 256, 16, 1}, // Q: head_dim=128, seqQ=256, num_heads=16
			{128, 512, 4, 1},  // K: head_dim=128, seqKV=512, num_kv_heads=4
		},
	}
	op, shape, _, _ := nodeToQueryShape(node)
	assert.Equal(t, "FLASH_ATTN_EXT", op)
	require.Len(t, shape, 3)
	assert.Equal(t, int64(256), shape[0], "seqQ")
	assert.Equal(t, int64(512), shape[1], "seqKV")
	assert.Equal(t, int64(16), shape[2], "numHeads from Q tensor")
}

func TestLookupLatency_FlashAttn_MultiHead(t *testing.T) {
	makePoints := func(scale float64) []LatencyPoint {
		return []LatencyPoint{
			{Shape: []int64{1, 64}, LatencyUs: 1.0 * scale},
			{Shape: []int64{1, 256}, LatencyUs: 2.0 * scale},
			{Shape: []int64{1, 1024}, LatencyUs: 4.0 * scale},
			{Shape: []int64{64, 64}, LatencyUs: 3.0 * scale},
			{Shape: []int64{256, 256}, LatencyUs: 20.0 * scale},
			{Shape: []int64{1024, 1024}, LatencyUs: 200.0 * scale},
		}
	}
	profile := &Profile{
		Operators: []OperatorCurve{
			{
				Op: "FLASH_ATTN_EXT", Backend: "Vulkan",
				FixedDims: map[string]int64{"num_heads": 8},
				Points:    makePoints(50.0),
			},
			{
				Op: "FLASH_ATTN_EXT", Backend: "Vulkan",
				FixedDims: map[string]int64{"num_heads": 32},
				Points:    makePoints(200.0),
			},
		},
	}
	// Query with 16 heads — should interpolate between 8 and 32 head curves
	lat, err := lookupLatency(profile, "FLASH_ATTN_EXT", []int64{256, 256, 16}, "f16", "", "Vulkan")
	assert.NoError(t, err)

	lat8, _ := lookupLatency(profile, "FLASH_ATTN_EXT", []int64{256, 256, 8}, "f16", "", "Vulkan")
	lat32, _ := lookupLatency(profile, "FLASH_ATTN_EXT", []int64{256, 256, 32}, "f16", "", "Vulkan")
	assert.Greater(t, lat, lat8)
	assert.Less(t, lat, lat32)
}
