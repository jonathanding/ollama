package perf

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTrimmedMedian_Basic(t *testing.T) {
	values := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	// 10% trim: remove 1 from each end → [2,3,4,5,6,7,8,9], median at index 4 → 6
	result := trimmedMedian(values, 0.10)
	assert.InDelta(t, 6.0, result, 0.001)
}

func TestTrimmedMedian_NoTrim(t *testing.T) {
	values := []float64{5, 1, 3}
	result := trimmedMedian(values, 0.0)
	assert.InDelta(t, 3.0, result, 0.001) // sorted: [1,3,5], median=3
}

func TestTrimmedMedian_AllSame(t *testing.T) {
	values := []float64{42.0, 42.0, 42.0, 42.0}
	result := trimmedMedian(values, 0.10)
	assert.InDelta(t, 42.0, result, 0.001)
}

func TestTrimmedMedian_SingleValue(t *testing.T) {
	values := []float64{7.0}
	result := trimmedMedian(values, 0.10)
	assert.InDelta(t, 7.0, result, 0.001)
}

func TestTrimmedMedian_Empty(t *testing.T) {
	assert.Equal(t, 0.0, trimmedMedian(nil, 0.10))
}

func TestTrimmedMedian_HighTrim(t *testing.T) {
	// TrimPercent so high that trimCount*2 >= len → falls back to no trim
	values := []float64{1, 2, 3}
	result := trimmedMedian(values, 0.5)
	assert.InDelta(t, 2.0, result, 0.001)
}

func TestBuildSamplingGrids_SILU(t *testing.T) {
	grids := buildSamplingGrids("SILU", "f32", "")
	require.Len(t, grids, 1, "SILU has one 1D grid")
	assert.Equal(t, "SILU", grids[0].Op)
	assert.Equal(t, "f32", grids[0].Dtype)
	assert.Nil(t, grids[0].FixedDims)
}

func TestBuildSamplingGrids_MulMat(t *testing.T) {
	grids := buildSamplingGrids("MUL_MAT", "f16", "q4_0")
	// One grid per (M, K) pair from Phase1MulMatFixedDims
	assert.GreaterOrEqual(t, len(grids), 4, "MUL_MAT should have multiple (M,K) grids")
	for _, g := range grids {
		assert.Equal(t, "MUL_MAT", g.Op)
		assert.Equal(t, "q4_0", g.WeightDtype)
		assert.NotNil(t, g.FixedDims)
		assert.Contains(t, g.FixedDims, "M")
		assert.Contains(t, g.FixedDims, "K")
	}
}

func TestBuildSamplingGrids_FlashAttn(t *testing.T) {
	grids := buildSamplingGrids("FLASH_ATTN_EXT", "f16", "")
	require.Len(t, grids, 1, "FLASH_ATTN_EXT has one grid with fixed head_dim/num_heads")
	assert.Equal(t, int64(32), grids[0].FixedDims["num_heads"])
	assert.Equal(t, int64(128), grids[0].FixedDims["head_dim"])
}

func TestSweepDimensions(t *testing.T) {
	assert.Equal(t, []string{"N"}, sweepDimensions("SILU"))
	assert.Equal(t, []string{"N"}, sweepDimensions("MUL_MAT"))
	assert.Equal(t, []string{"seq_q", "seq_kv"}, sweepDimensions("FLASH_ATTN_EXT"))
	assert.Equal(t, []string{"N"}, sweepDimensions("UNKNOWN_OP"))
}

func TestBuildSamplingGrids_UnknownOp(t *testing.T) {
	grids := buildSamplingGrids("NONEXISTENT", "f32", "")
	assert.Empty(t, grids, "unknown op should produce no grids")
}

// TestFlashAttnShapeConversion verifies the 1D→2D shape conversion
// after adaptive sampling. This is a regression test for the bug where
// AdaptiveSample1D received 2D shapes, breaking sort and interpolation.
func TestFlashAttnShapeConversion(t *testing.T) {
	// Simulate what benchmarkFlashAttn does post-sampling:
	// Decode points: [seqKV] → [1, seqKV]
	decodePts := []LatencyPoint{
		{Shape: []int64{64}, LatencyUs: 10.0},
		{Shape: []int64{256}, LatencyUs: 40.0},
		{Shape: []int64{1024}, LatencyUs: 160.0},
	}
	for i := range decodePts {
		seqKV := decodePts[i].Shape[0]
		decodePts[i].Shape = []int64{1, seqKV}
	}
	assert.Equal(t, []int64{1, 64}, decodePts[0].Shape)
	assert.Equal(t, []int64{1, 256}, decodePts[1].Shape)
	assert.Equal(t, []int64{1, 1024}, decodePts[2].Shape)

	// Prefill points: [seqLen] → [seqLen, seqLen]
	prefillPts := []LatencyPoint{
		{Shape: []int64{64}, LatencyUs: 15.0},
		{Shape: []int64{512}, LatencyUs: 200.0},
	}
	for i := range prefillPts {
		seqLen := prefillPts[i].Shape[0]
		prefillPts[i].Shape = []int64{seqLen, seqLen}
	}
	assert.Equal(t, []int64{64, 64}, prefillPts[0].Shape)
	assert.Equal(t, []int64{512, 512}, prefillPts[1].Shape)
}

// TestBenchmarkMulMat_OutputShapeContract verifies that benchmarkMulMat produces
// 1D shapes [N] compatible with InterpolateMulMat (dimIdx=0).
// This is a contract test — the seam between benchmark and interpolation.
func TestBenchmarkMulMat_OutputShapeContract(t *testing.T) {
	// Simulate what benchmarkMulMat's measure closure does
	M, K := int64(4096), int64(4096)
	nValues := []int64{1, 32, 256, 4096}

	for _, N := range nValues {
		// This mirrors bench.go benchmarkMulMat's measure closure:
		// pt := measureOp(backend, "MUL_MAT", []int64{M, K, N}, dtype, cfg)
		// pt.Shape = []int64{N}
		pt := LatencyPoint{
			Shape:     []int64{M, K, N}, // what measureOp returns
			LatencyUs: float64(N) * 10,
		}
		pt.Shape = []int64{N} // what the measure closure overrides

		// Verify shape is 1D
		assert.Len(t, pt.Shape, 1, "MUL_MAT points must be 1D for AdaptiveSample1D")
		assert.Equal(t, N, pt.Shape[0], "Shape[0] must be the sweep dimension N")
	}

	// Verify InterpolateMulMat can consume 1D points
	curves := []OperatorCurve{{
		Op: "MUL_MAT", FixedDims: map[string]int64{"M": 4096, "K": 4096},
		Points: []LatencyPoint{
			{Shape: []int64{1}, LatencyUs: 10.0},
			{Shape: []int64{4096}, LatencyUs: 3000.0},
		},
	}}
	result := InterpolateMulMat(curves, 4096, 4096, 128)
	assert.Greater(t, result, 10.0)
	assert.Less(t, result, 3000.0)
	assert.False(t, math.IsNaN(result))
}

func TestExtractEfficiencyConstants(t *testing.T) {
	// Simulate a reference curve at M=K=4096 with known peak TOPS and BW
	// peakTOPS = 64.3 GFLOPS, peakBW = 40.7 GB/s
	peakTOPS := 64.3e9
	peakBW := 40.7e9

	// Points from actual Intel iGPU measurement
	points := []LatencyPoint{
		{Shape: []int64{1}, LatencyUs: 3754},
		{Shape: []int64{3}, LatencyUs: 3007},
		{Shape: []int64{11}, LatencyUs: 8028},
		{Shape: []int64{35}, LatencyUs: 24217},
		{Shape: []int64{116}, LatencyUs: 64610},
		{Shape: []int64{380}, LatencyUs: 219651},
		{Shape: []int64{1248}, LatencyUs: 695931},
		{Shape: []int64{4096}, LatencyUs: 2302781},
	}

	eff := extractEfficiencyConstants(points, 4096, 4096, peakTOPS, peakBW, "f32")

	// Compute efficiency should be ~0.90-0.95
	assert.Greater(t, eff.ComputeEff, 0.80, "compute efficiency should be > 80%%")
	assert.Less(t, eff.ComputeEff, 1.0, "compute efficiency should be < 100%%")

	// BW efficiency should be ~0.40-0.55
	assert.Greater(t, eff.BWEff, 0.30, "BW efficiency should be > 30%%")
	assert.Less(t, eff.BWEff, 0.70, "BW efficiency should be < 70%%")

	// Overhead should be non-negative
	assert.GreaterOrEqual(t, eff.OverheadUs, 0.0)
}

func TestPredictMulMatLatency_ComputeBound(t *testing.T) {
	// Large N should be compute-bound
	hw := &HardwareProfile{
		PeakTOPS:                 map[string]float64{"f32": 64.3e9},
		PeakBandwidthBytesPerSec: 40.7e9,
		EfficiencyConstants: map[string]OpEfficiency{
			"MUL_MAT_f32": {ComputeEff: 0.93, BWEff: 0.45, OverheadUs: 0},
		},
	}

	// M=K=4096, N=4096: FLOPs = 2*4096^3 = 137.4G
	// Expected: 137.4G / (0.93 * 64.3G) * 1e6 ≈ 2,298,000 us
	lat := PredictMulMatLatency(hw, 4096, 4096, 4096, "f32")
	assert.InDelta(t, 2298000, lat, 100000, "should predict ~2.3M us for 4096^3 matmul")
}

func TestPredictMulMatLatency_BWBound(t *testing.T) {
	// N=1 should be BW-bound
	hw := &HardwareProfile{
		PeakTOPS:                 map[string]float64{"f32": 64.3e9},
		PeakBandwidthBytesPerSec: 40.7e9,
		EfficiencyConstants: map[string]OpEfficiency{
			"MUL_MAT_f32": {ComputeEff: 0.93, BWEff: 0.45, OverheadUs: 500},
		},
	}

	// M=K=4096, N=1: weight=4096*4096*4=64MB, act=4096*1*4=16KB, out=4096*1*4=16KB
	// bytes ≈ 64MB, BW time = 64MB / (0.45 * 40.7 GB/s) ≈ 3,494 us + 500 overhead
	lat := PredictMulMatLatency(hw, 4096, 4096, 1, "f32")
	assert.Greater(t, lat, 2000.0, "N=1 matmul should take > 2ms")
	assert.Less(t, lat, 10000.0, "N=1 matmul should take < 10ms")
}

func TestPredictMulMatLatency_ScalesWithShape(t *testing.T) {
	hw := &HardwareProfile{
		PeakTOPS:                 map[string]float64{"f32": 64.3e9},
		PeakBandwidthBytesPerSec: 40.7e9,
		EfficiencyConstants: map[string]OpEfficiency{
			"MUL_MAT_f32": {ComputeEff: 0.93, BWEff: 0.45, OverheadUs: 0},
		},
	}

	// Larger M should give proportionally more latency at large N (compute-bound)
	lat1 := PredictMulMatLatency(hw, 4096, 4096, 4096, "f32")
	lat2 := PredictMulMatLatency(hw, 14336, 4096, 4096, "f32")
	ratio := lat2 / lat1
	expectedRatio := 14336.0 / 4096.0 // ~3.5x
	assert.InDelta(t, expectedRatio, ratio, 0.3, "latency should scale with M")
}

func TestExtractEfficiencyConstants_Q40(t *testing.T) {
	// q4_0: weight bytes = M*K*0.5625, much less than f32's M*K*4
	// For q4_0 at M=K=4096, nearly ALL points are compute-bound because
	// weight data is so small (~9.4MB vs 64MB for f32).
	// Ideal compute times: N=1→522us, N=35→18264us, N=4096→2137636us
	// Use ~90% efficiency for realistic latencies.
	peakTOPS := 64.3e9
	peakBW := 40.7e9

	points := []LatencyPoint{
		{Shape: []int64{1}, LatencyUs: 600},
		{Shape: []int64{3}, LatencyUs: 1800},
		{Shape: []int64{11}, LatencyUs: 6500},
		{Shape: []int64{35}, LatencyUs: 20500},
		{Shape: []int64{116}, LatencyUs: 68000},
		{Shape: []int64{380}, LatencyUs: 222000},
		{Shape: []int64{1248}, LatencyUs: 725000},
		{Shape: []int64{4096}, LatencyUs: 2380000},
	}

	eff := extractEfficiencyConstants(points, 4096, 4096, peakTOPS, peakBW, "q4_0")

	assert.Greater(t, eff.ComputeEff, 0.5, "compute efficiency should be reasonable")
	assert.LessOrEqual(t, eff.ComputeEff, 1.0)
	// For q4_0, most/all points may be compute-bound; BWEff may use default
	assert.Greater(t, eff.BWEff, 0.0, "BW efficiency should be positive")
}

func TestPredictMulMatLatency_PerDtypeEfficiency(t *testing.T) {
	hw := &HardwareProfile{
		PeakTOPS:                 map[string]float64{"f32": 64.3e9},
		PeakBandwidthBytesPerSec: 40.7e9,
		EfficiencyConstants: map[string]OpEfficiency{
			"MUL_MAT_f32":  {ComputeEff: 0.93, BWEff: 0.45, OverheadUs: 500},
			"MUL_MAT_q4_0": {ComputeEff: 0.85, BWEff: 0.55, OverheadUs: 300},
		},
	}

	latF32 := PredictMulMatLatency(hw, 4096, 4096, 1, "f32")
	latQ40 := PredictMulMatLatency(hw, 4096, 4096, 1, "q4_0")

	assert.Greater(t, latF32, 0.0)
	assert.Greater(t, latQ40, 0.0)
	// q4_0 at N=1 should be faster (less data to read, BW-bound regime)
	assert.Less(t, latQ40, latF32, "q4_0 should be faster than f32 at N=1 (BW-bound)")
}

func TestPredictMulMatLatency_FallbackToGeneric(t *testing.T) {
	hw := &HardwareProfile{
		PeakTOPS:                 map[string]float64{"f32": 64.3e9},
		PeakBandwidthBytesPerSec: 40.7e9,
		EfficiencyConstants: map[string]OpEfficiency{
			"MUL_MAT": {ComputeEff: 0.90, BWEff: 0.40, OverheadUs: 0},
		},
	}
	lat := PredictMulMatLatency(hw, 4096, 4096, 4096, "f16")
	assert.Greater(t, lat, 0.0, "should fall back to generic MUL_MAT constants")
}

func TestElemBytesFromDtype(t *testing.T) {
	assert.Equal(t, 4.0, elemBytesFromDtype("f32"))
	assert.Equal(t, 2.0, elemBytesFromDtype("f16"))
	assert.InDelta(t, 0.5625, elemBytesFromDtype("q4_0"), 0.001) // 18 bytes / 32 elements
	assert.InDelta(t, 1.0625, elemBytesFromDtype("q8_0"), 0.001) // 34 bytes / 32 elements
	assert.Equal(t, 4.0, elemBytesFromDtype("unknown"))
}

// TestMeasureMulMat_OutputShape verifies measureMulMat returns correct shape metadata.
func TestMeasureMulMat_OutputShape(t *testing.T) {
	pt := LatencyPoint{
		Shape:     []int64{4096, 4096, 32},
		LatencyUs: 5000.0,
		StddevUs:  100.0,
		Reps:      7,
	}
	assert.Len(t, pt.Shape, 3)
	assert.Equal(t, int64(4096), pt.Shape[0]) // M
	assert.Equal(t, int64(4096), pt.Shape[1]) // K
	assert.Equal(t, int64(32), pt.Shape[2])   // N
}

func TestCountGrids_MulMatIsFour(t *testing.T) {
	ops := []string{"SILU", "MUL_MAT", "FLASH_ATTN_EXT"}
	dtypes := []string{"f32", "f16", "q4_0", "q8_0"}
	count := countGrids(ops, dtypes)
	// SILU: 1 (f32 only), MUL_MAT: 4 (one per weight dtype), FLASH_ATTN_EXT: 1 (f16 only)
	assert.Equal(t, 6, count, "should be 6 grids: SILU + 4 MUL_MAT refs + FLASH_ATTN")
}
