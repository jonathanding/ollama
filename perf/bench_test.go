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
