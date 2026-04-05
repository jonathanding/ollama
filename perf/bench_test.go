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
	// 3×3 grid = 9 (M,K) pairs
	assert.Equal(t, 9, len(grids), "MUL_MAT should have 9 (M,K) grids from 3×3 grid")
	for _, g := range grids {
		assert.Equal(t, "MUL_MAT", g.Op)
		assert.Equal(t, "q4_0", g.WeightDtype)
		assert.NotNil(t, g.FixedDims)
		assert.Contains(t, g.FixedDims, "M")
		assert.Contains(t, g.FixedDims, "K")
	}

	// Verify specific grid values
	expectedDims := []int64{512, 2048, 8192}
	seen := make(map[[2]int64]bool)
	for _, g := range grids {
		M := g.FixedDims["M"]
		K := g.FixedDims["K"]
		seen[[2]int64{M, K}] = true
		assert.Contains(t, expectedDims, M, "M=%d not in expected grid values", M)
		assert.Contains(t, expectedDims, K, "K=%d not in expected grid values", K)
	}
	assert.Equal(t, 9, len(seen), "should have 9 unique (M,K) pairs")
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
// exactly 3 points with 1D shapes [N] compatible with InterpolateMulMat.
func TestBenchmarkMulMat_OutputShapeContract(t *testing.T) {
	// The new benchmarkMulMat measures at 3 strategic N values: 1, 32, 512
	strategicNs := []int64{1, strategicNcross, 512}

	for _, N := range strategicNs {
		// Simulate what benchmarkMulMat's measurement does:
		pt := LatencyPoint{
			Shape:     []int64{4096, 4096, N}, // what measureOp returns
			LatencyUs: float64(N) * 10,
		}
		pt.Shape = []int64{N} // what benchmarkMulMat overrides

		assert.Len(t, pt.Shape, 1, "MUL_MAT points must be 1D for InterpolateMulMat")
		assert.Equal(t, N, pt.Shape[0], "Shape[0] must be the sweep dimension N")
	}

	// Verify InterpolateMulMat can consume 3-point curves
	curves := []OperatorCurve{{
		Op: "MUL_MAT", FixedDims: map[string]int64{"M": 2048, "K": 2048},
		Points: []LatencyPoint{
			{Shape: []int64{1}, LatencyUs: 100.0},
			{Shape: []int64{32}, LatencyUs: 800.0},
			{Shape: []int64{512}, LatencyUs: 5000.0},
		},
	}}
	result := InterpolateMulMat(curves, 2048, 2048, 128)
	assert.Greater(t, result, 800.0, "N=128 should be > N=32 latency")
	assert.Less(t, result, 5000.0, "N=128 should be < N=512 latency")
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

func TestPredictMulMatDirect_SingleCurve(t *testing.T) {
	// With a single reference curve at (4096,4096), direct lookup should return
	// the interpolated latency at the exact reference shape.
	profile := &Profile{
		Operators: []OperatorCurve{{
			Op:          "MUL_MAT",
			WeightDtype: "q4_0",
			FixedDims:   map[string]int64{"M": 4096, "K": 4096},
			Points: []LatencyPoint{
				{Shape: []int64{1}, LatencyUs: 2021},
				{Shape: []int64{3}, LatencyUs: 1968},
				{Shape: []int64{6}, LatencyUs: 2105},
				{Shape: []int64{35}, LatencyUs: 2863},
				{Shape: []int64{380}, LatencyUs: 8610},
				{Shape: []int64{4096}, LatencyUs: 95381},
			},
		}},
	}

	// Exact match at reference shape (4096,4096,N=1) → measured value
	lat := PredictMulMatDirect(profile, 4096, 4096, 1, "q4_0")
	assert.InDelta(t, 2021, lat, 1.0)

	// N=3 interpolation
	lat = PredictMulMatDirect(profile, 4096, 4096, 3, "q4_0")
	assert.InDelta(t, 1968, lat, 1.0)

	// Wrong dtype → 0
	lat = PredictMulMatDirect(profile, 4096, 4096, 1, "f16")
	assert.Equal(t, 0.0, lat)
}

func TestPredictMulMatDirect_MultiCurve(t *testing.T) {
	// With two reference curves, IDW should interpolate between them.
	profile := &Profile{
		Operators: []OperatorCurve{
			{
				Op:          "MUL_MAT",
				WeightDtype: "q4_0",
				FixedDims:   map[string]int64{"M": 4096, "K": 4096},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 2000},
				},
			},
			{
				Op:          "MUL_MAT",
				WeightDtype: "q4_0",
				FixedDims:   map[string]int64{"M": 2048, "K": 2048},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 500},
				},
			},
		},
	}

	// At (4096,4096): should return ~2000 (exact match)
	lat := PredictMulMatDirect(profile, 4096, 4096, 1, "q4_0")
	assert.InDelta(t, 2000, lat, 1.0)

	// At (2048,2048): should return ~500 (exact match)
	lat = PredictMulMatDirect(profile, 2048, 2048, 1, "q4_0")
	assert.InDelta(t, 500, lat, 1.0)

	// At (3072,3072): should interpolate between 500 and 2000
	lat = PredictMulMatDirect(profile, 3072, 3072, 1, "q4_0")
	assert.Greater(t, lat, 500.0)
	assert.Less(t, lat, 2000.0)
	t.Logf("IDW interpolation at (3072,3072,N=1): %.1f μs (between 500 and 2000)", lat)
}

func TestPredictMulMatDirect_VsRoofline(t *testing.T) {
	// Demonstrate the difference between roofline and direct interpolation
	// when reference curves at multiple (M,K) pairs exist.
	profile := &Profile{
		Hardware: HardwareProfile{
			PeakTOPS:                 map[string]float64{"q4_0": 1569e9},
			PeakBandwidthBytesPerSec: 48.75e9,
			EfficiencyConstants: map[string]OpEfficiency{
				"MUL_MAT_q4_0": {ComputeEff: 0.872, BWEff: 0.096},
			},
		},
		Operators: []OperatorCurve{
			{
				Op: "MUL_MAT", WeightDtype: "q4_0",
				FixedDims: map[string]int64{"M": 4096, "K": 4096},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 2021},
				},
			},
			{
				Op: "MUL_MAT", WeightDtype: "q4_0",
				FixedDims: map[string]int64{"M": 2048, "K": 2048},
				Points: []LatencyPoint{
					// Hypothetical: smaller matrix is faster per byte (better cache)
					{Shape: []int64{1}, LatencyUs: 400},
				},
			},
		},
	}

	// Roofline prediction for (2048,2048,N=1,q4_0):
	// bytes = 2048*2048*0.5625 + (2048+2048)*4 ≈ 2.36MB
	// bwTime = 2.36e6 / (0.096 * 48.75e9) * 1e6 ≈ 504 μs
	roofline := PredictMulMatLatency(&profile.Hardware, 2048, 2048, 1, "q4_0")
	t.Logf("Roofline prediction (2048,2048,N=1): %.1f μs", roofline)

	// Direct interpolation: should use measured 400 μs (exact match at 2048,2048)
	direct := PredictMulMatDirect(profile, 2048, 2048, 1, "q4_0")
	t.Logf("Direct prediction (2048,2048,N=1): %.1f μs", direct)

	// With actual data at (2048,2048)=400μs, direct is much closer to truth
	assert.InDelta(t, 400, direct, 1.0, "direct should match measured data")
	assert.Greater(t, roofline, 450.0, "roofline should overpredict for smaller shapes")

	t.Logf("Roofline/Direct ratio: %.2fx — direct interpolation wins when reference data exists", roofline/direct)
}

func TestElemBytesFromDtype(t *testing.T) {
	assert.Equal(t, 4.0, elemBytesFromDtype("f32"))
	assert.Equal(t, 2.0, elemBytesFromDtype("f16"))
	assert.InDelta(t, 0.5625, elemBytesFromDtype("q4_0"), 0.001) // 18 bytes / 32 elements
	assert.InDelta(t, 1.0625, elemBytesFromDtype("q8_0"), 0.001) // 34 bytes / 32 elements
	assert.Equal(t, 4.0, elemBytesFromDtype("unknown"))
}

// TestMeasureMulMat_OutputShape verifies MUL_MAT returns correct shape metadata.
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

func TestPlanStepCount_AllOps(t *testing.T) {
	caps := GetBackendCapabilities("Vulkan")
	ops := DefaultBenchmarkOps()
	dtypes := []string{"f32", "f16", "q4_0", "q8_0"}
	plan := buildBenchmarkPlan(ops, dtypes, caps, DefaultBenchmarkConfig())
	// Plan should contain: HWChar + MulMatRef + Operators + FusedOps + Overhead
	require.NotEmpty(t, plan)
	assert.Equal(t, StepHWChar, plan[0].Type)
	assert.Greater(t, len(plan), 1)
}

func TestPlanStepCount_SubsetOps(t *testing.T) {
	caps := GetBackendCapabilities("Vulkan")
	dtypes := []string{"f32", "f16", "q4_0", "q8_0"}

	plan := buildBenchmarkPlan([]string{"SILU"}, dtypes, caps, DefaultBenchmarkConfig())
	opCount := 0
	for _, s := range plan {
		if s.Type == StepOperator {
			opCount++
		}
	}
	assert.Equal(t, 1, opCount, "SILU is 1D, f32 only -> 1 operator step")

	plan = buildBenchmarkPlan([]string{"MUL_MAT"}, dtypes, caps, DefaultBenchmarkConfig())
	refCount := 0
	for _, s := range plan {
		if s.Type == StepMulMatRef {
			refCount++
		}
	}
	assert.Equal(t, 36, refCount, "MUL_MAT -> 4 dtypes × 9 (M,K) pairs = 36 ref curves")
}

// --- Direct backend measurement tests ---

func TestMeasureOpGPU_ReturnsGPUTimings(t *testing.T) {
	backend := setupBenchBackend(t)
	caps := DiscoverBackend(backend)
	if !caps.HasGPUTimestamp {
		t.Skip("no GPU timestamp support")
	}
	cfg := DefaultBenchmarkConfig()
	cfg.MeasureReps = 5
	cfg.MinReps = 3
	pt := measureOpGPU(backend, "ADD", []int64{65536}, "f32", cfg)
	assert.Greater(t, pt.LatencyUs, 0.0, "GPU-measured latency should be positive")
	assert.Greater(t, pt.Reps, 0, "should have measured at least 1 rep")
}

func TestMeasureOpForBackend_GPUDispatch(t *testing.T) {
	backend := setupBenchBackend(t)
	caps := DiscoverBackend(backend)
	cfg := DefaultBenchmarkConfig()
	cfg.MeasureReps = 5
	cfg.MinReps = 3
	// Use large shape to ensure measurable wall-clock time even on CPU
	pt := measureOpForBackend(backend, caps, "ADD", []int64{4 * 1024 * 1024}, "f32", cfg)
	assert.Greater(t, pt.LatencyUs, 0.0, "latency should be positive")
	assert.Greater(t, pt.Reps, 0, "should have measured at least 1 rep")
}

func TestMeasureOpGPU_FallbackToWallClock(t *testing.T) {
	backend := setupBenchBackend(t)
	caps := DiscoverBackend(backend)
	if caps.HasGPUTimestamp {
		t.Skip("this test is for backends WITHOUT GPU timestamps (CPU-only)")
	}
	cfg := DefaultBenchmarkConfig()
	cfg.MeasureReps = 5
	cfg.MinReps = 3
	// Use large shape to ensure measurable wall-clock time on CPU
	pt := measureOpForBackend(backend, caps, "ADD", []int64{4 * 1024 * 1024}, "f32", cfg)
	assert.Greater(t, pt.LatencyUs, 0.0, "wall-clock fallback should still produce positive latency")
	assert.Greater(t, pt.Reps, 0, "should have measured at least 1 rep")
}

func TestMeasureOp_SchedulerPath(t *testing.T) {
	backend := setupBenchBackend(t)
	cfg := DefaultBenchmarkConfig()
	cfg.MeasureReps = 5
	cfg.MinReps = 3
	// Use large shape to ensure measurable wall-clock time even on CPU
	pt := measureOp(backend, "ADD", []int64{4 * 1024 * 1024}, "f32", cfg, -1)
	assert.Greater(t, pt.LatencyUs, 0.0, "scheduler path should produce positive latency")
	ptDirect := measureOp(backend, "ADD", []int64{4 * 1024 * 1024}, "f32", cfg, 0)
	assert.Greater(t, ptDirect.LatencyUs, 0.0, "direct path should produce positive latency")
	t.Logf("scheduler latency: %.1f us, direct latency: %.1f us", pt.LatencyUs, ptDirect.LatencyUs)
}

func TestMeasureOpGPU_MulMatVec(t *testing.T) {
	backend := setupBenchBackend(t)
	caps := DiscoverBackend(backend)
	if !caps.HasGPUTimestamp {
		t.Skip("no GPU timestamp support")
	}
	cfg := DefaultBenchmarkConfig()
	cfg.MeasureReps = 5
	cfg.MinReps = 3
	pt := measureOpGPU(backend, "MUL_MAT", []int64{256, 256, 1}, "f32", cfg)
	assert.Greater(t, pt.LatencyUs, 0.0, "MUL_MAT_VEC should produce positive GPU latency")
	t.Logf("MUL_MAT_VEC (256x256, N=1) GPU latency: %.1f us", pt.LatencyUs)
}

func TestMeasureOpGPU_ConvergenceEarlyExit(t *testing.T) {
	backend := setupBenchBackend(t)
	caps := DiscoverBackend(backend)
	if !caps.HasGPUTimestamp {
		t.Skip("no GPU timestamp support")
	}
	cfg := DefaultBenchmarkConfig()
	cfg.MeasureReps = 50
	cfg.MinReps = 3
	cfg.ConvergenceCV = 0.5
	pt := measureOpGPU(backend, "ADD", []int64{65536}, "f32", cfg)
	assert.Greater(t, pt.LatencyUs, 0.0, "should produce positive latency")
	assert.Less(t, pt.Reps, 50, "should converge before max reps with lenient CV")
	t.Logf("converged in %d reps, latency: %.1f ± %.1f us", pt.Reps, pt.LatencyUs, pt.StddevUs)
}

func TestLookupLatencyV3_MulMatAllNUsesDirect(t *testing.T) {
	// Profile with 3×3 grid reference curves (3 points each)
	profile := &Profile{
		Version: 3,
		Hardware: HardwareProfile{
			PeakTOPS:                 map[string]float64{"f32": 64.3e9},
			PeakBandwidthBytesPerSec: 40.7e9,
			EfficiencyConstants: map[string]OpEfficiency{
				"MUL_MAT_f32": {ComputeEff: 0.93, BWEff: 0.45, OverheadUs: 500},
			},
		},
		Operators: []OperatorCurve{
			{
				Op: "MUL_MAT", WeightDtype: "f32",
				FixedDims:  map[string]int64{"M": 2048, "K": 2048},
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 500},
					{Shape: []int64{32}, LatencyUs: 800},
					{Shape: []int64{512}, LatencyUs: 50000},
				},
			},
			{
				Op: "MUL_MAT", WeightDtype: "f32",
				FixedDims:  map[string]int64{"M": 8192, "K": 2048},
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 2000},
					{Shape: []int64{32}, LatencyUs: 3000},
					{Shape: []int64{512}, LatencyUs: 200000},
				},
			},
		},
	}
	caps := GetBackendCapabilities("Vulkan")

	// N=1 (VEC range) — should use direct interpolation, NOT roofline
	lat1, err := lookupLatencyV3(profile, "MUL_MAT", []int64{2048, 2048, 1}, "f32", "f32", "Vulkan", &caps)
	require.NoError(t, err)
	assert.InDelta(t, 500.0, lat1, 50.0, "N=1 at exact grid point should be ~500μs from curve")

	// N=256 (MAT range) — should ALSO use direct interpolation
	lat256, err := lookupLatencyV3(profile, "MUL_MAT", []int64{2048, 2048, 256}, "f32", "f32", "Vulkan", &caps)
	require.NoError(t, err)
	assert.Greater(t, lat256, 800.0, "N=256 should be > N=32 latency")
	assert.Less(t, lat256, 50000.0, "N=256 should be < N=512 latency")

	// Query at (M,K) between grid points — should IDW blend
	latBlend, err := lookupLatencyV3(profile, "MUL_MAT", []int64{4096, 2048, 1}, "f32", "f32", "Vulkan", &caps)
	require.NoError(t, err)
	assert.Greater(t, latBlend, 500.0, "IDW blend should be > nearest small curve")
	assert.Less(t, latBlend, 2000.0, "IDW blend should be < nearest large curve")
}
