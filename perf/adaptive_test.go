package perf

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockMeasure returns a measurement function that follows a known mathematical function.
// This lets us test adaptive sampling convergence without a real backend.
func mockMeasure(f func(n int64) float64) func(shape []int64) LatencyPoint {
	return func(shape []int64) LatencyPoint {
		return LatencyPoint{
			Shape:     shape,
			LatencyUs: f(shape[0]),
			StddevUs:  0.01,
			Reps:      100,
		}
	}
}

func TestAdaptiveSample1D_SmoothPowerLaw(t *testing.T) {
	// f(N) = 0.01 * N^0.8 — smooth power law should converge quickly.
	measure := mockMeasure(func(n int64) float64 {
		return 0.01 * math.Pow(float64(n), 0.8)
	})
	cfg := BenchmarkConfig{
		ErrorThreshold: 0.05,
		MaxPointsPerOp: 20,
	}

	points := AdaptiveSample1D(measure, 1024, 8*1024*1024, 8, cfg)

	// Should converge with relatively few points (8 initial + a few refinements)
	assert.LessOrEqual(t, len(points), 12, "smooth function should converge quickly")
	assert.GreaterOrEqual(t, len(points), 8, "should have at least the initial grid")

	// Verify points are sorted by Shape[0]
	for i := 1; i < len(points); i++ {
		assert.Greater(t, points[i].Shape[0], points[i-1].Shape[0])
	}
}

func TestAdaptiveSample1D_SharpKnee(t *testing.T) {
	// f(N) has a sharp transition (knee) at N=10000:
	//   N < 10000: constant 5.0us (kernel launch overhead)
	//   N >= 10000: 5.0 + 0.001*(N-10000) (bandwidth-limited)
	measure := mockMeasure(func(n int64) float64 {
		if n < 10000 {
			return 5.0
		}
		return 5.0 + 0.001*float64(n-10000)
	})
	cfg := BenchmarkConfig{
		ErrorThreshold: 0.05,
		MaxPointsPerOp: 20,
	}

	points := AdaptiveSample1D(measure, 1024, 8*1024*1024, 8, cfg)

	// Should add extra points around the knee
	assert.Greater(t, len(points), 8, "should refine around the knee point")

	// Verify there are points near the knee (N ≈ 10000)
	hasNearKnee := false
	for _, pt := range points {
		if pt.Shape[0] >= 5000 && pt.Shape[0] <= 20000 {
			hasNearKnee = true
			break
		}
	}
	assert.True(t, hasNearKnee, "should have points near the knee at N=10000")
}

func TestAdaptiveSample1D_BudgetLimit(t *testing.T) {
	// Even if the function never converges, should stop at MaxPointsPerOp.
	measure := mockMeasure(func(n int64) float64 {
		// Fixed: use abs() to ensure positive values for log
		return math.Abs(math.Sin(float64(n)/1000))*float64(n) + 1.0
	})
	cfg := BenchmarkConfig{
		ErrorThreshold: 0.001, // very tight threshold — won't converge
		MaxPointsPerOp: 15,
	}

	points := AdaptiveSample1D(measure, 1024, 8*1024*1024, 8, cfg)
	assert.LessOrEqual(t, len(points), cfg.MaxPointsPerOp,
		"must respect budget limit")
}

func TestAdaptiveSample1D_AlreadyConverged(t *testing.T) {
	// Perfect power law — initial grid should be sufficient, no refinement needed.
	measure := mockMeasure(func(n int64) float64 {
		return math.Pow(float64(n), 1.0) // perfectly linear in log-log
	})
	cfg := BenchmarkConfig{
		ErrorThreshold: 0.05,
		MaxPointsPerOp: 20,
	}

	points := AdaptiveSample1D(measure, 1024, 8*1024*1024, 8, cfg)
	// Pure power law: initial 8 points + 1 convergence check midpoint
	assert.LessOrEqual(t, len(points), 9, "pure power law needs minimal refinement")
	assert.GreaterOrEqual(t, len(points), 8)
}

func TestAdaptiveSample1D_MinMaxRange(t *testing.T) {
	// First and last points should be at the specified min/max.
	measure := mockMeasure(func(n int64) float64 {
		return float64(n)
	})
	cfg := BenchmarkConfig{
		ErrorThreshold: 0.05,
		MaxPointsPerOp: 20,
	}

	points := AdaptiveSample1D(measure, 512, 1048576, 6, cfg)
	require.GreaterOrEqual(t, len(points), 6)
	assert.Equal(t, int64(512), points[0].Shape[0], "first point should be at min")
	assert.Equal(t, int64(1048576), points[len(points)-1].Shape[0], "last point should be at max")
}

func TestWorstInterval_FindsKnee(t *testing.T) {
	// Points with a sharp change at index 2: 1.0, 10.0, 50.0 (should be 100.0 for power law)
	points := makePoints([][2]float64{
		{100, 1.0},
		{1000, 10.0},
		{10000, 50.0}, // deviates from power law (expected ~100.0)
	})

	idx := worstInterval(points)
	assert.GreaterOrEqual(t, idx, 0)
	assert.Less(t, idx, len(points)-1)
}

func TestWorstInterval_TwoPoints(t *testing.T) {
	points := makePoints([][2]float64{
		{100, 1.0},
		{10000, 100.0},
	})
	idx := worstInterval(points)
	assert.Equal(t, 0, idx, "with 2 points, should return the only interval")
}

func TestWorstInterval_ZeroLatency(t *testing.T) {
	pts := []LatencyPoint{
		{Shape: []int64{100}, LatencyUs: 0.0},
		{Shape: []int64{1000}, LatencyUs: 0.0},
	}
	idx := worstInterval(pts)
	assert.Equal(t, -1, idx, "all-zero latencies should return -1")
}

func TestInterpolationError_ExactMatch(t *testing.T) {
	// Midpoint exactly on the log-linear interpolation → error ≈ 0
	left := LatencyPoint{Shape: []int64{100}, LatencyUs: 10.0}
	right := LatencyPoint{Shape: []int64{10000}, LatencyUs: 1000.0}
	mid := LatencyPoint{Shape: []int64{1000}, LatencyUs: 100.0} // geometric midpoint, power law
	err := interpolationError(left, right, mid)
	assert.InDelta(t, 0.0, err, 0.001)
}

func TestInterpolationError_Deviation(t *testing.T) {
	left := LatencyPoint{Shape: []int64{100}, LatencyUs: 10.0}
	right := LatencyPoint{Shape: []int64{10000}, LatencyUs: 1000.0}
	mid := LatencyPoint{Shape: []int64{1000}, LatencyUs: 50.0} // 2x off from expected 100
	err := interpolationError(left, right, mid)
	assert.Greater(t, err, 0.1, "50 vs expected 100 should show significant error")
}

func TestInterpolationError_ZeroLatency(t *testing.T) {
	left := LatencyPoint{Shape: []int64{100}, LatencyUs: 0.0}
	right := LatencyPoint{Shape: []int64{10000}, LatencyUs: 10.0}
	mid := LatencyPoint{Shape: []int64{1000}, LatencyUs: 5.0}
	err := interpolationError(left, right, mid)
	assert.Equal(t, 0.0, err, "zero-latency endpoint should return 0 error")
}

func TestLogMidpoint(t *testing.T) {
	// Geometric midpoint of 100 and 10000 = sqrt(100*10000) = 1000
	mid := logMidpoint(100, 10000)
	assert.Equal(t, int64(1000), mid)
}

func TestLogMidpoint_AdjacentValues(t *testing.T) {
	// Small gap
	mid := logMidpoint(1000, 1100)
	assert.Greater(t, mid, int64(1000))
	assert.Less(t, mid, int64(1100))
}

func TestInsertSorted(t *testing.T) {
	pts := makePoints([][2]float64{
		{100, 1.0}, {1000, 10.0}, {10000, 100.0},
	})
	newPt := LatencyPoint{Shape: []int64{500}, LatencyUs: 5.0}
	result := insertSorted(pts, newPt)
	require.Len(t, result, 4)
	assert.Equal(t, int64(100), result[0].Shape[0])
	assert.Equal(t, int64(500), result[1].Shape[0])
	assert.Equal(t, int64(1000), result[2].Shape[0])
	assert.Equal(t, int64(10000), result[3].Shape[0])
}

func TestInsertSorted_Duplicate(t *testing.T) {
	pts := makePoints([][2]float64{
		{100, 1.0}, {1000, 10.0},
	})
	dup := LatencyPoint{Shape: []int64{1000}, LatencyUs: 15.0}
	result := insertSorted(pts, dup)
	assert.Len(t, result, 2, "should not insert duplicate")
}

func TestAdaptiveSample1D_ZeroLatency(t *testing.T) {
	// If measure returns zero latency, adaptive sampling should not panic or produce NaN
	measure := func(shape []int64) LatencyPoint {
		return LatencyPoint{Shape: shape, LatencyUs: 0.0}
	}
	cfg := BenchmarkConfig{ErrorThreshold: 0.05, MaxPointsPerOp: 10}
	points := AdaptiveSample1D(measure, 100, 10000, 4, cfg)
	assert.GreaterOrEqual(t, len(points), 4)
	for _, pt := range points {
		assert.False(t, math.IsNaN(pt.LatencyUs))
		assert.False(t, math.IsInf(pt.LatencyUs, 0))
	}
}

func TestWidestInterval(t *testing.T) {
	// Three intervals: [100, 200], [200, 10000], [10000, 20000]
	// [200, 10000] is widest in log space
	points := makePoints([][2]float64{
		{100, 1.0}, {200, 2.0}, {10000, 100.0}, {20000, 200.0},
	})
	idx := widestInterval(points)
	assert.Equal(t, 1, idx, "interval [200, 10000] should be widest")
}
