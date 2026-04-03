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

	points := AdaptiveSample1D(measure, 1024, 64*1024*1024, 8, cfg)

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

	points := AdaptiveSample1D(measure, 1024, 64*1024*1024, 8, cfg)

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

	points := AdaptiveSample1D(measure, 1024, 64*1024*1024, 8, cfg)
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

	points := AdaptiveSample1D(measure, 1024, 64*1024*1024, 8, cfg)
	assert.Equal(t, 8, len(points), "pure power law needs no refinement")
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

func TestFindMaxInterpolationError(t *testing.T) {
	// Create points where interval 1-2 has the highest error.
	// Points: (100, 1.0), (1000, 10.0), (10000, 50.0)
	// Intervals: [100,1000] follows power law, [1000,10000] deviates
	points := makePoints([][2]float64{
		{100, 1.0},
		{1000, 10.0},  // consistent with power law
		{10000, 50.0},  // deviates (should be 100.0 for power law)
	})

	measure := mockMeasure(func(n int64) float64 {
		// True function: different from what the points suggest
		if n < 5000 {
			return 0.01 * float64(n)
		}
		return 0.005 * float64(n) // slope change
	})

	maxErr, maxIdx := findMaxInterpolationError(points, measure)
	assert.Greater(t, maxErr, 0.0, "should find some error")
	assert.GreaterOrEqual(t, maxIdx, 0)
	assert.Less(t, maxIdx, len(points)-1)
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

func TestFindMaxInterpolationError_ZeroLatency(t *testing.T) {
	// Points with zero latency should not cause NaN/Inf
	pts := []LatencyPoint{
		{Shape: []int64{100}, LatencyUs: 0.0},
		{Shape: []int64{1000}, LatencyUs: 0.0},
	}
	measure := func(shape []int64) LatencyPoint {
		return LatencyPoint{Shape: shape, LatencyUs: 0.0}
	}
	maxErr, _ := findMaxInterpolationError(pts, measure)
	assert.False(t, math.IsNaN(maxErr))
	assert.False(t, math.IsInf(maxErr, 0))
}
