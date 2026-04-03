package perf

import (
	"math"
	"sort"
)

// MeasureFunc is a callback that benchmarks an op at a given shape and returns the result.
// This abstraction allows testing with mock measurements.
type MeasureFunc func(shape []int64) LatencyPoint

// AdaptiveSample1D performs adaptive 1D sampling in log space.
//
// Algorithm:
//  1. Create nInitial log-spaced points between shapeMin and shapeMax
//  2. Measure latency at each point
//  3. Find the interval with highest interpolation error
//  4. If error > threshold, measure the midpoint and repeat from step 3
//  5. Stop when all errors < threshold or budget (MaxPointsPerOp) is exhausted
//
// The measure callback is invoked for each point. For real benchmarks, this
// calls the GGML backend. For tests, this can be a mock function.
func AdaptiveSample1D(measure MeasureFunc, shapeMin, shapeMax int64, nInitial int, cfg BenchmarkConfig) []LatencyPoint {
	// Step 1: Initial log-spaced grid
	logMin := math.Log(float64(shapeMin))
	logMax := math.Log(float64(shapeMax))
	points := make([]LatencyPoint, 0, cfg.MaxPointsPerOp)

	for i := 0; i < nInitial; i++ {
		logN := logMin + float64(i)*(logMax-logMin)/float64(nInitial-1)
		N := int64(math.Round(math.Exp(logN)))
		pt := measure([]int64{N})
		points = append(points, pt)
	}

	// Step 2: Adaptive refinement
	for len(points) < cfg.MaxPointsPerOp {
		maxErr, maxIdx := findMaxInterpolationError(points, measure)
		if maxErr < cfg.ErrorThreshold {
			break
		}
		// Measure midpoint of highest-error interval
		midN := logMidpoint(points[maxIdx].Shape[0], points[maxIdx+1].Shape[0])
		// Skip if midpoint is same as either endpoint (can't refine further)
		if midN == points[maxIdx].Shape[0] || midN == points[maxIdx+1].Shape[0] {
			break
		}
		pt := measure([]int64{midN})
		points = insertSorted(points, pt)
	}

	return points
}

// findMaxInterpolationError finds the interval with highest relative error
// between the interpolated value and the actual measured value at the midpoint.
//
// Returns (maxError, intervalIndex). Error is measured in log space:
//
//	relErr = |log(interpolated) - log(actual)| / |log(actual)|
func findMaxInterpolationError(points []LatencyPoint, measure MeasureFunc) (float64, int) {
	maxErr := 0.0
	maxIdx := 0

	for i := 0; i < len(points)-1; i++ {
		logX1 := math.Log(float64(points[i].Shape[0]))
		logX2 := math.Log(float64(points[i+1].Shape[0]))
		logY1 := math.Log(points[i].LatencyUs)
		logY2 := math.Log(points[i+1].LatencyUs)

		// Interpolated value at log-midpoint
		logMid := (logX1 + logX2) / 2
		logInterp := logY1 + (logY2-logY1)*(logMid-logX1)/(logX2-logX1)

		// Actual measurement at midpoint
		midN := int64(math.Round(math.Exp(logMid)))
		actual := measure([]int64{midN})
		actualLogY := math.Log(actual.LatencyUs)

		// Relative error in log space
		relErr := 0.0
		if actualLogY != 0 {
			relErr = math.Abs(logInterp-actualLogY) / math.Abs(actualLogY)
		}

		if relErr > maxErr {
			maxErr = relErr
			maxIdx = i
		}
	}

	return maxErr, maxIdx
}

// logMidpoint returns the geometric midpoint of two values: round(exp((log(a)+log(b))/2)).
func logMidpoint(a, b int64) int64 {
	logMid := (math.Log(float64(a)) + math.Log(float64(b))) / 2
	return int64(math.Round(math.Exp(logMid)))
}

// insertSorted inserts a LatencyPoint into a sorted slice (by Shape[0]).
func insertSorted(points []LatencyPoint, pt LatencyPoint) []LatencyPoint {
	idx := sort.Search(len(points), func(i int) bool {
		return points[i].Shape[0] >= pt.Shape[0]
	})
	// Don't insert duplicate shapes
	if idx < len(points) && points[idx].Shape[0] == pt.Shape[0] {
		return points
	}
	points = append(points, LatencyPoint{})
	copy(points[idx+1:], points[idx:])
	points[idx] = pt
	return points
}
