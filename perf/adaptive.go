package perf

import (
	"log/slog"
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
//  3. Find the interval with highest interpolation error (using existing points only)
//  4. Measure the midpoint of that interval and compute the actual error
//  5. If error > threshold, insert the midpoint and repeat from step 3
//  6. Stop when error < threshold or budget (MaxPointsPerOp) is exhausted
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
		slog.Info("sample", "point", i+1, "of", nInitial, "N", N, "latency_us", pt.LatencyUs)
	}

	// Step 2: Adaptive refinement — one measurement per round
	round := 0
	for len(points) < cfg.MaxPointsPerOp {
		// Find the worst interval using only existing points (zero extra measurements)
		splitIdx := worstInterval(points)
		if splitIdx < 0 {
			slog.Info("converged", "points", len(points), "reason", "no valid intervals")
			break
		}

		// Measure midpoint of worst interval
		midN := logMidpoint(points[splitIdx].Shape[0], points[splitIdx+1].Shape[0])
		if midN == points[splitIdx].Shape[0] || midN == points[splitIdx+1].Shape[0] {
			break
		}
		midPt := measure([]int64{midN})

		// Compute actual interpolation error at the measured midpoint
		err := interpolationError(points[splitIdx], points[splitIdx+1], midPt)
		if err < cfg.ErrorThreshold {
			slog.Info("converged", "points", len(points), "max_error", err)
			// Insert the point anyway (it's a useful data point) and stop
			points = insertSorted(points, midPt)
			break
		}

		points = insertSorted(points, midPt)
		round++
		slog.Info("refine", "round", round, "N", midN, "error", err, "points", len(points))
	}

	return points
}

// worstInterval finds the interval index that most likely has the highest
// interpolation error, using only already-measured points (zero measurements).
//
// Strategy: for each internal point p_i, check how well it's predicted by
// interpolating p_{i-1} and p_{i+1} in log-log space. The interval adjacent
// to the worst-predicted point is the one to split.
//
// Returns the interval index to split, or -1 if no valid intervals exist.
func worstInterval(points []LatencyPoint) int {
	if len(points) < 3 {
		// With fewer than 3 points, split the only/first interval
		if len(points) == 2 && points[0].LatencyUs > 0 && points[1].LatencyUs > 0 {
			return 0
		}
		return -1
	}

	maxErr := 0.0
	worstIdx := -1 // index of the worst-predicted internal point

	for i := 1; i < len(points)-1; i++ {
		if points[i-1].LatencyUs <= 0 || points[i].LatencyUs <= 0 || points[i+1].LatencyUs <= 0 {
			continue
		}

		logX0 := math.Log(float64(points[i-1].Shape[0]))
		logX1 := math.Log(float64(points[i].Shape[0]))
		logX2 := math.Log(float64(points[i+1].Shape[0]))
		logY0 := math.Log(points[i-1].LatencyUs)
		logY2 := math.Log(points[i+1].LatencyUs)

		// Linear interpolation in log-log space: predict logY1 from (logX0,logY0) and (logX2,logY2)
		t := (logX1 - logX0) / (logX2 - logX0)
		logInterp := logY0 + t*(logY2-logY0)
		actualLogY := math.Log(points[i].LatencyUs)

		relErr := 0.0
		if actualLogY != 0 {
			relErr = math.Abs(logInterp-actualLogY) / math.Abs(actualLogY)
		}
		if relErr > maxErr {
			maxErr = relErr
			worstIdx = i
		}
	}

	if worstIdx < 0 {
		// All internal points are well-predicted; split the widest interval
		return widestInterval(points)
	}

	// Split the wider half around the worst-predicted point
	leftSpan := math.Log(float64(points[worstIdx].Shape[0])) - math.Log(float64(points[worstIdx-1].Shape[0]))
	rightSpan := math.Log(float64(points[worstIdx+1].Shape[0])) - math.Log(float64(points[worstIdx].Shape[0]))
	if rightSpan > leftSpan {
		return worstIdx
	}
	return worstIdx - 1
}

// widestInterval returns the index of the interval with the largest span in log(N) space.
func widestInterval(points []LatencyPoint) int {
	maxSpan := 0.0
	maxIdx := 0
	for i := 0; i < len(points)-1; i++ {
		if points[i].Shape[0] <= 0 || points[i+1].Shape[0] <= 0 {
			continue
		}
		span := math.Log(float64(points[i+1].Shape[0])) - math.Log(float64(points[i].Shape[0]))
		if span > maxSpan {
			maxSpan = span
			maxIdx = i
		}
	}
	return maxIdx
}

// interpolationError computes the relative error between the log-linear interpolation
// of two endpoints and an actual measured midpoint.
func interpolationError(left, right, mid LatencyPoint) float64 {
	if left.LatencyUs <= 0 || right.LatencyUs <= 0 || mid.LatencyUs <= 0 {
		return 0
	}
	logX1 := math.Log(float64(left.Shape[0]))
	logX2 := math.Log(float64(right.Shape[0]))
	logXM := math.Log(float64(mid.Shape[0]))
	logY1 := math.Log(left.LatencyUs)
	logY2 := math.Log(right.LatencyUs)

	t := (logXM - logX1) / (logX2 - logX1)
	logInterp := logY1 + t*(logY2-logY1)
	actualLogY := math.Log(mid.LatencyUs)

	if actualLogY == 0 {
		return 0
	}
	return math.Abs(logInterp-actualLogY) / math.Abs(actualLogY)
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
