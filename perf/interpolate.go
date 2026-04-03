package perf

import (
	"math"
	"sort"
)

// Interpolate1D performs log-log piecewise linear interpolation on a 1D latency curve.
// Points must be sorted by Shape[0] in ascending order.
// Uses power-law extrapolation beyond the measured range.
func Interpolate1D(points []LatencyPoint, queryN int64) float64 {
	if len(points) == 0 {
		return 0
	}
	if len(points) == 1 {
		return points[0].LatencyUs
	}
	if queryN <= 0 {
		return points[0].LatencyUs
	}

	logQ := math.Log(float64(queryN))

	// Filter out points with non-positive latency (can't take log)
	valid := make([]LatencyPoint, 0, len(points))
	for _, p := range points {
		if p.LatencyUs > 0 && p.Shape[0] > 0 {
			valid = append(valid, p)
		}
	}
	if len(valid) == 0 {
		return 0
	}
	if len(valid) == 1 {
		return valid[0].LatencyUs
	}

	// Search for the interval containing queryN
	for i := 0; i < len(valid)-1; i++ {
		logX1 := math.Log(float64(valid[i].Shape[0]))
		logX2 := math.Log(float64(valid[i+1].Shape[0]))
		if logQ >= logX1 && logQ <= logX2 {
			logY1 := math.Log(valid[i].LatencyUs)
			logY2 := math.Log(valid[i+1].LatencyUs)
			t := (logQ - logX1) / (logX2 - logX1)
			return math.Exp(logY1 + t*(logY2-logY1))
		}
	}

	// Extrapolate left or right
	if logQ < math.Log(float64(valid[0].Shape[0])) {
		return extrapolateLeft(valid, logQ)
	}
	return extrapolateRight(valid, logQ)
}

func extrapolateLeft(points []LatencyPoint, logQ float64) float64 {
	if len(points) < 2 {
		return points[0].LatencyUs
	}
	logX1 := math.Log(float64(points[0].Shape[0]))
	logX2 := math.Log(float64(points[1].Shape[0]))
	logY1 := math.Log(points[0].LatencyUs)
	logY2 := math.Log(points[1].LatencyUs)
	slope := (logY2 - logY1) / (logX2 - logX1)
	return math.Exp(logY1 + slope*(logQ-logX1))
}

func extrapolateRight(points []LatencyPoint, logQ float64) float64 {
	n := len(points)
	if n < 2 {
		return points[0].LatencyUs
	}
	logX1 := math.Log(float64(points[n-2].Shape[0]))
	logX2 := math.Log(float64(points[n-1].Shape[0]))
	logY1 := math.Log(points[n-2].LatencyUs)
	logY2 := math.Log(points[n-1].LatencyUs)
	slope := (logY2 - logY1) / (logX2 - logX1)
	return math.Exp(logY2 + slope*(logQ-logX2))
}

// Interpolate1DByDim interpolates along a specific dimension of multi-dimensional points.
// Sorts points by the specified dimension before interpolating.
func Interpolate1DByDim(points []LatencyPoint, dimIdx int, queryVal int64) float64 {
	if len(points) == 0 {
		return 0
	}
	if len(points) == 1 {
		return points[0].LatencyUs
	}
	if queryVal <= 0 {
		return points[0].LatencyUs
	}

	// Filter out points with non-positive latency or shape (can't take log)
	valid := make([]LatencyPoint, 0, len(points))
	for _, p := range points {
		if p.LatencyUs > 0 && len(p.Shape) > dimIdx && p.Shape[dimIdx] > 0 {
			valid = append(valid, p)
		}
	}
	if len(valid) == 0 {
		return 0
	}
	if len(valid) == 1 {
		return valid[0].LatencyUs
	}

	// Sort by the specified dimension
	sort.Slice(valid, func(i, j int) bool {
		return valid[i].Shape[dimIdx] < valid[j].Shape[dimIdx]
	})

	logQ := math.Log(float64(queryVal))

	// Search for the interval containing queryVal
	for i := 0; i < len(valid)-1; i++ {
		logX1 := math.Log(float64(valid[i].Shape[dimIdx]))
		logX2 := math.Log(float64(valid[i+1].Shape[dimIdx]))
		if logQ >= logX1 && logQ <= logX2 {
			logY1 := math.Log(valid[i].LatencyUs)
			logY2 := math.Log(valid[i+1].LatencyUs)
			t := (logQ - logX1) / (logX2 - logX1)
			return math.Exp(logY1 + t*(logY2-logY1))
		}
	}

	// Extrapolate left or right
	if logQ < math.Log(float64(valid[0].Shape[dimIdx])) {
		return extrapolateLeftByDim(valid, dimIdx, logQ)
	}
	return extrapolateRightByDim(valid, dimIdx, logQ)
}

func extrapolateLeftByDim(points []LatencyPoint, dimIdx int, logQ float64) float64 {
	if len(points) < 2 {
		return points[0].LatencyUs
	}
	logX1 := math.Log(float64(points[0].Shape[dimIdx]))
	logX2 := math.Log(float64(points[1].Shape[dimIdx]))
	logY1 := math.Log(points[0].LatencyUs)
	logY2 := math.Log(points[1].LatencyUs)
	slope := (logY2 - logY1) / (logX2 - logX1)
	return math.Exp(logY1 + slope*(logQ-logX1))
}

func extrapolateRightByDim(points []LatencyPoint, dimIdx int, logQ float64) float64 {
	n := len(points)
	if n < 2 {
		return points[0].LatencyUs
	}
	logX1 := math.Log(float64(points[n-2].Shape[dimIdx]))
	logX2 := math.Log(float64(points[n-1].Shape[dimIdx]))
	logY1 := math.Log(points[n-2].LatencyUs)
	logY2 := math.Log(points[n-1].LatencyUs)
	slope := (logY2 - logY1) / (logX2 - logX1)
	return math.Exp(logY2 + slope*(logQ-logX2))
}

// InterpolateMulMat interpolates mul_mat latency using inverse distance weighting in (M,K) space.
// Each curve has fixed M and K values, varying N.
// Returns interpolated latency at (queryM, queryK, queryN).
func InterpolateMulMat(curves []OperatorCurve, queryM, queryK, queryN int64) float64 {
	if len(curves) == 0 {
		return 0
	}

	type candidate struct {
		curve   *OperatorCurve
		logDist float64
	}

	var candidates []candidate
	for i := range curves {
		curveM := curves[i].FixedDims["M"]
		curveK := curves[i].FixedDims["K"]
		dM := math.Log(float64(queryM)) - math.Log(float64(curveM))
		dK := math.Log(float64(queryK)) - math.Log(float64(curveK))
		dist := math.Sqrt(dM*dM + dK*dK)
		candidates = append(candidates, candidate{&curves[i], dist})
	}

	// Sort by distance
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].logDist < candidates[j].logDist
	})

	// Exact match or single curve
	// Points store 1D shapes [N] since M,K are fixed per curve
	if candidates[0].logDist == 0 || len(candidates) == 1 {
		return Interpolate1DByDim(candidates[0].curve.Points, 0, queryN)
	}

	// Inverse distance weighting between two nearest curves
	lat1 := Interpolate1DByDim(candidates[0].curve.Points, 0, queryN)
	lat2 := Interpolate1DByDim(candidates[1].curve.Points, 0, queryN)
	w1 := 1.0 / candidates[0].logDist
	w2 := 1.0 / candidates[1].logDist
	return (lat1*w1 + lat2*w2) / (w1 + w2)
}

// InterpolateFlashAttn interpolates flash_attn latency between decode and prefill regimes.
// Decode regime: seqQ=1, varying seqKV
// Prefill regime: seqQ=seqKV
// Blends between regimes using log-space interpolation on seqQ.
func InterpolateFlashAttn(curve *OperatorCurve, querySeqQ, querySeqKV int64) float64 {
	// Separate decode and prefill points
	var prefillPts, decodePts []LatencyPoint
	for _, pt := range curve.Points {
		if pt.Shape[0] == 1 {
			decodePts = append(decodePts, pt)
		} else if pt.Shape[0] == pt.Shape[1] {
			prefillPts = append(prefillPts, pt)
		}
	}

	// Guard against empty regimes — fall back to whichever has data
	if len(decodePts) == 0 && len(prefillPts) == 0 {
		return 0
	}
	if len(decodePts) == 0 {
		return Interpolate1DByDim(prefillPts, 1, querySeqKV)
	}
	if len(prefillPts) == 0 {
		return Interpolate1DByDim(decodePts, 1, querySeqKV)
	}

	// Pure decode regime
	if querySeqQ == 1 {
		return Interpolate1DByDim(decodePts, 1, querySeqKV)
	}

	// Pure prefill regime
	if querySeqQ == querySeqKV {
		return Interpolate1DByDim(prefillPts, 1, querySeqKV)
	}

	// Guard against invalid seqKV
	if querySeqKV <= 1 {
		return Interpolate1DByDim(decodePts, 1, querySeqKV)
	}

	// Blend between decode and prefill
	decodeLat := Interpolate1DByDim(decodePts, 1, querySeqKV)
	prefillLat := Interpolate1DByDim(prefillPts, 1, querySeqKV)

	// Compute blend factor: t=0 at decode, t=1 at prefill
	t := math.Log(float64(querySeqQ)) / math.Log(float64(querySeqKV))
	if t < 0 {
		t = 0
	}
	if t > 1 {
		t = 1
	}

	// Log-space blend
	return math.Exp(math.Log(decodeLat)*(1-t) + math.Log(prefillLat)*t)
}
