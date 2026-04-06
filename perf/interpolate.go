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
// For queries outside the grid (distance > ln(2)), uses physics-informed scaling fallback.
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

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].logDist < candidates[j].logDist
	})

	// Exact match or single curve
	if candidates[0].logDist == 0 || len(candidates) == 1 {
		lat := Interpolate1DByDim(candidates[0].curve.Points, 0, queryN)
		if candidates[0].logDist == 0 {
			return lat
		}
		// Single curve, may need scaling
		return scaleMulMatLatency(lat, candidates[0].curve, queryM, queryK)
	}

	// Check if query is outside the grid convex hull.
	// Heuristic: if nearest distance > ln(2) (~0.69) in combined log-(M,K) space,
	// the query is too far from the nearest grid point — use scaling.
	const extrapolationThreshold = 0.69 // ln(2)
	if candidates[0].logDist > extrapolationThreshold {
		lat := Interpolate1DByDim(candidates[0].curve.Points, 0, queryN)
		return scaleMulMatLatency(lat, candidates[0].curve, queryM, queryK)
	}

	// Inside grid: IDW blend between two nearest curves
	lat1 := Interpolate1DByDim(candidates[0].curve.Points, 0, queryN)
	lat2 := Interpolate1DByDim(candidates[1].curve.Points, 0, queryN)
	w1 := 1.0 / candidates[0].logDist
	w2 := 1.0 / candidates[1].logDist
	return (lat1*w1 + lat2*w2) / (w1 + w2)
}

// scaleMulMatLatency applies physics-informed scaling when extrapolating
// beyond the measured (M,K) grid. At any N, MUL_MAT latency scales
// proportionally to M*K (weight size dominates both BW and compute terms).
func scaleMulMatLatency(nearestLat float64, nearest *OperatorCurve, queryM, queryK int64) float64 {
	refM := nearest.FixedDims["M"]
	refK := nearest.FixedDims["K"]
	scaleFactor := float64(queryM*queryK) / float64(refM*refK)
	return nearestLat * scaleFactor
}

// InterpolateFlashAttnMultiHead interpolates flash_attn latency across multiple
// (numQHeads, numKVHeads) curves using inverse distance weighting in 2D log space.
// Each curve has FixedDims["num_heads"] (Q heads) and optionally FixedDims["num_kv_heads"].
// If num_kv_heads is absent, the curve is treated as MHA (KV = Q).
func InterpolateFlashAttnMultiHead(curves []OperatorCurve, querySeqQ, querySeqKV, queryNumQHeads, queryNumKVHeads int64) float64 {
	if len(curves) == 0 {
		return 0
	}
	if len(curves) == 1 {
		return InterpolateFlashAttn(&curves[0], querySeqQ, querySeqKV)
	}

	type candidate struct {
		curve *OperatorCurve
		dist  float64
	}

	logQ := math.Log(float64(queryNumQHeads))
	logKV := math.Log(float64(queryNumKVHeads))
	var candidates []candidate
	for i := range curves {
		nh := curves[i].FixedDims["num_heads"]
		if nh <= 0 {
			continue
		}
		nkv := curves[i].FixedDims["num_kv_heads"]
		if nkv <= 0 {
			nkv = nh // backward compat: MHA
		}
		dq := logQ - math.Log(float64(nh))
		dkv := logKV - math.Log(float64(nkv))
		dist := math.Sqrt(dq*dq + dkv*dkv)
		candidates = append(candidates, candidate{&curves[i], dist})
	}

	if len(candidates) == 0 {
		return 0
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].dist < candidates[j].dist
	})

	// Exact match
	if candidates[0].dist == 0 {
		return InterpolateFlashAttn(candidates[0].curve, querySeqQ, querySeqKV)
	}

	// Single candidate after filtering
	if len(candidates) == 1 {
		return InterpolateFlashAttn(candidates[0].curve, querySeqQ, querySeqKV)
	}

	lat1 := InterpolateFlashAttn(candidates[0].curve, querySeqQ, querySeqKV)
	lat2 := InterpolateFlashAttn(candidates[1].curve, querySeqQ, querySeqKV)

	if lat1 <= 0 || lat2 <= 0 {
		if lat1 > 0 {
			return lat1
		}
		return lat2
	}

	// Check if query is outside the grid — extrapolate using power-law from nearest two
	nh1 := float64(candidates[0].curve.FixedDims["num_heads"])
	nkv1 := float64(candidates[0].curve.FixedDims["num_kv_heads"])
	if nkv1 <= 0 {
		nkv1 = nh1
	}
	nh2 := float64(candidates[1].curve.FixedDims["num_heads"])
	nkv2 := float64(candidates[1].curve.FixedDims["num_kv_heads"])
	if nkv2 <= 0 {
		nkv2 = nh2
	}
	// Use 1D distance along the line connecting the two nearest curves for extrapolation check
	totalDist := candidates[0].dist + candidates[1].dist
	gridSpan := math.Sqrt(math.Pow(math.Log(nh2)-math.Log(nh1), 2) + math.Pow(math.Log(nkv2)-math.Log(nkv1), 2))
	if gridSpan > 0 && totalDist > gridSpan*1.1 {
		// Query is likely outside the grid — power-law extrapolation.
		// Direction: curve2 → curve1 → query (candidates sorted nearest-first).
		// slope = rate of log-latency increase per unit distance toward query.
		logLat1 := math.Log(lat1)
		logLat2 := math.Log(lat2)
		slope := (logLat1 - logLat2) / gridSpan
		return math.Exp(logLat1 + slope*candidates[0].dist)
	}

	// IDW blend between two nearest curves
	w1 := 1.0 / candidates[0].dist
	w2 := 1.0 / candidates[1].dist
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
