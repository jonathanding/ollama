package perf

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

// --- Helper functions ---

func makePoints(pairs [][2]float64) []LatencyPoint {
	pts := make([]LatencyPoint, len(pairs))
	for i, p := range pairs {
		pts[i] = LatencyPoint{Shape: []int64{int64(p[0])}, LatencyUs: p[1]}
	}
	return pts
}

func makePowerLawPoints(c, alpha float64, Ns []int64) []LatencyPoint {
	pts := make([]LatencyPoint, len(Ns))
	for i, N := range Ns {
		pts[i] = LatencyPoint{
			Shape:     []int64{N},
			LatencyUs: c * math.Pow(float64(N), alpha),
		}
	}
	return pts
}

func makePointsMultiDim(shapes [][]int64, latencies []float64) []LatencyPoint {
	pts := make([]LatencyPoint, len(shapes))
	for i := range shapes {
		pts[i] = LatencyPoint{
			Shape:     shapes[i],
			LatencyUs: latencies[i],
		}
	}
	return pts
}

func makeMulMatCurves(mkPairs [][2]int64, nValues []int64, latencyFunc func(m, k, n int64) float64) []OperatorCurve {
	curves := make([]OperatorCurve, len(mkPairs))
	for i, mk := range mkPairs {
		pts := make([]LatencyPoint, len(nValues))
		for j, n := range nValues {
			pts[j] = LatencyPoint{
				Shape:     []int64{n},
				LatencyUs: latencyFunc(mk[0], mk[1], n),
			}
		}
		curves[i] = OperatorCurve{
			Op:         "mul_mat",
			Dimensions: []string{"N"},
			FixedDims:  map[string]int64{"M": mk[0], "K": mk[1]},
			Points:     pts,
		}
	}
	return curves
}

func makeFlashAttnCurve(decodeSeqKVs, prefillSeqs []int64, latencyFunc func(seqQ, seqKV int64) float64) *OperatorCurve {
	var pts []LatencyPoint
	for _, seqKV := range decodeSeqKVs {
		pts = append(pts, LatencyPoint{
			Shape:     []int64{1, seqKV},
			LatencyUs: latencyFunc(1, seqKV),
		})
	}
	for _, seq := range prefillSeqs {
		pts = append(pts, LatencyPoint{
			Shape:     []int64{seq, seq},
			LatencyUs: latencyFunc(seq, seq),
		})
	}
	return &OperatorCurve{
		Op:         "flash_attn",
		Dimensions: []string{"SeqQ", "SeqKV"},
		Points:     pts,
	}
}

// --- Part 4a: Interpolate1D tests ---

func TestInterpolate1D_ExactMatch(t *testing.T) {
	pts := makePoints([][2]float64{{10, 100}, {20, 200}, {30, 300}})
	assert.InDelta(t, 100.0, Interpolate1D(pts, 10), 1e-9)
	assert.InDelta(t, 200.0, Interpolate1D(pts, 20), 1e-9)
	assert.InDelta(t, 300.0, Interpolate1D(pts, 30), 1e-9)
}

func TestInterpolate1D_InteriorInterpolation(t *testing.T) {
	pts := makePoints([][2]float64{{10, 100}, {100, 1000}})
	result := Interpolate1D(pts, 31)
	expected := math.Exp(math.Log(100) + 0.5*(math.Log(1000)-math.Log(100)))
	assert.InDelta(t, expected, result, 10.0)
}

func TestInterpolate1D_PowerLawExactness(t *testing.T) {
	pts := makePowerLawPoints(2.0, 1.5, []int64{10, 100, 1000})
	for _, queryN := range []int64{15, 31, 100, 316, 500} {
		expected := 2.0 * math.Pow(float64(queryN), 1.5)
		result := Interpolate1D(pts, queryN)
		relErr := math.Abs(result-expected) / expected
		assert.Less(t, relErr, 0.05, "queryN=%d: got %.2f, want %.2f", queryN, result, expected)
	}
}

func TestInterpolate1D_LogVsLinear(t *testing.T) {
	pts := makePoints([][2]float64{{10, 100}, {100, 200}})
	logResult := Interpolate1D(pts, 31)
	linearExpected := 100 + (200-100)*(31.0-10)/(100-10)
	assert.Greater(t, math.Abs(logResult-linearExpected), 5.0)
	assert.InDelta(t, 138.0, logResult, 5.0)
}

func TestInterpolate1D_ExtrapolateLeft(t *testing.T) {
	pts := makePoints([][2]float64{{100, 1000}, {1000, 10000}})
	result := Interpolate1D(pts, 10)
	logX1 := math.Log(100.0)
	logX2 := math.Log(1000.0)
	logY1 := math.Log(1000.0)
	logY2 := math.Log(10000.0)
	slope := (logY2 - logY1) / (logX2 - logX1)
	expected := math.Exp(logY1 + slope*(math.Log(10)-logX1))
	assert.InDelta(t, expected, result, 1.0)
}

func TestInterpolate1D_ExtrapolateRight(t *testing.T) {
	pts := makePoints([][2]float64{{10, 100}, {100, 1000}})
	result := Interpolate1D(pts, 1000)
	logX1 := math.Log(10.0)
	logX2 := math.Log(100.0)
	logY1 := math.Log(100.0)
	logY2 := math.Log(1000.0)
	slope := (logY2 - logY1) / (logX2 - logX1)
	expected := math.Exp(logY2 + slope*(math.Log(1000)-logX2))
	assert.InDelta(t, expected, result, 1.0)
}

func TestInterpolate1D_TwoPoints(t *testing.T) {
	pts := makePoints([][2]float64{{10, 100}, {100, 1000}})
	assert.InDelta(t, 100.0, Interpolate1D(pts, 10), 1e-9)
	assert.InDelta(t, 1000.0, Interpolate1D(pts, 100), 1e-9)
	midpoint := Interpolate1D(pts, 31)
	assert.Greater(t, midpoint, 100.0)
	assert.Less(t, midpoint, 1000.0)
}

func TestInterpolate1D_SinglePoint(t *testing.T) {
	pts := makePoints([][2]float64{{100, 500}})
	assert.InDelta(t, 500.0, Interpolate1D(pts, 10), 1e-9)
	assert.InDelta(t, 500.0, Interpolate1D(pts, 100), 1e-9)
	assert.InDelta(t, 500.0, Interpolate1D(pts, 1000), 1e-9)
}

func TestInterpolate1D_ManyPoints(t *testing.T) {
	pts := makePowerLawPoints(1.0, 2.0, []int64{10, 20, 30, 40, 50, 100})
	for _, queryN := range []int64{15, 25, 35, 75} {
		expected := math.Pow(float64(queryN), 2.0)
		result := Interpolate1D(pts, queryN)
		relErr := math.Abs(result-expected) / expected
		assert.Less(t, relErr, 0.05)
	}
}

func TestInterpolate1D_BoundaryFirstAndLast(t *testing.T) {
	pts := makePoints([][2]float64{{10, 100}, {20, 200}, {30, 300}})
	assert.InDelta(t, 100.0, Interpolate1D(pts, 10), 1e-9)
	assert.InDelta(t, 300.0, Interpolate1D(pts, 30), 1e-9)
}

func TestInterpolate1D_NonPowerLaw(t *testing.T) {
	pts := makePoints([][2]float64{{10, 100}, {20, 250}, {30, 450}, {40, 700}})
	result := Interpolate1D(pts, 25)
	assert.Greater(t, result, 250.0)
	assert.Less(t, result, 450.0)
}

// --- Part 4b: Interpolate1DByDim tests ---

func TestInterpolate1DByDim_DimIdx0_MatchesInterpolate1D(t *testing.T) {
	pts := makePoints([][2]float64{{10, 100}, {100, 1000}})
	result1D := Interpolate1D(pts, 31)
	resultByDim := Interpolate1DByDim(pts, 0, 31)
	assert.InDelta(t, result1D, resultByDim, 1e-9)
}

func TestInterpolate1DByDim_DimIdx1(t *testing.T) {
	pts := makePointsMultiDim(
		[][]int64{{10, 100}, {10, 200}, {10, 300}},
		[]float64{1000, 2000, 3000},
	)
	result := Interpolate1DByDim(pts, 1, 150)
	logQ := math.Log(150.0)
	logX1 := math.Log(100.0)
	logX2 := math.Log(200.0)
	t_val := (logQ - logX1) / (logX2 - logX1)
	expected := math.Exp(math.Log(1000) + t_val*(math.Log(2000)-math.Log(1000)))
	assert.InDelta(t, expected, result, 1.0)
}

func TestInterpolate1DByDim_SinglePoint(t *testing.T) {
	pts := makePointsMultiDim([][]int64{{10, 20, 30}}, []float64{500})
	assert.InDelta(t, 500.0, Interpolate1DByDim(pts, 1, 100), 1e-9)
}

func TestInterpolate1DByDim_SortsByDimIdx(t *testing.T) {
	pts := makePointsMultiDim(
		[][]int64{{10, 300}, {10, 100}, {10, 200}},
		[]float64{3000, 1000, 2000},
	)
	result := Interpolate1DByDim(pts, 1, 150)
	logQ := math.Log(150.0)
	logX1 := math.Log(100.0)
	logX2 := math.Log(200.0)
	t_val := (logQ - logX1) / (logX2 - logX1)
	expected := math.Exp(math.Log(1000) + t_val*(math.Log(2000)-math.Log(1000)))
	assert.InDelta(t, expected, result, 1.0)
}

// --- Part 4c: InterpolateMulMat tests ---

func TestInterpolateMulMat_ExactMKMatch(t *testing.T) {
	curves := makeMulMatCurves(
		[][2]int64{{128, 4096}, {256, 4096}},
		[]int64{1, 32},
		func(m, k, n int64) float64 { return float64(m*k*n) / 1e6 },
	)
	result := InterpolateMulMat(curves, 128, 4096, 16)
	expected := Interpolate1DByDim(curves[0].Points, 0, 16)
	assert.InDelta(t, expected, result, 1e-9)
}

func TestInterpolateMulMat_BetweenMKPairs(t *testing.T) {
	curves := makeMulMatCurves(
		[][2]int64{{100, 1000}, {200, 2000}},
		[]int64{10, 100},
		func(m, k, n int64) float64 { return float64(m*k*n) / 1e6 },
	)
	result := InterpolateMulMat(curves, 150, 1500, 31)
	lat1 := Interpolate1DByDim(curves[0].Points, 0, 31)
	lat2 := Interpolate1DByDim(curves[1].Points, 0, 31)
	dM := math.Log(150.0) - math.Log(100.0)
	dK := math.Log(1500.0) - math.Log(1000.0)
	dist1 := math.Sqrt(dM*dM + dK*dK)
	dM2 := math.Log(150.0) - math.Log(200.0)
	dK2 := math.Log(1500.0) - math.Log(2000.0)
	dist2 := math.Sqrt(dM2*dM2 + dK2*dK2)
	w1 := 1.0 / dist1
	w2 := 1.0 / dist2
	expected := (lat1*w1 + lat2*w2) / (w1 + w2)
	assert.InDelta(t, expected, result, 1.0)
}

func TestInterpolateMulMat_SingleCurve(t *testing.T) {
	curves := makeMulMatCurves(
		[][2]int64{{128, 4096}},
		[]int64{1, 32},
		func(m, k, n int64) float64 { return float64(m*k*n) / 1e6 },
	)
	result := InterpolateMulMat(curves, 256, 8192, 16)
	expected := Interpolate1DByDim(curves[0].Points, 0, 16)
	assert.InDelta(t, expected, result, 1e-9)
}

func TestInterpolateMulMat_ManyCurves(t *testing.T) {
	curves := makeMulMatCurves(
		[][2]int64{{100, 1000}, {200, 2000}, {300, 3000}, {400, 4000}},
		[]int64{10, 100},
		func(m, k, n int64) float64 { return float64(m*k*n) / 1e6 },
	)
	result := InterpolateMulMat(curves, 250, 2500, 50)
	assert.Greater(t, result, 0.0)
}

func TestInterpolateMulMat_AsymmetricMK(t *testing.T) {
	curves := makeMulMatCurves(
		[][2]int64{{100, 1000}, {200, 1000}},
		[]int64{10, 100},
		func(m, k, n int64) float64 { return float64(m*k*n) / 1e6 },
	)
	result := InterpolateMulMat(curves, 150, 1000, 50)
	lat1 := Interpolate1DByDim(curves[0].Points, 0, 50)
	lat2 := Interpolate1DByDim(curves[1].Points, 0, 50)
	assert.Greater(t, result, 0.0)
	assert.Greater(t, result, math.Min(lat1, lat2)*0.9)
	assert.Less(t, result, math.Max(lat1, lat2)*1.1)
}

func TestInterpolateMulMat_InverseDistanceWeighting(t *testing.T) {
	curves := makeMulMatCurves(
		[][2]int64{{100, 1000}, {1000, 10000}},
		[]int64{10, 100},
		func(m, k, n int64) float64 { return float64(m*k*n) / 1e6 },
	)
	result := InterpolateMulMat(curves, 150, 1500, 50)
	lat1 := Interpolate1DByDim(curves[0].Points, 0, 50)
	lat2 := Interpolate1DByDim(curves[1].Points, 0, 50)
	assert.Greater(t, result, 0.0)
	// Result should be weighted between the two curves
	assert.Greater(t, result, math.Min(lat1, lat2))
	assert.Less(t, result, math.Max(lat1, lat2))
}

// --- Part 4d: InterpolateFlashAttn tests ---

func TestInterpolateFlashAttn_DecodeRegime(t *testing.T) {
	curve := makeFlashAttnCurve(
		[]int64{128, 256, 512},
		[]int64{128, 256, 512},
		func(seqQ, seqKV int64) float64 {
			if seqQ == 1 {
				return 100.0 * float64(seqKV)
			}
			return 10.0 * float64(seqQ*seqKV)
		},
	)
	result := InterpolateFlashAttn(curve, 1, 384)
	expected := Interpolate1DByDim(curve.Points[:3], 1, 384)
	assert.InDelta(t, expected, result, 1.0)
}

func TestInterpolateFlashAttn_PrefillRegime(t *testing.T) {
	curve := makeFlashAttnCurve(
		[]int64{128, 256, 512},
		[]int64{128, 256, 512},
		func(seqQ, seqKV int64) float64 {
			if seqQ == 1 {
				return 100.0 * float64(seqKV)
			}
			return 10.0 * float64(seqQ*seqKV)
		},
	)
	result := InterpolateFlashAttn(curve, 384, 384)
	var prefillPts []LatencyPoint
	for _, pt := range curve.Points {
		if pt.Shape[0] == pt.Shape[1] {
			prefillPts = append(prefillPts, pt)
		}
	}
	expected := Interpolate1DByDim(prefillPts, 1, 384)
	assert.InDelta(t, expected, result, 1.0)
}

func TestInterpolateFlashAttn_BetweenRegimes(t *testing.T) {
	curve := makeFlashAttnCurve(
		[]int64{128, 256, 512},
		[]int64{128, 256, 512},
		func(seqQ, seqKV int64) float64 {
			if seqQ == 1 {
				return 100.0 * float64(seqKV)
			}
			return 10.0 * float64(seqQ*seqKV)
		},
	)
	decodeLat := InterpolateFlashAttn(curve, 1, 384)
	prefillLat := InterpolateFlashAttn(curve, 384, 384)
	result := InterpolateFlashAttn(curve, 64, 384)
	assert.Greater(t, result, math.Min(decodeLat, prefillLat))
	assert.Less(t, result, math.Max(decodeLat, prefillLat))
}

func TestInterpolateFlashAttn_SeqKV1_Guard(t *testing.T) {
	curve := makeFlashAttnCurve(
		[]int64{128, 256},
		[]int64{128, 256},
		func(seqQ, seqKV int64) float64 { return float64(seqQ * seqKV) },
	)
	result := InterpolateFlashAttn(curve, 100, 1)
	assert.False(t, math.IsNaN(result), "result should not be NaN")
	assert.Greater(t, result, 0.0)
}

func TestInterpolateFlashAttn_SeqQ1_SeqKV1(t *testing.T) {
	curve := makeFlashAttnCurve(
		[]int64{1, 128, 256},
		[]int64{128, 256},
		func(seqQ, seqKV int64) float64 {
			if seqQ == 1 && seqKV == 1 {
				return 50.0
			}
			return float64(seqQ * seqKV)
		},
	)
	result := InterpolateFlashAttn(curve, 1, 1)
	assert.InDelta(t, 50.0, result, 1e-9)
}

func TestInterpolateFlashAttn_DecodeExtrapolation(t *testing.T) {
	curve := makeFlashAttnCurve(
		[]int64{128, 256},
		[]int64{128, 256},
		func(seqQ, seqKV int64) float64 {
			if seqQ == 1 {
				return 100.0 * float64(seqKV)
			}
			return 10.0 * float64(seqQ*seqKV)
		},
	)
	result := InterpolateFlashAttn(curve, 1, 512)
	assert.Greater(t, result, 100.0*256)
}

func TestInterpolateFlashAttn_PrefillHighSeqQ(t *testing.T) {
	curve := makeFlashAttnCurve(
		[]int64{128, 256},
		[]int64{128, 256},
		func(seqQ, seqKV int64) float64 {
			if seqQ == 1 {
				return 100.0 * float64(seqKV)
			}
			return 10.0 * float64(seqQ*seqKV)
		},
	)
	// When seqQ >> seqKV, we're in pure prefill regime (t=1)
	// Result is based on interpolating seqKV in the prefill curve
	result := InterpolateFlashAttn(curve, 1024, 384)
	prefillBase := InterpolateFlashAttn(curve, 256, 256)
	// At seqKV=384 (extrapolated), latency should be higher than at seqKV=256
	assert.Greater(t, result, prefillBase)
}

func TestInterpolateFlashAttn_BlendMonotonicity(t *testing.T) {
	curve := makeFlashAttnCurve(
		[]int64{128, 256, 512},
		[]int64{128, 256, 512},
		func(seqQ, seqKV int64) float64 {
			if seqQ == 1 {
				return 100.0 * float64(seqKV)
			}
			return 10.0 * float64(seqQ*seqKV)
		},
	)
	seqKV := int64(384)
	var prevResult float64
	for _, seqQ := range []int64{1, 32, 64, 128, 256, 384} {
		result := InterpolateFlashAttn(curve, seqQ, seqKV)
		if seqQ > 1 {
			assert.GreaterOrEqual(t, result, prevResult*0.9, "seqQ=%d should not decrease too much", seqQ)
		}
		prevResult = result
	}
}
