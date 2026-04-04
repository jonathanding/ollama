package perf

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHWCharResult_BalancePoint(t *testing.T) {
	// Balance point = TOPS / BW
	result := HWCharResult{
		PeakTOPS:     map[string]float64{"f16": 330e12},
		PeakBW:       1008e9,
		BalancePoint: map[string]float64{},
	}
	result.BalancePoint["f16"] = result.PeakTOPS["f16"] / result.PeakBW
	assert.InDelta(t, 327.38, result.BalancePoint["f16"], 1.0)
}

func TestHWCharResult_MultiDtype(t *testing.T) {
	result := HWCharResult{
		PeakTOPS: map[string]float64{
			"f16": 330e12,
			"f32": 82.6e12,
		},
		PeakBW:       1008e9,
		BalancePoint: make(map[string]float64),
	}
	for dtype, tops := range result.PeakTOPS {
		result.BalancePoint[dtype] = tops / result.PeakBW
	}
	assert.InDelta(t, 327.38, result.BalancePoint["f16"], 1.0)
	assert.InDelta(t, 81.94, result.BalancePoint["f32"], 0.1)
}

func TestHWCharResult_ZeroBandwidth(t *testing.T) {
	// Edge case: zero bandwidth should not cause division by zero
	result := HWCharResult{
		PeakTOPS:     map[string]float64{"f16": 330e12},
		PeakBW:       0,
		BalancePoint: make(map[string]float64),
	}
	// CharacterizeHardware guards with `if result.PeakBW > 0`
	// so BalancePoint should remain zero-value (not set)
	assert.Empty(t, result.BalancePoint)
}

func TestParseDType_HWChar(t *testing.T) {
	tests := []struct {
		input string
		valid bool
	}{
		{"f32", true},
		{"f16", true},
		{"q4_0", true},
		{"q8_0", true},
		{"invalid", false},
		{"", false},
		{"F16", false}, // case-sensitive
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			_, ok := parseDType(tt.input)
			assert.Equal(t, tt.valid, ok)
		})
	}
}

func TestBenchPeakTOPS_ConfigPropagation(t *testing.T) {
	cfg := DefaultBenchmarkConfig()
	assert.Equal(t, 0.05, cfg.ConvergenceCV, "convergence CV should propagate to hwchar")
	assert.Equal(t, 5, cfg.MinReps, "MinReps should propagate to hwchar")
}

func TestTrimmedMedian_SmallSample(t *testing.T) {
	// 5 samples with 10% trim: trimCount = round(0.5) = 1
	values := []float64{100, 102, 101, 103, 99}
	median := trimmedMedian(values, 0.1)
	assert.InDelta(t, 101.0, median, 1.0)

	// 7 samples with 10% trim: trimCount = round(0.7) = 1
	values = []float64{100, 102, 101, 103, 99, 104, 98}
	median = trimmedMedian(values, 0.1)
	assert.InDelta(t, 101.0, median, 2.0)
}

// TestConvergentMeasure_ConvergesEarly verifies that stable measurements stop before maxReps.
func TestConvergentMeasure_ConvergesEarly(t *testing.T) {
	callCount := 0
	compute := func() float64 {
		callCount++
		noise := float64(callCount%3-1) * 10.0
		return 1000.0 + noise
	}
	cfg := BenchmarkConfig{
		MeasureReps:   50,
		TrimPercent:   0.1,
		ConvergenceCV: 0.05,
		MinReps:       5,
	}
	median, stddev, reps := convergentMeasure(compute, cfg)

	assert.Greater(t, median, 900.0)
	assert.Less(t, median, 1100.0)
	assert.Less(t, stddev, 50.0, "stddev should be small for stable measurements")
	assert.GreaterOrEqual(t, reps, 5, "must run at least MinReps")
	assert.Less(t, reps, 15, "stable measurement should converge well before 50 reps")
}

// TestConvergentMeasure_NoisyDoesNotConverge verifies that noisy measurements run to maxReps.
func TestConvergentMeasure_NoisyDoesNotConverge(t *testing.T) {
	callCount := 0
	compute := func() float64 {
		callCount++
		if callCount%2 == 0 {
			return 500.0
		}
		return 1500.0
	}
	cfg := BenchmarkConfig{
		MeasureReps:   20,
		TrimPercent:   0.1,
		ConvergenceCV: 0.05,
		MinReps:       5,
	}
	_, _, reps := convergentMeasure(compute, cfg)

	assert.Equal(t, 20, reps, "noisy measurement should run all maxReps")
}

// TestConvergentMeasure_TieredMaxReps verifies that slow ops get reduced maxReps.
func TestConvergentMeasure_TieredMaxReps(t *testing.T) {
	callCount := 0
	compute := func() float64 {
		callCount++
		return 2e6 + float64(callCount%5)*4e5
	}
	cfg := BenchmarkConfig{
		MeasureReps:   50,
		TrimPercent:   0.1,
		ConvergenceCV: 0.05,
		MinReps:       5,
	}
	_, _, reps := convergentMeasure(compute, cfg)

	assert.LessOrEqual(t, reps, 10, "ops >1s should cap at 10 reps")
}

// TestConvergentMeasure_VerySlowOp verifies >5s ops cap at MinReps.
func TestConvergentMeasure_VerySlowOp(t *testing.T) {
	callCount := 0
	compute := func() float64 {
		callCount++
		return 6e6 + float64(callCount%3)*1e6
	}
	cfg := BenchmarkConfig{
		MeasureReps:   50,
		TrimPercent:   0.1,
		ConvergenceCV: 0.05,
		MinReps:       5,
	}
	_, _, reps := convergentMeasure(compute, cfg)

	assert.Equal(t, 5, reps, "ops >5s should cap at MinReps")
}

// TestConvergentMeasure_MinRepsRespected verifies we never stop before MinReps.
func TestConvergentMeasure_MinRepsRespected(t *testing.T) {
	compute := func() float64 {
		return 1000.0
	}
	cfg := BenchmarkConfig{
		MeasureReps:   50,
		TrimPercent:   0.1,
		ConvergenceCV: 0.05,
		MinReps:       8,
	}
	_, _, reps := convergentMeasure(compute, cfg)

	assert.GreaterOrEqual(t, reps, 8, "must run at least MinReps even if stable")
}

// TestConvergentMeasure_TrimmedCV verifies CV is computed on trimmed data, not raw.
func TestConvergentMeasure_TrimmedCV(t *testing.T) {
	callCount := 0
	compute := func() float64 {
		callCount++
		if callCount%7 == 0 {
			return 50000.0 // outlier
		}
		return 1000.0 + float64(callCount%3)*5.0
	}
	cfg := BenchmarkConfig{
		MeasureReps:   30,
		TrimPercent:   0.1,
		ConvergenceCV: 0.05,
		MinReps:       5,
	}
	median, _, reps := convergentMeasure(compute, cfg)

	assert.Greater(t, median, 900.0)
	assert.Less(t, median, 1100.0)
	assert.Less(t, reps, 30, "should converge after trimming removes outliers")
}

// Integration tests for benchPeakTOPS and benchPeakBandwidth require a real
// GGML backend. They are in integration_test.go (Task 14).
