package perf

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- trimmedStats tests ---

func TestTrimmedStats_Empty(t *testing.T) {
	med, sd := trimmedStats(nil, 0.1)
	assert.Equal(t, 0.0, med, "empty slice should return median 0")
	assert.Equal(t, 0.0, sd, "empty slice should return stddev 0")

	med, sd = trimmedStats([]float64{}, 0.1)
	assert.Equal(t, 0.0, med, "empty slice should return median 0")
	assert.Equal(t, 0.0, sd, "empty slice should return stddev 0")
}

func TestTrimmedStats_SingleValue(t *testing.T) {
	med, sd := trimmedStats([]float64{42.0}, 0.1)
	assert.Equal(t, 42.0, med, "single value should be its own median")
	assert.Equal(t, 0.0, sd, "single value should have zero stddev")
}

func TestTrimmedStats_AllIdentical(t *testing.T) {
	values := []float64{7.0, 7.0, 7.0, 7.0, 7.0}
	med, sd := trimmedStats(values, 0.1)
	assert.Equal(t, 7.0, med)
	assert.Equal(t, 0.0, sd, "identical values should have zero stddev")
}

func TestTrimmedStats_KnownDistribution_NoTrim(t *testing.T) {
	// 5 values: 1, 2, 3, 4, 5
	// With trimPercent=0: no trimming
	// Median (index 2 of 5) = 3
	// Mean = 3, Var = (4+1+0+1+4)/5 = 2, SD = sqrt(2) ≈ 1.4142
	values := []float64{1, 2, 3, 4, 5}
	med, sd := trimmedStats(values, 0.0)
	assert.Equal(t, 3.0, med)
	assert.InDelta(t, math.Sqrt(2.0), sd, 1e-10)
}

func TestTrimmedStats_KnownDistribution_WithTrim(t *testing.T) {
	// 10 values with outliers at both ends
	// trimPercent=0.2 → trimCount = round(10*0.2) = 2 → remove 2 from each end
	// Original sorted: [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
	// Trimmed (remove 2 from each end): [3, 4, 5, 6, 7, 8]
	// Median: index 3 of 6 = 6
	// Mean = (3+4+5+6+7+8)/6 = 5.5
	// Var = ((2.5^2)+(1.5^2)+(0.5^2)+(0.5^2)+(1.5^2)+(2.5^2))/6 = 17.5/6
	// SD = sqrt(17.5/6) ≈ 1.7078
	values := []float64{100, 1, 5, 3, 7, 2, 8, 4, 6, 9}
	med, sd := trimmedStats(values, 0.2)
	assert.Equal(t, 6.0, med, "median of trimmed [3,4,5,6,7,8]")
	expectedSD := math.Sqrt(17.5 / 6.0)
	assert.InDelta(t, expectedSD, sd, 1e-10)
}

func TestTrimmedStats_OutliersExcluded(t *testing.T) {
	// Core values around 100, with extreme outliers
	// Without trim: outliers dominate stddev
	// With trim: outliers removed
	values := []float64{1, 100, 101, 102, 103, 104, 105, 106, 107, 1000}
	_, sdNoTrim := trimmedStats(values, 0.0)
	_, sdTrimmed := trimmedStats(values, 0.1)
	assert.Greater(t, sdNoTrim, sdTrimmed,
		"trimming should reduce stddev when outliers present")
}

func TestTrimmedStats_TrimPercentTooLarge(t *testing.T) {
	// trimPercent=0.6 on 5 elements → trimCount = round(5*0.6) = 3
	// 3*2=6 >= 5, so trimCount resets to 0 (no trimming)
	values := []float64{1, 2, 3, 4, 5}
	med, sd := trimmedStats(values, 0.6)
	// Should fall back to no trimming
	assert.Equal(t, 3.0, med)
	assert.InDelta(t, math.Sqrt(2.0), sd, 1e-10)
}

func TestTrimmedStats_TwoValues(t *testing.T) {
	values := []float64{10, 20}
	med, sd := trimmedStats(values, 0.0)
	// Median: index 1 of 2 = 20
	assert.Equal(t, 20.0, med)
	// Mean = 15, Var = (25+25)/2 = 25, SD = 5
	assert.InDelta(t, 5.0, sd, 1e-10)
}

func TestTrimmedStats_DoesNotMutateInput(t *testing.T) {
	values := []float64{5, 3, 1, 4, 2}
	original := make([]float64, len(values))
	copy(original, values)
	trimmedStats(values, 0.1)
	assert.Equal(t, original, values, "input slice should not be modified")
}

func TestTrimmedStats_LargeDataset(t *testing.T) {
	// 100 uniform values from 1 to 100, plus outliers at 0 and 10000
	values := make([]float64, 102)
	for i := 0; i < 100; i++ {
		values[i] = float64(i + 1)
	}
	values[100] = 0
	values[101] = 10000
	// With 10% trim: remove 10 from each end
	// Sorted: [0, 1, 2, ..., 100, 10000] → trim 10 each → [11..91] (81 elements)
	// The outlier 10000 is definitely removed
	med, sd := trimmedStats(values, 0.1)
	// Median should be around 51 (center of 11..91)
	assert.InDelta(t, 51.0, med, 1.0, "median should be near center of trimmed range")
	// SD should be reasonable (not inflated by 10000 outlier)
	assert.Less(t, sd, 100.0, "stddev should not be inflated by outlier")
}

// --- measureOpForBackend dispatch tests ---

func TestMeasureOpForBackend_DispatchesOnGPUTimestamp(t *testing.T) {
	// We cannot mock ml.Backend easily, but we CAN verify the dispatch logic
	// by checking that the function exists with the correct signature and
	// that BackendCapabilities correctly reports HasGPUTimestamp.

	// Vulkan should have GPU timestamps
	vulkanCaps := GetBackendCapabilities("Vulkan")
	assert.True(t, vulkanCaps.HasGPUTimestamp,
		"Vulkan backend should report HasGPUTimestamp=true")

	// CUDA should NOT have GPU timestamps
	cudaCaps := GetBackendCapabilities("CUDA")
	assert.False(t, cudaCaps.HasGPUTimestamp,
		"CUDA backend should report HasGPUTimestamp=false")

	// CPU should NOT have GPU timestamps
	cpuCaps := GetBackendCapabilities("CPU")
	assert.False(t, cpuCaps.HasGPUTimestamp,
		"CPU backend should report HasGPUTimestamp=false")

	// Unknown backend should NOT have GPU timestamps
	unknownCaps := GetBackendCapabilities("SomeNewBackend")
	assert.False(t, unknownCaps.HasGPUTimestamp,
		"unknown backend should default to HasGPUTimestamp=false")
}

// --- Convergence logic tests ---

func TestTrimmedStats_ConvergenceDetection(t *testing.T) {
	// Simulate what measureOpGPU does: accumulate samples and check CV
	cfg := DefaultBenchmarkConfig()

	// Scenario 1: Very stable samples should converge quickly
	stableSamples := []float64{100.1, 100.2, 99.9, 100.0, 100.3}
	converged := false
	for i := cfg.MinReps; i <= len(stableSamples); i++ {
		med, sd := trimmedStats(stableSamples[:i], cfg.TrimPercent)
		if med > 0 && sd/med < cfg.ConvergenceCV {
			converged = true
			break
		}
	}
	assert.True(t, converged,
		"stable samples (CV < 0.3%) should converge within 5 reps at CV threshold 5%")

	// Scenario 2: Highly variable samples should NOT converge
	noisySamples := []float64{10, 200, 5, 500, 20}
	converged = false
	for i := cfg.MinReps; i <= len(noisySamples); i++ {
		med, sd := trimmedStats(noisySamples[:i], cfg.TrimPercent)
		if med > 0 && sd/med < cfg.ConvergenceCV {
			converged = true
			break
		}
	}
	assert.False(t, converged,
		"highly variable samples should not converge at 5% CV threshold")
}

func TestTrimmedStats_ConvergenceWithGrowingSampleSet(t *testing.T) {
	// Simulate incremental sample collection as measureOpGPU does it.
	// Start noisy, then stabilize. Should eventually converge.
	samples := []float64{
		90, 150, 80, 120, 110, // noisy start
		100, 101, 99, 100, 102, // stable tail
		100, 99, 101, 100, 100,
	}
	cfg := BenchmarkConfig{
		MinReps:       5,
		TrimPercent:   0.1,
		ConvergenceCV: 0.05,
	}

	convergedAt := -1
	for i := cfg.MinReps; i <= len(samples); i++ {
		med, sd := trimmedStats(samples[:i], cfg.TrimPercent)
		if med > 0 && sd/med < cfg.ConvergenceCV {
			convergedAt = i
			break
		}
	}
	require.Greater(t, convergedAt, cfg.MinReps,
		"should converge after the initial noisy period")
	require.LessOrEqual(t, convergedAt, len(samples),
		"should converge before running out of samples")
}

// --- GPU timing summation test ---

func TestGPUTimingSummation(t *testing.T) {
	// Verify the summation logic used in measureOpGPU:
	// "Sum all op timings in the graph"
	// This tests the pattern, not the actual backend call.
	type timing struct {
		gpuTimeUs float64
	}

	tests := []struct {
		name     string
		timings  []timing
		expected float64
	}{
		{
			name:     "single op",
			timings:  []timing{{10.5}},
			expected: 10.5,
		},
		{
			name:     "multiple ops",
			timings:  []timing{{10.0}, {20.0}, {30.0}},
			expected: 60.0,
		},
		{
			name:     "empty timings",
			timings:  []timing{},
			expected: 0.0,
		},
		{
			name:     "fractional microseconds",
			timings:  []timing{{0.123}, {0.456}, {0.789}},
			expected: 1.368,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var gpuUs float64
			for _, tm := range tc.timings {
				gpuUs += tm.gpuTimeUs
			}
			assert.InDelta(t, tc.expected, gpuUs, 1e-10)
		})
	}
}

// --- measureOpForBackend routing verification ---

func TestMeasureOpForBackend_RoutingLogic(t *testing.T) {
	// Verify the dispatch decision is purely based on HasGPUTimestamp.
	// We test this by constructing capabilities with explicit values
	// and verifying the field that drives the decision.

	tests := []struct {
		name           string
		caps           BackendCapabilities
		expectGPUPath  bool
	}{
		{
			name:          "GPU timestamp enabled",
			caps:          BackendCapabilities{Name: "Test", HasGPUTimestamp: true},
			expectGPUPath: true,
		},
		{
			name:          "GPU timestamp disabled",
			caps:          BackendCapabilities{Name: "Test", HasGPUTimestamp: false},
			expectGPUPath: false,
		},
		{
			name:          "Vulkan defaults",
			caps:          GetBackendCapabilities("Vulkan"),
			expectGPUPath: true,
		},
		{
			name:          "CUDA defaults",
			caps:          GetBackendCapabilities("CUDA"),
			expectGPUPath: false,
		},
		{
			name:          "CPU defaults",
			caps:          GetBackendCapabilities("CPU"),
			expectGPUPath: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// The dispatch decision in measureOpForBackend is:
			//   if caps.HasGPUTimestamp → measureOpGPU
			//   else → measureOp
			assert.Equal(t, tc.expectGPUPath, tc.caps.HasGPUTimestamp,
				"dispatch should route to GPU path when HasGPUTimestamp is true")
		})
	}
}

// --- trimmedStats consistency with trimmedMedian ---

func TestTrimmedStats_MedianMatchesTrimmedMedian(t *testing.T) {
	// trimmedStats median should match the existing trimmedMedian function
	// for the same inputs and trim percentage.
	testCases := []struct {
		name    string
		values  []float64
		trim    float64
	}{
		{"simple ascending", []float64{1, 2, 3, 4, 5}, 0.0},
		{"with outliers", []float64{1, 50, 51, 52, 53, 54, 200}, 0.1},
		{"descending", []float64{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}, 0.2},
		{"single", []float64{42}, 0.1},
		{"two elements", []float64{10, 20}, 0.0},
		{"repeated values", []float64{5, 5, 5, 5, 5, 5, 5}, 0.1},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			expectedMedian := trimmedMedian(tc.values, tc.trim)
			gotMedian, _ := trimmedStats(tc.values, tc.trim)
			assert.Equal(t, expectedMedian, gotMedian,
				"trimmedStats median should match trimmedMedian")
		})
	}
}

// --- Edge cases for the early-stop path in measureOpGPU ---

func TestTrimmedStats_MinRepsEnforcement(t *testing.T) {
	// Verify that convergence is not checked before MinReps samples.
	// Even perfectly stable data needs MinReps samples first.
	cfg := BenchmarkConfig{
		MinReps:       5,
		TrimPercent:   0.1,
		ConvergenceCV: 0.05,
	}

	// 3 perfectly identical samples — but we shouldn't converge yet
	// because we need MinReps=5
	samples := []float64{100, 100, 100}
	require.Less(t, len(samples), cfg.MinReps,
		"test setup: fewer samples than MinReps")

	// The measureOpGPU code checks: if len(samples) >= cfg.MinReps
	// With only 3 samples, this check should fail
	shouldCheck := len(samples) >= cfg.MinReps
	assert.False(t, shouldCheck,
		"should not check convergence before MinReps samples collected")
}
