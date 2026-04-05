package perf

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/ml"
	_ "github.com/ollama/ollama/ml/backend" // register backends
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// skipIfNoBackend skips the test if no GGML backend is available.
func skipIfNoBackend(t *testing.T) ml.Backend {
	t.Helper()
	if os.Getenv("DAOP_INTEGRATION") == "" {
		t.Skip("set DAOP_INTEGRATION=1 to run GGML integration tests")
	}
	// Try to create a backend
	b, err := ml.NewBackendForBench(ml.BackendParams{AllocMemory: true})
	if err != nil {
		t.Skipf("no GGML backend available: %v", err)
	}
	return b
}

func TestGGML_MeasureOp_SILU(t *testing.T) {
	backend := skipIfNoBackend(t)
	defer backend.Close()

	cfg := BenchmarkConfig{WarmupReps: 2, MeasureReps: 10, TrimPercent: 0.1}
	pt := measureOp(backend, "SILU", []int64{65536}, "f32", cfg, -1)

	assert.Greater(t, pt.LatencyUs, 0.0, "SILU should have measurable latency")
	assert.Greater(t, pt.Reps, 0)
	t.Logf("SILU N=65536: %.2f us (stddev: %.2f us)", pt.LatencyUs, pt.StddevUs)
}

func TestGGML_MeasureOp_SILU_Monotonic(t *testing.T) {
	backend := skipIfNoBackend(t)
	defer backend.Close()

	cfg := BenchmarkConfig{WarmupReps: 2, MeasureReps: 20, TrimPercent: 0.1}
	sizes := []int64{1024, 16384, 262144, 4194304}
	var prev float64
	for _, N := range sizes {
		pt := measureOp(backend, "SILU", []int64{N}, "f32", cfg, -1)
		t.Logf("SILU N=%d: %.2f us", N, pt.LatencyUs)
		// Sanity check: ensure we got a valid measurement
		if pt.LatencyUs <= 0 {
			t.Logf("WARNING: got zero/negative latency for N=%d, skipping monotonicity check", N)
			continue
		}
		if prev > 0 {
			assert.Greater(t, pt.LatencyUs, prev*0.5,
				"SILU N=%d should not be much faster than N=%d", N, N/16)
		}
		prev = pt.LatencyUs
	}
}

func TestGGML_MeasureOp_MulMat(t *testing.T) {
	backend := skipIfNoBackend(t)
	defer backend.Close()

	cfg := BenchmarkConfig{WarmupReps: 2, MeasureReps: 10, TrimPercent: 0.1}
	pt := measureOp(backend, "MUL_MAT", []int64{4096, 4096, 1}, "f16", cfg, -1)

	assert.Greater(t, pt.LatencyUs, 0.0)
	t.Logf("MUL_MAT [4096,4096,1] f16: %.2f us", pt.LatencyUs)
}

func TestGGML_AdaptiveSample_SILU(t *testing.T) {
	backend := skipIfNoBackend(t)
	defer backend.Close()

	cfg := BenchmarkConfig{
		WarmupReps: 2, MeasureReps: 20, TrimPercent: 0.1,
		ErrorThreshold: 0.05, MaxPointsPerOp: 15,
	}

	measure := func(shape []int64) LatencyPoint {
		return measureOp(backend, "SILU", shape, "f32", cfg, -1)
	}

	points := AdaptiveSample1D(measure, 1024, 16*1024*1024, 6, cfg)

	assert.GreaterOrEqual(t, len(points), 6)
	assert.LessOrEqual(t, len(points), 15)

	// Points should be sorted
	for i := 1; i < len(points); i++ {
		assert.Greater(t, points[i].Shape[0], points[i-1].Shape[0])
	}

	t.Logf("Adaptive SILU: %d points", len(points))
	for _, pt := range points {
		t.Logf("  N=%d: %.2f us", pt.Shape[0], pt.LatencyUs)
	}
}

func TestGGML_HardwareCharacterization(t *testing.T) {
	backend := skipIfNoBackend(t)
	defer backend.Close()

	cfg := BenchmarkConfig{WarmupReps: 2, MeasureReps: 20, TrimPercent: 0.1}
	result, err := CharacterizeHardware(backend, cfg)
	require.NoError(t, err)

	assert.Greater(t, result.PeakBW, 0.0, "should measure non-zero bandwidth")
	assert.NotEmpty(t, result.PeakTOPS, "should measure at least one dtype TOPS")

	t.Logf("Peak BW: %.1f GB/s", result.PeakBW/1e9)
	for dtype, tops := range result.PeakTOPS {
		t.Logf("Peak TOPS (%s): %.1f TOPS", dtype, tops/1e12)
		bp := result.BalancePoint[dtype]
		t.Logf("Balance point (%s): %.1f FLOP/byte", dtype, bp)
	}
}

func TestGGML_FullPipeline_SILU(t *testing.T) {
	backend := skipIfNoBackend(t)
	defer backend.Close()

	cfg := BenchmarkConfig{
		WarmupReps: 2, MeasureReps: 20, TrimPercent: 0.1,
		ErrorThreshold: 0.10, MaxPointsPerOp: 10,
	}

	// Run benchmark for SILU only
	profile, err := RunBenchmark(backend, []string{"SILU"}, []string{"f32"}, cfg)
	require.NoError(t, err)

	assert.Equal(t, 2, profile.Version)
	assert.NotEmpty(t, profile.Operators)

	// Find SILU curve
	var siluCurve *OperatorCurve
	for i := range profile.Operators {
		if profile.Operators[i].Op == "SILU" {
			siluCurve = &profile.Operators[i]
			break
		}
	}
	require.NotNil(t, siluCurve)
	assert.Equal(t, []string{"N"}, siluCurve.Dimensions)
	assert.GreaterOrEqual(t, len(siluCurve.Points), 6)

	// Test interpolation on the real curve
	lat := Interpolate1D(siluCurve.Points, 100000)
	assert.Greater(t, lat, 0.0)

	// Save and reload
	dir := t.TempDir()
	path := filepath.Join(dir, "profile.json")
	require.NoError(t, WriteProfile(path, profile))
	loaded, err := LoadProfile(path)
	require.NoError(t, err)
	assert.Equal(t, len(profile.Operators), len(loaded.Operators))

	// Generate HTML viewer
	htmlPath := filepath.Join(dir, "viewer.html")
	require.NoError(t, GenerateHTMLViewer(profile, htmlPath))
	info, err := os.Stat(htmlPath)
	require.NoError(t, err)
	assert.Greater(t, info.Size(), int64(1000))

	t.Logf("Full SILU pipeline: %d curves, %d total points",
		len(profile.Operators),
		func() int {
			n := 0
			for _, c := range profile.Operators {
				n += len(c.Points)
			}
			return n
		}())
}
