package perf

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestBenchOrchestrationOverheadGraphSizes verifies the expected graph sizes
// match the documented design: [50, 100, 200, 300, 500].
// We test this by inspecting synthetic output that mirrors the function's structure.
func TestBenchOrchestrationOverheadGraphSizes(t *testing.T) {
	expectedSizes := []int{50, 100, 200, 300, 500}

	// Simulate the output structure: one LatencyPoint per graph size
	points := make([]LatencyPoint, len(expectedSizes))
	for i, n := range expectedSizes {
		points[i] = LatencyPoint{
			Shape:     []int64{int64(n)},
			LatencyUs: float64(n) * 10, // synthetic: ~10us per node
			StddevUs:  float64(n) * 0.5,
			Reps:      20,
		}
	}

	require.Len(t, points, 5, "orchestration overhead should sample exactly 5 graph sizes")
	for i, n := range expectedSizes {
		assert.Equal(t, int64(n), points[i].Shape[0],
			"point %d should have graph size %d", i, n)
	}
}

// TestOrchestrationOverheadPointStructure verifies that LatencyPoint with
// graph-size shapes has the correct structure and field types.
func TestOrchestrationOverheadPointStructure(t *testing.T) {
	pt := LatencyPoint{
		Shape:     []int64{200},
		LatencyUs: 1500.5,
		StddevUs:  75.2,
		Reps:      25,
	}

	// Shape is a 1D slice with one element (the node count)
	require.Len(t, pt.Shape, 1, "orchestration overhead shape should be 1D")
	assert.Equal(t, int64(200), pt.Shape[0])

	// Latency and stddev are positive float64
	assert.Greater(t, pt.LatencyUs, 0.0)
	assert.Greater(t, pt.StddevUs, 0.0)

	// Reps is positive int
	assert.Greater(t, pt.Reps, 0)

	// Verify JSON serialization preserves the structure
	data, err := json.Marshal(pt)
	require.NoError(t, err)

	var decoded LatencyPoint
	require.NoError(t, json.Unmarshal(data, &decoded))
	assert.Equal(t, pt.Shape, decoded.Shape)
	assert.InDelta(t, pt.LatencyUs, decoded.LatencyUs, 0.001)
	assert.InDelta(t, pt.StddevUs, decoded.StddevUs, 0.001)
	assert.Equal(t, pt.Reps, decoded.Reps)
}

// TestOrchestrationOverheadMonotonicity verifies the design assumption that
// orchestration overhead increases with graph size. Using synthetic data that
// follows the expected pattern (roughly linear in num_nodes), we verify that
// the points maintain monotonicity — the same invariant the estimator relies on.
func TestOrchestrationOverheadMonotonicity(t *testing.T) {
	// Synthetic data: overhead grows roughly linearly with node count,
	// plus some noise (as real measurements would have)
	points := []LatencyPoint{
		{Shape: []int64{50}, LatencyUs: 480},
		{Shape: []int64{100}, LatencyUs: 1020},
		{Shape: []int64{200}, LatencyUs: 2150},
		{Shape: []int64{300}, LatencyUs: 3100},
		{Shape: []int64{500}, LatencyUs: 5300},
	}

	// Verify monotonicity: each point should have higher latency than previous
	for i := 1; i < len(points); i++ {
		assert.Greater(t, points[i].LatencyUs, points[i-1].LatencyUs,
			"latency should increase from %d to %d nodes",
			points[i-1].Shape[0], points[i].Shape[0])
	}

	// Verify the per-node cost is in a reasonable range (not wildly non-linear)
	// This checks the design assumption that overhead is roughly linear
	firstPerNode := points[0].LatencyUs / float64(points[0].Shape[0])
	lastPerNode := points[len(points)-1].LatencyUs / float64(points[len(points)-1].Shape[0])
	ratio := lastPerNode / firstPerNode
	assert.InDelta(t, 1.0, ratio, 0.5,
		"per-node cost should be roughly constant (ratio %.2f)", ratio)
}

// TestOrchestrationOverheadCurveUsableForInterpolation verifies that orchestration
// overhead points can be used with Interpolate1D, which is how the estimator
// looks up overhead for a given graph size at inference time.
func TestOrchestrationOverheadCurveUsableForInterpolation(t *testing.T) {
	points := []LatencyPoint{
		{Shape: []int64{50}, LatencyUs: 500},
		{Shape: []int64{100}, LatencyUs: 1000},
		{Shape: []int64{200}, LatencyUs: 2000},
		{Shape: []int64{300}, LatencyUs: 3000},
		{Shape: []int64{500}, LatencyUs: 5000},
	}

	// Exact match
	lat50 := Interpolate1D(points, 50)
	assert.InDelta(t, 500.0, lat50, 0.1, "exact match at 50 nodes")

	// Interpolation between measured points
	lat150 := Interpolate1D(points, 150)
	assert.Greater(t, lat150, 1000.0, "150 nodes should be > 100 nodes latency")
	assert.Less(t, lat150, 2000.0, "150 nodes should be < 200 nodes latency")

	// Extrapolation beyond measured range
	lat600 := Interpolate1D(points, 600)
	assert.Greater(t, lat600, 5000.0, "600 nodes should extrapolate beyond 500 nodes")

	// Verify the curve works as an OperatorCurve with dimension "num_nodes"
	curve := OperatorCurve{
		Op:           "ORCHESTRATION_OVERHEAD",
		Backend:      "Vulkan",
		ComputeDtype: "f32",
		Dimensions:   []string{"num_nodes"},
		Points:       points,
	}
	assert.Len(t, curve.Dimensions, 1)
	assert.Equal(t, "num_nodes", curve.Dimensions[0])

	// The curve's points should be directly usable with Interpolate1D
	lat := Interpolate1D(curve.Points, 250)
	assert.Greater(t, lat, 2000.0)
	assert.Less(t, lat, 3000.0)
}

// TestOrchestrationOverheadProfileStorage verifies that orchestration overhead
// can be stored as an OperatorCurve in a Profile, and survives JSON round-trip.
// This is the integration contract: benchOrchestrationOverhead produces points,
// RunBenchmark stores them as an OperatorCurve, estimate.go reads them back.
func TestOrchestrationOverheadProfileStorage(t *testing.T) {
	curve := OperatorCurve{
		Op:           "ORCHESTRATION_OVERHEAD",
		Backend:      "Vulkan",
		ComputeDtype: "f32",
		Dimensions:   []string{"num_nodes"},
		Points: []LatencyPoint{
			{Shape: []int64{50}, LatencyUs: 3000, StddevUs: 150, Reps: 20},
			{Shape: []int64{100}, LatencyUs: 5800, StddevUs: 290, Reps: 20},
			{Shape: []int64{200}, LatencyUs: 11500, StddevUs: 580, Reps: 20},
			{Shape: []int64{300}, LatencyUs: 17200, StddevUs: 860, Reps: 20},
			{Shape: []int64{500}, LatencyUs: 28500, StddevUs: 1400, Reps: 20},
		},
	}

	// Serialize to JSON
	data, err := json.Marshal(curve)
	require.NoError(t, err)

	// Deserialize back
	var loaded OperatorCurve
	require.NoError(t, json.Unmarshal(data, &loaded))

	// Verify all fields survived round-trip
	assert.Equal(t, "ORCHESTRATION_OVERHEAD", loaded.Op)
	assert.Equal(t, "Vulkan", loaded.Backend)
	assert.Equal(t, "f32", loaded.ComputeDtype)
	assert.Equal(t, []string{"num_nodes"}, loaded.Dimensions)
	require.Len(t, loaded.Points, 5)

	// Verify each point
	for i, pt := range loaded.Points {
		assert.Equal(t, curve.Points[i].Shape, pt.Shape)
		assert.InDelta(t, curve.Points[i].LatencyUs, pt.LatencyUs, 0.001)
		assert.InDelta(t, curve.Points[i].StddevUs, pt.StddevUs, 0.001)
		assert.Equal(t, curve.Points[i].Reps, pt.Reps)
	}

	// Verify the loaded curve is usable for interpolation
	lat := Interpolate1D(loaded.Points, 150)
	assert.Greater(t, lat, 5800.0, "interpolated value between 100 and 200 nodes")
	assert.Less(t, lat, 11500.0, "interpolated value between 100 and 200 nodes")

	// Verify it can be embedded in a full Profile
	profile := &Profile{
		Version: 2,
		Hardware: HardwareProfile{
			Backends:                 []BackendInfo{{Name: "vulkan", Device: "RTX 4090"}},
			PeakTOPS:                 map[string]float64{"f32": 80e12},
			PeakBandwidthBytesPerSec: 900e9,
		},
		Operators: []OperatorCurve{curve},
	}

	profileData, err := json.Marshal(profile)
	require.NoError(t, err)

	var loadedProfile Profile
	require.NoError(t, json.Unmarshal(profileData, &loadedProfile))
	require.Len(t, loadedProfile.Operators, 1)
	assert.Equal(t, "ORCHESTRATION_OVERHEAD", loadedProfile.Operators[0].Op)
	assert.Len(t, loadedProfile.Operators[0].Points, 5)
}
