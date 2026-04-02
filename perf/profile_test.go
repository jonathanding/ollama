package perf

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProfileRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "profile.json")

	p := newTestProfile()
	err := WriteProfile(path, p)
	require.NoError(t, err)

	loaded, err := LoadProfile(path)
	require.NoError(t, err)
	assert.Equal(t, len(p.Hardware.Backends), len(loaded.Hardware.Backends))
	assert.Equal(t, p.Hardware.Backends[0].Name, loaded.Hardware.Backends[0].Name)
	assert.InDelta(t, p.Operators[0].Eta, loaded.Operators[0].Eta, 0.001)
}

func TestRawDataRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "raw.json")

	raw := &RawData{
		Version: 1,
		HardwareBenchmarks: []HardwareBenchmark{
			{Backend: "cuda", Dtype: "f16", Test: "peak_flops", Value: 82.6e12, Unit: "FLOPS"},
		},
		OperatorBenchmarks: []OperatorBenchmark{
			{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Points: []BenchmarkPoint{
					{LatencyUs: 12.5, FLOPs: 33554432, BytesMoved: 8396800, Intensity: 4.0, Reps: 200},
				},
			},
		},
	}
	err := WriteRawData(path, raw)
	require.NoError(t, err)

	loaded, err := LoadRawData(path)
	require.NoError(t, err)
	assert.Equal(t, 1, len(loaded.HardwareBenchmarks))
}

func TestComputeEta(t *testing.T) {
	peakFLOPS := 82.6e12
	peakBW := 1008e9

	points := []BenchmarkPoint{
		{FLOPs: 1e9, BytesMoved: 1e7, LatencyUs: 15.0},
		{FLOPs: 1e10, BytesMoved: 1e8, LatencyUs: 130.0},
	}
	eta, variance := ComputeEtaFromPoints(points, peakFLOPS, peakBW)
	assert.Greater(t, eta, 0.0)
	assert.LessOrEqual(t, eta, 1.0)
	_ = variance
}

func TestBenchDir(t *testing.T) {
	dir := BenchDir()
	assert.Contains(t, dir, "bench")
}
