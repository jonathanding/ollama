package perf

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProfileJSON_RoundTrip(t *testing.T) {
	p := Profile{
		Version:   2,
		Timestamp: time.Date(2026, 4, 3, 12, 0, 0, 0, time.UTC),
		Hardware: HardwareProfile{
			Backends: []BackendInfo{
				{Name: "cuda", Device: "RTX 4090", VRAMBytes: 24_000_000_000},
			},
			PeakTOPS:                  map[string]float64{"f16": 330e12},
			PeakBandwidthBytesPerSec:  1008e9,
			InterconnectBWBytesPerSec: 0,
			BalancePoints:             map[string]float64{"f16": 327.38},
		},
		Operators: []OperatorCurve{
			{
				Op: "SILU", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 2.5, StddevUs: 0.1, Reps: 100},
					{Shape: []int64{1048576}, LatencyUs: 45.0, StddevUs: 1.2, Reps: 100},
				},
			},
		},
	}

	data, err := json.MarshalIndent(p, "", "  ")
	require.NoError(t, err)

	var decoded Profile
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, 2, decoded.Version)
	assert.Equal(t, "cuda", decoded.Hardware.Backends[0].Name)
	assert.InDelta(t, 330e12, decoded.Hardware.PeakTOPS["f16"], 1e6)
	assert.Len(t, decoded.Operators, 1)
	assert.Equal(t, "SILU", decoded.Operators[0].Op)
	assert.Equal(t, []string{"N"}, decoded.Operators[0].Dimensions)
	assert.Len(t, decoded.Operators[0].Points, 2)
	assert.Equal(t, int64(1024), decoded.Operators[0].Points[0].Shape[0])
}

func TestOperatorCurveJSON_WithFixedDims(t *testing.T) {
	curve := OperatorCurve{
		Op: "MUL_MAT", Backend: "cuda",
		ComputeDtype: "f16", WeightDtype: "q4_0",
		Dimensions: []string{"N"},
		FixedDims:  map[string]int64{"M": 4096, "K": 4096},
		Points: []LatencyPoint{
			{Shape: []int64{1}, LatencyUs: 10.0, StddevUs: 0.5, Reps: 100},
			{Shape: []int64{4096}, LatencyUs: 500.0, StddevUs: 5.0, Reps: 100},
		},
	}

	data, err := json.Marshal(curve)
	require.NoError(t, err)

	var decoded OperatorCurve
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, "MUL_MAT", decoded.Op)
	assert.Equal(t, "q4_0", decoded.WeightDtype)
	assert.Equal(t, []string{"N"}, decoded.Dimensions)
	assert.Equal(t, int64(4096), decoded.FixedDims["M"])
	assert.Equal(t, int64(4096), decoded.FixedDims["K"])
	assert.Len(t, decoded.Points, 2)
}

func TestOperatorCurveJSON_FixedDimsOmittedWhenNil(t *testing.T) {
	curve := OperatorCurve{
		Op: "SILU", Backend: "cuda", ComputeDtype: "f32",
		Dimensions: []string{"N"},
	}
	data, err := json.Marshal(curve)
	require.NoError(t, err)
	assert.NotContains(t, string(data), "fixed_dims")
}

func TestOperatorCurveJSON_WeightDtypeOmittedWhenEmpty(t *testing.T) {
	curve := OperatorCurve{
		Op: "SILU", Backend: "cuda", ComputeDtype: "f32",
		Dimensions: []string{"N"},
	}
	data, err := json.Marshal(curve)
	require.NoError(t, err)
	assert.NotContains(t, string(data), "weight_dtype")
}

func TestLatencyPointJSON(t *testing.T) {
	pt := LatencyPoint{
		Shape:     []int64{128, 2048},
		LatencyUs: 150.5,
		StddevUs:  3.2,
		Reps:      100,
	}
	data, err := json.Marshal(pt)
	require.NoError(t, err)

	var decoded LatencyPoint
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, []int64{128, 2048}, decoded.Shape)
	assert.InDelta(t, 150.5, decoded.LatencyUs, 0.01)
}

func TestEstimateResultJSON(t *testing.T) {
	r := EstimateResult{
		Model:                   "llama3:8b-q4_0",
		PrefillLatencyUs:        50000,
		PrefillMs:               50.0,
		DecodeLatencyUsPerToken: 5000,
		DecodeTokensPerSec:      200,
		Prefill: PhaseEstimation{
			TotalLatencyMs: 50.0,
			TokensPerSec:   10240,
			TopOps: []OpBreakdown{
				{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16",
					WeightDtype: "q4_0", Count: 224, TotalUs: 45000, Percentage: 0.9},
			},
		},
		Warnings: []string{"uncalibrated: ROPE"},
	}
	data, err := json.Marshal(r)
	require.NoError(t, err)

	var decoded EstimateResult
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, "llama3:8b-q4_0", decoded.Model)
	assert.InDelta(t, 200.0, decoded.DecodeTokensPerSec, 0.01)
	assert.Len(t, decoded.Prefill.TopOps, 1)
	assert.Equal(t, 224, decoded.Prefill.TopOps[0].Count)
	assert.Len(t, decoded.Warnings, 1)
}

func TestHWCharResult(t *testing.T) {
	result := HWCharResult{
		PeakTOPS:     map[string]float64{"f16": 330e12, "f32": 165e12},
		PeakBW:       1008e9,
		BalancePoint: map[string]float64{"f16": 327.38, "f32": 163.69},
	}

	assert.InDelta(t, 330e12, result.PeakTOPS["f16"], 1e6)
	assert.InDelta(t, 1008e9, result.PeakBW, 1e3)
	assert.InDelta(t, 327.38, result.BalancePoint["f16"], 0.01)
}
