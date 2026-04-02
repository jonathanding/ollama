package perf

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestProfile() *Profile {
	return &Profile{
		Hardware: HardwareProfile{
			Backends: []BackendProfile{
				{
					Name:          "cuda",
					Device:        "RTX 4090",
					PeakFLOPS:     map[string]float64{"f16": 82.6e12, "f32": 41.3e12},
					PeakBandwidth: 1008e9,
					BalancePoints: map[string]float64{"f16": 81.9, "f32": 41.0},
				},
			},
		},
		Operators: []OperatorProfile{
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0", Eta: 0.62},
			{Op: "RMS_NORM", Backend: "cuda", ComputeDtype: "f32", Eta: 0.85},
		},
		Interconnects: []InterconnectInfo{
			{From: "cuda:0", To: "cpu", Bandwidth: 25.6e9},
		},
	}
}

func TestEstimateOpCost_ComputeBound(t *testing.T) {
	p := newTestProfile()
	cost, err := EstimateOpCost(p, OpKey{"MUL_MAT", "cuda", "f16", "q4_0"},
		1e12, 1e6,
	)
	require.NoError(t, err)
	assert.Equal(t, "compute", cost.Bound)
	assert.InDelta(t, 0.62, cost.Eta, 0.001)
	assert.InDelta(t, 0.0195, cost.TActual, 0.001)
}

func TestEstimateOpCost_MemoryBound(t *testing.T) {
	p := newTestProfile()
	cost, err := EstimateOpCost(p, OpKey{"MUL_MAT", "cuda", "f16", "q4_0"},
		1e6, 1e9,
	)
	require.NoError(t, err)
	assert.Equal(t, "memory", cost.Bound)
}

func TestEstimateOpCost_Uncalibrated(t *testing.T) {
	p := newTestProfile()
	cost, err := EstimateOpCost(p, OpKey{"SOFTMAX", "cuda", "f32", ""},
		1e6, 1e6,
	)
	require.NoError(t, err)
	assert.True(t, cost.Uncalibrated)
	assert.InDelta(t, 1.0, cost.Eta, 0.001)
}

func TestLookupEta(t *testing.T) {
	p := newTestProfile()
	eta, found := LookupEta(p, OpKey{"MUL_MAT", "cuda", "f16", "q4_0"})
	assert.True(t, found)
	assert.InDelta(t, 0.62, eta, 0.001)

	_, found = LookupEta(p, OpKey{"UNKNOWN", "cuda", "f32", ""})
	assert.False(t, found)
}
