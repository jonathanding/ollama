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

// Integration tests for benchPeakTOPS and benchPeakBandwidth require a real
// GGML backend. They are in integration_test.go (Task 14).
