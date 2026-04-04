package perf

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMapWeightDtype_DirectlyMeasured(t *testing.T) {
	for _, dt := range []string{"f32", "f16", "q4_0", "q8_0"} {
		assert.Equal(t, dt, mapWeightDtype(dt), "measured dtype %s should map to itself", dt)
	}
}

func TestMapWeightDtype_KQuants(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"q4_K", "q4_0"},
		{"q4_1", "q4_0"},
		{"q5_K", "q8_0"},
		{"q5_0", "q8_0"},
		{"q5_1", "q8_0"},
		{"q6_K", "q8_0"},
		{"q3_K", "q4_0"},
		{"q2_K", "q4_0"},
		{"q8_K", "q8_0"},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			assert.Equal(t, tt.expected, mapWeightDtype(tt.input))
		})
	}
}

func TestMapWeightDtype_UnknownFallback(t *testing.T) {
	assert.Equal(t, "f16", mapWeightDtype("bf16"))
	assert.Equal(t, "f16", mapWeightDtype("unknown_type"))
	assert.Equal(t, "f16", mapWeightDtype(""))
}
