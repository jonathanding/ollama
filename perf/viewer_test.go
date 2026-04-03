package perf

import (
	"bytes"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFormatSI(t *testing.T) {
	tests := []struct {
		val      float64
		unit     string
		expected string
	}{
		{330e12, "OPS", "330.0 TOPS"},
		{1008e9, "B/s", "1.0 TB/s"},
		{82.6e6, "FLOPS", "82.6 MFLOPS"},
		{1.5e3, "B/s", "1.5 KB/s"},
		{42.0, "B/s", "42.0 B/s"},
		{0.0, "B/s", "0.0 B/s"},
	}
	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			result := formatSI(tt.val, tt.unit)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestTruncate(t *testing.T) {
	assert.Equal(t, "hello", truncate("hello", 10))
	assert.Equal(t, "hello", truncate("hello", 5))
	assert.Equal(t, "hel...", truncate("hello world", 6))
	assert.Equal(t, "h...", truncate("hello", 4))
}

func TestPrintProfile_Summary(t *testing.T) {
	p := &Profile{
		Hardware: HardwareProfile{
			Backends: []BackendInfo{
				{Name: "cuda", Device: "RTX 4090", VRAMBytes: 24_000_000_000},
			},
			PeakTOPS:                 map[string]float64{"f16": 330e12},
			PeakBandwidthBytesPerSec: 1008e9,
			BalancePoints:            map[string]float64{"f16": 327.38},
		},
		Operators: []OperatorCurve{
			{Op: "SILU", Backend: "cuda", ComputeDtype: "f32", Dimensions: []string{"N"},
				Points: []LatencyPoint{{Shape: []int64{1024}, LatencyUs: 2.5}}},
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Dimensions: []string{"N"}, FixedDims: map[string]int64{"M": 4096, "K": 4096},
				Points: []LatencyPoint{{Shape: []int64{1}, LatencyUs: 10.0}}},
		},
	}

	var buf bytes.Buffer
	PrintProfile(&buf, p, false)
	output := buf.String()

	assert.Contains(t, output, "Hardware Profile (v2)")
	assert.Contains(t, output, "cuda")
	assert.Contains(t, output, "RTX 4090")
	assert.Contains(t, output, "Operator Curves: 2")
	assert.Contains(t, output, "SILU")
	assert.Contains(t, output, "MUL_MAT")
}

func TestPrintProfile_Detail(t *testing.T) {
	p := &Profile{
		Hardware: HardwareProfile{
			Backends:                 []BackendInfo{{Name: "cuda", Device: "GPU"}},
			PeakTOPS:                 map[string]float64{"f16": 100e12},
			PeakBandwidthBytesPerSec: 500e9,
			BalancePoints:            map[string]float64{"f16": 200.0},
		},
		Operators: []OperatorCurve{
			{Op: "SILU", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points:     []LatencyPoint{{Shape: []int64{1024}, LatencyUs: 2.5}, {Shape: []int64{2048}, LatencyUs: 5.0}}},
		},
	}

	var buf bytes.Buffer
	PrintProfile(&buf, p, true)
	output := buf.String()

	assert.Contains(t, output, "2 points")
	assert.Contains(t, output, "SILU (f32)")
}

func TestPrintEstimateResult(t *testing.T) {
	r := &EstimateResult{
		Model: "llama3:8b-q4_0",
		Prefill: PhaseEstimation{
			TotalLatencyMs: 45.2,
			TokensPerSec:   2210.0,
			TopOps: []OpBreakdown{
				{Op: "MUL_MAT", ComputeDtype: "f16", WeightDtype: "q4_0", Count: 32, TotalUs: 30000, Percentage: 0.66},
			},
		},
		Decode: PhaseEstimation{
			TotalLatencyMs: 8.5,
			TokensPerSec:   117.6,
			TopOps: []OpBreakdown{
				{Op: "MUL_MAT", ComputeDtype: "f16", WeightDtype: "q4_0", Count: 32, TotalUs: 7000, Percentage: 0.82},
			},
		},
		Warnings: []string{"FLASH_ATTN_EXT not calibrated, using fallback"},
	}

	var buf bytes.Buffer
	PrintEstimateResult(&buf, r, false)
	output := buf.String()

	assert.Contains(t, output, "llama3:8b-q4_0")
	assert.Contains(t, output, "Prefill")
	assert.Contains(t, output, "45.2ms")
	assert.Contains(t, output, "Decode")
	assert.Contains(t, output, "Warnings:")
	assert.Contains(t, output, "FLASH_ATTN_EXT not calibrated")
}

func TestPrintEstimateResult_NoWarnings(t *testing.T) {
	r := &EstimateResult{
		Model:   "test-model",
		Prefill: PhaseEstimation{TotalLatencyMs: 10, TokensPerSec: 1000},
		Decode:  PhaseEstimation{TotalLatencyMs: 5, TokensPerSec: 200},
	}

	var buf bytes.Buffer
	PrintEstimateResult(&buf, r, false)
	output := buf.String()

	assert.NotContains(t, output, "Warnings")
}

func TestPrintTopOps_Empty(t *testing.T) {
	var buf bytes.Buffer
	printTopOps(&buf, nil, false)
	assert.Equal(t, "", buf.String())
}

func TestPrintTopOps_LimitedTo5(t *testing.T) {
	ops := make([]OpBreakdown, 8)
	for i := range ops {
		ops[i] = OpBreakdown{Op: "OP", ComputeDtype: "f32", Count: 1, TotalUs: 100, Percentage: 0.1}
	}
	var buf bytes.Buffer
	printTopOps(&buf, ops, false)
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	assert.Equal(t, 6, len(lines))
}

func TestPrintTopOps_DetailShows10(t *testing.T) {
	ops := make([]OpBreakdown, 12)
	for i := range ops {
		ops[i] = OpBreakdown{Op: "OP", ComputeDtype: "f32", Count: 1, TotalUs: 100, Percentage: 0.05}
	}
	var buf bytes.Buffer
	printTopOps(&buf, ops, true)
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	assert.Equal(t, 11, len(lines))
}
