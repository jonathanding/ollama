package perf

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIsZeroCostOp(t *testing.T) {
	tests := []struct {
		op   string
		want bool
	}{
		{"VIEW", true},
		{"RESHAPE", true},
		{"PERMUTE", true},
		{"MUL_MAT", false},
		{"SILU", false},
		{"ADD", false},
		{"FLASH_ATTN_EXT", false},
		{"CONT", false},
		{"", false},
	}
	for _, tt := range tests {
		t.Run(tt.op, func(t *testing.T) {
			assert.Equal(t, tt.want, IsZeroCostOp(tt.op))
		})
	}
}

func TestElemSize(t *testing.T) {
	tests := []struct {
		dtype string
		want  float64
	}{
		{"f32", 4},
		{"f16", 2},
		{"bf16", 2},
		{"q4_0", 0.5625},
		{"q4_K", 0.5625},
		{"q5_K", 0.6875},
		{"q6_K", 0.8125},
		{"q8_0", 1.0625},
		{"unknown_dtype", 4}, // default fallback
	}
	for _, tt := range tests {
		t.Run(tt.dtype, func(t *testing.T) {
			assert.InDelta(t, tt.want, elemSize(tt.dtype), 0.001)
		})
	}
}

func TestProduct(t *testing.T) {
	tests := []struct {
		name  string
		shape []int64
		want  float64
	}{
		{"scalar", []int64{}, 1},
		{"1d", []int64{1024}, 1024},
		{"2d", []int64{32, 128}, 4096},
		{"4d", []int64{128, 32, 512, 1}, 2097152},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.InDelta(t, tt.want, product(tt.shape), 0.01)
		})
	}
}
