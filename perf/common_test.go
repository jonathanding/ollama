package perf

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestOpVariantProfileKey(t *testing.T) {
	tests := []struct {
		name     string
		variant  OpVariant
		expected string
	}{
		{
			name:     "plain op",
			variant:  OpVariant{Op: "MUL_MAT"},
			expected: "MUL_MAT",
		},
		{
			name:     "with variant",
			variant:  OpVariant{Op: "MUL_MAT", Variant: "VEC"},
			expected: "MUL_MAT_VEC",
		},
		{
			name:     "fused op",
			variant:  OpVariant{Op: "RMS_NORM", Variant: "MUL"},
			expected: "RMS_NORM_MUL",
		},
		{
			name:     "variant with weight dtype ignored in key",
			variant:  OpVariant{Op: "MUL_MAT", Variant: "VEC", WeightDtype: "f16", Backend: "Vulkan"},
			expected: "MUL_MAT_VEC",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, tt.variant.ProfileKey())
		})
	}
}

func TestGetBackendCapabilities(t *testing.T) {
	vk := GetBackendCapabilities("Vulkan")
	assert.Equal(t, "Vulkan", vk.Name)
	assert.True(t, vk.HasGPUTimestamp)
	assert.True(t, vk.HasMulMatVec)
	assert.Equal(t, 8, vk.MulMatVecMaxN)
	assert.Len(t, vk.FusionRules, 3, "Vulkan should have 3 fusion rules")

	cpu := GetBackendCapabilities("CPU")
	assert.Equal(t, "CPU", cpu.Name)
	assert.False(t, cpu.HasGPUTimestamp)
	assert.False(t, cpu.HasMulMatVec)
	assert.Nil(t, cpu.FusionRules)

	unknown := GetBackendCapabilities("FutureBackend")
	assert.Equal(t, "FutureBackend", unknown.Name)
	assert.False(t, unknown.HasGPUTimestamp)
}

func TestBackendCapabilitiesToJSON(t *testing.T) {
	caps := GetBackendCapabilities("Vulkan")
	j := caps.ToJSON()
	assert.Equal(t, "Vulkan", j.Name)
	assert.True(t, j.HasGPUTimestamp)
	assert.True(t, j.HasMulMatVec)
	assert.Equal(t, 8, j.MulMatVecMaxN)
}
