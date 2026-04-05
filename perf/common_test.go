package perf

import (
	"encoding/json"
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

func TestGetBackendCapabilities_CUDA(t *testing.T) {
	cuda := GetBackendCapabilities("CUDA")
	assert.Equal(t, "CUDA", cuda.Name)
	assert.False(t, cuda.HasGPUTimestamp, "CUDA does not yet expose GPU timestamps")
	assert.True(t, cuda.HasMulMatVec, "CUDA has MUL_MAT_VEC kernel")
	assert.Equal(t, 8, cuda.MulMatVecMaxN)
	assert.Nil(t, cuda.FusionRules, "CUDA has no fusion rules yet")
}

func TestBackendCapabilitiesToJSON(t *testing.T) {
	// Vulkan: all features enabled
	caps := GetBackendCapabilities("Vulkan")
	j := caps.ToJSON()
	assert.Equal(t, "Vulkan", j.Name)
	assert.True(t, j.HasGPUTimestamp)
	assert.True(t, j.HasMulMatVec)
	assert.Equal(t, 8, j.MulMatVecMaxN)

	// CPU: all features disabled
	cpuCaps := GetBackendCapabilities("CPU")
	cpuJ := cpuCaps.ToJSON()
	assert.Equal(t, "CPU", cpuJ.Name)
	assert.False(t, cpuJ.HasGPUTimestamp)
	assert.False(t, cpuJ.HasMulMatVec)
	assert.Equal(t, 0, cpuJ.MulMatVecMaxN)
}

func TestBackendCapabilitiesJSON_RoundTrip(t *testing.T) {
	// Verify JSON serialization/deserialization round-trip
	caps := GetBackendCapabilities("Vulkan")
	j := caps.ToJSON()

	// Marshal to JSON
	data, err := json.Marshal(j)
	assert.NoError(t, err)

	// Unmarshal back
	var loaded BackendCapabilitiesJSON
	err = json.Unmarshal(data, &loaded)
	assert.NoError(t, err)

	assert.Equal(t, j.Name, loaded.Name)
	assert.Equal(t, j.HasGPUTimestamp, loaded.HasGPUTimestamp)
	assert.Equal(t, j.HasMulMatVec, loaded.HasMulMatVec)
	assert.Equal(t, j.MulMatVecMaxN, loaded.MulMatVecMaxN)
}
