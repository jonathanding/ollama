package perf

import "github.com/ollama/ollama/ml"

// OpVariant uniquely identifies a benchmark/estimate operation variant.
// ProfileKey() ensures consistent naming between benchmark storage and estimate lookup.
type OpVariant struct {
	Op          string // GGML op name: "MUL_MAT", "RMS_NORM", etc.
	Variant     string // Kernel variant: "", "VEC", "MUL", "MUL_ROPE", etc.
	WeightDtype string // For MUL_MAT: "f16", "q4_0", etc.
	Backend     string // "Vulkan", "CUDA", "CPU"
}

// ProfileKey returns the canonical key for profile storage/lookup.
// Guarantees benchmark writes and estimate reads use the same string.
func (v OpVariant) ProfileKey() string {
	key := v.Op
	if v.Variant != "" {
		key += "_" + v.Variant
	}
	return key
}

// BackendCapabilities describes features supported by a compute backend.
// Shared between benchmark (what to measure) and estimate (how to predict).
type BackendCapabilities struct {
	Name            string       // "Vulkan", "CUDA", "CPU"
	HasGPUTimestamp bool         // Supports GPU timestamp for accurate per-op timing
	FusionRules     []FusionRule // Op fusion patterns this backend applies
	HasMulMatVec    bool         // Has dedicated MUL_MAT_VEC kernel
	MulMatVecMaxN   int          // MUL_MAT_VEC triggered when N <= this value
}

// BackendCapabilitiesJSON is the JSON-serializable subset of BackendCapabilities.
// FusionRules contain function pointers and cannot be serialized;
// they are rebuilt at load time via GetBackendCapabilities(Name).
type BackendCapabilitiesJSON struct {
	Name            string `json:"name"`
	HasGPUTimestamp bool   `json:"has_gpu_timestamp"`
	HasMulMatVec    bool   `json:"has_mul_mat_vec"`
	MulMatVecMaxN   int    `json:"mul_mat_vec_max_n"`
}

// ToJSON converts to the serializable subset.
func (c BackendCapabilities) ToJSON() BackendCapabilitiesJSON {
	return BackendCapabilitiesJSON{
		Name:            c.Name,
		HasGPUTimestamp: c.HasGPUTimestamp,
		HasMulMatVec:    c.HasMulMatVec,
		MulMatVecMaxN:   c.MulMatVecMaxN,
	}
}

// GetBackendCapabilities returns the known capabilities for a backend.
// Called by both benchmark and estimate to ensure consistent behavior.
func GetBackendCapabilities(backendName string) BackendCapabilities {
	switch backendName {
	case "Vulkan":
		return BackendCapabilities{
			Name:            "Vulkan",
			HasGPUTimestamp: true,
			FusionRules:     VulkanFusionRules(),
			HasMulMatVec:    true,
			MulMatVecMaxN:   8,
		}
	case "CUDA":
		return BackendCapabilities{
			Name:            "CUDA",
			HasGPUTimestamp: false,
			FusionRules:     nil,
			HasMulMatVec:    true,
			MulMatVecMaxN:   8,
		}
	case "CPU":
		return BackendCapabilities{
			Name:            "CPU",
			HasGPUTimestamp: false,
			FusionRules:     CPUFusionRules(),
			HasMulMatVec:    false,
			MulMatVecMaxN:   0,
		}
	default:
		return BackendCapabilities{Name: backendName}
	}
}

// DiscoverBackend detects the primary backend from the runtime environment.
// Returns capabilities for the first GPU device, or CPU if none available.
func DiscoverBackend(backend ml.Backend) BackendCapabilities {
	devices := backend.BackendDevices()
	if len(devices) == 0 {
		return GetBackendCapabilities("CPU")
	}
	return GetBackendCapabilities(devices[0].Library)
}
