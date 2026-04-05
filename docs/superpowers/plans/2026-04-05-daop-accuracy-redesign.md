# DAOP Accuracy Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 18x estimate error (1338ms predicted vs 75ms actual for qwen3:1.7b decode) by introducing GPU timestamps, op fusion simulation, MUL_MAT_VEC routing, and CPU orchestration overhead modeling. Target: <2x error.

**Architecture:** GPU timestamp C API exposes per-op Vulkan execution times to Go, eliminating dispatch overhead from benchmarks. Shared infrastructure (`perf/common.go`) ensures naming consistency between benchmark and estimate. Go-side fusion simulation replaces unfused per-op accumulation. Orchestration overhead model adds CPU-side cost as `f(num_nodes)`.

**Tech Stack:** Go 1.24, C/C++ (ggml-vulkan.cpp), CGO, Vulkan timestamp queries, testify

**Design Spec:** `docs/superpowers/specs/2026-04-05-daop-accuracy-redesign.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|----------------|
| `perf/common.go` | OpVariant, BackendCapabilities, BackendCapabilitiesJSON, GetBackendCapabilities, DiscoverBackend |
| `perf/common_test.go` | Unit tests for shared types and backend discovery |
| `perf/fusion.go` | FusionRule, ApplyFusion, VulkanFusionRules, CPUFusionRules |
| `perf/fusion_test.go` | Unit tests for fusion pattern matching and graph rewriting |

### Modified Files

| File | Changes |
|------|---------|
| `ml/backend/ggml/ggml/include/ggml-vulkan.h` | Declare `ggml_vk_op_timing`, `ggml_vk_enable_timestamps()`, `ggml_vk_get_op_timings()` |
| `ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Implement timestamp storage + C API functions |
| `ml/backend.go` | Add `OpTiming` type, `EnableGPUTimestamps()`, `GetOpTimings()` to Backend interface |
| `ml/backend/ggml/ggml.go` | CGO bindings for GPU timestamp API |
| `perf/types.go` | Profile version 3, add `BackendCaps` field to Profile |
| `perf/profile.go` | Version 3 validation in LoadProfile |
| `perf/registry.go` | Add fused op entries: RMS_NORM_MUL, RMS_NORM_MUL_ROPE, MUL_MAT_ADD |
| `perf/bench.go` | `measureOpGPU()`, `measureOpForBackend()`, `benchOrchestrationOverhead()`, RunBenchmark wiring |
| `perf/estimate.go` | Fusion integration, MUL_MAT_VEC routing, orchestration overhead |
| `perf/estimate_test.go` | New tests for fusion/VEC/overhead in estimation |

---

## Phase 1: Shared Infrastructure + GPU Timestamp C API

### Task 1: perf/common.go — Shared Types and Backend Capabilities

**Files:**
- Create: `perf/common.go`
- Create: `perf/common_test.go`

This task creates the shared module that both benchmark and estimate depend on. It defines `OpVariant` (consistent naming between benchmark storage and estimate lookup), `BackendCapabilities` (per-backend feature flags), and `DiscoverBackend()` (runtime detection).

- [ ] **Step 1: Write tests for OpVariant.ProfileKey()**

```go
// perf/common_test.go
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./perf/ -run TestOpVariantProfileKey -v`
Expected: FAIL — `OpVariant` type not defined

- [ ] **Step 3: Write tests for GetBackendCapabilities**

Add to `perf/common_test.go`:

```go
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
```

- [ ] **Step 4: Implement perf/common.go**

```go
// perf/common.go
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
```

**Note:** This file references `FusionRule`, `VulkanFusionRules()`, and `CPUFusionRules()` from `perf/fusion.go` (Task 2). To compile independently, either implement Task 2 first or create stub declarations. Recommended: implement Task 1 and Task 2 together, then run tests.

- [ ] **Step 5: Run tests to verify they pass**

Run: `go test ./perf/ -run "TestOpVariant|TestGetBackend|TestBackendCapabilities" -v`
Expected: PASS (after Task 2 fusion.go exists)

- [ ] **Step 6: Commit**

```bash
git add perf/common.go perf/common_test.go
git commit -m "perf: add shared infrastructure for benchmark/estimate consistency"
```

---

### Task 2: perf/fusion.go — Fusion Rules and Graph Rewriting

**Files:**
- Create: `perf/fusion.go`
- Create: `perf/fusion_test.go`

Implements Go-side op fusion simulation. Three core rules cover >95% of Vulkan decode fusion:
1. RMS_NORM_MUL_ROPE (3-op, must be first)
2. RMS_NORM_MUL (2-op)
3. MUL_MAT_ADD (2-op, mat-vec only N≤8)

- [ ] **Step 1: Write tests for ApplyFusion**

```go
// perf/fusion_test.go
package perf

import (
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/stretchr/testify/assert"
)

func node(op, backend string) ml.GraphNode {
	return ml.GraphNode{Op: op, Backend: backend, ComputeDtype: "f32"}
}

func nodeWithInputShapes(op, backend string, inputShapes [][]int64) ml.GraphNode {
	return ml.GraphNode{Op: op, Backend: backend, ComputeDtype: "f32", InputShapes: inputShapes}
}

func TestApplyFusionEmpty(t *testing.T) {
	result := ApplyFusion(nil, VulkanFusionRules())
	assert.Nil(t, result)
}

func TestApplyFusionNoRules(t *testing.T) {
	nodes := []ml.GraphNode{node("RMS_NORM", "Vulkan"), node("MUL", "Vulkan")}
	result := ApplyFusion(nodes, nil)
	assert.Equal(t, nodes, result)
}

func TestApplyFusionRMSNormMul(t *testing.T) {
	nodes := []ml.GraphNode{
		node("RMS_NORM", "Vulkan"),
		node("MUL", "Vulkan"),
		node("ADD", "Vulkan"),
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 2)
	assert.Equal(t, "RMS_NORM_MUL", result[0].Op)
	assert.Equal(t, "ADD", result[1].Op)
}

func TestApplyFusionRMSNormMulRope(t *testing.T) {
	nodes := []ml.GraphNode{
		node("RMS_NORM", "Vulkan"),
		node("MUL", "Vulkan"),
		node("ROPE", "Vulkan"),
		node("ADD", "Vulkan"),
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 2)
	assert.Equal(t, "RMS_NORM_MUL_ROPE", result[0].Op)
	assert.Equal(t, "ADD", result[1].Op)
}

func TestApplyFusionMulMatAdd(t *testing.T) {
	// MUL_MAT with N=1 (mat-vec) followed by ADD -> fused
	nodes := []ml.GraphNode{
		nodeWithInputShapes("MUL_MAT", "Vulkan", [][]int64{{4096, 1024}, {4096, 1}}),
		node("ADD", "Vulkan"),
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 1)
	assert.Equal(t, "MUL_MAT_ADD", result[0].Op)
}

func TestApplyFusionMulMatAddNotMatVec(t *testing.T) {
	// MUL_MAT with N=512 (not mat-vec) -> NOT fused
	nodes := []ml.GraphNode{
		nodeWithInputShapes("MUL_MAT", "Vulkan", [][]int64{{4096, 1024}, {4096, 512}}),
		node("ADD", "Vulkan"),
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 2)
	assert.Equal(t, "MUL_MAT", result[0].Op)
	assert.Equal(t, "ADD", result[1].Op)
}

func TestApplyFusionDifferentBackends(t *testing.T) {
	// RMS_NORM on Vulkan + MUL on CPU -> NOT fused
	nodes := []ml.GraphNode{
		node("RMS_NORM", "Vulkan"),
		node("MUL", "CPU"),
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 2)
	assert.Equal(t, "RMS_NORM", result[0].Op)
}

func TestApplyFusion3OpPriorityOver2Op(t *testing.T) {
	// RMS_NORM + MUL + ROPE should match 3-op rule, not 2-op
	nodes := []ml.GraphNode{
		node("RMS_NORM", "Vulkan"),
		node("MUL", "Vulkan"),
		node("ROPE", "Vulkan"),
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 1)
	assert.Equal(t, "RMS_NORM_MUL_ROPE", result[0].Op)
}

func TestVulkanFusionRulesOrder(t *testing.T) {
	rules := VulkanFusionRules()
	assert.Equal(t, 3, len(rules))
	// 3-op rule must come before 2-op rules
	assert.Len(t, rules[0].Pattern, 3, "first rule should be 3-op pattern")
	assert.Len(t, rules[1].Pattern, 2)
	assert.Len(t, rules[2].Pattern, 2)
}

func TestCPUFusionRulesEmpty(t *testing.T) {
	assert.Nil(t, CPUFusionRules())
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./perf/ -run "TestApplyFusion|TestVulkanFusion|TestCPUFusion" -v`
Expected: FAIL — `FusionRule`, `ApplyFusion`, `VulkanFusionRules` not defined

- [ ] **Step 3: Implement perf/fusion.go**

```go
// perf/fusion.go
package perf

import "github.com/ollama/ollama/ml"

// FusionRule defines an op fusion pattern that a backend applies at dispatch time.
// The estimate uses these rules to simulate fusion before summing per-op latencies.
type FusionRule struct {
	Name    string   // Fused op name: "RMS_NORM_MUL", "MUL_MAT_ADD", etc.
	Pattern []string // Sequential op names to match: ["RMS_NORM", "MUL"]
	// Match checks whether nodes[start:start+len(Pattern)] satisfy fusion constraints.
	// Called only after op names match. Check backend consistency, shapes, etc.
	Match func(nodes []ml.GraphNode, start int) bool
}

// VulkanFusionRules returns the 3 core fusion rules for Vulkan LLM decode.
// Rules are ordered by pattern length descending: 3-op patterns must be tried
// before 2-op patterns to avoid partial matches (e.g., RMS_NORM+MUL matching
// the first two ops of a RMS_NORM+MUL+ROPE sequence).
func VulkanFusionRules() []FusionRule {
	return []FusionRule{
		{
			Name:    "RMS_NORM_MUL_ROPE",
			Pattern: []string{"RMS_NORM", "MUL", "ROPE"},
			Match: func(nodes []ml.GraphNode, i int) bool {
				return nodes[i].Backend == nodes[i+1].Backend &&
					nodes[i+1].Backend == nodes[i+2].Backend
			},
		},
		{
			Name:    "RMS_NORM_MUL",
			Pattern: []string{"RMS_NORM", "MUL"},
			Match: func(nodes []ml.GraphNode, i int) bool {
				return nodes[i].Backend == nodes[i+1].Backend
			},
		},
		{
			Name:    "MUL_MAT_ADD",
			Pattern: []string{"MUL_MAT", "ADD"},
			Match: func(nodes []ml.GraphNode, i int) bool {
				if len(nodes[i].InputShapes) < 2 || len(nodes[i].InputShapes[1]) < 2 {
					return false
				}
				N := nodes[i].InputShapes[1][1]
				return N <= 8 && nodes[i].Backend == nodes[i+1].Backend
			},
		},
	}
}

// CPUFusionRules returns fusion rules for CPU backend. CPU has no op fusion.
func CPUFusionRules() []FusionRule { return nil }

// ApplyFusion scans graph nodes and replaces fusable patterns with single fused nodes.
// Returns a new node list (potentially shorter than input).
//
// Limitation: only checks op name sequence and backend consistency, not data dependencies
// (i.e., does not verify MUL's src[0] comes from RMS_NORM's output). In LLM graphs,
// RMS_NORM is always followed by a MUL consuming its output, so false positives are
// extremely unlikely.
func ApplyFusion(nodes []ml.GraphNode, rules []FusionRule) []ml.GraphNode {
	if len(rules) == 0 || len(nodes) == 0 {
		return nodes
	}
	result := make([]ml.GraphNode, 0, len(nodes))
	i := 0
	for i < len(nodes) {
		fused := false
		for _, rule := range rules {
			pLen := len(rule.Pattern)
			if i+pLen > len(nodes) {
				continue
			}
			match := true
			for j, opName := range rule.Pattern {
				if nodes[i+j].Op != opName {
					match = false
					break
				}
			}
			if match && rule.Match(nodes, i) {
				fusedNode := nodes[i]
				fusedNode.Op = rule.Name
				result = append(result, fusedNode)
				i += pLen
				fused = true
				break
			}
		}
		if !fused {
			result = append(result, nodes[i])
			i++
		}
	}
	return result
}
```

- [ ] **Step 4: Run all tests for Task 1 + Task 2**

Run: `go test ./perf/ -run "TestOpVariant|TestGetBackend|TestBackendCapabilities|TestApplyFusion|TestVulkanFusion|TestCPUFusion" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add perf/common.go perf/common_test.go perf/fusion.go perf/fusion_test.go
git commit -m "perf: add shared backend capabilities and op fusion simulation"
```

---

### Task 3: C API — GPU Timestamp Exposure

**Files:**
- Modify: `ml/backend/ggml/ggml/include/ggml-vulkan.h:14-25`
- Modify: `ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp`

Adds two C functions that expose Vulkan GPU timestamps through a structured API instead of stderr. The existing `vk_perf_logger` infrastructure (query pools, `writeTimestamp()`) is reused; we add a structured output path alongside the existing stderr logging.

**Important context:**
- `ggml_backend_vk_context` struct (line ~1731) has `std::unique_ptr<vk_perf_logger> perf_logger`
- Query pool created at line ~13162 when `vk_perf_logger_enabled`
- Timestamps written at line ~13304 per op
- Results read at line ~13345-13349

- [ ] **Step 1: Add C API declarations to ggml-vulkan.h**

Add after the existing function declarations (after `ggml_backend_vk_reg`):

```c
// --- GPU Timestamp API for structured per-op timing ---

struct ggml_vk_op_timing {
    const char * op_name;    // Actual kernel name: "MUL_MAT_VEC", "RMS_NORM_MUL", etc.
    int          node_idx;   // Index in the computation graph
    float        gpu_time_us; // GPU execution time in microseconds
};

// Enable/disable GPU timestamp collection for this backend.
// When enabled, the next graph_compute will record per-op GPU timestamps.
// Default: disabled. Has small performance overhead when enabled.
GGML_API void ggml_vk_enable_timestamps(ggml_backend_t backend, bool enable);

// Return per-op GPU timestamps from the most recent graph_compute.
// Caller does NOT own the returned memory -- it is managed by the backend
// and valid until the next graph_compute or backend destruction.
// *n_timings is set to the number of entries returned.
// Returns NULL if timestamps are not enabled or backend is not Vulkan.
GGML_API struct ggml_vk_op_timing * ggml_vk_get_op_timings(
    ggml_backend_t backend, int * n_timings);
```

- [ ] **Step 2: Add storage to ggml_backend_vk_context**

In `ggml-vulkan.cpp`, find the `ggml_backend_vk_context` struct (around line 1731). Add these fields:

```cpp
    // Structured timestamp API (separate from vk_perf_logger stderr output)
    bool timestamps_enabled = false;
    std::vector<ggml_vk_op_timing> op_timings;
    std::vector<std::string> op_timing_names; // owns the name strings
```

- [ ] **Step 3: Enable query pool when timestamps_enabled**

In `ggml_backend_vk_graph_compute` (around line 13156), the existing code creates a query pool only when `vk_perf_logger_enabled`. Change the condition to also trigger when `ctx->timestamps_enabled`:

Find the block that checks `vk_perf_logger_enabled` for query pool creation and change:
```cpp
// Before:
if (vk_perf_logger_enabled) {
// After:
if (vk_perf_logger_enabled || ctx->timestamps_enabled) {
```

Apply this same change to ALL guarded blocks in `ggml_backend_vk_graph_compute` that use `vk_perf_logger_enabled` for:
- Query pool allocation/resize (~line 13162)
- Query pool reset (~line 13170)
- Initial `writeTimestamp()` (~line 13180)
- Per-op `writeTimestamp()` (~line 13304)
- Query result retrieval (~line 13345)

- [ ] **Step 4: Populate op_timings after query results**

In the query result retrieval section (around line 13345-13349), after the existing perf_logger logging, add code to populate `ctx->op_timings`:

```cpp
// After reading query pool results and computing per-op times:
if (ctx->timestamps_enabled) {
    ctx->op_timings.clear();
    ctx->op_timing_names.clear();
    
    // The existing loop already computes per-op GPU time as:
    //   (timestamps[i] - timestamps[i-1]) * device.properties.limits.timestampPeriod
    // and calls perf_logger->log_timing(node, fusion_name, time_ns).
    // We replicate the naming logic and store structured results.
    
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        // Build op name (same logic as vk_perf_logger::log_timing)
        std::string name = ggml_op_name(node->op);
        // If fusion was applied, use fusion name instead
        // (The existing code tracks this via the fusion_name parameter)
        
        float gpu_time_us = /* computed from timestamp delta */ ;
        
        ctx->op_timing_names.push_back(std::move(name));
        ctx->op_timings.push_back({
            ctx->op_timing_names.back().c_str(),
            i,
            gpu_time_us
        });
    }
}
```

**Note:** The exact implementation must follow the existing perf_logger loop structure in `ggml_backend_vk_graph_compute`. The implementer must trace the existing perf_logger code path to replicate the correct op naming (including `_VEC` suffix for MUL_MAT_VEC, fused op names, etc.) and timestamp delta calculation.

- [ ] **Step 5: Implement the two C API functions**

Add at the end of `ggml-vulkan.cpp` (near the other `ggml_backend_vk_*` public functions):

```cpp
void ggml_vk_enable_timestamps(ggml_backend_t backend, bool enable) {
    if (!ggml_backend_is_vk(backend)) {
        return;
    }
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    ctx->timestamps_enabled = enable;
    if (!enable) {
        ctx->op_timings.clear();
        ctx->op_timing_names.clear();
    }
}

struct ggml_vk_op_timing * ggml_vk_get_op_timings(
        ggml_backend_t backend, int * n_timings) {
    if (!ggml_backend_is_vk(backend) || n_timings == nullptr) {
        if (n_timings) *n_timings = 0;
        return nullptr;
    }
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    *n_timings = (int)ctx->op_timings.size();
    if (ctx->op_timings.empty()) {
        return nullptr;
    }
    return ctx->op_timings.data();
}
```

- [ ] **Step 6: Build to verify compilation**

Run: `cmake --build build --config Release 2>&1 | tail -20`
Expected: Successful build (no errors from new code)

- [ ] **Step 7: Commit**

```bash
git add ml/backend/ggml/ggml/include/ggml-vulkan.h ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp
git commit -m "ml: add GPU timestamp C API for structured per-op Vulkan timing"
```

---

### Task 4: Go CGO Bindings + ml.Backend Interface

**Files:**
- Modify: `ml/backend.go:16-32`
- Modify: `ml/backend/ggml/ggml.go`

Exposes the C timestamp API to Go. Adds `OpTiming` type and two methods to `ml.Backend`.

- [ ] **Step 1: Add OpTiming type and interface methods to ml/backend.go**

Add `OpTiming` type after the `GraphNode` struct (after line 119):

```go
// OpTiming represents the GPU execution time for one operation in a computation graph.
// Returned by GetOpTimings() after a Compute() call with GPU timestamps enabled.
type OpTiming struct {
	OpName    string  // Actual kernel name: "MUL_MAT_VEC", "RMS_NORM_MUL", etc.
	NodeIdx   int     // Index in the computation graph
	GPUTimeUs float64 // GPU execution time in microseconds
}
```

Add two methods to the `Backend` interface (inside the interface block, after `BackendDevices()`):

```go
	// EnableGPUTimestamps enables/disables per-op GPU timestamp collection.
	// When enabled, the next Compute() records GPU-side execution times.
	// Default: disabled. No-op for backends without GPU timestamp support.
	EnableGPUTimestamps(enable bool)

	// GetOpTimings returns per-op GPU timestamps from the most recent Compute().
	// Returns nil if timestamps are not enabled or not supported by this backend.
	GetOpTimings() []OpTiming
```

- [ ] **Step 2: Implement CGO bindings in ml/backend/ggml/ggml.go**

First, add the CGO include for ggml-vulkan.h. In the CGO preamble (around line 3-10), add:

```go
// #include "ggml-vulkan.h"
```

Then add the two methods to the `Backend` struct:

```go
// EnableGPUTimestamps enables per-op GPU timestamp collection on the Vulkan backend.
func (b *Backend) EnableGPUTimestamps(enable bool) {
	for _, be := range b.schedBackends {
		if C.ggml_backend_is_vk(be) {
			C.ggml_vk_enable_timestamps(be, C.bool(enable))
		}
	}
}

// GetOpTimings returns per-op GPU timestamps from the most recent Compute().
func (b *Backend) GetOpTimings() []ml.OpTiming {
	for _, be := range b.schedBackends {
		if C.ggml_backend_is_vk(be) {
			var nTimings C.int
			timings := C.ggml_vk_get_op_timings(be, &nTimings)
			if timings == nil || nTimings == 0 {
				return nil
			}
			result := make([]ml.OpTiming, int(nTimings))
			for i := 0; i < int(nTimings); i++ {
				t := (*C.struct_ggml_vk_op_timing)(
					unsafe.Pointer(uintptr(unsafe.Pointer(timings)) +
						uintptr(i)*unsafe.Sizeof(*timings)))
				result[i] = ml.OpTiming{
					OpName:    C.GoString(t.op_name),
					NodeIdx:   int(t.node_idx),
					GPUTimeUs: float64(t.gpu_time_us),
				}
			}
			return result
		}
	}
	return nil
}
```

**Note:** Ensure `unsafe` is imported at the top of `ggml.go` (it likely already is).

- [ ] **Step 3: Build to verify compilation**

Run: `go build ./ml/... 2>&1`
Expected: Successful build

- [ ] **Step 4: Commit**

```bash
git add ml/backend.go ml/backend/ggml/ggml.go
git commit -m "ml: add Go CGO bindings for GPU timestamp API"
```

---

### Task 5: Profile v3 — Types and Loading

**Files:**
- Modify: `perf/types.go:9-14`
- Modify: `perf/profile.go:47-61`

Bumps profile version to 3 and adds `BackendCaps` field. Version 2 profiles remain loadable.

- [ ] **Step 1: Write test for version 3 profile loading**

Add to `perf/profile_test.go`:

```go
func TestLoadProfileV3(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "profile.json")

	p := &Profile{
		Version:   3,
		Timestamp: time.Now(),
		Hardware:  HardwareProfile{PeakTOPS: map[string]float64{"f32": 100}},
		BackendCaps: map[string]BackendCapabilitiesJSON{
			"Vulkan": {
				Name:            "Vulkan",
				HasGPUTimestamp: true,
				HasMulMatVec:    true,
				MulMatVecMaxN:   8,
			},
		},
	}
	err := WriteProfile(path, p)
	require.NoError(t, err)

	loaded, err := LoadProfile(path)
	require.NoError(t, err)
	assert.Equal(t, 3, loaded.Version)
	assert.True(t, loaded.BackendCaps["Vulkan"].HasGPUTimestamp)
	assert.Equal(t, 8, loaded.BackendCaps["Vulkan"].MulMatVecMaxN)
}

func TestLoadProfileV2StillWorks(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "profile.json")

	p := &Profile{
		Version:   2,
		Timestamp: time.Now(),
		Hardware:  HardwareProfile{PeakTOPS: map[string]float64{"f32": 100}},
	}
	err := WriteProfile(path, p)
	require.NoError(t, err)

	loaded, err := LoadProfile(path)
	require.NoError(t, err)
	assert.Equal(t, 2, loaded.Version)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./perf/ -run "TestLoadProfileV3|TestLoadProfileV2Still" -v`
Expected: FAIL — `BackendCaps` field doesn't exist, v3 rejected by version check

- [ ] **Step 3: Add BackendCaps to Profile struct**

In `perf/types.go`, modify the `Profile` struct (lines 9-14):

```go
type Profile struct {
	Version     int                                `json:"version"`
	Timestamp   time.Time                          `json:"timestamp"`
	Hardware    HardwareProfile                    `json:"hardware"`
	Operators   []OperatorCurve                    `json:"operators"`
	BackendCaps map[string]BackendCapabilitiesJSON `json:"backend_caps,omitempty"`
}
```

- [ ] **Step 4: Update LoadProfile to accept v2 and v3**

In `perf/profile.go`, change the version check (line 57-59):

```go
	if p.Version != 2 && p.Version != 3 {
		return nil, fmt.Errorf("unsupported profile version %d (expected 2 or 3)", p.Version)
	}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `go test ./perf/ -run "TestLoadProfile" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add perf/types.go perf/profile.go perf/profile_test.go
git commit -m "perf: add Profile v3 with BackendCaps field, keep v2 loadable"
```

---

## Phase 2: Op Fusion + Orchestration Overhead + MUL_MAT_VEC

### Task 6: perf/bench.go — measureOpGPU and Backend Dispatch

**Files:**
- Modify: `perf/bench.go:51-124`

Adds `measureOpGPU()` which uses GPU timestamps instead of wall-clock, and `measureOpForBackend()` which dispatches to the correct measurement method based on backend capabilities.

- [ ] **Step 1: Implement measureOpGPU**

Add to `perf/bench.go` (after the existing `measureOp` function):

```go
// measureOpGPU benchmarks an operator using GPU timestamps instead of wall-clock.
// This eliminates Vulkan dispatch overhead from measurements.
// Requires backend that implements EnableGPUTimestamps/GetOpTimings.
func measureOpGPU(backend ml.Backend, op string, gridPoint []int64, computeDtype string, cfg BenchmarkConfig) LatencyPoint {
	runner, ok := LookupRegistry(op)
	if !ok {
		slog.Warn("unknown op in registry", "op", op)
		return LatencyPoint{Shape: gridPoint}
	}

	dt, ok := parseDType(computeDtype)
	if !ok {
		slog.Warn("unsupported dtype", "dtype", computeDtype)
		return LatencyPoint{Shape: gridPoint}
	}

	backend.EnableGPUTimestamps(true)
	defer backend.EnableGPUTimestamps(false)

	ctx := backend.NewContext()
	defer ctx.Close()

	var inputs []ml.Tensor
	if runner.CreateInputs != nil {
		inputs = runner.CreateInputs(ctx, computeDtype, gridPoint)
	} else {
		tensorShapes := expandShapes(op, gridPoint)
		inputs = make([]ml.Tensor, len(tensorShapes))
		for i, shape := range tensorShapes {
			intShape := make([]int, len(shape))
			for j, s := range shape {
				intShape[j] = int(s)
			}
			inputs[i] = randomTensor(ctx, dt, intShape...)
		}
	}

	out := runner.Run(ctx, inputs)
	if out == nil {
		slog.Warn("op returned nil", "op", op)
		return LatencyPoint{Shape: gridPoint}
	}
	ctx.Forward(out)

	// Warmup (2 iterations — GPU timestamps make warmup fast)
	for range 2 {
		ctx.Compute(out)
	}

	// Measure using GPU timestamps
	samples := make([]float64, 0, cfg.MeasureReps)
	for i := 0; i < cfg.MeasureReps; i++ {
		ctx.Compute(out)
		timings := backend.GetOpTimings()
		if len(timings) == 0 {
			slog.Warn("no GPU timings returned", "op", op)
			break
		}
		// Find our target op in the timings (may differ from graph op name
		// due to kernel selection, e.g., MUL_MAT -> MUL_MAT_VEC)
		var gpuUs float64
		for _, t := range timings {
			gpuUs += t.GPUTimeUs // Sum all ops in the graph
		}
		samples = append(samples, gpuUs)

		// Early stopping with convergence check
		if len(samples) >= cfg.MinReps {
			med, sd := trimmedStats(samples, cfg.TrimPercent)
			if med > 0 && sd/med < cfg.ConvergenceCV {
				break
			}
		}
	}

	if len(samples) == 0 {
		return LatencyPoint{Shape: gridPoint}
	}

	med, sd := trimmedStats(samples, cfg.TrimPercent)
	return LatencyPoint{
		Shape:     gridPoint,
		LatencyUs: med,
		StddevUs:  sd,
		Reps:      len(samples),
	}
}
```

**Note:** `trimmedStats` is the existing helper in bench.go that computes trimmed median and stddev. If it doesn't exist as a standalone function, extract it from `convergentMeasure`. The implementer should check the exact function name — it may be called `trimmedMedian` or be inline in `convergentMeasure`.

- [ ] **Step 2: Implement measureOpForBackend dispatch**

Add to `perf/bench.go`:

```go
// measureOpForBackend dispatches to GPU timestamp or wall-clock measurement
// based on backend capabilities.
func measureOpForBackend(backend ml.Backend, caps BackendCapabilities,
	op string, gridPoint []int64, computeDtype string, cfg BenchmarkConfig) LatencyPoint {
	if caps.HasGPUTimestamp {
		return measureOpGPU(backend, op, gridPoint, computeDtype, cfg)
	}
	return measureOp(backend, op, gridPoint, computeDtype, cfg)
}
```

- [ ] **Step 3: Build to verify compilation**

Run: `go build ./perf/ 2>&1`
Expected: Successful build

- [ ] **Step 4: Commit**

```bash
git add perf/bench.go
git commit -m "perf: add GPU timestamp measurement path for Vulkan benchmarks"
```

---

### Task 7: perf/registry.go — Fused Op Benchmark Entries

**Files:**
- Modify: `perf/registry.go:58-153`

Adds registry entries for fused ops so the benchmark can measure their GPU execution times. These entries construct small graphs containing fusable patterns — when Vulkan processes them with GPU timestamps enabled, the returned timings reflect the fused kernel.

- [ ] **Step 1: Add fused op entries to opRegistry**

Add these entries to `opRegistry` in `perf/registry.go` (after the existing entries, before the closing `}`):

```go
	// --- Fused op entries (Vulkan fusion simulation) ---
	// These create graphs with fusable patterns. When benchmarked with GPU timestamps
	// on Vulkan, the backend fuses them automatically and returns fused kernel timing.

	"RMS_NORM_MUL": {
		Dimensions: []string{"N"},
		CreateInputs: func(ctx ml.Context, dtypeStr string, gridPoint []int64) []ml.Tensor {
			N := int(gridPoint[0])
			input := randomTensor(ctx, ml.DTypeF32, N)
			scale := randomTensor(ctx, ml.DTypeF32, N)
			return []ml.Tensor{input, scale}
		},
		Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
			normed := in[0].RMSNorm(ctx, nil, 1e-5)
			return normed.Mul(ctx, in[1])
		},
	},
	"RMS_NORM_MUL_ROPE": {
		Dimensions: []string{"N"},
		CreateInputs: func(ctx ml.Context, dtypeStr string, gridPoint []int64) []ml.Tensor {
			// N = total elements. Shape: [headDim=128, 1, seqLen, 1]
			shape, seqLen := ropeInputParams(gridPoint[0])
			input := randomTensor(ctx, ml.DTypeF32, shape...)
			scale := randomTensor(ctx, ml.DTypeF32, shape...)
			pos := ropePositions(seqLen)
			posTensor := ctx.Input().FromInts(pos, int(seqLen))
			return []ml.Tensor{input, scale, posTensor}
		},
		Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
			normed := in[0].RMSNorm(ctx, nil, 1e-5)
			scaled := normed.Mul(ctx, in[1])
			type roper interface {
				RoPE(ctx ml.Context, positions ml.Tensor, dim int, base, scale float32, options ...func(*rope.Options)) ml.Tensor
			}
			if t, ok := scaled.(roper); ok {
				return t.RoPE(ctx, in[2], 128, 10000.0, 1.0)
			}
			return nil
		},
	},
	"MUL_MAT_ADD": {
		Dimensions: []string{"M", "K", "N"},
		CreateInputs: func(ctx ml.Context, dtypeStr string, gridPoint []int64) []ml.Tensor {
			dt, _ := parseDType(dtypeStr)
			wShape, aShape := mulMatInputShapes(gridPoint)
			weight := randomTensor(ctx, dt, wShape...)
			activation := randomTensor(ctx, ml.DTypeF32, aShape...)
			// Bias has shape [M] (output dim)
			M := int(gridPoint[0])
			bias := randomTensor(ctx, ml.DTypeF32, M)
			return []ml.Tensor{weight, activation, bias}
		},
		Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
			mm := in[0].Mulmat(ctx, in[1])
			return mm.Add(ctx, in[2])
		},
	},
```

- [ ] **Step 2: Write test to verify registry entries exist**

Add to `perf/registry_test.go`:

```go
func TestFusedOpRegistryEntries(t *testing.T) {
	fusedOps := []string{"RMS_NORM_MUL", "RMS_NORM_MUL_ROPE", "MUL_MAT_ADD"}
	for _, op := range fusedOps {
		runner, ok := LookupRegistry(op)
		assert.True(t, ok, "missing registry entry for %s", op)
		assert.NotNil(t, runner.Run, "%s must have Run", op)
		assert.NotNil(t, runner.CreateInputs, "%s must have CreateInputs", op)
		assert.NotEmpty(t, runner.Dimensions, "%s must have Dimensions", op)
	}
}
```

- [ ] **Step 3: Run tests**

Run: `go test ./perf/ -run TestFusedOpRegistry -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add perf/registry.go perf/registry_test.go
git commit -m "perf: add fused op benchmark entries for RMS_NORM_MUL, MUL_MAT_ADD"
```

---

### Task 8: perf/bench.go — Orchestration Overhead Benchmark

**Files:**
- Modify: `perf/bench.go`

Measures CPU orchestration overhead as `f(num_nodes)` using a synthetic graph of N chained trivial ADD ops. GPU compute time is ~0, so wall-clock ≈ pure CPU overhead.

- [ ] **Step 1: Implement benchOrchestrationOverhead**

Add to `perf/bench.go`:

```go
// benchOrchestrationOverhead measures CPU orchestration overhead for different graph sizes.
// Builds N chained trivial ops (16-element ADD), GPU compute ≈ 0, wall-clock ≈ CPU overhead.
// Used by estimate to add CPU-side cost to GPU op time sum.
func benchOrchestrationOverhead(backend ml.Backend, cfg BenchmarkConfig) []LatencyPoint {
	graphSizes := []int{50, 100, 200, 300, 500}
	var points []LatencyPoint

	for _, n := range graphSizes {
		ctx := backend.NewContext()

		a := randomTensor(ctx, ml.DTypeF32, 16)
		b := randomTensor(ctx, ml.DTypeF32, 16)
		last := a.Add(ctx, b)
		for i := 1; i < n; i++ {
			last = last.Add(ctx, b) // Chain: each op depends on the previous output
		}
		ctx.Forward(last)

		// Warmup
		for range 3 {
			ctx.Compute(last)
		}

		// Measure wall-clock (GPU compute ≈ 0 for 16-element ADD)
		med, sd, reps := convergentMeasure(func() float64 {
			start := time.Now()
			ctx.Compute(last)
			return float64(time.Since(start).Microseconds())
		}, cfg)

		points = append(points, LatencyPoint{
			Shape:     []int64{int64(n)},
			LatencyUs: med,
			StddevUs:  sd,
			Reps:      reps,
		})

		ctx.Close()
		slog.Info("orchestration overhead", "nodes", n, "latency_us", fmt.Sprintf("%.0f", med))
	}

	return points
}
```

- [ ] **Step 2: Build to verify compilation**

Run: `go build ./perf/ 2>&1`
Expected: Successful build

- [ ] **Step 3: Commit**

```bash
git add perf/bench.go
git commit -m "perf: add orchestration overhead benchmark for CPU-side cost modeling"
```

---

### Task 9: perf/bench.go — RunBenchmark Integration

**Files:**
- Modify: `perf/bench.go:131-282`

Wires the new benchmark components into `RunBenchmark()`:
1. Discover backend capabilities
2. Use `measureOpForBackend()` (GPU timestamp path) instead of `measureOp()`
3. Benchmark fused ops when backend has fusion
4. Benchmark orchestration overhead for GPU backends
5. Store `BackendCaps` in profile
6. Bump version to 3

- [ ] **Step 1: Modify RunBenchmark to discover backend and set version 3**

At the top of `RunBenchmark()` (after hardware characterization, around line 141), add:

```go
	// Discover backend capabilities
	caps := DiscoverBackend(backend)
	slog.Info("backend capabilities", "name", caps.Name,
		"gpu_timestamp", caps.HasGPUTimestamp, "fusion_rules", len(caps.FusionRules),
		"mul_mat_vec", caps.HasMulMatVec)
```

Change the profile version from 2 to 3:

```go
	profile := &Profile{
		Version:   3,
		Timestamp: time.Now(),
		Hardware:  hwProfile,
		BackendCaps: map[string]BackendCapabilitiesJSON{
			caps.Name: caps.ToJSON(),
		},
	}
```

- [ ] **Step 2: Replace measureOp calls with measureOpForBackend**

In the benchmark loops where `measureOp` is called (inside `benchmarkElementwise`, `benchmarkMulMat`, etc.), replace the measurement function passed to `AdaptiveSample1D`. The measure function closure currently calls `measureOp` — change it to `measureOpForBackend`.

For example, in `benchmarkElementwise` (find the `AdaptiveSample1D` call):
```go
// Before:
measure := func(n int64) LatencyPoint {
    return measureOp(backend, op, []int64{n}, dtype, cfg)
}
// After:
measure := func(n int64) LatencyPoint {
    return measureOpForBackend(backend, caps, op, []int64{n}, dtype, cfg)
}
```

Apply the same change in `benchmarkMulMat` and `benchmarkFlashAttn`.

**Note:** `caps` must be passed to these functions. Either add `caps BackendCapabilities` as a parameter, or capture it from the `RunBenchmark` closure scope. The simplest approach: add `caps` as a parameter to `benchmarkElementwise`, `benchmarkMulMat`, `benchmarkFlashAttn`.

- [ ] **Step 3: Add fused op benchmarking to RunBenchmark**

After the existing operator loop (around line 276), add:

```go
	// Step 3: Benchmark fused ops (if backend supports fusion)
	if len(caps.FusionRules) > 0 {
		fusedOps := []string{"RMS_NORM_MUL", "RMS_NORM_MUL_ROPE", "MUL_MAT_ADD"}
		for _, fop := range fusedOps {
			if _, ok := LookupRegistry(fop); !ok {
				continue
			}
			slog.Info("benchmarking fused op", "op", fop)

			switch fop {
			case "MUL_MAT_ADD":
				// MUL_MAT_ADD uses same grid as MUL_MAT but only N≤8
				for _, wdt := range Phase1Dtypes() {
					points := benchmarkMulMat(backend, caps, wdt,
						map[string]int64{"M": 4096, "K": 4096}, cfg)
					// Filter to N≤8 points only
					var vecPoints []LatencyPoint
					for _, p := range points {
						if len(p.Shape) > 0 && p.Shape[0] <= 8 {
							vecPoints = append(vecPoints, p)
						}
					}
					if len(vecPoints) > 0 {
						profile.Operators = append(profile.Operators, OperatorCurve{
							Op:           fop,
							Backend:      caps.Name,
							ComputeDtype: "f32",
							WeightDtype:  wdt,
							Dimensions:   []string{"N"},
							FixedDims:    map[string]int64{"M": 4096, "K": 4096},
							Points:       vecPoints,
						})
					}
				}
			default:
				// 1D fused ops (RMS_NORM_MUL, RMS_NORM_MUL_ROPE)
				measure := func(n int64) LatencyPoint {
					return measureOpForBackend(backend, caps, fop, []int64{n}, "f32", cfg)
				}
				points := AdaptiveSample1D(measure, 1024, 8*1024*1024, 8, cfg)
				if len(points) > 0 {
					profile.Operators = append(profile.Operators, OperatorCurve{
						Op:           fop,
						Backend:      caps.Name,
						ComputeDtype: "f32",
						Dimensions:   []string{"N"},
						Points:       points,
					})
				}
			}
		}
	}
```

- [ ] **Step 4: Add orchestration overhead benchmarking**

After fused ops:

```go
	// Step 4: Benchmark orchestration overhead (GPU backends only)
	if caps.HasGPUTimestamp {
		slog.Info("benchmarking orchestration overhead")
		ohPoints := benchOrchestrationOverhead(backend, cfg)
		if len(ohPoints) > 0 {
			profile.Operators = append(profile.Operators, OperatorCurve{
				Op:           "ORCHESTRATION_OVERHEAD",
				Backend:      caps.Name,
				ComputeDtype: "f32",
				Dimensions:   []string{"num_nodes"},
				Points:       ohPoints,
			})
		}
	}
```

- [ ] **Step 5: Build to verify compilation**

Run: `go build ./perf/ 2>&1`
Expected: Successful build

- [ ] **Step 6: Commit**

```bash
git add perf/bench.go
git commit -m "perf: integrate GPU timestamps, fused ops, and overhead into RunBenchmark"
```

---

### Task 10: perf/estimate.go — MUL_MAT_VEC Routing, Fusion, and Overhead

**Files:**
- Modify: `perf/estimate.go:186-296`
- Modify: `perf/estimate_test.go`

The core estimate changes: (1) MUL_MAT routes to MUL_MAT_VEC when N≤maxN, (2) fusion is applied before per-op accumulation, (3) orchestration overhead is added.

- [ ] **Step 1: Write tests for MUL_MAT_VEC routing in lookupLatency**

Add to `perf/estimate_test.go`:

```go
func TestLookupLatencyMulMatVecRouting(t *testing.T) {
	profile := &Profile{
		Version: 3,
		Hardware: HardwareProfile{
			PeakTOPS:            map[string]float64{"f16": 55e9, "f32": 59e9},
			PeakBandwidthBytesPerSec: 37e9,
			BalancePoints:       map[string]float64{"f16": 1500, "f32": 1600},
			EfficiencyConstants: map[string]OpEfficiency{
				"f16": {ComputeEff: 0.8, BWEff: 0.5, OverheadUs: 100},
				// MUL_MAT_VEC has its own efficiency constants
				"MUL_MAT_VEC_f16": {ComputeEff: 0.5, BWEff: 0.7, OverheadUs: 10},
			},
		},
	}
	caps := GetBackendCapabilities("Vulkan")

	// N=1 should route to MUL_MAT_VEC
	latVec, err := lookupLatencyV3(profile, "MUL_MAT", []int64{1024, 2048, 1}, "f32", "f16", "Vulkan", &caps)
	assert.NoError(t, err)
	assert.Greater(t, latVec, 0.0)

	// N=512 should use regular MUL_MAT
	latMat, err := lookupLatencyV3(profile, "MUL_MAT", []int64{1024, 2048, 512}, "f32", "f16", "Vulkan", &caps)
	assert.NoError(t, err)
	assert.Greater(t, latMat, 0.0)

	// VEC at N=1 should be faster than MAT at N=512
	assert.Less(t, latVec, latMat)
}
```

- [ ] **Step 2: Write tests for fusion in estimatePhase**

```go
func TestEstimatePhaseFusion(t *testing.T) {
	profile := &Profile{
		Version: 3,
		Hardware: HardwareProfile{
			PeakTOPS:            map[string]float64{"f32": 59e9},
			PeakBandwidthBytesPerSec: 37e9,
		},
		Operators: []OperatorCurve{
			{
				Op: "RMS_NORM_MUL", Backend: "Vulkan", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 12},
					{Shape: []int64{4096}, LatencyUs: 15},
				},
			},
		},
	}
	caps := GetBackendCapabilities("Vulkan")

	// Graph with RMS_NORM + MUL should be fused
	nodes := []ml.GraphNode{
		{Op: "RMS_NORM", Backend: "Vulkan", ComputeDtype: "f32", Shape: [4]int64{2048, 1, 1, 1}},
		{Op: "MUL", Backend: "Vulkan", ComputeDtype: "f32", Shape: [4]int64{2048, 1, 1, 1}},
	}
	var warnings []string
	result := estimatePhaseV3(profile, nodes, &caps, &warnings)

	// Should produce a result (fused node looked up as RMS_NORM_MUL)
	assert.Greater(t, result.TotalLatencyMs, 0.0)
	// Should have 1 op in breakdown (fused), not 2
	assert.Len(t, result.TopOps, 1)
	assert.Equal(t, "RMS_NORM_MUL", result.TopOps[0].Op)
}
```

- [ ] **Step 3: Write test for orchestration overhead**

```go
func TestEstimatePhaseOrchestrationOverhead(t *testing.T) {
	profile := &Profile{
		Version: 3,
		Hardware: HardwareProfile{
			PeakTOPS:            map[string]float64{"f32": 59e9},
			PeakBandwidthBytesPerSec: 37e9,
		},
		Operators: []OperatorCurve{
			{
				Op: "ADD", Backend: "Vulkan", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 5},
					{Shape: []int64{4096}, LatencyUs: 8},
				},
			},
			{
				Op: "ORCHESTRATION_OVERHEAD", Backend: "Vulkan", ComputeDtype: "f32",
				Dimensions: []string{"num_nodes"},
				Points: []LatencyPoint{
					{Shape: []int64{50}, LatencyUs: 3000},
					{Shape: []int64{100}, LatencyUs: 5500},
					{Shape: []int64{300}, LatencyUs: 15000},
				},
			},
		},
	}
	caps := GetBackendCapabilities("Vulkan")

	// 10 ADD nodes
	nodes := make([]ml.GraphNode, 10)
	for i := range nodes {
		nodes[i] = ml.GraphNode{Op: "ADD", Backend: "Vulkan", ComputeDtype: "f32",
			Shape: [4]int64{2048, 1, 1, 1}}
	}

	var warnings []string
	result := estimatePhaseV3(profile, nodes, &caps, &warnings)

	// Total should include GPU time + orchestration overhead
	// GPU time: 10 * interpolate(2048) ≈ ~65μs
	// Overhead: interpolate(10 nodes) from overhead curve
	assert.Greater(t, result.TotalLatencyMs, 0.0)
}
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `go test ./perf/ -run "TestLookupLatencyMulMatVec|TestEstimatePhaseFusion|TestEstimatePhaseOrchestration" -v`
Expected: FAIL — `lookupLatencyV3` and `estimatePhaseV3` not defined

- [ ] **Step 5: Implement lookupLatencyV3 with MUL_MAT_VEC routing**

Add to `perf/estimate.go`:

```go
// lookupLatencyV3 extends lookupLatency with MUL_MAT_VEC routing based on backend caps.
func lookupLatencyV3(profile *Profile, op string, shape []int64,
	computeDtype, weightDtype, backend string, caps *BackendCapabilities) (float64, error) {

	switch op {
	case "MUL_MAT":
		if len(shape) < 3 {
			return 0, fmt.Errorf("MUL_MAT requires 3 shape dims, got %d", len(shape))
		}
		M, K, N := shape[0], shape[1], shape[2]
		mappedWdt := mapWeightDtype(weightDtype)

		// Route to MUL_MAT_VEC when N is small enough and backend supports it
		if caps != nil && caps.HasMulMatVec && N <= int64(caps.MulMatVecMaxN) {
			vecKey := "MUL_MAT_VEC_" + mappedWdt
			lat := PredictMulMatLatency(&profile.Hardware, M, K, N, vecKey)
			if lat > 0 {
				return lat, nil
			}
			// Fallback to regular MUL_MAT roofline if no VEC constants
		}
		lat := PredictMulMatLatency(&profile.Hardware, M, K, N, mappedWdt)
		if lat == 0 {
			return 0, fmt.Errorf("no efficiency constants for MUL_MAT — run daop-bench first")
		}
		return lat, nil

	default:
		// Delegate to existing lookupLatency for all other ops
		return lookupLatency(profile, op, shape, computeDtype, weightDtype, backend)
	}
}
```

- [ ] **Step 6: Implement estimatePhaseV3 with fusion and overhead**

Add to `perf/estimate.go`:

```go
// lookupOrchestrationOverhead queries the CPU overhead curve for a given graph size.
func lookupOrchestrationOverhead(profile *Profile, numNodes int, backend string) float64 {
	for _, c := range profile.Operators {
		if c.Op == "ORCHESTRATION_OVERHEAD" && c.Backend == backend {
			return Interpolate1D(c.Points, int64(numNodes))
		}
	}
	return 0
}

// estimatePhaseV3 computes total latency with fusion simulation and orchestration overhead.
func estimatePhaseV3(profile *Profile, nodes []ml.GraphNode, caps *BackendCapabilities, warnings *[]string) PhaseEstimation {
	// 1. Apply fusion
	var fusedNodes []ml.GraphNode
	if caps != nil {
		fusedNodes = ApplyFusion(nodes, caps.FusionRules)
	} else {
		fusedNodes = nodes
	}

	// 2. Sum per-op GPU time
	opStats := make(map[OpKey]*OpBreakdown)
	var totalGPUUs float64

	for _, fnode := range fusedNodes {
		if IsZeroCostOp(fnode.Op) {
			continue
		}
		op, shape, cdt, wdt := nodeToQueryShape(fnode)
		lat, err := lookupLatencyV3(profile, op, shape, cdt, wdt, fnode.Backend, caps)
		if err != nil {
			*warnings = append(*warnings, err.Error())
			continue
		}
		totalGPUUs += lat

		key := OpKey{op, fnode.Backend, cdt, wdt}
		if s, ok := opStats[key]; ok {
			s.Count++
			s.TotalUs += lat
		} else {
			opStats[key] = &OpBreakdown{
				Op: op, Backend: fnode.Backend,
				ComputeDtype: cdt, WeightDtype: wdt,
				Count: 1, TotalUs: lat,
			}
		}
	}

	// 3. Add orchestration overhead
	var overheadUs float64
	if caps != nil && caps.HasGPUTimestamp {
		overheadUs = lookupOrchestrationOverhead(profile, len(nodes), caps.Name)
	}
	totalUs := totalGPUUs + overheadUs

	// 4. Build top-ops breakdown
	var topOps []OpBreakdown
	for _, s := range opStats {
		if totalUs > 0 {
			s.Percentage = s.TotalUs / totalUs
		}
		topOps = append(topOps, *s)
	}
	sort.Slice(topOps, func(i, j int) bool { return topOps[i].TotalUs > topOps[j].TotalUs })
	if len(topOps) > 10 {
		topOps = topOps[:10]
	}

	tokPerSec := 0.0
	if totalUs > 0 {
		tokPerSec = 1e6 / totalUs
	}

	return PhaseEstimation{
		TotalLatencyMs: totalUs / 1000,
		TokensPerSec:   tokPerSec,
		TopOps:         topOps,
	}
}
```

- [ ] **Step 7: Update EstimateModel to use v3 path**

Modify `EstimateModel` in `perf/estimate.go` (lines 299-322):

```go
func EstimateModel(profile *Profile, modelPath string) (*EstimateResult, error) {
	schedule, err := discoverModelSchedule(modelPath)
	if err != nil {
		return nil, err
	}

	prefillNodes, decodeNodes, err := buildModelGraphNodes(modelPath, schedule)
	if err != nil {
		return nil, err
	}

	result := &EstimateResult{Model: modelPath}

	// Use v3 path if backend caps are available, otherwise fall back to v2
	if profile.Version >= 3 && len(profile.BackendCaps) > 0 {
		// Find the primary backend caps
		var caps *BackendCapabilities
		for name := range profile.BackendCaps {
			c := GetBackendCapabilities(name)
			caps = &c
			break
		}
		result.Prefill = estimatePhaseV3(profile, prefillNodes, caps, &result.Warnings)
		result.Decode = estimatePhaseV3(profile, decodeNodes, caps, &result.Warnings)
	} else {
		// Legacy v2 path
		result.Prefill = estimatePhase(profile, prefillNodes, &result.Warnings)
		result.Decode = estimatePhase(profile, decodeNodes, &result.Warnings)
	}

	result.PrefillLatencyUs = result.Prefill.TotalLatencyMs * 1000
	result.PrefillMs = result.Prefill.TotalLatencyMs
	if result.Decode.TotalLatencyMs > 0 {
		result.DecodeLatencyUsPerToken = result.Decode.TotalLatencyMs * 1000
		result.DecodeTokensPerSec = 1e6 / result.DecodeLatencyUsPerToken
	}

	return result, nil
}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `go test ./perf/ -run "TestLookupLatencyMulMatVec|TestEstimatePhaseFusion|TestEstimatePhaseOrchestration" -v`
Expected: PASS

- [ ] **Step 9: Run full test suite**

Run: `go test ./perf/ -v 2>&1 | tail -30`
Expected: All existing tests still pass (v2 path unchanged)

- [ ] **Step 10: Commit**

```bash
git add perf/estimate.go perf/estimate_test.go
git commit -m "perf: add MUL_MAT_VEC routing, fusion simulation, and overhead to estimate"
```

---

## Phase 3: End-to-End Validation

### Task 11: Validation — GPU Timestamp vs GGML_VK_PERF_LOGGER

**Files:** No new files — manual validation

This task validates the GPU timestamp C API against the existing `GGML_VK_PERF_LOGGER=1` stderr output to confirm they produce consistent results.

- [ ] **Step 1: Run benchmark with new code**

```bash
go run . daop-bench
```

Verify:
- Profile version is 3
- New operator curves appear: `ORCHESTRATION_OVERHEAD`, `RMS_NORM_MUL`, `RMS_NORM_MUL_ROPE`, `MUL_MAT_ADD`
- `BackendCaps` section is present in `profile.json`
- If Vulkan is available, GPU timestamp path was used (check log output)

- [ ] **Step 2: Compare GPU timestamp benchmark with GGML_VK_PERF_LOGGER**

```bash
# Run a small model with perf logger
GGML_VK_PERF_LOGGER=1 go run . run qwen3:1.7b --verbose 2>&1 | grep "MUL_MAT_VEC"
```

Compare the per-op GPU times from our benchmark with the perf logger values. They should be in the same order of magnitude (within 2x).

- [ ] **Step 3: Run estimate and compare**

```bash
go run . daop-estimate qwen3:1.7b
```

Compare estimate result with actual inference:
- Actual: ~75 ms/tok (13.3 tok/s)
- Target: estimate within 2x → 37-150 ms/tok (6.7-27 tok/s)

- [ ] **Step 4: Tune if needed**

If estimate error > 2x, investigate:
1. Check top-ops breakdown — which ops contribute most error?
2. If fusion rules miss patterns, add rules to `VulkanFusionRules()`
3. If orchestration overhead model is off, adjust curve or add more sample points
4. If MUL_MAT_VEC efficiency constants are wrong, re-check GPU timestamp data

- [ ] **Step 5: Commit final state**

```bash
git add -A
git commit -m "perf: complete accuracy redesign validation"
```

---

## Self-Review Notes

### Bugs to fix during implementation

1. **[BUG] Task 4 — CGO Vulkan include 跨平台问题**: `ggml.go` 不能直接 `#include "ggml-vulkan.h"` — 无 Vulkan 平台会 link 失败。修正方案：用 build tag 隔离 Vulkan CGO 到单独文件（如 `ggml_vulkan.go` with `//go:build vulkan`），或将 timestamp API 提升到 `ggml-backend.h` 层面。对我们的 fork（总是编译 Vulkan）可以先直接 include。

2. **[BUG] Task 9 Step 3 — MUL_MAT_ADD benchmark 逻辑**: 当前代码调用 `benchmarkMulMat()` 测量纯 MUL_MAT 然后过滤 N≤8。这测的是 MUL_MAT 不是融合的 MUL_MAT+ADD。**修正**：改为直接调用 `measureOpForBackend(backend, caps, "MUL_MAT_ADD", []int64{M, K, N}, wdt, cfg)` 用 registry entry 创建融合图。或 MVP 更简单方案：不单独 benchmark MUL_MAT_ADD，estimate 中 MUL_MAT_ADD → 复用 MUL_MAT_VEC 效率常量（ADD 开销 <1μs，可忽略）。

3. **[BUG] Task 6 — `trimmedStats` 函数不存在**: `measureOpGPU` 引用了 `trimmedStats(samples, cfg.TrimPercent)` 但这个函数在现有代码中不存在。可能叫 `trimmedMedian` 或内联在 `convergentMeasure` 中。实现时需要先 grep 确认现有函数名，可能需要提取为独立 helper。

### Gaps and clarifications

4. **MUL_MAT_VEC efficiency constants extraction** (Task 9): 当用 GPU timestamp benchmark MUL_MAT reference curve 时，N≤8 的点反映 MUL_MAT_VEC kernel 性能。实现时必须**在 N=8 处分割曲线**，对两个子集分别调用 `extractEfficiencyConstants`。存储为 `"MUL_MAT_VEC_f16"` 等。

5. **MUL_MAT_ADD shape extraction** (Task 10): 融合后 `MUL_MAT_ADD` 节点保留 MUL_MAT 的 `InputShapes`。`nodeToQueryShape` 需要处理 `"MUL_MAT_ADD"` — 和 `"MUL_MAT"` 一样提取 M, K, N。`lookupLatencyV3` 也需要新增 `case "MUL_MAT_ADD"` 路由到 MUL_MAT_VEC 效率常量。

6. **GPU timestamp driver fallback**: 如果 `GetOpTimings()` 返回 nil，`measureOpGPU` 应 fallback 到 wall-clock 测量并 warn。Task 6 部分处理了（warning），但应显式回退到 `measureOp` 返回值。

7. **Scope**: Vulkan only。CPU 现有方案不变。CUDA 推迟（spec Section 1.2）。

8. **Profile version compatibility**: v3 向后兼容 v2。`EstimateModel` 根据 profile version 自动选择 v2 或 v3 路径。
