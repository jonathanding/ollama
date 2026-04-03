# DAOP v2: Empirical Performance Estimation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Roofline-based performance estimator (v1) with an empirical latency model that directly measures `latency = f(op, shape, dtype)` via log-space piecewise linear interpolation, for three representative operators (SILU, MUL_MAT, FLASH_ATTN_EXT).

**Architecture:** Benchmark runner measures operator latency at log-spaced shapes with adaptive refinement. Results are stored as `OperatorCurve` objects (each a 1D function of a sweep dimension, with fixed dimensions captured in `FixedDims`). Estimation loads a model's computation graph, maps each node to the nearest calibrated curve via log-space interpolation, and sums latencies. An HTML viewer visualizes the curves.

**Tech Stack:** Go 1.24, GGML backends via `ml/backend.go` interfaces, `github.com/stretchr/testify` for tests, Plotly.js (CDN) for HTML visualization, `//go:embed` for HTML template.

**Spec:** `docs/superpowers/specs/2026-04-03-daop-v2-empirical-design.md`
**Design doc:** `docs/daop/design.md`

---

## CRITICAL NOTES FOR IMPLEMENTERS

> **READ THESE BEFORE STARTING ANY TASK.** These are cross-cutting concerns that affect every task.

### 1. DType Mapping
The `ml` package uses `ml.DType` constants, NOT strings. Phase 1 supported dtypes and their constants:
```go
// ml/backend.go constants:
// ml.DTypeF32, ml.DTypeF16, ml.DTypeQ80, ml.DTypeQ40, ml.DTypeI32, ml.DTypeMXFP4

// String-to-DType mapping needed throughout:
func parseDType(s string) (ml.DType, bool) {
    switch s {
    case "f32":  return ml.DTypeF32, true
    case "f16":  return ml.DTypeF16, true
    case "q4_0": return ml.DTypeQ40, true
    case "q8_0": return ml.DTypeQ80, true
    default:     return 0, false
    }
}
```
Other quantized types (q4_K, q5_K, q6_K, etc.) do NOT have `ml.DType` constants — they cannot be created via `ctx.Zeros()`. Phase 1 benchmarks only f32, f16, q4_0, q8_0.

### 2. Tensor Creation
All tensors are created via `ctx.Zeros(dtype, shape...)` where shape params are `int` (not `int64`). Cast explicitly: `ctx.Zeros(dt, int(M), int(K))`.

### 3. SILU Signature
`SILU(ctx Context, up ...Tensor) Tensor` — the `up` parameter is variadic and optional. For benchmarking, call `in[0].SILU(ctx)` with no extra args.

### 4. FLASH_ATTN_EXT via ScaledDotProductAttention
The `ml.Tensor` interface does NOT include SDPA. You must type-assert:
```go
sdpa, ok := in[0].(ml.ScaledDotProductAttention)
```
The `ScaledDotProductAttention` interface signature:
```go
ScaledDotProductAttention(ctx Context, key, value, mask, sinks Tensor, vmla Tensor, scale float64, cacheConfigApplied bool) Tensor
```
For benchmarking: `sdpa.ScaledDotProductAttention(ctx, in[1], in[2], nil, nil, nil, scale, false)`.

### 5. Context.Forward() and Reserve()
`ctx.Forward(tensor)` returns a `Context` (chainable). `Reserve()` is called on that returned context:
```go
ctx.Forward(t).Reserve()   // NOT ctx.Forward(t); ctx.Reserve()
```
`ctx.GraphNodes()` returns `[]ml.GraphNode` — the captured computation graph.

### 6. BackendParams.AllocMemory
Set `AllocMemory: false` when you only need the computation graph (no actual tensor memory). This is used in `buildModelGraphNodes` to avoid loading multi-GB model weights.

### 7. GraphNode Fields
```go
type GraphNode struct {
    Op           string      // "MUL_MAT", "SILU", etc.
    Name         string      // tensor name
    Backend      string      // "cuda", "cpu", etc.
    Shape        [4]int64    // output tensor shape (GGML column-major: ne[0..3])
    ComputeDtype string      // "f32", "f16", etc. (string, not ml.DType)
    WeightDtype  string      // for MUL_MAT: "q4_0", etc.
    InputShapes  [][]int64   // shapes of input tensors
}
```

### 8. Test Patterns
This project uses `github.com/stretchr/testify` (`assert`, `require`). Follow TDD strictly:
- Write the **failing test first** (Red)
- Write **minimal code** to pass (Green)
- Refactor
- Tests must cover: happy path, edge cases, error cases, boundary conditions
- Use `t.TempDir()` for file I/O tests
- Integration tests requiring GGML backend: use build tag `// go:build integration` or skip with runtime check

### 9. Existing Code Context
The `perf/` package exists with v1 code. You are REWRITING it. Key files to understand:
- `perf/types.go` — v1 data structures (will be replaced)
- `perf/ops.go` — `IsZeroCostOp()`, `elemSize()`, `product()` are KEPT; `ComputeFLOPs()`, `ComputeBytes()`, `CanComputeFLOPs()` are REMOVED
- `perf/roofline.go` — DELETED entirely (LookupBackend logic moves to profile.go)
- `perf/bench.go` — REWRITTEN (keep `benchPeakFLOPS`, `benchPeakBandwidth` concept but move to hwchar.go)
- `perf/profile.go` — REWRITTEN for v2 format
- `perf/estimate.go` — REWRITTEN for curve lookup
- `perf/viewer.go` — UPDATED for v2 types
- `perf/resolve.go` — KEPT as-is

### 10. Import Paths
```go
import (
    "github.com/ollama/ollama/ml"
    "github.com/ollama/ollama/model"
    "github.com/ollama/ollama/model/input"
)
```

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `perf/types.go` | **Rewrite** | v2 data structures: Profile, HardwareProfile, OperatorCurve, LatencyPoint, OpRunner, SamplingGrid, EstimateResult, PhaseEstimation, OpBreakdown, OpKey, BenchmarkConfig |
| `perf/ops.go` | **Trim** | Keep: `IsZeroCostOp()`, `elemSize()`, `product()`. Remove: `ComputeFLOPs()`, `ComputeBytes()`, `CanComputeFLOPs()` |
| `perf/registry.go` | **Create** | Operator registry (`opRegistry` map), `expandShapes()`, `parseDType()`, `dtypeToString()` |
| `perf/interpolate.go` | **Create** | `Interpolate1D`, `Interpolate1DByDim`, `InterpolateMulMat`, `InterpolateFlashAttn`, `extrapolateLeft`, `extrapolateRight` |
| `perf/adaptive.go` | **Create** | `adaptiveSample1D()`, `findMaxInterpolationError()`, `logMidpoint()`, `insertSorted()` |
| `perf/hwchar.go` | **Create** | `CharacterizeHardware()`, `benchPeakTOPS()`, `benchPeakBandwidth()` (moved from bench.go) |
| `perf/bench.go` | **Rewrite** | `RunBenchmark()` using registry + adaptive sampling, `measureOp()`, `benchmarkOp()` |
| `perf/profile.go` | **Rewrite** | `LoadProfile()`, `WriteProfile()`, `MergeProfile()`, `ProfilePath()`, `BenchDir()`, `LookupBackend()` |
| `perf/estimate.go` | **Rewrite** | `EstimateModel()`, `estimatePhase()`, `lookupLatency()`, `nodeToQueryShape()`, `buildModelGraphNodes()` |
| `perf/viewer.go` | **Update** | CLI viewer updated for v2 types (OperatorCurve instead of OperatorProfile) |
| `perf/viewer_html.go` | **Create** | `GenerateHTMLViewer()` — generates self-contained HTML with Plotly.js |
| `perf/viewer.html` | **Create** | HTML template (embedded via `//go:embed`) |
| `perf/resolve.go` | **Keep** | No changes (model path resolution) |
| `perf/roofline.go` | **Delete** | Entire file removed |
| | | |
| **Test files** | | |
| `perf/types_test.go` | **Create** | Profile/OperatorCurve JSON round-trip, FixedDims serialization |
| `perf/ops_test.go` | **Rewrite** | Keep IsZeroCostOp tests, remove ComputeFLOPs/ComputeBytes tests |
| `perf/registry_test.go` | **Create** | Registry completeness, expandShapes correctness, parseDType |
| `perf/interpolate_test.go` | **Create** | Comprehensive interpolation tests (most critical test file) |
| `perf/adaptive_test.go` | **Create** | Adaptive sampling with mock measurement |
| `perf/hwchar_test.go` | **Create** | Integration test with real GGML backend |
| `perf/bench_test.go` | **Rewrite** | Benchmark runner tests |
| `perf/profile_test.go` | **Rewrite** | v2 profile round-trip |
| `perf/estimate_test.go` | **Rewrite** | Estimation pipeline tests |
| `perf/viewer_html_test.go` | **Create** | HTML generation tests |
| `perf/integration_test.go` | **Rewrite** | End-to-end with real GGML backend |

---

## Task 1: v2 Data Structures (`types.go`)

**Files:**
- Rewrite: `perf/types.go`
- Create: `perf/types_test.go`

This task replaces ALL v1 data structures with v2 equivalents. Every subsequent task depends on this.

- [ ] **Step 1: Write failing tests for v2 types**

Create `perf/types_test.go` with JSON round-trip tests:

```go
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
		// FixedDims is nil — should be omitted from JSON
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

func TestBenchmarkConfigDefaults(t *testing.T) {
	cfg := DefaultBenchmarkConfig()
	assert.Equal(t, 5, cfg.WarmupReps)
	assert.Equal(t, 100, cfg.MeasureReps)
	assert.InDelta(t, 0.10, cfg.TrimPercent, 0.001)
	assert.InDelta(t, 0.05, cfg.ErrorThreshold, 0.001)
	assert.Equal(t, 20, cfg.MaxPointsPerOp)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./perf/ -run "TestProfileJSON|TestOperatorCurve|TestLatencyPoint|TestEstimateResult|TestBenchmarkConfig" -v`

Expected: compilation errors — v2 types don't exist yet.

- [ ] **Step 3: Rewrite `types.go` with v2 data structures**

Replace the entire contents of `perf/types.go` with:

```go
package perf

import "time"

// --- Profile (v2): empirical latency curves ---

// Profile stores calibrated operator latency curves for a specific hardware configuration.
// Version 2 replaces v1's Roofline-based eta coefficients with direct latency measurements.
type Profile struct {
	Version   int              `json:"version"`   // 2
	Timestamp time.Time        `json:"timestamp"`
	Hardware  HardwareProfile  `json:"hardware"`
	Operators []OperatorCurve  `json:"operators"`
}

// HardwareProfile captures hardware characteristics used for initial sampling grid placement.
type HardwareProfile struct {
	Backends                  []BackendInfo      `json:"backends"`
	PeakTOPS                  map[string]float64 `json:"peak_tops"`                    // dtype -> TOPS
	PeakBandwidthBytesPerSec  float64            `json:"peak_bandwidth_bytes_sec"`
	InterconnectBWBytesPerSec float64            `json:"interconnect_bandwidth_bytes_sec"` // Phase 2
	BalancePoints             map[string]float64 `json:"balance_points"`               // dtype -> FLOPs/byte
}

// BackendInfo identifies a compute backend (GPU, CPU, etc.).
type BackendInfo struct {
	Name      string `json:"name"`
	Device    string `json:"device"`
	VRAMBytes int64  `json:"vram_bytes"`
}

// OperatorCurve stores measured latency points for one operator configuration.
//
// For 1D ops (SILU, ADD, etc.): Dimensions=["N"], FixedDims=nil.
// For MUL_MAT: each (M, K, weight_dtype) is a SEPARATE OperatorCurve with
//   FixedDims={"M": m, "K": k} and Dimensions=["N"].
// For FLASH_ATTN_EXT: FixedDims={"num_heads": 32, "head_dim": 128},
//   Dimensions=["seq_q", "seq_kv"]. Points contain two regimes:
//   decode (seq_q=1, sweep seq_kv) and prefill (seq_q=seq_kv, sweep both).
type OperatorCurve struct {
	Op           string           `json:"op"`
	Backend      string           `json:"backend"`
	ComputeDtype string           `json:"compute_dtype"`
	WeightDtype  string           `json:"weight_dtype,omitempty"`
	Dimensions   []string         `json:"dimensions"`
	FixedDims    map[string]int64 `json:"fixed_dims,omitempty"`
	Points       []LatencyPoint   `json:"points"`
}

// LatencyPoint is one measured (shape, latency) pair.
// Shape values correspond to the OperatorCurve.Dimensions order.
type LatencyPoint struct {
	Shape     []int64 `json:"shape"`
	LatencyUs float64 `json:"latency_us"`
	StddevUs  float64 `json:"stddev_us"`
	Reps      int     `json:"reps"`
}

// --- Operator Registry ---

// OpRunner defines how to benchmark an operator.
// NumInputs: how many input tensors the op requires.
// Dimensions: ALL performance-relevant shape dimensions.
//   When creating OperatorCurves, some dims become FixedDims and the rest
//   become OperatorCurve.Dimensions (sweep dims).
//   e.g., MUL_MAT OpRunner.Dimensions=["M","K","N"] ->
//     OperatorCurve{FixedDims={"M":4096,"K":4096}, Dimensions=["N"]}
// Run: invokes the operator on given inputs, returns output tensor.
type OpRunner struct {
	NumInputs  int
	Dimensions []string
	Run        func(ctx interface{}, inputs []interface{}) interface{}
	// Note: Run is typed as interface{} here because types.go does not import ml.
	// The ACTUAL registry type is OpRunnerML in registry.go.
	// This OpRunner type is NOT used at runtime — it exists only for documentation.
	// All runtime code uses OpRunnerML directly.
}

// SamplingGrid defines the points to benchmark for one operator.
type SamplingGrid struct {
	Op          string
	Dtype       string
	WeightDtype string
	Dimensions  []string
	Points      [][]int64
}

// --- Estimation output ---

// EstimateResult is the output of EstimateModel().
type EstimateResult struct {
	Model                   string          `json:"model"`
	PrefillLatencyUs        float64         `json:"prefill_latency_us"`
	PrefillMs               float64         `json:"prefill_ms"`
	DecodeLatencyUsPerToken float64         `json:"decode_latency_us_per_token"`
	DecodeTokensPerSec      float64         `json:"decode_tokens_per_sec"`
	Prefill                 PhaseEstimation `json:"prefill"`
	Decode                  PhaseEstimation `json:"decode"`
	Warnings                []string        `json:"warnings,omitempty"`
}

// PhaseEstimation breaks down latency for one inference phase (prefill or decode).
type PhaseEstimation struct {
	TotalLatencyMs float64       `json:"total_latency_ms"`
	TokensPerSec   float64       `json:"tokens_per_sec"`
	TopOps         []OpBreakdown `json:"top_ops"`
}

// OpBreakdown shows the contribution of one operator type to total latency.
type OpBreakdown struct {
	Op           string  `json:"op"`
	Backend      string  `json:"backend"`
	ComputeDtype string  `json:"compute_dtype"`
	WeightDtype  string  `json:"weight_dtype,omitempty"`
	Count        int     `json:"count"`
	TotalUs      float64 `json:"total_us"`
	Percentage   float64 `json:"percentage"`
}

// --- Internal helper types ---

// OpKey uniquely identifies an operator configuration.
type OpKey struct {
	Op           string
	Backend      string
	ComputeDtype string
	WeightDtype  string
}

// --- Benchmark configuration ---

// BenchmarkConfig controls benchmark behavior.
type BenchmarkConfig struct {
	WarmupReps     int
	MeasureReps    int
	TrimPercent    float64 // fraction of outliers to trim (0.1 = 10%)
	ErrorThreshold float64 // adaptive sampling convergence (0.05 = 5%)
	MaxPointsPerOp int     // budget limit per (op, dtype)
}

// DefaultBenchmarkConfig returns sensible defaults.
func DefaultBenchmarkConfig() BenchmarkConfig {
	return BenchmarkConfig{
		WarmupReps:     5,
		MeasureReps:    100,
		TrimPercent:    0.10,
		ErrorThreshold: 0.05,
		MaxPointsPerOp: 20,
	}
}

// HWCharResult holds hardware characterization results.
type HWCharResult struct {
	PeakTOPS     map[string]float64 // dtype -> TOPS
	PeakBW       float64            // bytes/sec
	BalancePoint map[string]float64 // dtype -> FLOPs/byte
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./perf/ -run "TestProfileJSON|TestOperatorCurve|TestLatencyPoint|TestEstimateResult|TestBenchmarkConfig" -v`

Expected: All 7 tests PASS. Some other tests in perf/ may fail due to removed v1 types — that is expected and will be fixed in subsequent tasks.

- [ ] **Step 5: Commit**

```bash
git add perf/types.go perf/types_test.go
git commit -m "perf: rewrite types.go with v2 empirical data structures"
```

---

## Task 2: Trim `ops.go` — Remove Roofline Functions

**Files:**
- Modify: `perf/ops.go` (remove `ComputeFLOPs`, `ComputeBytes`, `CanComputeFLOPs`)
- Rewrite: `perf/ops_test.go` (remove tests for removed functions, keep rest)

v2 no longer needs FLOPs/bytes computation — latency is measured directly. Keep only the utilities needed by the rest of the codebase.

- [ ] **Step 1: Rewrite `ops_test.go` for remaining functions**

Replace `perf/ops_test.go` with tests for the kept functions only:

```go
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
```

- [ ] **Step 2: Run tests to verify they pass with current code**

Run: `go test ./perf/ -run "TestIsZeroCostOp|TestElemSize|TestProduct" -v`

Expected: PASS (these functions still exist in ops.go).

- [ ] **Step 3: Trim `ops.go` — remove Roofline functions**

Replace `perf/ops.go` with only the kept functions:

```go
package perf

// elemSize returns the bytes per element for a dtype string.
func elemSize(dtype string) float64 {
	switch dtype {
	case "f32":
		return 4
	case "f16":
		return 2
	case "bf16":
		return 2
	case "i8", "int8":
		return 1
	case "i32", "int32":
		return 4
	case "q4_0":
		return 0.5625
	case "q4_1":
		return 0.625
	case "q5_0":
		return 0.6875
	case "q5_1":
		return 0.75
	case "q8_0":
		return 1.0625
	case "q4_K":
		return 0.5625
	case "q5_K":
		return 0.6875
	case "q6_K":
		return 0.8125
	case "q3_K":
		return 0.4375
	case "iq4_nl":
		return 0.5625
	default:
		return 4
	}
}

// product returns the product of all elements in a shape slice.
func product(shape []int64) float64 {
	p := float64(1)
	for _, v := range shape {
		p *= float64(v)
	}
	return p
}

// IsZeroCostOp returns true for ops that don't consume compute time
// (metadata-only operations like view, reshape, permute).
func IsZeroCostOp(op string) bool {
	switch op {
	case "VIEW", "RESHAPE", "PERMUTE":
		return true
	default:
		return false
	}
}
```

- [ ] **Step 4: Run all perf tests to see what breaks**

Run: `go test ./perf/ -run "TestIsZeroCostOp|TestElemSize|TestProduct" -v`

Expected: PASS. Other test files will have compilation errors because they reference removed types/functions — those will be fixed in subsequent tasks.

- [ ] **Step 5: Commit**

```bash
git add perf/ops.go perf/ops_test.go
git commit -m "perf: trim ops.go to keep only IsZeroCostOp, elemSize, product"
```

---

## Task 3: Operator Registry (`registry.go`)

**Files:**
- Create: `perf/registry.go`
- Create: `perf/registry_test.go`

The operator registry maps GGML op names to benchmark functions. It defines HOW to create tensors and invoke each op, and HOW to expand grid dimensions into tensor shapes.

- [ ] **Step 1: Write failing tests for registry**

Create `perf/registry_test.go`:

```go
package perf

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRegistryContainsPhase1Ops(t *testing.T) {
	required := []string{"SILU", "MUL_MAT", "FLASH_ATTN_EXT"}
	for _, op := range required {
		t.Run(op, func(t *testing.T) {
			runner, ok := opRegistry[op]
			assert.True(t, ok, "op %q must be in registry", op)
			assert.Greater(t, runner.NumInputs, 0)
			assert.NotEmpty(t, runner.Dimensions)
		})
	}
}

func TestRegistryDimensions(t *testing.T) {
	tests := []struct {
		op   string
		dims []string
	}{
		{"SILU", []string{"N"}},
		{"MUL_MAT", []string{"M", "K", "N"}},
		{"FLASH_ATTN_EXT", []string{"seq_q", "seq_kv"}},
	}
	for _, tt := range tests {
		t.Run(tt.op, func(t *testing.T) {
			runner := opRegistry[tt.op]
			assert.Equal(t, tt.dims, runner.Dimensions)
		})
	}
}

func TestRegistryNumInputs(t *testing.T) {
	tests := []struct {
		op   string
		want int
	}{
		{"SILU", 1},
		{"MUL_MAT", 2},
		{"FLASH_ATTN_EXT", 3},
	}
	for _, tt := range tests {
		t.Run(tt.op, func(t *testing.T) {
			assert.Equal(t, tt.want, opRegistry[tt.op].NumInputs)
		})
	}
}

func TestExpandShapes_SILU(t *testing.T) {
	shapes := expandShapes("SILU", []int64{65536})
	require.Len(t, shapes, 1, "SILU needs 1 input tensor")
	assert.Equal(t, []int64{65536}, shapes[0])
}

func TestExpandShapes_MulMat(t *testing.T) {
	shapes := expandShapes("MUL_MAT", []int64{4096, 4096, 32})
	require.Len(t, shapes, 2, "MUL_MAT needs 2 input tensors")
	assert.Equal(t, []int64{4096, 4096}, shapes[0]) // weight [M, K]
	assert.Equal(t, []int64{4096, 32}, shapes[1])    // activation [K, N]
}

func TestExpandShapes_FlashAttn(t *testing.T) {
	shapes := expandShapes("FLASH_ATTN_EXT", []int64{1, 2048})
	require.Len(t, shapes, 3, "FLASH_ATTN_EXT needs 3 input tensors (Q, K, V)")
	// Q: [head_dim, num_heads, seq_q, 1]
	assert.Equal(t, []int64{128, 32, 1, 1}, shapes[0])
	// K: [head_dim, num_heads, seq_kv, 1]
	assert.Equal(t, []int64{128, 32, 2048, 1}, shapes[1])
	// V: [head_dim, num_heads, seq_kv, 1]
	assert.Equal(t, []int64{128, 32, 2048, 1}, shapes[2])
}

func TestExpandShapes_FlashAttn_Prefill(t *testing.T) {
	shapes := expandShapes("FLASH_ATTN_EXT", []int64{512, 512})
	assert.Equal(t, int64(512), shapes[0][2]) // seq_q
	assert.Equal(t, int64(512), shapes[1][2]) // seq_kv
	assert.Equal(t, int64(512), shapes[2][2]) // V seq_kv
}

func TestParseDType(t *testing.T) {
	tests := []struct {
		input string
		ok    bool
	}{
		{"f32", true},
		{"f16", true},
		{"q4_0", true},
		{"q8_0", true},
		{"bf16", false},  // not supported in Phase 1
		{"q4_K", false},  // no ml.DType constant
		{"", false},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			_, ok := parseDType(tt.input)
			assert.Equal(t, tt.ok, ok)
		})
	}
}

func TestDtypeToString(t *testing.T) {
	for _, s := range []string{"f32", "f16", "q4_0", "q8_0"} {
		dt, ok := parseDType(s)
		require.True(t, ok)
		assert.Equal(t, s, dtypeToString(dt))
	}
}

func TestPhase1Dtypes(t *testing.T) {
	dtypes := Phase1Dtypes()
	assert.Contains(t, dtypes, "f32")
	assert.Contains(t, dtypes, "f16")
	assert.Contains(t, dtypes, "q4_0")
	assert.Contains(t, dtypes, "q8_0")
}

func TestPhase1MulMatShapePairs(t *testing.T) {
	pairs := Phase1MulMatFixedDims()
	assert.NotEmpty(t, pairs)
	// Should include the standard 4096x4096 pair
	found := false
	for _, p := range pairs {
		if p[0] == 4096 && p[1] == 4096 {
			found = true
			break
		}
	}
	assert.True(t, found, "must include (4096, 4096) pair")
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./perf/ -run "TestRegistry|TestExpandShapes|TestParseDType|TestDtypeToString|TestPhase1" -v`

Expected: compilation errors — registry.go doesn't exist yet.

- [ ] **Step 3: Implement `registry.go`**

Create `perf/registry.go`:

```go
package perf

import (
	"math"

	"github.com/ollama/ollama/ml"
)

// OpRunnerML is the concrete registry entry type using ml package types.
// OpRunner in types.go uses interface{} to avoid circular imports;
// this is the real implementation.
type OpRunnerML struct {
	NumInputs  int
	Dimensions []string
	Run        func(ctx ml.Context, inputs []ml.Tensor) ml.Tensor
}

// opRegistry maps GGML op names to their benchmark definitions.
// To add a new operator:
//  1. Add an entry: "OP_NAME": {NumInputs, Dimensions, RunFunc}
//  2. Add shape expansion in expandShapes() if not 1D
//  3. Add tests in registry_test.go
var opRegistry = map[string]OpRunnerML{
	"SILU": {
		NumInputs:  1,
		Dimensions: []string{"N"},
		Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
			return in[0].SILU(ctx)
		},
	},
	"MUL_MAT": {
		NumInputs:  2,
		Dimensions: []string{"M", "K", "N"},
		Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
			return in[0].Mulmat(ctx, in[1])
		},
	},
	"FLASH_ATTN_EXT": {
		NumInputs:  3,
		Dimensions: []string{"seq_q", "seq_kv"},
		Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
			sdpa, ok := in[0].(ml.ScaledDotProductAttention)
			if !ok {
				return nil
			}
			// Scale = 1/sqrt(head_dim). head_dim is always in[0].Shape()[0].
			headDim := in[0].Shape()[0]
			scale := 1.0 / math.Sqrt(float64(headDim))
			return sdpa.ScaledDotProductAttention(ctx, in[1], in[2], nil, nil, nil, scale, false)
		},
	},
}

// expandShapes converts grid dimensions to full tensor shapes per op.
// gridPoint contains values for OpRunner.Dimensions in order.
func expandShapes(op string, gridPoint []int64) [][]int64 {
	switch op {
	case "FLASH_ATTN_EXT":
		// gridPoint = [seq_q, seq_kv], fixed head_dim=128, num_heads=32
		seqQ, seqKV := gridPoint[0], gridPoint[1]
		return [][]int64{
			{128, 32, seqQ, 1},  // Q
			{128, 32, seqKV, 1}, // K
			{128, 32, seqKV, 1}, // V
		}
	case "MUL_MAT":
		// gridPoint = [M, K, N]
		M, K, N := gridPoint[0], gridPoint[1], gridPoint[2]
		return [][]int64{
			{M, K}, // weight
			{K, N}, // activation
		}
	default:
		// 1D ops: gridPoint = [N]
		return [][]int64{gridPoint}
	}
}

// parseDType converts a string dtype to ml.DType.
// Returns (dtype, true) if supported, (0, false) otherwise.
func parseDType(s string) (ml.DType, bool) {
	switch s {
	case "f32":
		return ml.DTypeF32, true
	case "f16":
		return ml.DTypeF16, true
	case "q4_0":
		return ml.DTypeQ40, true
	case "q8_0":
		return ml.DTypeQ80, true
	default:
		return 0, false
	}
}

// dtypeToString converts ml.DType to its string name.
func dtypeToString(dt ml.DType) string {
	switch dt {
	case ml.DTypeF32:
		return "f32"
	case ml.DTypeF16:
		return "f16"
	case ml.DTypeQ40:
		return "q4_0"
	case ml.DTypeQ80:
		return "q8_0"
	default:
		return "unknown"
	}
}

// Phase1Dtypes returns the dtypes supported in Phase 1 benchmarks.
func Phase1Dtypes() []string {
	return []string{"f32", "f16", "q4_0", "q8_0"}
}

// Phase1MulMatFixedDims returns representative (M, K) pairs for MUL_MAT benchmarks.
// These cover common transformer architectures (Llama 7B/8B, 13B, 70B).
func Phase1MulMatFixedDims() [][2]int64 {
	return [][2]int64{
		{4096, 4096},   // Llama-7B/8B: hidden_dim
		{14336, 4096},  // Llama-8B: FFN up/gate (intermediate_size × hidden_size)
		{4096, 14336},  // Llama-8B: FFN down (hidden_size × intermediate_size)
		{8192, 8192},   // Llama-70B: hidden_dim
		{28672, 8192},  // Llama-70B: FFN up/gate
		{8192, 28672},  // Llama-70B: FFN down
	}
}

// LookupRegistry returns the OpRunnerML for a given op name.
// Returns (runner, true) if found, (zero, false) otherwise.
func LookupRegistry(op string) (OpRunnerML, bool) {
	r, ok := opRegistry[op]
	return r, ok
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./perf/ -run "TestRegistry|TestExpandShapes|TestParseDType|TestDtypeToString|TestPhase1" -v`

Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add perf/registry.go perf/registry_test.go
git commit -m "perf: add operator registry with Phase 1 ops (SILU, MUL_MAT, FLASH_ATTN_EXT)"
```

---

## Task 4: Log-Space Interpolation (`interpolate.go`)

**Files:**
- Create: `perf/interpolate.go`
- Create: `perf/interpolate_test.go`

**THIS IS THE MOST CRITICAL TASK.** The interpolation functions are the mathematical core of the v2 estimator. They must be thoroughly tested with edge cases, not just happy paths.

**Key invariant:** All interpolation happens in log-log space. For a power law `latency = c * N^alpha`, log-space interpolation is exact. This is why it works well for GPU ops where performance follows piecewise power laws.

### 4a: Interpolate1D

- [ ] **Step 1: Write comprehensive failing tests for Interpolate1D**

Create `perf/interpolate_test.go`:

```go
package perf

import (
	"math"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- Test helpers ---

// makePoints creates LatencyPoints from (shape, latency) pairs for 1D interpolation.
func makePoints(pairs [][2]float64) []LatencyPoint {
	pts := make([]LatencyPoint, len(pairs))
	for i, p := range pairs {
		pts[i] = LatencyPoint{Shape: []int64{int64(p[0])}, LatencyUs: p[1]}
	}
	return pts
}

// makePowerLawPoints generates points following latency = c * N^alpha.
// This is the ideal case for log-log interpolation — it should be exact.
func makePowerLawPoints(c, alpha float64, Ns []int64) []LatencyPoint {
	pts := make([]LatencyPoint, len(Ns))
	for i, N := range Ns {
		pts[i] = LatencyPoint{
			Shape:     []int64{N},
			LatencyUs: c * math.Pow(float64(N), alpha),
		}
	}
	return pts
}

// --- Interpolate1D tests ---

func TestInterpolate1D_ExactMatch(t *testing.T) {
	// Query at a measured point should return the exact value.
	pts := makePoints([][2]float64{
		{1024, 10.0},
		{4096, 40.0},
		{16384, 160.0},
	})
	assert.InDelta(t, 10.0, Interpolate1D(pts, 1024), 0.001)
	assert.InDelta(t, 40.0, Interpolate1D(pts, 4096), 0.001)
	assert.InDelta(t, 160.0, Interpolate1D(pts, 16384), 0.001)
}

func TestInterpolate1D_InteriorInterpolation(t *testing.T) {
	// Query between two points. For a power law, log-log interp is exact.
	// latency = 0.01 * N^1.0 (linear relationship)
	pts := makePowerLawPoints(0.01, 1.0, []int64{1000, 10000})
	// Query at N=3162 (geometric midpoint of 1000 and 10000)
	result := Interpolate1D(pts, 3162)
	expected := 0.01 * 3162.0
	assert.InDelta(t, expected, result, expected*0.01, "should recover linear power law")
}

func TestInterpolate1D_PowerLawExactness(t *testing.T) {
	// For f(N) = 5.0 * N^0.7, log-log interpolation should be nearly exact
	// at any query point between the measured points.
	c, alpha := 5.0, 0.7
	pts := makePowerLawPoints(c, alpha, []int64{100, 1000, 10000, 100000})
	queries := []int64{200, 500, 2000, 5000, 50000}
	for _, q := range queries {
		result := Interpolate1D(pts, q)
		expected := c * math.Pow(float64(q), alpha)
		relErr := math.Abs(result-expected) / expected
		assert.Less(t, relErr, 0.01,
			"query N=%d: got %.2f, expected %.2f (relErr=%.4f)", q, result, expected, relErr)
	}
}

func TestInterpolate1D_LogVsLinear(t *testing.T) {
	// Demonstrate that log-space interpolation differs from linear interpolation.
	// For points (100, 1.0) and (10000, 100.0):
	//   Linear midpoint at N=5050:  lat = 50.5
	//   Log midpoint at N=1000:     lat = 10.0 (geometric mean)
	pts := makePoints([][2]float64{
		{100, 1.0},
		{10000, 100.0},
	})
	// At N=1000 (geometric midpoint), log-space should give ~10.0
	result := Interpolate1D(pts, 1000)
	assert.InDelta(t, 10.0, result, 0.5, "log-space interpolation at geometric midpoint")

	// At N=5050 (arithmetic midpoint), should NOT give 50.5
	result2 := Interpolate1D(pts, 5050)
	assert.NotInDelta(t, 50.5, result2, 5.0, "should NOT behave like linear interpolation")
}

func TestInterpolate1D_ExtrapolateLeft(t *testing.T) {
	// Query below the smallest measured point — extends the leftmost segment's slope.
	pts := makePowerLawPoints(0.01, 1.0, []int64{1000, 10000, 100000})
	result := Interpolate1D(pts, 100)
	expected := 0.01 * 100.0
	assert.InDelta(t, expected, result, expected*0.05,
		"left extrapolation should extend power law")
}

func TestInterpolate1D_ExtrapolateRight(t *testing.T) {
	// Query above the largest measured point.
	pts := makePowerLawPoints(0.01, 1.0, []int64{1000, 10000, 100000})
	result := Interpolate1D(pts, 1000000)
	expected := 0.01 * 1000000.0
	assert.InDelta(t, expected, result, expected*0.05,
		"right extrapolation should extend power law")
}

func TestInterpolate1D_TwoPoints(t *testing.T) {
	// Minimum viable interpolation: just two points.
	pts := makePoints([][2]float64{
		{256, 5.0},
		{65536, 1000.0},
	})
	// Geometric midpoint: sqrt(256 * 65536) = 4096
	result := Interpolate1D(pts, 4096)
	// In log-log: interpolated value at log-midpoint = geometric mean of latencies
	// = sqrt(5.0 * 1000.0) = 70.7
	assert.InDelta(t, math.Sqrt(5.0*1000.0), result, 1.0)
}

func TestInterpolate1D_SinglePoint(t *testing.T) {
	// Edge case: only one measured point. Should return that point's latency
	// for any query (best we can do).
	pts := makePoints([][2]float64{{1024, 42.0}})
	assert.InDelta(t, 42.0, Interpolate1D(pts, 1024), 0.001)
	assert.InDelta(t, 42.0, Interpolate1D(pts, 100), 0.001)
	assert.InDelta(t, 42.0, Interpolate1D(pts, 100000), 0.001)
}

func TestInterpolate1D_ManyPoints(t *testing.T) {
	// 10 points following a power law — verify all intermediate queries.
	c, alpha := 2.0, 0.8
	Ns := []int64{64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}
	pts := makePowerLawPoints(c, alpha, Ns)
	// Query at midpoints between every consecutive pair
	for i := 0; i < len(Ns)-1; i++ {
		midN := int64(math.Sqrt(float64(Ns[i]) * float64(Ns[i+1])))
		result := Interpolate1D(pts, midN)
		expected := c * math.Pow(float64(midN), alpha)
		relErr := math.Abs(result-expected) / expected
		assert.Less(t, relErr, 0.01,
			"midpoint N=%d between %d and %d", midN, Ns[i], Ns[i+1])
	}
}

func TestInterpolate1D_BoundaryFirstAndLast(t *testing.T) {
	pts := makePoints([][2]float64{
		{100, 1.0},
		{1000, 10.0},
		{10000, 100.0},
	})
	// First point
	assert.InDelta(t, 1.0, Interpolate1D(pts, 100), 0.001)
	// Last point
	assert.InDelta(t, 100.0, Interpolate1D(pts, 10000), 0.001)
}

func TestInterpolate1D_NonPowerLaw(t *testing.T) {
	// Real GPU latency often has a flat region (kernel launch overhead)
	// then a rising region. Log-log won't be exact but should be reasonable.
	pts := makePoints([][2]float64{
		{64, 5.0},     // flat (kernel launch dominates)
		{256, 5.2},    // still mostly flat
		{1024, 8.0},   // starting to rise
		{4096, 25.0},  // rising
		{16384, 95.0}, // compute-dominated
	})
	result := Interpolate1D(pts, 2048)
	// Should be between 8.0 and 25.0
	assert.Greater(t, result, 8.0)
	assert.Less(t, result, 25.0)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./perf/ -run "TestInterpolate1D" -v`

Expected: compilation errors — Interpolate1D doesn't exist.

- [ ] **Step 3: Implement Interpolate1D**

Create `perf/interpolate.go`:

```go
package perf

import (
	"math"
	"sort"
)

// Interpolate1D performs piecewise linear interpolation in log-log space.
// Points must be sorted by Shape[0] ascending.
//
// For a power law latency = c * N^alpha, this interpolation is exact.
// For non-power-law relationships, it provides a reasonable piecewise approximation.
//
// Edge cases:
//   - Single point: returns that point's latency for any query
//   - Query below range: extrapolates using leftmost segment slope
//   - Query above range: extrapolates using rightmost segment slope
func Interpolate1D(points []LatencyPoint, queryN int64) float64 {
	if len(points) == 0 {
		return 0
	}
	if len(points) == 1 {
		return points[0].LatencyUs
	}

	logQ := math.Log(float64(queryN))

	// Find bracketing interval
	for i := 0; i < len(points)-1; i++ {
		logX1 := math.Log(float64(points[i].Shape[0]))
		logX2 := math.Log(float64(points[i+1].Shape[0]))
		if logQ >= logX1 && logQ <= logX2 {
			logY1 := math.Log(points[i].LatencyUs)
			logY2 := math.Log(points[i+1].LatencyUs)
			t := (logQ - logX1) / (logX2 - logX1)
			return math.Exp(logY1 + t*(logY2-logY1))
		}
	}

	// Extrapolation
	if logQ < math.Log(float64(points[0].Shape[0])) {
		return extrapolateLeft(points, logQ)
	}
	return extrapolateRight(points, logQ)
}

// extrapolateLeft extends the slope of the leftmost segment.
func extrapolateLeft(points []LatencyPoint, logQ float64) float64 {
	if len(points) < 2 {
		return points[0].LatencyUs
	}
	logX1 := math.Log(float64(points[0].Shape[0]))
	logX2 := math.Log(float64(points[1].Shape[0]))
	logY1 := math.Log(points[0].LatencyUs)
	logY2 := math.Log(points[1].LatencyUs)
	slope := (logY2 - logY1) / (logX2 - logX1)
	return math.Exp(logY1 + slope*(logQ-logX1))
}

// extrapolateRight extends the slope of the rightmost segment.
func extrapolateRight(points []LatencyPoint, logQ float64) float64 {
	n := len(points)
	if n < 2 {
		return points[0].LatencyUs
	}
	logX1 := math.Log(float64(points[n-2].Shape[0]))
	logX2 := math.Log(float64(points[n-1].Shape[0]))
	logY1 := math.Log(points[n-2].LatencyUs)
	logY2 := math.Log(points[n-1].LatencyUs)
	slope := (logY2 - logY1) / (logX2 - logX1)
	return math.Exp(logY2 + slope*(logQ-logX2))
}
```

- [ ] **Step 4: Run Interpolate1D tests**

Run: `go test ./perf/ -run "TestInterpolate1D" -v`

Expected: All 11 Interpolate1D tests PASS.

- [ ] **Step 5: Commit**

```bash
git add perf/interpolate.go perf/interpolate_test.go
git commit -m "perf: add Interpolate1D with log-log piecewise linear interpolation"
```

### 4b: Interpolate1DByDim

- [ ] **Step 6: Write failing tests for Interpolate1DByDim**

Append to `perf/interpolate_test.go`:

```go
// --- Interpolate1DByDim tests ---

// makePointsMultiDim creates LatencyPoints with multi-dimensional shapes.
func makePointsMultiDim(shapes [][]int64, latencies []float64) []LatencyPoint {
	pts := make([]LatencyPoint, len(shapes))
	for i := range shapes {
		pts[i] = LatencyPoint{Shape: shapes[i], LatencyUs: latencies[i]}
	}
	return pts
}

func TestInterpolate1DByDim_DimIdx0_MatchesInterpolate1D(t *testing.T) {
	// When dimIdx=0, should behave identically to Interpolate1D.
	pts := makePoints([][2]float64{
		{100, 1.0}, {1000, 10.0}, {10000, 100.0},
	})
	for _, q := range []int64{100, 500, 1000, 5000, 10000} {
		a := Interpolate1D(pts, q)
		b := Interpolate1DByDim(pts, 0, q)
		assert.InDelta(t, a, b, 0.001, "dimIdx=0 should match Interpolate1D at N=%d", q)
	}
}

func TestInterpolate1DByDim_DimIdx1(t *testing.T) {
	// Points with Shape=[seq_q, seq_kv]. Interpolate over dim 1 (seq_kv).
	pts := makePointsMultiDim(
		[][]int64{{1, 128}, {1, 512}, {1, 2048}, {1, 8192}},
		[]float64{5.0, 15.0, 55.0, 200.0},
	)
	// Query seq_kv=1024 (between 512 and 2048)
	result := Interpolate1DByDim(pts, 1, 1024)
	assert.Greater(t, result, 15.0)
	assert.Less(t, result, 55.0)
}

func TestInterpolate1DByDim_SinglePoint(t *testing.T) {
	pts := makePointsMultiDim(
		[][]int64{{1, 1024}},
		[]float64{42.0},
	)
	assert.InDelta(t, 42.0, Interpolate1DByDim(pts, 1, 500), 0.001)
}

func TestInterpolate1DByDim_SortsByDimIdx(t *testing.T) {
	// Points deliberately out of order for dimIdx=1
	pts := makePointsMultiDim(
		[][]int64{{1, 2048}, {1, 128}, {1, 8192}, {1, 512}},
		[]float64{55.0, 5.0, 200.0, 15.0},
	)
	// Should still work correctly despite unsorted input
	result := Interpolate1DByDim(pts, 1, 1024)
	assert.Greater(t, result, 15.0)
	assert.Less(t, result, 55.0)
}
```

- [ ] **Step 7: Implement Interpolate1DByDim**

Append to `perf/interpolate.go`:

```go
// Interpolate1DByDim is like Interpolate1D but reads Shape[dimIdx] instead of Shape[0].
// Points are sorted internally by Shape[dimIdx].
func Interpolate1DByDim(points []LatencyPoint, dimIdx int, queryVal int64) float64 {
	if len(points) == 0 {
		return 0
	}
	if len(points) == 1 {
		return points[0].LatencyUs
	}

	// Sort by the target dimension
	sorted := make([]LatencyPoint, len(points))
	copy(sorted, points)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Shape[dimIdx] < sorted[j].Shape[dimIdx]
	})

	logQ := math.Log(float64(queryVal))

	// Find bracketing interval
	for i := 0; i < len(sorted)-1; i++ {
		logX1 := math.Log(float64(sorted[i].Shape[dimIdx]))
		logX2 := math.Log(float64(sorted[i+1].Shape[dimIdx]))
		if logQ >= logX1 && logQ <= logX2 {
			logY1 := math.Log(sorted[i].LatencyUs)
			logY2 := math.Log(sorted[i+1].LatencyUs)
			t := (logQ - logX1) / (logX2 - logX1)
			return math.Exp(logY1 + t*(logY2-logY1))
		}
	}

	// Extrapolation
	if logQ < math.Log(float64(sorted[0].Shape[dimIdx])) {
		return extrapolateLeftByDim(sorted, dimIdx, logQ)
	}
	return extrapolateRightByDim(sorted, dimIdx, logQ)
}

func extrapolateLeftByDim(points []LatencyPoint, dimIdx int, logQ float64) float64 {
	if len(points) < 2 {
		return points[0].LatencyUs
	}
	logX1 := math.Log(float64(points[0].Shape[dimIdx]))
	logX2 := math.Log(float64(points[1].Shape[dimIdx]))
	logY1 := math.Log(points[0].LatencyUs)
	logY2 := math.Log(points[1].LatencyUs)
	slope := (logY2 - logY1) / (logX2 - logX1)
	return math.Exp(logY1 + slope*(logQ-logX1))
}

func extrapolateRightByDim(points []LatencyPoint, dimIdx int, logQ float64) float64 {
	n := len(points)
	if n < 2 {
		return points[0].LatencyUs
	}
	logX1 := math.Log(float64(points[n-2].Shape[dimIdx]))
	logX2 := math.Log(float64(points[n-1].Shape[dimIdx]))
	logY1 := math.Log(points[n-2].LatencyUs)
	logY2 := math.Log(points[n-1].LatencyUs)
	slope := (logY2 - logY1) / (logX2 - logX1)
	return math.Exp(logY2 + slope*(logQ-logX2))
}
```

- [ ] **Step 8: Run tests**

Run: `go test ./perf/ -run "TestInterpolate1DByDim" -v`

Expected: All 4 Interpolate1DByDim tests PASS.

- [ ] **Step 9: Commit**

```bash
git add perf/interpolate.go perf/interpolate_test.go
git commit -m "perf: add Interpolate1DByDim for multi-dimensional latency points"
```

### 4c: InterpolateMulMat

- [ ] **Step 10: Write failing tests for InterpolateMulMat**

Append to `perf/interpolate_test.go`:

```go
// --- InterpolateMulMat tests ---

// makeMulMatCurves creates a set of MUL_MAT OperatorCurves for testing.
// Each curve has fixed (M, K) and points along N.
func makeMulMatCurves(configs []struct {
	M, K int64
	// points: (N, latency) pairs
	points [][2]float64
}) []OperatorCurve {
	curves := make([]OperatorCurve, len(configs))
	for i, cfg := range configs {
		pts := make([]LatencyPoint, len(cfg.points))
		for j, p := range cfg.points {
			pts[j] = LatencyPoint{Shape: []int64{int64(p[0])}, LatencyUs: p[1]}
		}
		curves[i] = OperatorCurve{
			Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16",
			Dimensions: []string{"N"},
			FixedDims:  map[string]int64{"M": cfg.M, "K": cfg.K},
			Points:     pts,
		}
	}
	return curves
}

func TestInterpolateMulMat_ExactMKMatch(t *testing.T) {
	curves := makeMulMatCurves([]struct {
		M, K   int64
		points [][2]float64
	}{
		{4096, 4096, [][2]float64{{1, 10.0}, {32, 50.0}, {256, 200.0}, {4096, 3000.0}}},
		{8192, 8192, [][2]float64{{1, 20.0}, {32, 100.0}, {256, 400.0}, {4096, 6000.0}}},
	})
	// Query at exact (M=4096, K=4096) — should use first curve only
	result := InterpolateMulMat(curves, 4096, 4096, 32)
	assert.InDelta(t, 50.0, result, 0.001, "exact M,K match should use Interpolate1D directly")
}

func TestInterpolateMulMat_BetweenMKPairs(t *testing.T) {
	curves := makeMulMatCurves([]struct {
		M, K   int64
		points [][2]float64
	}{
		{4096, 4096, [][2]float64{{1, 10.0}, {4096, 1000.0}}},
		{8192, 8192, [][2]float64{{1, 20.0}, {4096, 2000.0}}},
	})
	// Query at M=6000, K=6000 — between the two curves
	result := InterpolateMulMat(curves, 6000, 6000, 1)
	// Should be between 10.0 and 20.0 (N=1 latency from each curve)
	assert.Greater(t, result, 10.0)
	assert.Less(t, result, 20.0)
}

func TestInterpolateMulMat_SingleCurve(t *testing.T) {
	curves := makeMulMatCurves([]struct {
		M, K   int64
		points [][2]float64
	}{
		{4096, 4096, [][2]float64{{1, 10.0}, {4096, 1000.0}}},
	})
	// Only one curve — should use it regardless of query M,K
	result := InterpolateMulMat(curves, 9999, 9999, 1)
	assert.InDelta(t, 10.0, result, 0.001, "single curve should be used directly")
}

func TestInterpolateMulMat_ManyCurves(t *testing.T) {
	curves := makeMulMatCurves([]struct {
		M, K   int64
		points [][2]float64
	}{
		{1024, 1024, [][2]float64{{1, 2.0}, {4096, 200.0}}},
		{4096, 4096, [][2]float64{{1, 10.0}, {4096, 1000.0}}},
		{8192, 4096, [][2]float64{{1, 15.0}, {4096, 1500.0}}},
		{8192, 8192, [][2]float64{{1, 20.0}, {4096, 2000.0}}},
	})
	// Query at exact (4096, 4096) with many available curves
	result := InterpolateMulMat(curves, 4096, 4096, 1)
	assert.InDelta(t, 10.0, result, 0.001, "should find exact match among many curves")
}

func TestInterpolateMulMat_AsymmetricMK(t *testing.T) {
	curves := makeMulMatCurves([]struct {
		M, K   int64
		points [][2]float64
	}{
		{14336, 4096, [][2]float64{{1, 15.0}, {4096, 1500.0}}},
		{4096, 14336, [][2]float64{{1, 12.0}, {4096, 1200.0}}},
	})
	// Query at (14336, 4096) — exact match on first curve
	result := InterpolateMulMat(curves, 14336, 4096, 1)
	assert.InDelta(t, 15.0, result, 0.001)
	// Query at (4096, 14336) — exact match on second curve
	result2 := InterpolateMulMat(curves, 4096, 14336, 1)
	assert.InDelta(t, 12.0, result2, 0.001)
}

func TestInterpolateMulMat_InverseDistanceWeighting(t *testing.T) {
	// Two curves at equal log-distance from query: result should be average.
	curves := makeMulMatCurves([]struct {
		M, K   int64
		points [][2]float64
	}{
		{1000, 1000, [][2]float64{{100, 10.0}}},
		{10000, 10000, [][2]float64{{100, 20.0}}},
	})
	// Query at geometric midpoint: sqrt(1000*10000) = ~3162
	result := InterpolateMulMat(curves, 3162, 3162, 100)
	// Equal log-distance → equal weights → average of 10.0 and 20.0 = 15.0
	assert.InDelta(t, 15.0, result, 0.5)
}
```

- [ ] **Step 11: Implement InterpolateMulMat**

Append to `perf/interpolate.go`:

```go
// InterpolateMulMat looks up latency for MUL_MAT with query (M, K, N).
//
// Each MUL_MAT OperatorCurve has FixedDims={"M": m, "K": k} and Dimensions=["N"].
// Strategy:
//  1. Find the two closest (M, K) curves by Euclidean distance in log space
//  2. For each curve, do 1D interpolation over N
//  3. Weight-average by inverse log-distance
func InterpolateMulMat(curves []OperatorCurve, queryM, queryK, queryN int64) float64 {
	if len(curves) == 0 {
		return 0
	}

	type candidate struct {
		curve   *OperatorCurve
		logDist float64
	}

	var candidates []candidate
	for i := range curves {
		curveM := curves[i].FixedDims["M"]
		curveK := curves[i].FixedDims["K"]
		dM := math.Log(float64(queryM)) - math.Log(float64(curveM))
		dK := math.Log(float64(queryK)) - math.Log(float64(curveK))
		dist := math.Sqrt(dM*dM + dK*dK)
		candidates = append(candidates, candidate{&curves[i], dist})
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].logDist < candidates[j].logDist
	})

	// Exact match or only one curve
	if candidates[0].logDist == 0 || len(candidates) == 1 {
		return Interpolate1D(candidates[0].curve.Points, queryN)
	}

	// Inverse-distance weighted average of two nearest curves
	lat1 := Interpolate1D(candidates[0].curve.Points, queryN)
	lat2 := Interpolate1D(candidates[1].curve.Points, queryN)
	w1 := 1.0 / candidates[0].logDist
	w2 := 1.0 / candidates[1].logDist
	return (lat1*w1 + lat2*w2) / (w1 + w2)
}
```

- [ ] **Step 12: Run tests**

Run: `go test ./perf/ -run "TestInterpolateMulMat" -v`

Expected: All 7 InterpolateMulMat tests PASS.

- [ ] **Step 13: Commit**

```bash
git add perf/interpolate.go perf/interpolate_test.go
git commit -m "perf: add InterpolateMulMat with inverse-distance weighted multi-curve lookup"
```

### 4d: InterpolateFlashAttn

- [ ] **Step 14: Write failing tests for InterpolateFlashAttn**

Append to `perf/interpolate_test.go`:

```go
// --- InterpolateFlashAttn tests ---

// makeFlashAttnCurve creates a FLASH_ATTN_EXT OperatorCurve with decode and prefill points.
// decodePts: (seq_kv, latency) pairs with seq_q=1
// prefillPts: (seq_kv, latency) pairs with seq_q=seq_kv
func makeFlashAttnCurve(decodePts, prefillPts [][2]float64) *OperatorCurve {
	var points []LatencyPoint
	for _, p := range decodePts {
		points = append(points, LatencyPoint{
			Shape:     []int64{1, int64(p[0])},
			LatencyUs: p[1],
		})
	}
	for _, p := range prefillPts {
		seqKV := int64(p[0])
		points = append(points, LatencyPoint{
			Shape:     []int64{seqKV, seqKV},
			LatencyUs: p[1],
		})
	}
	return &OperatorCurve{
		Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
		Dimensions: []string{"seq_q", "seq_kv"},
		FixedDims:  map[string]int64{"num_heads": 32, "head_dim": 128},
		Points:     points,
	}
}

func TestInterpolateFlashAttn_DecodeRegime(t *testing.T) {
	// seq_q=1 → should use decode curve only
	curve := makeFlashAttnCurve(
		[][2]float64{{128, 5.0}, {512, 15.0}, {2048, 55.0}, {8192, 200.0}},
		[][2]float64{{128, 20.0}, {512, 100.0}, {2048, 500.0}, {8192, 3000.0}},
	)
	result := InterpolateFlashAttn(curve, 1, 1024)
	// Between decode 512 (15.0) and 2048 (55.0)
	assert.Greater(t, result, 15.0)
	assert.Less(t, result, 55.0)
}

func TestInterpolateFlashAttn_PrefillRegime(t *testing.T) {
	// seq_q=seq_kv → should use prefill curve only
	curve := makeFlashAttnCurve(
		[][2]float64{{128, 5.0}, {512, 15.0}, {2048, 55.0}, {8192, 200.0}},
		[][2]float64{{128, 20.0}, {512, 100.0}, {2048, 500.0}, {8192, 3000.0}},
	)
	result := InterpolateFlashAttn(curve, 1024, 1024)
	// Between prefill 512 (100.0) and 2048 (500.0)
	assert.Greater(t, result, 100.0)
	assert.Less(t, result, 500.0)
}

func TestInterpolateFlashAttn_BetweenRegimes(t *testing.T) {
	// seq_q between 1 and seq_kv → blend of decode and prefill
	curve := makeFlashAttnCurve(
		[][2]float64{{1024, 30.0}},
		[][2]float64{{1024, 300.0}},
	)
	// seq_q=32, seq_kv=1024 → t = log(32)/log(1024) = 5/10 = 0.5
	result := InterpolateFlashAttn(curve, 32, 1024)
	// Blend in log space: exp(log(30)*(1-0.5) + log(300)*0.5) = sqrt(30*300) = 94.87
	expected := math.Sqrt(30.0 * 300.0)
	assert.InDelta(t, expected, result, 1.0)
}

func TestInterpolateFlashAttn_SeqKV1_Guard(t *testing.T) {
	// seq_kv=1 → would cause log(1)=0 → division by zero.
	// Guard should fall back to decode curve.
	curve := makeFlashAttnCurve(
		[][2]float64{{1, 1.0}, {128, 5.0}, {1024, 30.0}},
		[][2]float64{{128, 20.0}, {1024, 300.0}},
	)
	result := InterpolateFlashAttn(curve, 1, 1)
	assert.InDelta(t, 1.0, result, 0.1, "seq_kv=1 should use decode curve")
}

func TestInterpolateFlashAttn_SeqQ1_SeqKV1(t *testing.T) {
	// Degenerate case: both 1
	curve := makeFlashAttnCurve(
		[][2]float64{{1, 0.5}, {128, 5.0}},
		[][2]float64{{128, 20.0}},
	)
	result := InterpolateFlashAttn(curve, 1, 1)
	assert.InDelta(t, 0.5, result, 0.1)
}

func TestInterpolateFlashAttn_DecodeExtrapolation(t *testing.T) {
	// Query beyond measured range
	curve := makeFlashAttnCurve(
		[][2]float64{{128, 5.0}, {2048, 55.0}},
		[][2]float64{{128, 20.0}, {2048, 500.0}},
	)
	result := InterpolateFlashAttn(curve, 1, 16384)
	// Should extrapolate beyond 2048
	assert.Greater(t, result, 55.0)
}

func TestInterpolateFlashAttn_PrefillHighSeqQ(t *testing.T) {
	// seq_q very close to seq_kv but not equal → should be close to prefill
	curve := makeFlashAttnCurve(
		[][2]float64{{1024, 30.0}},
		[][2]float64{{1024, 300.0}},
	)
	result := InterpolateFlashAttn(curve, 1000, 1024)
	// t = log(1000)/log(1024) ≈ 0.997 → very close to prefill
	assert.InDelta(t, 300.0, result, 10.0, "near-equal seq_q/seq_kv should be close to prefill")
}

func TestInterpolateFlashAttn_BlendMonotonicity(t *testing.T) {
	// As seq_q increases from 1 to seq_kv, result should increase monotonically
	// (prefill is always slower than decode for same seq_kv).
	curve := makeFlashAttnCurve(
		[][2]float64{{1024, 30.0}},
		[][2]float64{{1024, 300.0}},
	)
	seqQs := []int64{1, 2, 4, 16, 64, 256, 512, 1024}
	prev := 0.0
	for _, sq := range seqQs {
		result := InterpolateFlashAttn(curve, sq, 1024)
		assert.Greater(t, result, prev, "seq_q=%d should give higher latency than seq_q=%d", sq, sq/2)
		prev = result
	}
}
```

- [ ] **Step 15: Implement InterpolateFlashAttn**

Append to `perf/interpolate.go`:

```go
// InterpolateFlashAttn looks up latency for FLASH_ATTN_EXT with query (seq_q, seq_kv).
//
// The curve contains two regimes of points:
//   - Decode: Shape[0]=1 (seq_q=1), varying Shape[1] (seq_kv)
//   - Prefill: Shape[0]=Shape[1] (seq_q=seq_kv), varying both
//
// For queries:
//   - seq_q=1 → use decode points, 1D interpolation over seq_kv
//   - seq_q=seq_kv → use prefill points, 1D interpolation over seq_kv
//   - Otherwise → blend between decode and prefill using t = log(seq_q)/log(seq_kv)
//
// Edge case: seq_kv <= 1 → fall back to decode curve (avoids log(1)=0 division).
func InterpolateFlashAttn(curve *OperatorCurve, querySeqQ, querySeqKV int64) float64 {
	// Separate points into decode (seq_q=1) and prefill (seq_q=seq_kv)
	var prefillPts, decodePts []LatencyPoint
	for _, pt := range curve.Points {
		if pt.Shape[0] == 1 {
			decodePts = append(decodePts, pt)
		} else if pt.Shape[0] == pt.Shape[1] {
			prefillPts = append(prefillPts, pt)
		}
	}

	// Pure decode
	if querySeqQ == 1 {
		return Interpolate1DByDim(decodePts, 1, querySeqKV)
	}

	// Pure prefill
	if querySeqQ == querySeqKV {
		return Interpolate1DByDim(prefillPts, 1, querySeqKV)
	}

	// Guard: seq_kv <= 1 would cause log(1)=0 → division by zero
	if querySeqKV <= 1 {
		return Interpolate1DByDim(decodePts, 1, querySeqKV)
	}

	// Between regimes: weighted blend in log space
	// t=0 → decode, t=1 → prefill
	decodeLat := Interpolate1DByDim(decodePts, 1, querySeqKV)
	prefillLat := Interpolate1DByDim(prefillPts, 1, querySeqKV)
	t := math.Log(float64(querySeqQ)) / math.Log(float64(querySeqKV))
	if t < 0 {
		t = 0
	}
	if t > 1 {
		t = 1
	}
	return math.Exp(math.Log(decodeLat)*(1-t) + math.Log(prefillLat)*t)
}
```

- [ ] **Step 16: Run all interpolation tests**

Run: `go test ./perf/ -run "TestInterpolate" -v`

Expected: All 30+ interpolation tests PASS.

- [ ] **Step 17: Commit**

```bash
git add perf/interpolate.go perf/interpolate_test.go
git commit -m "perf: add InterpolateFlashAttn with two-regime blending and edge guards"
```

---

## Task 5: Hardware Characterization (`hwchar.go`)

**Files:**
- Create: `perf/hwchar.go`
- Create: `perf/hwchar_test.go`

Measures peak TOPS and peak bandwidth for each backend device. These values are NOT used for latency prediction — they only inform the initial sampling grid placement and serve as sanity checks.

This code is extracted and refactored from v1's `bench.go` (`benchPeakFLOPS`, `benchPeakBandwidth`).

- [ ] **Step 1: Write failing tests**

Create `perf/hwchar_test.go`:

```go
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

// Integration tests for benchPeakTOPS and benchPeakBandwidth require a real
// GGML backend. They are in integration_test.go.
```

- [ ] **Step 2: Implement `hwchar.go`**

Create `perf/hwchar.go`:

```go
package perf

import (
	"fmt"
	"log/slog"
	"math"
	"sort"
	"time"

	"github.com/ollama/ollama/ml"
)

// CharacterizeHardware measures peak TOPS and bandwidth for all backend devices.
// Returns an HWCharResult and populates the given HardwareProfile.
func CharacterizeHardware(backend ml.Backend, cfg BenchmarkConfig) (*HWCharResult, error) {
	devices := backend.BackendDevices()
	if len(devices) == 0 {
		return nil, fmt.Errorf("no backend devices found")
	}

	result := &HWCharResult{
		PeakTOPS:     make(map[string]float64),
		BalancePoint: make(map[string]float64),
	}

	slog.Info("hardware characterization", "devices", len(devices))

	// Measure peak TOPS for f16 and f32
	for _, dtypeStr := range []string{"f16", "f32"} {
		dt, ok := parseDType(dtypeStr)
		if !ok {
			continue
		}
		tops, err := benchPeakTOPS(backend, dt, cfg)
		if err != nil {
			slog.Warn("peak TOPS failed", "dtype", dtypeStr, "error", err)
			continue
		}
		result.PeakTOPS[dtypeStr] = tops
		slog.Info("peak TOPS", "dtype", dtypeStr, "TOPS", tops)
	}

	// Measure peak bandwidth
	bw, err := benchPeakBandwidth(backend, cfg)
	if err != nil {
		return nil, fmt.Errorf("peak bandwidth failed: %w", err)
	}
	result.PeakBW = bw
	slog.Info("peak bandwidth", "bytes_per_sec", bw)

	// Compute balance points
	for dtype, tops := range result.PeakTOPS {
		if result.PeakBW > 0 {
			result.BalancePoint[dtype] = tops / result.PeakBW
		}
	}

	return result, nil
}

// benchPeakTOPS measures peak TOPS via large MUL_MAT (M=K=N=4096).
// TOPS = FLOPs / latency, where FLOPs = 2 * M * K * N.
func benchPeakTOPS(backend ml.Backend, dtype ml.DType, cfg BenchmarkConfig) (float64, error) {
	const M, K, N = 4096, 4096, 4096
	ctx := backend.NewContext()
	defer ctx.Close()

	a := ctx.Zeros(dtype, M, K)
	b := ctx.Zeros(dtype, K, N)
	out := a.Mulmat(ctx, b)
	ctx.Forward(out)

	// Warmup
	for i := 0; i < cfg.WarmupReps; i++ {
		ctx.Compute(out)
	}

	// Measure with trimming
	latencies := make([]float64, cfg.MeasureReps)
	for i := 0; i < cfg.MeasureReps; i++ {
		start := time.Now()
		ctx.Compute(out)
		latencies[i] = time.Since(start).Seconds()
	}

	median := trimmedMedian(latencies, cfg.TrimPercent)
	flops := 2.0 * M * K * N
	return flops / median, nil
}

// benchPeakBandwidth measures peak memory bandwidth via large CONT (copy).
// Size: 64M elements * 4 bytes = 256MB. Bytes = 2 * 256MB (read + write).
func benchPeakBandwidth(backend ml.Backend, cfg BenchmarkConfig) (float64, error) {
	const size = 64 * 1024 * 1024 // 64M elements
	ctx := backend.NewContext()
	defer ctx.Close()

	src := ctx.Zeros(ml.DTypeF32, size)
	dst := src.Contiguous(ctx)
	ctx.Forward(dst)

	// Warmup
	for i := 0; i < cfg.WarmupReps; i++ {
		ctx.Compute(dst)
	}

	// Measure with trimming
	latencies := make([]float64, cfg.MeasureReps)
	for i := 0; i < cfg.MeasureReps; i++ {
		start := time.Now()
		ctx.Compute(dst)
		latencies[i] = time.Since(start).Seconds()
	}

	median := trimmedMedian(latencies, cfg.TrimPercent)
	bytesTotal := 2.0 * size * 4 // read + write, 4 bytes per f32
	return bytesTotal / median, nil
}

// trimmedMedian sorts the values, trims outliers, and returns the median.
func trimmedMedian(values []float64, trimPercent float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	trimCount := int(math.Round(float64(len(sorted)) * trimPercent))
	if trimCount*2 >= len(sorted) {
		trimCount = 0
	}
	trimmed := sorted[trimCount : len(sorted)-trimCount]
	if len(trimmed) == 0 {
		return sorted[len(sorted)/2]
	}
	return trimmed[len(trimmed)/2]
}

// HWCharResultToHardwareProfile converts benchmark results into a HardwareProfile.
func HWCharResultToHardwareProfile(result *HWCharResult, backend ml.Backend) HardwareProfile {
	hp := HardwareProfile{
		PeakTOPS:                 result.PeakTOPS,
		PeakBandwidthBytesPerSec: result.PeakBW,
		BalancePoints:            result.BalancePoint,
	}

	for _, dev := range backend.BackendDevices() {
		hp.Backends = append(hp.Backends, BackendInfo{
			Name:   dev.Library,
			Device: dev.Name,
		})
	}

	return hp
}
```

- [ ] **Step 3: Run tests**

Run: `go test ./perf/ -run "TestHWCharResult" -v`

Expected: PASS (it's a pure Go test on the data structure).

- [ ] **Step 4: Commit**

```bash
git add perf/hwchar.go perf/hwchar_test.go
git commit -m "perf: add hardware characterization (peak TOPS, bandwidth, balance point)"
```

---

## Task 6: Adaptive Sampling (`adaptive.go`)

**Files:**
- Create: `perf/adaptive.go`
- Create: `perf/adaptive_test.go`

The adaptive sampling algorithm starts with a log-spaced grid, then iteratively refines by adding points where interpolation error is highest. This is the key to getting accurate curves with minimal benchmark time.

**IMPORTANT:** The adaptive logic uses a `measureFunc` callback so it can be tested with mock measurements (no GGML required).

- [ ] **Step 1: Write comprehensive failing tests with mock measurement**

Create `perf/adaptive_test.go`:

```go
package perf

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockMeasure returns a measurement function that follows a known mathematical function.
// This lets us test adaptive sampling convergence without a real backend.
func mockMeasure(f func(n int64) float64) func(shape []int64) LatencyPoint {
	return func(shape []int64) LatencyPoint {
		return LatencyPoint{
			Shape:     shape,
			LatencyUs: f(shape[0]),
			StddevUs:  0.01,
			Reps:      100,
		}
	}
}

func TestAdaptiveSample1D_SmoothPowerLaw(t *testing.T) {
	// f(N) = 0.01 * N^0.8 — smooth power law should converge quickly.
	measure := mockMeasure(func(n int64) float64 {
		return 0.01 * math.Pow(float64(n), 0.8)
	})
	cfg := BenchmarkConfig{
		ErrorThreshold: 0.05,
		MaxPointsPerOp: 20,
	}

	points := AdaptiveSample1D(measure, 1024, 64*1024*1024, 8, cfg)

	// Should converge with relatively few points (8 initial + a few refinements)
	assert.LessOrEqual(t, len(points), 12, "smooth function should converge quickly")
	assert.GreaterOrEqual(t, len(points), 8, "should have at least the initial grid")

	// Verify points are sorted by Shape[0]
	for i := 1; i < len(points); i++ {
		assert.Greater(t, points[i].Shape[0], points[i-1].Shape[0])
	}
}

func TestAdaptiveSample1D_SharpKnee(t *testing.T) {
	// f(N) has a sharp transition (knee) at N=10000:
	//   N < 10000: constant 5.0us (kernel launch overhead)
	//   N >= 10000: 5.0 + 0.001*(N-10000) (bandwidth-limited)
	measure := mockMeasure(func(n int64) float64 {
		if n < 10000 {
			return 5.0
		}
		return 5.0 + 0.001*float64(n-10000)
	})
	cfg := BenchmarkConfig{
		ErrorThreshold: 0.05,
		MaxPointsPerOp: 20,
	}

	points := AdaptiveSample1D(measure, 1024, 64*1024*1024, 8, cfg)

	// Should add extra points around the knee
	assert.Greater(t, len(points), 8, "should refine around the knee point")

	// Verify there are points near the knee (N ≈ 10000)
	hasNearKnee := false
	for _, pt := range points {
		if pt.Shape[0] >= 5000 && pt.Shape[0] <= 20000 {
			hasNearKnee = true
			break
		}
	}
	assert.True(t, hasNearKnee, "should have points near the knee at N=10000")
}

func TestAdaptiveSample1D_BudgetLimit(t *testing.T) {
	// Even if the function never converges, should stop at MaxPointsPerOp.
	measure := mockMeasure(func(n int64) float64 {
		// Chaotic function that will never converge
		return math.Sin(float64(n)/1000) * float64(n)
	})
	cfg := BenchmarkConfig{
		ErrorThreshold: 0.001, // very tight threshold — won't converge
		MaxPointsPerOp: 15,
	}

	points := AdaptiveSample1D(measure, 1024, 64*1024*1024, 8, cfg)
	assert.LessOrEqual(t, len(points), cfg.MaxPointsPerOp,
		"must respect budget limit")
}

func TestAdaptiveSample1D_AlreadyConverged(t *testing.T) {
	// Perfect power law — initial grid should be sufficient, no refinement needed.
	measure := mockMeasure(func(n int64) float64 {
		return math.Pow(float64(n), 1.0) // perfectly linear in log-log
	})
	cfg := BenchmarkConfig{
		ErrorThreshold: 0.05,
		MaxPointsPerOp: 20,
	}

	points := AdaptiveSample1D(measure, 1024, 64*1024*1024, 8, cfg)
	assert.Equal(t, 8, len(points), "pure power law needs no refinement")
}

func TestAdaptiveSample1D_MinMaxRange(t *testing.T) {
	// First and last points should be at the specified min/max.
	measure := mockMeasure(func(n int64) float64 {
		return float64(n)
	})
	cfg := BenchmarkConfig{
		ErrorThreshold: 0.05,
		MaxPointsPerOp: 20,
	}

	points := AdaptiveSample1D(measure, 512, 1048576, 6, cfg)
	require.GreaterOrEqual(t, len(points), 6)
	assert.Equal(t, int64(512), points[0].Shape[0], "first point should be at min")
	assert.Equal(t, int64(1048576), points[len(points)-1].Shape[0], "last point should be at max")
}

func TestFindMaxInterpolationError(t *testing.T) {
	// Create points where interval 1-2 has the highest error.
	// Points: (100, 1.0), (1000, 10.0), (10000, 50.0)
	// Intervals: [100,1000] follows power law, [1000,10000] deviates
	points := makePoints([][2]float64{
		{100, 1.0},
		{1000, 10.0},  // consistent with power law
		{10000, 50.0},  // deviates (should be 100.0 for power law)
	})

	measure := mockMeasure(func(n int64) float64 {
		// True function: different from what the points suggest
		if n < 5000 {
			return 0.01 * float64(n)
		}
		return 0.005 * float64(n) // slope change
	})

	maxErr, maxIdx := findMaxInterpolationError(points, measure)
	assert.Greater(t, maxErr, 0.0, "should find some error")
	assert.GreaterOrEqual(t, maxIdx, 0)
	assert.Less(t, maxIdx, len(points)-1)
}

func TestLogMidpoint(t *testing.T) {
	// Geometric midpoint of 100 and 10000 = sqrt(100*10000) = 1000
	mid := logMidpoint(100, 10000)
	assert.Equal(t, int64(1000), mid)
}

func TestLogMidpoint_AdjacentValues(t *testing.T) {
	// Small gap
	mid := logMidpoint(1000, 1100)
	assert.Greater(t, mid, int64(1000))
	assert.Less(t, mid, int64(1100))
}

func TestInsertSorted(t *testing.T) {
	pts := makePoints([][2]float64{
		{100, 1.0}, {1000, 10.0}, {10000, 100.0},
	})
	newPt := LatencyPoint{Shape: []int64{500}, LatencyUs: 5.0}
	result := insertSorted(pts, newPt)
	require.Len(t, result, 4)
	assert.Equal(t, int64(100), result[0].Shape[0])
	assert.Equal(t, int64(500), result[1].Shape[0])
	assert.Equal(t, int64(1000), result[2].Shape[0])
	assert.Equal(t, int64(10000), result[3].Shape[0])
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./perf/ -run "TestAdaptive|TestFindMax|TestLogMidpoint|TestInsertSorted" -v`

Expected: compilation errors — functions don't exist.

- [ ] **Step 3: Implement `adaptive.go`**

Create `perf/adaptive.go`:

```go
package perf

import (
	"math"
	"sort"
)

// MeasureFunc is a callback that benchmarks an op at a given shape and returns the result.
// This abstraction allows testing with mock measurements.
type MeasureFunc func(shape []int64) LatencyPoint

// AdaptiveSample1D performs adaptive 1D sampling in log space.
//
// Algorithm:
//  1. Create nInitial log-spaced points between shapeMin and shapeMax
//  2. Measure latency at each point
//  3. Find the interval with highest interpolation error
//  4. If error > threshold, measure the midpoint and repeat from step 3
//  5. Stop when all errors < threshold or budget (MaxPointsPerOp) is exhausted
//
// The measure callback is invoked for each point. For real benchmarks, this
// calls the GGML backend. For tests, this can be a mock function.
func AdaptiveSample1D(measure MeasureFunc, shapeMin, shapeMax int64, nInitial int, cfg BenchmarkConfig) []LatencyPoint {
	// Step 1: Initial log-spaced grid
	logMin := math.Log(float64(shapeMin))
	logMax := math.Log(float64(shapeMax))
	points := make([]LatencyPoint, 0, cfg.MaxPointsPerOp)

	for i := 0; i < nInitial; i++ {
		logN := logMin + float64(i)*(logMax-logMin)/float64(nInitial-1)
		N := int64(math.Round(math.Exp(logN)))
		pt := measure([]int64{N})
		points = append(points, pt)
	}

	// Step 2: Adaptive refinement
	for len(points) < cfg.MaxPointsPerOp {
		maxErr, maxIdx := findMaxInterpolationError(points, measure)
		if maxErr < cfg.ErrorThreshold {
			break
		}
		// Measure midpoint of highest-error interval
		midN := logMidpoint(points[maxIdx].Shape[0], points[maxIdx+1].Shape[0])
		// Skip if midpoint is same as either endpoint (can't refine further)
		if midN == points[maxIdx].Shape[0] || midN == points[maxIdx+1].Shape[0] {
			break
		}
		pt := measure([]int64{midN})
		points = insertSorted(points, pt)
	}

	return points
}

// findMaxInterpolationError finds the interval with highest relative error
// between the interpolated value and the actual measured value at the midpoint.
//
// Returns (maxError, intervalIndex). Error is measured in log space:
//   relErr = |log(interpolated) - log(actual)| / |log(actual)|
func findMaxInterpolationError(points []LatencyPoint, measure MeasureFunc) (float64, int) {
	maxErr := 0.0
	maxIdx := 0

	for i := 0; i < len(points)-1; i++ {
		logX1 := math.Log(float64(points[i].Shape[0]))
		logX2 := math.Log(float64(points[i+1].Shape[0]))
		logY1 := math.Log(points[i].LatencyUs)
		logY2 := math.Log(points[i+1].LatencyUs)

		// Interpolated value at log-midpoint
		logMid := (logX1 + logX2) / 2
		logInterp := logY1 + (logY2-logY1)*(logMid-logX1)/(logX2-logX1)

		// Actual measurement at midpoint
		midN := int64(math.Round(math.Exp(logMid)))
		actual := measure([]int64{midN})
		actualLogY := math.Log(actual.LatencyUs)

		// Relative error in log space
		relErr := 0.0
		if actualLogY != 0 {
			relErr = math.Abs(logInterp-actualLogY) / math.Abs(actualLogY)
		}

		if relErr > maxErr {
			maxErr = relErr
			maxIdx = i
		}
	}

	return maxErr, maxIdx
}

// logMidpoint returns the geometric midpoint of two values: round(exp((log(a)+log(b))/2)).
func logMidpoint(a, b int64) int64 {
	logMid := (math.Log(float64(a)) + math.Log(float64(b))) / 2
	return int64(math.Round(math.Exp(logMid)))
}

// insertSorted inserts a LatencyPoint into a sorted slice (by Shape[0]).
func insertSorted(points []LatencyPoint, pt LatencyPoint) []LatencyPoint {
	idx := sort.Search(len(points), func(i int) bool {
		return points[i].Shape[0] >= pt.Shape[0]
	})
	// Don't insert duplicate shapes
	if idx < len(points) && points[idx].Shape[0] == pt.Shape[0] {
		return points
	}
	points = append(points, LatencyPoint{})
	copy(points[idx+1:], points[idx:])
	points[idx] = pt
	return points
}
```

- [ ] **Step 4: Run tests**

Run: `go test ./perf/ -run "TestAdaptive|TestFindMax|TestLogMidpoint|TestInsertSorted" -v`

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add perf/adaptive.go perf/adaptive_test.go
git commit -m "perf: add adaptive sampling with log-space refinement and mock-testable interface"
```

---

## Task 7: Benchmark Runner (`bench.go` Rewrite)

**Files:**
- Rewrite: `perf/bench.go`
- Rewrite: `perf/bench_test.go`

The benchmark runner orchestrates the full calibration: for each operator × dtype, it runs adaptive sampling to produce OperatorCurves. This replaces v1's `RunFullBenchmark`, `benchSingleOp`, `SelectBenchmarkShapes`.

**IMPORTANT:** `bench.go` is the glue between registry, adaptive sampling, and hardware characterization. It uses real GGML backends — most tests here need either mocking or an integration test tag.

- [ ] **Step 1: Write tests for `measureOp` and `RunBenchmark`**

Replace `perf/bench_test.go`:

```go
package perf

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTrimmedMedian_Basic(t *testing.T) {
	values := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	// 10% trim: remove 1 from each end → [2,3,4,5,6,7,8,9], median at index 4 → 6
	result := trimmedMedian(values, 0.10)
	assert.InDelta(t, 6.0, result, 0.001)
}

func TestTrimmedMedian_NoTrim(t *testing.T) {
	values := []float64{5, 1, 3}
	result := trimmedMedian(values, 0.0)
	assert.InDelta(t, 3.0, result, 0.001) // sorted: [1,3,5], median=3
}

func TestTrimmedMedian_AllSame(t *testing.T) {
	values := []float64{42.0, 42.0, 42.0, 42.0}
	result := trimmedMedian(values, 0.10)
	assert.InDelta(t, 42.0, result, 0.001)
}

func TestTrimmedMedian_SingleValue(t *testing.T) {
	values := []float64{7.0}
	result := trimmedMedian(values, 0.10)
	assert.InDelta(t, 7.0, result, 0.001)
}

func TestTrimmedMedian_Empty(t *testing.T) {
	assert.Equal(t, 0.0, trimmedMedian(nil, 0.10))
}

func TestTrimmedMedian_HighTrim(t *testing.T) {
	// TrimPercent so high that trimCount*2 >= len → falls back to no trim
	values := []float64{1, 2, 3}
	result := trimmedMedian(values, 0.5)
	assert.InDelta(t, 2.0, result, 0.001)
}

func TestBuildSamplingGrids_SILU(t *testing.T) {
	grids := buildSamplingGrids("SILU", "f32", "")
	require.Len(t, grids, 1, "SILU has one 1D grid")
	assert.Equal(t, "SILU", grids[0].Op)
	assert.Equal(t, "f32", grids[0].Dtype)
	assert.Nil(t, grids[0].FixedDims)
}

func TestBuildSamplingGrids_MulMat(t *testing.T) {
	grids := buildSamplingGrids("MUL_MAT", "f16", "q4_0")
	// One grid per (M, K) pair from Phase1MulMatFixedDims
	assert.GreaterOrEqual(t, len(grids), 4, "MUL_MAT should have multiple (M,K) grids")
	for _, g := range grids {
		assert.Equal(t, "MUL_MAT", g.Op)
		assert.Equal(t, "q4_0", g.WeightDtype)
		assert.NotNil(t, g.FixedDims)
		assert.Contains(t, g.FixedDims, "M")
		assert.Contains(t, g.FixedDims, "K")
	}
}

func TestBuildSamplingGrids_FlashAttn(t *testing.T) {
	grids := buildSamplingGrids("FLASH_ATTN_EXT", "f16", "")
	require.Len(t, grids, 1, "FLASH_ATTN_EXT has one grid with fixed head_dim/num_heads")
	assert.Equal(t, int64(32), grids[0].FixedDims["num_heads"])
	assert.Equal(t, int64(128), grids[0].FixedDims["head_dim"])
}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `go test ./perf/ -run "TestTrimmedMedian|TestBuildSamplingGrids" -v`

Expected: compilation errors for `buildSamplingGrids`.

- [ ] **Step 3: Rewrite `bench.go`**

Replace `perf/bench.go` entirely:

```go
package perf

import (
	"fmt"
	"log/slog"
	"math"
	"sort"
	"time"

	"github.com/ollama/ollama/ml"
)

// SamplingGridWithFixed extends SamplingGrid with fixed dimensions for multi-dim ops.
type SamplingGridWithFixed struct {
	Op          string
	Dtype       string
	WeightDtype string
	FixedDims   map[string]int64 // nil for 1D ops
}

// buildSamplingGrids creates the grid specifications for one operator + dtype combo.
// For 1D ops: one grid. For MUL_MAT: one grid per (M, K) pair. For FLASH_ATTN: one grid.
func buildSamplingGrids(op, computeDtype, weightDtype string) []SamplingGridWithFixed {
	switch op {
	case "MUL_MAT":
		pairs := Phase1MulMatFixedDims()
		grids := make([]SamplingGridWithFixed, len(pairs))
		for i, pair := range pairs {
			grids[i] = SamplingGridWithFixed{
				Op: op, Dtype: computeDtype, WeightDtype: weightDtype,
				FixedDims: map[string]int64{"M": pair[0], "K": pair[1]},
			}
		}
		return grids
	case "FLASH_ATTN_EXT":
		return []SamplingGridWithFixed{{
			Op: op, Dtype: computeDtype,
			FixedDims: map[string]int64{"num_heads": 32, "head_dim": 128},
		}}
	default:
		return []SamplingGridWithFixed{{
			Op: op, Dtype: computeDtype,
		}}
	}
}

// measureOp benchmarks an operator at one shape point using the GGML backend.
// It creates tensors, runs warmup+measurement, trims outliers, and returns the median latency.
func measureOp(backend ml.Backend, op string, gridPoint []int64, computeDtype string, cfg BenchmarkConfig) LatencyPoint {
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

	tensorShapes := expandShapes(op, gridPoint)

	ctx := backend.NewContext()
	defer ctx.Close()

	// Create input tensors
	inputs := make([]ml.Tensor, runner.NumInputs)
	for i := 0; i < runner.NumInputs; i++ {
		if i < len(tensorShapes) {
			shape := tensorShapes[i]
			intShape := make([]int, len(shape))
			for j, s := range shape {
				intShape[j] = int(s)
			}
			inputs[i] = ctx.Zeros(dt, intShape...)
		}
	}

	// Build computation graph
	out := runner.Run(ctx, inputs)
	if out == nil {
		slog.Warn("op runner returned nil", "op", op)
		return LatencyPoint{Shape: gridPoint}
	}
	ctx.Forward(out)

	// Warmup
	for i := 0; i < cfg.WarmupReps; i++ {
		ctx.Compute(out)
	}

	// Measure
	latencies := make([]float64, cfg.MeasureReps)
	for i := 0; i < cfg.MeasureReps; i++ {
		start := time.Now()
		ctx.Compute(out)
		latencies[i] = float64(time.Since(start).Microseconds())
	}

	median := trimmedMedian(latencies, cfg.TrimPercent)

	// Compute stddev of trimmed set
	sort.Float64s(latencies)
	trimCount := int(math.Round(float64(len(latencies)) * cfg.TrimPercent))
	if trimCount*2 >= len(latencies) {
		trimCount = 0
	}
	trimmed := latencies[trimCount : len(latencies)-trimCount]
	mean := 0.0
	for _, l := range trimmed {
		mean += l
	}
	mean /= float64(len(trimmed))
	variance := 0.0
	for _, l := range trimmed {
		d := l - mean
		variance += d * d
	}
	stddev := math.Sqrt(variance / float64(len(trimmed)))

	return LatencyPoint{
		Shape:     gridPoint,
		LatencyUs: median,
		StddevUs:  stddev,
		Reps:      cfg.MeasureReps,
	}
}

// RunBenchmark executes the full v2 calibration pipeline:
// 1. Hardware characterization (peak TOPS, BW)
// 2. For each op × dtype: adaptive sampling → OperatorCurves
// Returns a complete Profile ready for estimation.
func RunBenchmark(backend ml.Backend, ops []string, dtypes []string, cfg BenchmarkConfig) (*Profile, error) {
	// Step 1: Hardware characterization
	hwResult, err := CharacterizeHardware(backend, cfg)
	if err != nil {
		return nil, fmt.Errorf("hardware characterization: %w", err)
	}
	hwProfile := HWCharResultToHardwareProfile(hwResult, backend)

	profile := &Profile{
		Version:   2,
		Timestamp: time.Now(),
		Hardware:  hwProfile,
	}

	slog.Info("starting operator calibration", "ops", len(ops), "dtypes", len(dtypes))

	// Step 2: Benchmark each op × dtype
	for _, op := range ops {
		opDtypes := dtypes
		// FLASH_ATTN_EXT only uses f16
		if op == "FLASH_ATTN_EXT" {
			opDtypes = []string{"f16"}
		}
		// 1D element-wise ops only use f32
		runner, ok := LookupRegistry(op)
		if !ok {
			slog.Warn("skipping unknown op", "op", op)
			continue
		}
		if len(runner.Dimensions) == 1 && runner.Dimensions[0] == "N" && op != "MUL_MAT" {
			opDtypes = []string{"f32"}
		}

		for _, dtype := range opDtypes {
			weightDtype := ""
			if op == "MUL_MAT" {
				weightDtype = dtype
			}

			grids := buildSamplingGrids(op, dtype, weightDtype)

			for _, grid := range grids {
				slog.Info("benchmarking", "op", op, "dtype", dtype, "fixed", grid.FixedDims)

				var curve OperatorCurve
				curve.Op = op
				curve.ComputeDtype = dtype
				curve.WeightDtype = weightDtype
				curve.Dimensions = sweepDimensions(op)
				curve.FixedDims = grid.FixedDims

				// Determine backend name from first device
				devices := backend.BackendDevices()
				if len(devices) > 0 {
					curve.Backend = devices[0].Library
				}

				switch op {
				case "FLASH_ATTN_EXT":
					curve.Points = benchmarkFlashAttn(backend, dtype, grid.FixedDims, cfg)
				case "MUL_MAT":
					curve.Points = benchmarkMulMat(backend, dtype, grid.FixedDims, cfg)
				default:
					curve.Points = benchmarkElementwise(backend, op, dtype, cfg)
				}

				if len(curve.Points) > 0 {
					profile.Operators = append(profile.Operators, curve)
				}
			}
		}
	}

	return profile, nil
}

// sweepDimensions returns the sweep (non-fixed) dimensions for an op.
func sweepDimensions(op string) []string {
	switch op {
	case "MUL_MAT":
		return []string{"N"}
	case "FLASH_ATTN_EXT":
		return []string{"seq_q", "seq_kv"}
	default:
		return []string{"N"}
	}
}

// benchmarkElementwise uses AdaptiveSample1D for 1D ops.
func benchmarkElementwise(backend ml.Backend, op, dtype string, cfg BenchmarkConfig) []LatencyPoint {
	measure := func(shape []int64) LatencyPoint {
		return measureOp(backend, op, shape, dtype, cfg)
	}
	return AdaptiveSample1D(measure, 1024, 64*1024*1024, 8, cfg)
}

// benchmarkMulMat uses AdaptiveSample1D over N with fixed (M, K).
func benchmarkMulMat(backend ml.Backend, dtype string, fixedDims map[string]int64, cfg BenchmarkConfig) []LatencyPoint {
	M := fixedDims["M"]
	K := fixedDims["K"]
	measure := func(shape []int64) LatencyPoint {
		N := shape[0]
		return measureOp(backend, "MUL_MAT", []int64{M, K, N}, dtype, cfg)
	}
	// N range: 1 to 4096 (batch sizes in inference)
	return AdaptiveSample1D(measure, 1, 4096, 8, cfg)
}

// benchmarkFlashAttn samples two regimes: decode and prefill.
func benchmarkFlashAttn(backend ml.Backend, dtype string, fixedDims map[string]int64, cfg BenchmarkConfig) []LatencyPoint {
	var points []LatencyPoint

	// Decode: seq_q=1, sweep seq_kv
	decodeMeasure := func(shape []int64) LatencyPoint {
		seqKV := shape[0]
		pt := measureOp(backend, "FLASH_ATTN_EXT", []int64{1, seqKV}, dtype, cfg)
		pt.Shape = []int64{1, seqKV}
		return pt
	}
	decodePts := AdaptiveSample1D(decodeMeasure, 64, 16384, 8, cfg)
	points = append(points, decodePts...)

	// Prefill: seq_q=seq_kv, sweep both
	prefillMeasure := func(shape []int64) LatencyPoint {
		seqLen := shape[0]
		pt := measureOp(backend, "FLASH_ATTN_EXT", []int64{seqLen, seqLen}, dtype, cfg)
		pt.Shape = []int64{seqLen, seqLen}
		return pt
	}
	prefillPts := AdaptiveSample1D(prefillMeasure, 64, 16384, 8, cfg)
	points = append(points, prefillPts...)

	return points
}
```

- [ ] **Step 4: Run tests**

Run: `go test ./perf/ -run "TestTrimmedMedian|TestBuildSamplingGrids" -v`

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add perf/bench.go perf/bench_test.go
git commit -m "perf: rewrite bench.go with registry-driven adaptive benchmarking"
```

---

## Task 8: Profile Management (`profile.go` Rewrite)

**Files:**
- Rewrite: `perf/profile.go`
- Rewrite: `perf/profile_test.go`

Profile management handles loading, saving, and merging v2 profiles (JSON files with OperatorCurves).

- [ ] **Step 1: Write failing tests for v2 profile**

Replace `perf/profile_test.go`:

```go
package perf

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestProfile() *Profile {
	return &Profile{
		Version:   2,
		Timestamp: time.Now(),
		Hardware: HardwareProfile{
			Backends: []BackendInfo{
				{Name: "cuda", Device: "RTX 4090", VRAMBytes: 24_000_000_000},
			},
			PeakTOPS:                 map[string]float64{"f16": 330e12, "f32": 82.6e12},
			PeakBandwidthBytesPerSec: 1008e9,
			BalancePoints:            map[string]float64{"f16": 327.38, "f32": 81.94},
		},
		Operators: []OperatorCurve{
			{
				Op: "SILU", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 2.5, Reps: 100},
					{Shape: []int64{65536}, LatencyUs: 15.0, Reps: 100},
					{Shape: []int64{1048576}, LatencyUs: 200.0, Reps: 100},
				},
			},
			{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Dimensions: []string{"N"},
				FixedDims:  map[string]int64{"M": 4096, "K": 4096},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 10.0, Reps: 100},
					{Shape: []int64{32}, LatencyUs: 50.0, Reps: 100},
					{Shape: []int64{4096}, LatencyUs: 3000.0, Reps: 100},
				},
			},
		},
	}
}

func TestProfileRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "profile.json")

	original := newTestProfile()
	err := WriteProfile(path, original)
	require.NoError(t, err)

	loaded, err := LoadProfile(path)
	require.NoError(t, err)

	assert.Equal(t, 2, loaded.Version)
	assert.Equal(t, "cuda", loaded.Hardware.Backends[0].Name)
	assert.InDelta(t, 330e12, loaded.Hardware.PeakTOPS["f16"], 1e6)
	assert.InDelta(t, 1008e9, loaded.Hardware.PeakBandwidthBytesPerSec, 1e6)
	assert.Len(t, loaded.Operators, 2)

	// Verify SILU curve
	silu := loaded.Operators[0]
	assert.Equal(t, "SILU", silu.Op)
	assert.Equal(t, []string{"N"}, silu.Dimensions)
	assert.Nil(t, silu.FixedDims)
	assert.Len(t, silu.Points, 3)

	// Verify MUL_MAT curve with FixedDims
	mm := loaded.Operators[1]
	assert.Equal(t, "MUL_MAT", mm.Op)
	assert.Equal(t, "q4_0", mm.WeightDtype)
	assert.Equal(t, int64(4096), mm.FixedDims["M"])
	assert.Equal(t, int64(4096), mm.FixedDims["K"])
}

func TestLoadProfile_NotFound(t *testing.T) {
	_, err := LoadProfile("/nonexistent/path.json")
	assert.Error(t, err)
}

func TestLoadProfile_InvalidJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.json")
	require.NoError(t, writeFile(path, []byte("not json")))
	_, err := LoadProfile(path)
	assert.Error(t, err)
}

func TestMergeProfile(t *testing.T) {
	existing := newTestProfile()
	update := &Profile{
		Version:   2,
		Timestamp: time.Now(),
		Hardware:  existing.Hardware,
		Operators: []OperatorCurve{
			// New curve not in existing
			{
				Op: "SILU", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{67108864}, LatencyUs: 5000.0, Reps: 100},
				},
			},
			// Duplicate of existing (same OpKey) — should NOT be added
			{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Dimensions: []string{"N"},
				FixedDims:  map[string]int64{"M": 4096, "K": 4096},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 99.0, Reps: 100},
				},
			},
			// New MUL_MAT with different FixedDims
			{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Dimensions: []string{"N"},
				FixedDims:  map[string]int64{"M": 8192, "K": 8192},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 20.0, Reps: 100},
				},
			},
		},
	}

	merged := MergeProfile(existing, update)

	// Should have: 2 original + 1 new SILU + 1 new MUL_MAT(8192) = 4
	// The duplicate MUL_MAT(4096) should be kept from existing (not replaced)
	assert.GreaterOrEqual(t, len(merged.Operators), 3)
}

func TestLookupBackend(t *testing.T) {
	p := newTestProfile()
	info, err := LookupBackendInfo(p, "cuda")
	require.NoError(t, err)
	assert.Equal(t, "RTX 4090", info.Device)
}

func TestLookupBackend_NotFound(t *testing.T) {
	p := newTestProfile()
	_, err := LookupBackendInfo(p, "nonexistent")
	assert.Error(t, err)
}

func TestBenchDir(t *testing.T) {
	dir := BenchDir()
	assert.Contains(t, dir, ".ollama")
	assert.Contains(t, dir, "bench")
}

func TestProfilePath(t *testing.T) {
	path := ProfilePath()
	assert.Contains(t, path, "profile.json")
}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `go test ./perf/ -run "TestProfileRoundTrip|TestLoadProfile|TestMergeProfile|TestLookupBackend|TestBenchDir|TestProfilePath" -v`

Expected: compilation errors — `LookupBackendInfo`, `writeFile` don't exist.

- [ ] **Step 3: Rewrite `profile.go`**

Replace `perf/profile.go`:

```go
package perf

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// BenchDir returns the directory for benchmark data.
func BenchDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, ".ollama", "bench")
}

// ProfilePath returns the default profile file path.
func ProfilePath() string {
	return filepath.Join(BenchDir(), "profile.json")
}

// LoadProfile reads a v2 profile from disk.
func LoadProfile(path string) (*Profile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading profile: %w", err)
	}
	var p Profile
	if err := json.Unmarshal(data, &p); err != nil {
		return nil, fmt.Errorf("parsing profile: %w", err)
	}
	return &p, nil
}

// WriteProfile writes a v2 profile to disk.
func WriteProfile(path string, p *Profile) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// MergeProfile merges new operator curves into an existing profile.
// Curves with matching (Op, Backend, ComputeDtype, WeightDtype, FixedDims) are skipped.
// New curves are appended. Hardware profile is taken from the existing profile.
func MergeProfile(existing, update *Profile) *Profile {
	merged := &Profile{
		Version:   existing.Version,
		Timestamp: update.Timestamp,
		Hardware:  existing.Hardware,
	}

	// Copy existing curves
	merged.Operators = make([]OperatorCurve, len(existing.Operators))
	copy(merged.Operators, existing.Operators)

	// Build lookup of existing curves
	type curveKey struct {
		op, backend, cdt, wdt string
		fixedM, fixedK        int64 // for MUL_MAT
	}
	seen := make(map[curveKey]bool)
	for _, c := range existing.Operators {
		k := curveKey{c.Op, c.Backend, c.ComputeDtype, c.WeightDtype,
			c.FixedDims["M"], c.FixedDims["K"]}
		seen[k] = true
	}

	// Add new curves that don't conflict
	for _, c := range update.Operators {
		k := curveKey{c.Op, c.Backend, c.ComputeDtype, c.WeightDtype,
			c.FixedDims["M"], c.FixedDims["K"]}
		if !seen[k] {
			merged.Operators = append(merged.Operators, c)
			seen[k] = true
		}
	}

	return merged
}

// LookupBackendInfo finds a backend by name in the profile.
func LookupBackendInfo(p *Profile, backendName string) (*BackendInfo, error) {
	for i := range p.Hardware.Backends {
		if p.Hardware.Backends[i].Name == backendName {
			return &p.Hardware.Backends[i], nil
		}
	}
	return nil, fmt.Errorf("backend %q not found in profile", backendName)
}

// writeFile is a helper for tests.
func writeFile(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}
```

- [ ] **Step 4: Run tests**

Run: `go test ./perf/ -run "TestProfileRoundTrip|TestLoadProfile|TestMergeProfile|TestLookupBackend|TestBenchDir|TestProfilePath" -v`

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add perf/profile.go perf/profile_test.go
git commit -m "perf: rewrite profile.go for v2 format with OperatorCurve storage"
```

---

## Task 9: Graph Capture & Shape Extraction (`estimate.go` — Part 1)

**Files:**
- Rewrite: `perf/estimate.go` (first half: `nodeToQueryShape`, `buildModelGraphNodes`)
- Rewrite: `perf/estimate_test.go` (first half: shape extraction tests)

This task implements the bridge between model computation graphs and latency lookup. `nodeToQueryShape` maps a GraphNode to the dimensions needed by the interpolation functions. `buildModelGraphNodes` captures the computation graph from a model.

- [ ] **Step 1: Write failing tests for nodeToQueryShape**

Replace `perf/estimate_test.go`:

```go
package perf

import (
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNodeToQueryShape_SILU(t *testing.T) {
	node := ml.GraphNode{
		Op:           "SILU",
		Backend:      "cuda",
		Shape:        [4]int64{4096, 1, 1, 1},
		ComputeDtype: "f32",
		InputShapes:  [][]int64{{4096, 1, 1, 1}},
	}
	op, shape, cdt, wdt := nodeToQueryShape(node)
	assert.Equal(t, "SILU", op)
	assert.Equal(t, []int64{4096}, shape) // total elements
	assert.Equal(t, "f32", cdt)
	assert.Equal(t, "", wdt)
}

func TestNodeToQueryShape_SILU_MultiDim(t *testing.T) {
	// SILU with multi-dim output: total elements = 4096 * 32 = 131072
	node := ml.GraphNode{
		Op:           "SILU",
		Backend:      "cuda",
		Shape:        [4]int64{4096, 32, 1, 1},
		ComputeDtype: "f32",
		InputShapes:  [][]int64{{4096, 32, 1, 1}},
	}
	op, shape, _, _ := nodeToQueryShape(node)
	assert.Equal(t, "SILU", op)
	assert.Equal(t, []int64{131072}, shape)
}

func TestNodeToQueryShape_MulMat(t *testing.T) {
	// MUL_MAT: InputShapes[0] = weight [M, K], InputShapes[1] = activation [K, N]
	node := ml.GraphNode{
		Op:           "MUL_MAT",
		Backend:      "cuda",
		Shape:        [4]int64{4096, 32, 1, 1}, // output [M, N, ...]
		ComputeDtype: "f16",
		WeightDtype:  "q4_0",
		InputShapes:  [][]int64{{4096, 4096}, {4096, 32}},
	}
	op, shape, cdt, wdt := nodeToQueryShape(node)
	assert.Equal(t, "MUL_MAT", op)
	require.Len(t, shape, 3)
	assert.Equal(t, int64(4096), shape[0]) // M
	assert.Equal(t, int64(4096), shape[1]) // K
	assert.Equal(t, int64(32), shape[2])   // N
	assert.Equal(t, "f16", cdt)
	assert.Equal(t, "q4_0", wdt)
}

func TestNodeToQueryShape_MulMat_LargeN(t *testing.T) {
	node := ml.GraphNode{
		Op:           "MUL_MAT",
		Backend:      "cuda",
		ComputeDtype: "f16",
		WeightDtype:  "q4_0",
		InputShapes:  [][]int64{{14336, 4096}, {4096, 512}},
	}
	_, shape, _, _ := nodeToQueryShape(node)
	assert.Equal(t, int64(14336), shape[0]) // M
	assert.Equal(t, int64(4096), shape[1])  // K
	assert.Equal(t, int64(512), shape[2])   // N (prefill batch)
}

func TestNodeToQueryShape_FlashAttn(t *testing.T) {
	// FLASH_ATTN_EXT: InputShapes[0]=Q [head_dim, num_heads, seq_q, 1]
	//                 InputShapes[1]=K [head_dim, num_heads, seq_kv, 1]
	node := ml.GraphNode{
		Op:           "FLASH_ATTN_EXT",
		Backend:      "cuda",
		ComputeDtype: "f16",
		InputShapes: [][]int64{
			{128, 32, 1, 1},    // Q: head_dim=128, num_heads=32, seq_q=1
			{128, 32, 2048, 1}, // K: seq_kv=2048
		},
	}
	op, shape, _, _ := nodeToQueryShape(node)
	assert.Equal(t, "FLASH_ATTN_EXT", op)
	require.Len(t, shape, 2)
	assert.Equal(t, int64(1), shape[0])    // seq_q
	assert.Equal(t, int64(2048), shape[1]) // seq_kv
}

func TestNodeToQueryShape_FlashAttn_Prefill(t *testing.T) {
	node := ml.GraphNode{
		Op:           "FLASH_ATTN_EXT",
		Backend:      "cuda",
		ComputeDtype: "f16",
		InputShapes: [][]int64{
			{128, 32, 512, 1}, // Q: seq_q=512
			{128, 32, 512, 1}, // K: seq_kv=512
		},
	}
	_, shape, _, _ := nodeToQueryShape(node)
	assert.Equal(t, int64(512), shape[0]) // seq_q
	assert.Equal(t, int64(512), shape[1]) // seq_kv
}

func TestNodeToQueryShape_UnknownOp(t *testing.T) {
	// Unknown op should use total output elements as 1D shape
	node := ml.GraphNode{
		Op:           "CUSTOM_OP",
		Backend:      "cuda",
		Shape:        [4]int64{256, 32, 1, 1},
		ComputeDtype: "f32",
	}
	_, shape, _, _ := nodeToQueryShape(node)
	assert.Equal(t, []int64{8192}, shape) // 256*32
}

func TestNodeToQueryShape_MulMat_InsufficientInputShapes(t *testing.T) {
	// Edge case: MUL_MAT with missing input shapes
	node := ml.GraphNode{
		Op:           "MUL_MAT",
		Backend:      "cuda",
		ComputeDtype: "f16",
		InputShapes:  [][]int64{{4096, 4096}}, // only one input shape
	}
	_, shape, _, _ := nodeToQueryShape(node)
	// Should fall back to output shape total elements
	assert.NotEmpty(t, shape)
}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `go test ./perf/ -run "TestNodeToQueryShape" -v`

Expected: compilation errors.

- [ ] **Step 3: Implement `nodeToQueryShape` and `buildModelGraphNodes`**

Rewrite `perf/estimate.go`:

```go
package perf

import (
	"fmt"
	"log/slog"
	"sort"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

// nodeToQueryShape extracts the performance-relevant dimensions from a GraphNode.
// Returns (op, shape, computeDtype, weightDtype) where shape corresponds to
// the OpRunner.Dimensions for that op type.
func nodeToQueryShape(node ml.GraphNode) (op string, shape []int64, computeDtype, weightDtype string) {
	op = node.Op
	computeDtype = node.ComputeDtype
	weightDtype = node.WeightDtype

	switch op {
	case "MUL_MAT":
		// InputShapes[0] = weight [M, K], InputShapes[1] = activation [K, N]
		if len(node.InputShapes) >= 2 && len(node.InputShapes[0]) >= 2 && len(node.InputShapes[1]) >= 2 {
			M := node.InputShapes[0][0]
			K := node.InputShapes[0][1]
			N := node.InputShapes[1][1]
			shape = []int64{M, K, N}
			return
		}
		// Fallback: use output shape total elements
		shape = []int64{totalElements(node.Shape)}
		return

	case "FLASH_ATTN_EXT":
		// InputShapes[0] = Q [head_dim, num_heads, seq_q, batch]
		// InputShapes[1] = K [head_dim, num_heads, seq_kv, batch]
		if len(node.InputShapes) >= 2 && len(node.InputShapes[0]) >= 3 && len(node.InputShapes[1]) >= 3 {
			seqQ := node.InputShapes[0][2]
			seqKV := node.InputShapes[1][2]
			shape = []int64{seqQ, seqKV}
			return
		}
		shape = []int64{totalElements(node.Shape)}
		return

	default:
		// 1D ops: total elements
		shape = []int64{totalElements(node.Shape)}
		return
	}
}

// totalElements computes the product of non-zero dimensions in a GraphNode shape.
func totalElements(shape [4]int64) int64 {
	total := int64(1)
	for _, d := range shape {
		if d > 0 {
			total *= d
		}
	}
	return total
}

// buildModelGraphNodes loads a model and captures prefill+decode computation graphs.
//
// Uses AllocMemory:false to avoid loading model weights (MB not GB).
// Pattern follows runner/ollamarunner/runner.go:reserveWorstCaseGraph().
//
// Returns separate prefill (batch=512) and decode (batch=1) graphs.
func buildModelGraphNodes(modelPath string) (prefill, decode []ml.GraphNode, err error) {
	m, err := model.New(modelPath, ml.BackendParams{AllocMemory: false})
	if err != nil {
		return nil, nil, fmt.Errorf("load model: %w", err)
	}
	defer m.Backend().Close()

	captureGraph := func(batchSize int) ([]ml.GraphNode, error) {
		ctx := m.Backend().NewContext()
		defer ctx.Close()

		// Construct dummy input batch
		batchInputs := make([]int32, batchSize)
		positions := make([]int32, batchSize)
		sequences := make([]int, batchSize)
		for i := 0; i < batchSize; i++ {
			positions[i] = int32(i)
		}
		batch := input.Batch{
			Inputs:    ctx.Input().FromInts(batchInputs, batchSize),
			Outputs:   ctx.Input().Empty(ml.DTypeI32, 1),
			Positions: positions,
			Sequences: sequences,
		}

		// Initialize cache for graph capture (reserve=true)
		if cache := m.Config().Cache; cache != nil {
			if err := cache.StartForward(ctx, batch, true); err != nil {
				return nil, fmt.Errorf("cache start: %w", err)
			}
		}

		// Build computation graph
		t, err := m.Forward(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("forward: %w", err)
		}

		// Capture graph structure
		ctx.SetBatchSize(batchSize)
		ctx.Forward(t).Reserve()

		return ctx.GraphNodes(), nil
	}

	prefill, err = captureGraph(512)
	if err != nil {
		return nil, nil, fmt.Errorf("prefill graph: %w", err)
	}
	decode, err = captureGraph(1)
	if err != nil {
		return nil, nil, fmt.Errorf("decode graph: %w", err)
	}

	return prefill, decode, nil
}

// lookupLatency finds the estimated latency for one graph node operation.
// Dispatches to the appropriate interpolation function based on op type.
func lookupLatency(profile *Profile, op string, shape []int64,
	computeDtype, weightDtype, backend string) (float64, error) {

	switch op {
	case "MUL_MAT":
		if len(shape) < 3 {
			return 0, fmt.Errorf("MUL_MAT requires 3 shape dims, got %d", len(shape))
		}
		var curves []OperatorCurve
		for _, c := range profile.Operators {
			if c.Op == op && c.ComputeDtype == computeDtype &&
				c.WeightDtype == weightDtype && c.Backend == backend {
				curves = append(curves, c)
			}
		}
		if len(curves) == 0 {
			return 0, fmt.Errorf("uncalibrated: %s(%s/%s on %s)", op, computeDtype, weightDtype, backend)
		}
		return InterpolateMulMat(curves, shape[0], shape[1], shape[2]), nil

	case "FLASH_ATTN_EXT":
		if len(shape) < 2 {
			return 0, fmt.Errorf("FLASH_ATTN_EXT requires 2 shape dims, got %d", len(shape))
		}
		for i := range profile.Operators {
			c := &profile.Operators[i]
			if c.Op == op && c.ComputeDtype == computeDtype && c.Backend == backend {
				return InterpolateFlashAttn(c, shape[0], shape[1]), nil
			}
		}
		return 0, fmt.Errorf("uncalibrated: %s(%s on %s)", op, computeDtype, backend)

	default:
		if len(shape) < 1 {
			return 0, fmt.Errorf("op %s requires at least 1 shape dim", op)
		}
		for _, c := range profile.Operators {
			if c.Op == op && c.ComputeDtype == computeDtype && c.Backend == backend {
				return Interpolate1D(c.Points, shape[0]), nil
			}
		}
		return 0, fmt.Errorf("uncalibrated: %s(%s on %s)", op, computeDtype, backend)
	}
}

// estimatePhase computes total latency for a set of graph nodes with per-op breakdown.
func estimatePhase(profile *Profile, nodes []ml.GraphNode, warnings *[]string) PhaseEstimation {
	opStats := make(map[OpKey]*OpBreakdown)
	var totalUs float64

	for _, node := range nodes {
		if IsZeroCostOp(node.Op) {
			continue
		}
		op, shape, cdt, wdt := nodeToQueryShape(node)
		lat, err := lookupLatency(profile, op, shape, cdt, wdt, node.Backend)
		if err != nil {
			*warnings = append(*warnings, err.Error())
			continue
		}
		totalUs += lat

		key := OpKey{op, node.Backend, cdt, wdt}
		if s, ok := opStats[key]; ok {
			s.Count++
			s.TotalUs += lat
		} else {
			opStats[key] = &OpBreakdown{
				Op: op, Backend: node.Backend,
				ComputeDtype: cdt, WeightDtype: wdt,
				Count: 1, TotalUs: lat,
			}
		}
	}

	// Build top-ops list sorted by TotalUs descending
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

// EstimateModel estimates inference performance for a model using a calibrated profile.
func EstimateModel(profile *Profile, modelPath string) (*EstimateResult, error) {
	prefillNodes, decodeNodes, err := buildModelGraphNodes(modelPath)
	if err != nil {
		return nil, err
	}

	result := &EstimateResult{Model: modelPath}
	result.Prefill = estimatePhase(profile, prefillNodes, &result.Warnings)
	result.Decode = estimatePhase(profile, decodeNodes, &result.Warnings)

	result.PrefillLatencyUs = result.Prefill.TotalLatencyMs * 1000
	result.PrefillMs = result.Prefill.TotalLatencyMs
	if result.Decode.TotalLatencyMs > 0 {
		result.DecodeLatencyUsPerToken = result.Decode.TotalLatencyMs * 1000
		result.DecodeTokensPerSec = 1e6 / result.DecodeLatencyUsPerToken
	}

	return result, nil
}

// RunEstimate is the CLI entry point for estimation.
func RunEstimate(modelRef string, profilePath string) (*EstimateResult, error) {
	if profilePath == "" {
		profilePath = ProfilePath()
	}
	profile, err := LoadProfile(profilePath)
	if err != nil {
		return nil, fmt.Errorf("load profile: %w (have you run 'ollama daop-bench'?)", err)
	}

	ggufPath, err := ResolveModelPath(modelRef)
	if err != nil {
		return nil, err
	}

	return EstimateModel(profile, ggufPath)
}

// Suppress unused import warnings
var _ = slog.Info
```

- [ ] **Step 4: Run nodeToQueryShape tests**

Run: `go test ./perf/ -run "TestNodeToQueryShape" -v`

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add perf/estimate.go perf/estimate_test.go
git commit -m "perf: add nodeToQueryShape, lookupLatency, estimatePhase, buildModelGraphNodes"
```

---

## Task 10: Estimation Pipeline Tests (`estimate.go` — Part 2)

**Files:**
- Modify: `perf/estimate_test.go` (add lookupLatency, estimatePhase tests)

Tests for the complete estimation pipeline using synthetic profiles (no GGML required).

- [ ] **Step 1: Add lookupLatency and estimatePhase tests**

Append to `perf/estimate_test.go`:

```go
// --- lookupLatency tests ---

func makeTestProfileForEstimation() *Profile {
	return &Profile{
		Version: 2,
		Hardware: HardwareProfile{
			Backends:                 []BackendInfo{{Name: "cuda", Device: "RTX 4090"}},
			PeakTOPS:                 map[string]float64{"f16": 330e12},
			PeakBandwidthBytesPerSec: 1008e9,
			BalancePoints:            map[string]float64{"f16": 327.38},
		},
		Operators: []OperatorCurve{
			// SILU 1D curve
			{
				Op: "SILU", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 2.5},
					{Shape: []int64{65536}, LatencyUs: 15.0},
					{Shape: []int64{1048576}, LatencyUs: 200.0},
					{Shape: []int64{16777216}, LatencyUs: 3000.0},
				},
			},
			// MUL_MAT curve 1: (M=4096, K=4096)
			{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Dimensions: []string{"N"},
				FixedDims:  map[string]int64{"M": 4096, "K": 4096},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 10.0},
					{Shape: []int64{32}, LatencyUs: 50.0},
					{Shape: []int64{256}, LatencyUs: 200.0},
					{Shape: []int64{4096}, LatencyUs: 3000.0},
				},
			},
			// MUL_MAT curve 2: (M=14336, K=4096)
			{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Dimensions: []string{"N"},
				FixedDims:  map[string]int64{"M": 14336, "K": 4096},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 25.0},
					{Shape: []int64{32}, LatencyUs: 120.0},
					{Shape: []int64{4096}, LatencyUs: 8000.0},
				},
			},
			// FLASH_ATTN_EXT curve
			{
				Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
				Dimensions: []string{"seq_q", "seq_kv"},
				FixedDims:  map[string]int64{"num_heads": 32, "head_dim": 128},
				Points: []LatencyPoint{
					// Decode points (seq_q=1)
					{Shape: []int64{1, 128}, LatencyUs: 5.0},
					{Shape: []int64{1, 512}, LatencyUs: 15.0},
					{Shape: []int64{1, 2048}, LatencyUs: 55.0},
					// Prefill points (seq_q=seq_kv)
					{Shape: []int64{128, 128}, LatencyUs: 20.0},
					{Shape: []int64{512, 512}, LatencyUs: 100.0},
					{Shape: []int64{2048, 2048}, LatencyUs: 500.0},
				},
			},
		},
	}
}

func TestLookupLatency_SILU(t *testing.T) {
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "SILU", []int64{65536}, "f32", "", "cuda")
	require.NoError(t, err)
	assert.InDelta(t, 15.0, lat, 0.001, "exact match at measured point")
}

func TestLookupLatency_SILU_Interpolated(t *testing.T) {
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "SILU", []int64{500000}, "f32", "", "cuda")
	require.NoError(t, err)
	assert.Greater(t, lat, 15.0)
	assert.Less(t, lat, 200.0)
}

func TestLookupLatency_MulMat_ExactMK(t *testing.T) {
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "MUL_MAT", []int64{4096, 4096, 1}, "f16", "q4_0", "cuda")
	require.NoError(t, err)
	assert.InDelta(t, 10.0, lat, 0.001, "exact M,K,N match")
}

func TestLookupLatency_MulMat_InterpolatedN(t *testing.T) {
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "MUL_MAT", []int64{4096, 4096, 128}, "f16", "q4_0", "cuda")
	require.NoError(t, err)
	assert.Greater(t, lat, 50.0)
	assert.Less(t, lat, 200.0)
}

func TestLookupLatency_FlashAttn_Decode(t *testing.T) {
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "FLASH_ATTN_EXT", []int64{1, 512}, "f16", "", "cuda")
	require.NoError(t, err)
	assert.InDelta(t, 15.0, lat, 0.5)
}

func TestLookupLatency_FlashAttn_Prefill(t *testing.T) {
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "FLASH_ATTN_EXT", []int64{512, 512}, "f16", "", "cuda")
	require.NoError(t, err)
	assert.InDelta(t, 100.0, lat, 1.0)
}

func TestLookupLatency_Uncalibrated(t *testing.T) {
	p := makeTestProfileForEstimation()
	_, err := lookupLatency(p, "GELU", []int64{4096}, "f32", "", "cuda")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "uncalibrated")
}

func TestLookupLatency_WrongBackend(t *testing.T) {
	p := makeTestProfileForEstimation()
	_, err := lookupLatency(p, "SILU", []int64{4096}, "f32", "", "cpu")
	assert.Error(t, err)
}

// --- estimatePhase tests ---

func TestEstimatePhase_SimpleGraph(t *testing.T) {
	p := makeTestProfileForEstimation()
	nodes := []ml.GraphNode{
		{Op: "SILU", Backend: "cuda", Shape: [4]int64{65536, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "SILU", Backend: "cuda", Shape: [4]int64{65536, 1, 1, 1}, ComputeDtype: "f32"},
	}
	var warnings []string
	result := estimatePhase(p, nodes, &warnings)
	assert.Empty(t, warnings)
	// 2 SILU at N=65536, each 15.0us → total 30.0us = 0.03ms
	assert.InDelta(t, 0.03, result.TotalLatencyMs, 0.001)
	assert.Len(t, result.TopOps, 1)
	assert.Equal(t, "SILU", result.TopOps[0].Op)
	assert.Equal(t, 2, result.TopOps[0].Count)
}

func TestEstimatePhase_MixedOps(t *testing.T) {
	p := makeTestProfileForEstimation()
	nodes := []ml.GraphNode{
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		{Op: "SILU", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{14336, 4096}, {4096, 1}}},
	}
	var warnings []string
	result := estimatePhase(p, nodes, &warnings)
	assert.Empty(t, warnings)
	assert.Greater(t, result.TotalLatencyMs, 0.0)
	// MUL_MAT should dominate
	assert.Equal(t, "MUL_MAT", result.TopOps[0].Op)
	assert.Greater(t, result.TopOps[0].Percentage, 0.5)
}

func TestEstimatePhase_SkipsZeroCostOps(t *testing.T) {
	p := makeTestProfileForEstimation()
	nodes := []ml.GraphNode{
		{Op: "VIEW", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "RESHAPE", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "SILU", Backend: "cuda", Shape: [4]int64{65536, 1, 1, 1}, ComputeDtype: "f32"},
	}
	var warnings []string
	result := estimatePhase(p, nodes, &warnings)
	assert.Len(t, result.TopOps, 1)
	assert.Equal(t, "SILU", result.TopOps[0].Op)
}

func TestEstimatePhase_UncalibratedWarning(t *testing.T) {
	p := makeTestProfileForEstimation()
	nodes := []ml.GraphNode{
		{Op: "GELU", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
	}
	var warnings []string
	result := estimatePhase(p, nodes, &warnings)
	assert.NotEmpty(t, warnings)
	assert.Contains(t, warnings[0], "uncalibrated")
	assert.InDelta(t, 0.0, result.TotalLatencyMs, 0.001, "uncalibrated op should not contribute")
}

func TestEstimatePhase_EmptyGraph(t *testing.T) {
	p := makeTestProfileForEstimation()
	var warnings []string
	result := estimatePhase(p, nil, &warnings)
	assert.InDelta(t, 0.0, result.TotalLatencyMs, 0.001)
	assert.Empty(t, result.TopOps)
}

func TestEstimatePhase_LlamaLikeDecodeLayer(t *testing.T) {
	// Simulate one Llama transformer layer in decode mode (batch=1):
	// 4 MUL_MAT (Q, K, V, O projection) + 1 FLASH_ATTN + 2 MUL_MAT (FFN up, down) + SILU
	p := makeTestProfileForEstimation()
	nodes := []ml.GraphNode{
		// Attention QKV projections: [4096, 4096] × [4096, 1]
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		// FLASH_ATTN: decode with seq_kv=2048
		{Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
			InputShapes: [][]int64{{128, 32, 1, 1}, {128, 32, 2048, 1}}},
		// Output projection
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		// FFN: up projection [14336, 4096] × [4096, 1]
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{14336, 4096}, {4096, 1}}},
		// SILU activation
		{Op: "SILU", Backend: "cuda", Shape: [4]int64{14336, 1, 1, 1}, ComputeDtype: "f32"},
		// FFN: down projection [4096, 14336] × [14336, 1]
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 14336}, {14336, 1}}},
		// RMS_NORM and other small ops are VIEW/RESHAPE — ignored
		{Op: "VIEW", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
	}

	var warnings []string
	result := estimatePhase(p, nodes, &warnings)

	// MUL_MAT should dominate in decode (memory-bound, large weights)
	assert.Greater(t, result.TotalLatencyMs, 0.0)
	assert.Equal(t, "MUL_MAT", result.TopOps[0].Op)
	assert.Greater(t, result.TopOps[0].Percentage, 0.5,
		"MUL_MAT should be >50%% of decode latency")

	// Should have warnings only for uncalibrated ops (none here — we have
	// all required curves except for the MUL_MAT(4096,14336) which will use
	// InterpolateMulMat's inverse-distance weighting)
	// Actually MUL_MAT(4096,14336) doesn't exactly match any curve, but
	// InterpolateMulMat will interpolate between available curves.
	// No warnings expected if all ops have matching curves.
}
```

- [ ] **Step 2: Run all estimate tests**

Run: `go test ./perf/ -run "TestNodeToQueryShape|TestLookupLatency|TestEstimatePhase" -v`

Expected: All 19 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add perf/estimate.go perf/estimate_test.go
git commit -m "perf: add estimation pipeline tests (lookupLatency, estimatePhase, Llama layer)"
```

---

## Task 11: CLI Viewer Update (`viewer.go`)

**Files:**
- Rewrite: `perf/viewer.go`

Update the CLI viewer to work with v2 types. The viewer prints human-readable profile summaries and estimation results to the terminal.

- [ ] **Step 1: Rewrite `viewer.go` for v2 types**

Replace `perf/viewer.go`:

```go
package perf

import (
	"fmt"
	"io"
	"sort"
	"strings"
)

// PrintProfile prints a human-readable v2 profile summary to w.
func PrintProfile(w io.Writer, p *Profile, detail bool) {
	fmt.Fprintln(w, "Hardware Profile (v2)")
	fmt.Fprintln(w, strings.Repeat("-", 60))

	for _, bi := range p.Hardware.Backends {
		fmt.Fprintf(w, "  Backend: %s (%s)\n", bi.Name, bi.Device)
		if bi.VRAMBytes > 0 {
			fmt.Fprintf(w, "  VRAM: %s\n", formatSI(float64(bi.VRAMBytes), "B"))
		}
	}

	for dtype, tops := range p.Hardware.PeakTOPS {
		bp := p.Hardware.BalancePoints[dtype]
		fmt.Fprintf(w, "  %s: %s peak, %.1f FLOP/byte balance\n",
			dtype, formatSI(tops, "OPS"), bp)
	}
	fmt.Fprintf(w, "  Bandwidth: %s\n", formatSI(p.Hardware.PeakBandwidthBytesPerSec, "B/s"))

	fmt.Fprintf(w, "\nOperator Curves: %d\n", len(p.Operators))
	fmt.Fprintln(w, strings.Repeat("-", 60))

	if detail {
		sorted := make([]OperatorCurve, len(p.Operators))
		copy(sorted, p.Operators)
		sort.Slice(sorted, func(i, j int) bool {
			if sorted[i].Op != sorted[j].Op {
				return sorted[i].Op < sorted[j].Op
			}
			return sorted[i].ComputeDtype < sorted[j].ComputeDtype
		})

		for _, c := range sorted {
			label := c.Op
			if c.WeightDtype != "" {
				label += fmt.Sprintf(" (%s->%s)", c.WeightDtype, c.ComputeDtype)
			} else {
				label += fmt.Sprintf(" (%s)", c.ComputeDtype)
			}
			if len(c.FixedDims) > 0 {
				label += fmt.Sprintf(" fixed=%v", c.FixedDims)
			}
			fmt.Fprintf(w, "  %-50s %d points\n", label, len(c.Points))
		}
	} else {
		// Summary: count curves per op
		opCounts := make(map[string]int)
		for _, c := range p.Operators {
			opCounts[c.Op]++
		}
		for op, count := range opCounts {
			fmt.Fprintf(w, "  %-20s %d curves\n", op, count)
		}
	}
}

// PrintEstimateResult prints a human-readable v2 estimation result to w.
func PrintEstimateResult(w io.Writer, r *EstimateResult, detail bool) {
	fmt.Fprintf(w, "Model: %s\n\n", r.Model)

	fmt.Fprintln(w, "Prefill")
	fmt.Fprintln(w, strings.Repeat("-", 50))
	fmt.Fprintf(w, "  Latency: %.1fms (%.0f tok/s)\n",
		r.Prefill.TotalLatencyMs, r.Prefill.TokensPerSec)
	printTopOps(w, r.Prefill.TopOps, detail)

	fmt.Fprintln(w)

	fmt.Fprintln(w, "Decode (per token)")
	fmt.Fprintln(w, strings.Repeat("-", 50))
	fmt.Fprintf(w, "  Latency: %.3fms/tok (%.0f tok/s)\n",
		r.Decode.TotalLatencyMs, r.Decode.TokensPerSec)
	printTopOps(w, r.Decode.TopOps, detail)

	if len(r.Warnings) > 0 {
		fmt.Fprintln(w)
		fmt.Fprintln(w, "Warnings:")
		for _, w2 := range r.Warnings {
			fmt.Fprintf(w, "  ! %s\n", w2)
		}
	}
}

func printTopOps(w io.Writer, ops []OpBreakdown, detail bool) {
	if len(ops) == 0 {
		return
	}
	fmt.Fprintln(w, "  Top ops:")
	limit := 5
	if detail {
		limit = 10
	}
	for i, op := range ops {
		if i >= limit {
			break
		}
		dtype := op.ComputeDtype
		if op.WeightDtype != "" {
			dtype = op.WeightDtype
		}
		fmt.Fprintf(w, "    %-16s %-8s %4dx  %8.1fus  %5.1f%%\n",
			op.Op, dtype, op.Count, op.TotalUs, op.Percentage*100)
	}
}

func formatSI(val float64, unit string) string {
	switch {
	case val >= 1e12:
		return fmt.Sprintf("%.1f T%s", val/1e12, unit)
	case val >= 1e9:
		return fmt.Sprintf("%.1f G%s", val/1e9, unit)
	case val >= 1e6:
		return fmt.Sprintf("%.1f M%s", val/1e6, unit)
	case val >= 1e3:
		return fmt.Sprintf("%.1f K%s", val/1e3, unit)
	default:
		return fmt.Sprintf("%.1f %s", val, unit)
	}
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-1] + "..."
}
```

- [ ] **Step 2: Verify compilation**

Run: `go build ./perf/`

Expected: Compiles without errors.

- [ ] **Step 3: Commit**

```bash
git add perf/viewer.go
git commit -m "perf: update CLI viewer for v2 OperatorCurve types"
```

---

## Task 12: HTML Viewer (`viewer_html.go` + `viewer.html`)

**Files:**
- Create: `perf/viewer.html`
- Create: `perf/viewer_html.go`
- Create: `perf/viewer_html_test.go`

The HTML viewer generates a self-contained HTML file with Plotly.js charts for interactive visualization of benchmark data.

- [ ] **Step 1: Write failing tests for HTML generation**

Create `perf/viewer_html_test.go`:

```go
package perf

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGenerateHTMLViewer_ProducesValidHTML(t *testing.T) {
	p := newTestProfile()
	dir := t.TempDir()
	path := filepath.Join(dir, "viewer.html")

	err := GenerateHTMLViewer(p, path)
	require.NoError(t, err)

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	html := string(data)

	assert.Contains(t, html, "<!DOCTYPE html>")
	assert.Contains(t, html, "</html>")
	assert.Contains(t, html, "plotly")
	assert.Contains(t, html, "PROFILE_DATA")
}

func TestGenerateHTMLViewer_ContainsProfileData(t *testing.T) {
	p := newTestProfile()
	dir := t.TempDir()
	path := filepath.Join(dir, "viewer.html")

	err := GenerateHTMLViewer(p, path)
	require.NoError(t, err)

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	html := string(data)

	// Should contain op names from the profile
	assert.Contains(t, html, "SILU")
	assert.Contains(t, html, "MUL_MAT")
}

func TestGenerateHTMLViewer_ContainsChartElements(t *testing.T) {
	p := newTestProfile()
	dir := t.TempDir()
	path := filepath.Join(dir, "viewer.html")

	err := GenerateHTMLViewer(p, path)
	require.NoError(t, err)

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	html := string(data)

	// Should have chart container elements
	assert.Contains(t, html, "chart-container")
	// Should have op selector
	assert.Contains(t, html, "select")
}

func TestGenerateHTMLViewer_EmptyProfile(t *testing.T) {
	p := &Profile{Version: 2}
	dir := t.TempDir()
	path := filepath.Join(dir, "viewer.html")

	err := GenerateHTMLViewer(p, path)
	require.NoError(t, err)

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.True(t, len(data) > 0)
}

func TestGenerateHTMLViewer_FileSize(t *testing.T) {
	p := newTestProfile()
	dir := t.TempDir()
	path := filepath.Join(dir, "viewer.html")

	err := GenerateHTMLViewer(p, path)
	require.NoError(t, err)

	info, err := os.Stat(path)
	require.NoError(t, err)
	// HTML file should be reasonable size (not empty, not enormous)
	assert.Greater(t, info.Size(), int64(1000), "HTML should be > 1KB")
	assert.Less(t, info.Size(), int64(1000000), "HTML should be < 1MB")
}

func TestGenerateHTMLViewer_SpecialCharsInData(t *testing.T) {
	// Ensure JSON embedding handles special characters
	p := &Profile{
		Version: 2,
		Hardware: HardwareProfile{
			Backends: []BackendInfo{{Name: "cuda", Device: "NVIDIA RTX 4090 <test>"}},
		},
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "viewer.html")

	err := GenerateHTMLViewer(p, path)
	require.NoError(t, err)

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	// Should not have unescaped HTML tags from device name
	assert.False(t, strings.Contains(string(data), "<test>"),
		"device name should be JSON-escaped, not raw HTML")
}
```

- [ ] **Step 2: Create the HTML template**

Create `perf/viewer.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DAOP Benchmark Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f8f9fa; }
  .header { margin-bottom: 20px; }
  .header h1 { margin: 0; color: #333; }
  .header .meta { color: #666; font-size: 14px; }
  .controls { margin: 15px 0; display: flex; gap: 15px; align-items: center; }
  .controls select, .controls button { padding: 6px 12px; border: 1px solid #ccc; border-radius: 4px; }
  .controls button { cursor: pointer; background: #fff; }
  .controls button.active { background: #007bff; color: #fff; border-color: #007bff; }
  #chart-container { background: #fff; border-radius: 8px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .hw-info { background: #fff; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .hw-info table { border-collapse: collapse; }
  .hw-info td { padding: 4px 12px 4px 0; }
  .hw-info td:first-child { font-weight: 600; color: #555; }
</style>
</head>
<body>

<div class="header">
  <h1>DAOP Benchmark Viewer</h1>
  <div class="meta" id="profile-meta"></div>
</div>

<div class="hw-info" id="hw-info"></div>

<div class="controls">
  <label>Op: <select id="op-select"></select></label>
  <button id="btn-log" class="active" onclick="toggleScale('log')">Log Scale</button>
  <button id="btn-linear" onclick="toggleScale('linear')">Linear Scale</button>
</div>

<div id="chart-container"></div>

<script>
const PROFILE_DATA = __PROFILE_JSON__;

let currentScale = 'log';

function init() {
  const meta = document.getElementById('profile-meta');
  meta.textContent = `Version ${PROFILE_DATA.version} | ${PROFILE_DATA.timestamp || 'N/A'}`;

  renderHWInfo();
  populateOpSelect();
  renderChart();
}

function renderHWInfo() {
  const hw = PROFILE_DATA.hardware;
  if (!hw) return;
  let html = '<table>';
  if (hw.backends) {
    hw.backends.forEach(b => {
      html += `<tr><td>Backend</td><td>${escapeHtml(b.name)} (${escapeHtml(b.device)})</td></tr>`;
    });
  }
  if (hw.peak_tops) {
    Object.entries(hw.peak_tops).forEach(([dt, v]) => {
      html += `<tr><td>Peak TOPS (${dt})</td><td>${formatSI(v, 'OPS')}</td></tr>`;
    });
  }
  if (hw.peak_bandwidth_bytes_sec) {
    html += `<tr><td>Peak BW</td><td>${formatSI(hw.peak_bandwidth_bytes_sec, 'B/s')}</td></tr>`;
  }
  html += '</table>';
  document.getElementById('hw-info').innerHTML = html;
}

function populateOpSelect() {
  const ops = new Set();
  (PROFILE_DATA.operators || []).forEach(c => ops.add(c.op));
  const select = document.getElementById('op-select');
  select.innerHTML = '';
  ops.forEach(op => {
    const opt = document.createElement('option');
    opt.value = op; opt.textContent = op;
    select.appendChild(opt);
  });
  select.onchange = renderChart;
}

function renderChart() {
  const op = document.getElementById('op-select').value;
  const curves = (PROFILE_DATA.operators || []).filter(c => c.op === op);
  if (curves.length === 0) {
    Plotly.purge('chart-container');
    return;
  }

  const traces = curves.map((curve, idx) => {
    let label = curve.op;
    if (curve.weight_dtype) label += ` ${curve.weight_dtype}`;
    if (curve.fixed_dims) {
      const dims = Object.entries(curve.fixed_dims).map(([k,v]) => `${k}=${v}`).join(',');
      label += ` [${dims}]`;
    }

    const xs = curve.points.map(p => p.shape[p.shape.length - 1]);
    const ys = curve.points.map(p => p.latency_us);
    const texts = curve.points.map(p =>
      `Shape: [${p.shape.join(', ')}]<br>Latency: ${p.latency_us.toFixed(2)} us` +
      (p.stddev_us ? `<br>Stddev: ${p.stddev_us.toFixed(2)} us` : '')
    );

    return {
      x: xs, y: ys, text: texts,
      mode: 'lines+markers', name: label,
      hovertemplate: '%{text}<extra></extra>',
    };
  });

  const xLabel = curves[0].dimensions ? curves[0].dimensions[curves[0].dimensions.length - 1] : 'N';
  const layout = {
    title: `${op} Latency Curves`,
    xaxis: { title: xLabel, type: currentScale },
    yaxis: { title: 'Latency (us)', type: currentScale },
    hovermode: 'closest',
    height: 500,
  };

  Plotly.newPlot('chart-container', traces, layout, {responsive: true});
}

function toggleScale(scale) {
  currentScale = scale;
  document.getElementById('btn-log').classList.toggle('active', scale === 'log');
  document.getElementById('btn-linear').classList.toggle('active', scale === 'linear');
  renderChart();
}

function formatSI(val, unit) {
  if (val >= 1e12) return (val/1e12).toFixed(1) + ' T' + unit;
  if (val >= 1e9) return (val/1e9).toFixed(1) + ' G' + unit;
  if (val >= 1e6) return (val/1e6).toFixed(1) + ' M' + unit;
  if (val >= 1e3) return (val/1e3).toFixed(1) + ' K' + unit;
  return val.toFixed(1) + ' ' + unit;
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(s));
  return div.innerHTML;
}

init();
</script>
</body>
</html>
```

- [ ] **Step 3: Implement `viewer_html.go`**

Create `perf/viewer_html.go`:

```go
package perf

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

//go:embed viewer.html
var htmlTemplate string

// GenerateHTMLViewer creates a self-contained HTML file with interactive charts.
// The profile data is embedded as JSON in a <script> tag.
func GenerateHTMLViewer(profile *Profile, outputPath string) error {
	profileJSON, err := json.Marshal(profile)
	if err != nil {
		return fmt.Errorf("marshal profile: %w", err)
	}

	// Replace the placeholder with actual JSON data
	html := strings.Replace(htmlTemplate, "__PROFILE_JSON__", string(profileJSON), 1)

	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return err
	}

	return os.WriteFile(outputPath, []byte(html), 0o644)
}

// OpenHTMLViewer opens the HTML file in the default browser.
func OpenHTMLViewer(path string) error {
	// Platform-specific open command
	absPath, err := filepath.Abs(path)
	if err != nil {
		return err
	}

	// Use url format for cross-platform compatibility
	fmt.Printf("Open in browser: file://%s\n", filepath.ToSlash(absPath))
	return nil
}
```

- [ ] **Step 4: Run HTML viewer tests**

Run: `go test ./perf/ -run "TestGenerateHTMLViewer" -v`

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add perf/viewer.html perf/viewer_html.go perf/viewer_html_test.go
git commit -m "perf: add interactive HTML viewer with Plotly.js charts"
```

---

## Task 13: Delete Roofline, Clean Up Old Tests, Final Compilation

**Files:**
- Delete: `perf/roofline.go`
- Delete: `perf/roofline_test.go`
- Rewrite: `perf/integration_test.go` (v2 integration tests)

This task removes all remaining v1 code and ensures the entire `perf/` package compiles and tests pass.

- [ ] **Step 1: Delete `roofline.go` and `roofline_test.go`**

```bash
git rm perf/roofline.go perf/roofline_test.go
```

- [ ] **Step 2: Rewrite `integration_test.go` for v2**

Replace `perf/integration_test.go`:

```go
package perf

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/ml"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestProfileRoundTripIntegration tests the full profile write/load cycle.
func TestProfileRoundTripIntegration(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "profile.json")

	original := &Profile{
		Version:   2,
		Timestamp: time.Now(),
		Hardware: HardwareProfile{
			Backends:                 []BackendInfo{{Name: "cuda", Device: "RTX 4090", VRAMBytes: 24_000_000_000}},
			PeakTOPS:                 map[string]float64{"f16": 330e12, "f32": 82.6e12},
			PeakBandwidthBytesPerSec: 1008e9,
			BalancePoints:            map[string]float64{"f16": 327.38},
		},
		Operators: []OperatorCurve{
			{
				Op: "SILU", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 2.5, StddevUs: 0.1, Reps: 100},
					{Shape: []int64{1048576}, LatencyUs: 200.0, StddevUs: 5.0, Reps: 100},
				},
			},
			{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Dimensions: []string{"N"},
				FixedDims:  map[string]int64{"M": 4096, "K": 4096},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 10.0, StddevUs: 0.5, Reps: 100},
					{Shape: []int64{4096}, LatencyUs: 3000.0, StddevUs: 50.0, Reps: 100},
				},
			},
			{
				Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
				Dimensions: []string{"seq_q", "seq_kv"},
				FixedDims:  map[string]int64{"num_heads": 32, "head_dim": 128},
				Points: []LatencyPoint{
					{Shape: []int64{1, 128}, LatencyUs: 5.0, Reps: 100},
					{Shape: []int64{1, 2048}, LatencyUs: 55.0, Reps: 100},
					{Shape: []int64{512, 512}, LatencyUs: 100.0, Reps: 100},
				},
			},
		},
	}

	err := WriteProfile(path, original)
	require.NoError(t, err)

	loaded, err := LoadProfile(path)
	require.NoError(t, err)

	assert.Equal(t, 2, loaded.Version)
	assert.Len(t, loaded.Operators, 3)
	assert.Equal(t, "SILU", loaded.Operators[0].Op)
	assert.Equal(t, "MUL_MAT", loaded.Operators[1].Op)
	assert.Equal(t, "FLASH_ATTN_EXT", loaded.Operators[2].Op)
	assert.Equal(t, int64(4096), loaded.Operators[1].FixedDims["M"])
}

// TestEndToEndEstimation_Synthetic tests the full estimation pipeline
// with a synthetic profile and manually constructed graph nodes.
func TestEndToEndEstimation_Synthetic(t *testing.T) {
	p := makeTestProfileForEstimation()

	// Simulate a 32-layer Llama-8B decode pass
	var nodes []ml.GraphNode
	for layer := 0; layer < 32; layer++ {
		// 4 attention MUL_MATs: [4096, 4096] × [4096, 1]
		for i := 0; i < 4; i++ {
			nodes = append(nodes, ml.GraphNode{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				InputShapes: [][]int64{{4096, 4096}, {4096, 1}},
			})
		}
		// FLASH_ATTN decode
		nodes = append(nodes, ml.GraphNode{
			Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
			InputShapes: [][]int64{{128, 32, 1, 1}, {128, 32, 2048, 1}},
		})
		// FFN: up [14336, 4096] × [4096, 1]
		nodes = append(nodes, ml.GraphNode{
			Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{14336, 4096}, {4096, 1}},
		})
		// SILU
		nodes = append(nodes, ml.GraphNode{
			Op: "SILU", Backend: "cuda", Shape: [4]int64{14336, 1, 1, 1}, ComputeDtype: "f32",
		})
		// FFN: down [4096, 14336] × [14336, 1]
		nodes = append(nodes, ml.GraphNode{
			Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 14336}, {14336, 1}},
		})
		// VIEW, RESHAPE — should be skipped
		nodes = append(nodes, ml.GraphNode{Op: "VIEW", Backend: "cuda"})
		nodes = append(nodes, ml.GraphNode{Op: "RESHAPE", Backend: "cuda"})
	}

	var warnings []string
	result := estimatePhase(p, nodes, &warnings)

	// Sanity checks
	assert.Greater(t, result.TotalLatencyMs, 0.0)

	// MUL_MAT should dominate (>50% in decode)
	require.NotEmpty(t, result.TopOps)
	assert.Equal(t, "MUL_MAT", result.TopOps[0].Op)
	assert.Greater(t, result.TopOps[0].Percentage, 0.5)

	// tok/s should be in a reasonable range for decode
	// With synthetic data, we just check it's positive
	assert.Greater(t, result.TokensPerSec, 0.0)

	t.Logf("Synthetic Llama-8B decode: %.2f ms/tok, %.0f tok/s",
		result.TotalLatencyMs, result.TokensPerSec)
	t.Logf("Top ops:")
	for _, op := range result.TopOps {
		t.Logf("  %-16s %4dx  %8.1fus  %.1f%%",
			op.Op, op.Count, op.TotalUs, op.Percentage*100)
	}
}

// TestHTMLViewerIntegration tests generating HTML viewer from a full profile.
func TestHTMLViewerIntegration(t *testing.T) {
	p := makeTestProfileForEstimation()
	dir := t.TempDir()
	path := filepath.Join(dir, "viewer.html")

	err := GenerateHTMLViewer(p, path)
	require.NoError(t, err)

	// Verify the file is non-trivial
	info, err := os.Stat(path)
	require.NoError(t, err)
	assert.Greater(t, info.Size(), int64(5000), "HTML viewer should have substantial content")
}
```

Add the missing `os` import by checking if it's needed — it's already used via `os.Stat` in the test above. Add to imports:

```go
import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/ml"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)
```

- [ ] **Step 3: Run all tests**

Run: `go test ./perf/ -v -count=1`

Expected: All tests PASS. If any compilation errors remain from v1 references, fix them.

**Common issues to watch for:**
- If `resolve_test.go` references old types, update it
- If any file still imports `perf/roofline`, update the import
- If `HardwareBenchmark`, `OperatorBenchmark`, `RawData`, etc. are still referenced, remove those references

- [ ] **Step 4: Clean up any remaining v1 references**

Check for any remaining references to removed types:

```bash
grep -r "RawData\|HardwareBenchmark\|OperatorBenchmark\|OperatorProfile\|ComputeFLOPs\|ComputeBytes\|CanComputeFLOPs\|EstimateOpCost\|LookupEta\|OpCost\|BackendProfile\|InterconnectInfo\|EstimateConfig\|OpStats\|EstimateGraphLatency\|ComputePhaseEstimation\|BuildSummary\|ProcessRawToProfile\|ComputeEtaFromPoints\|LookupInterconnectBW\|EstimateTransferCost\|PredefinedOps\|RunFullBenchmark\|RunUpdateBenchmark\|SelectBenchmarkShapes\|ShouldAdaptiveExtend" perf/
```

Fix any references found. This may involve updating `resolve_test.go` or other test files.

- [ ] **Step 5: Verify clean build**

```bash
go build ./perf/
go vet ./perf/
go test ./perf/ -v -count=1
```

Expected: All pass with no warnings.

- [ ] **Step 6: Commit**

```bash
git add -A perf/
git commit -m "perf: remove roofline.go, rewrite integration tests for v2 empirical model"
```

---

## Task 14: Final Integration — Real GGML Backend Tests

**Files:**
- Create: `perf/ggml_integration_test.go`

These tests require a GGML backend (cmake build). They verify the full pipeline works with real hardware. They are skipped when no backend is available.

**IMPORTANT:** The user has confirmed they will `cmake` build the GGML DLLs. These tests should use a runtime check to skip if no backend is found, NOT a build tag (since the code compiles either way — it's the runtime that needs the DLLs).

- [ ] **Step 1: Write integration tests**

Create `perf/ggml_integration_test.go`:

```go
package perf

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/ml"
	_ "github.com/ollama/ollama/ml/backend" // register backends
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// skipIfNoBackend skips the test if no GGML backend is available.
func skipIfNoBackend(t *testing.T) ml.Backend {
	t.Helper()
	if os.Getenv("DAOP_INTEGRATION") == "" {
		t.Skip("set DAOP_INTEGRATION=1 to run GGML integration tests")
	}
	// Try to create a backend
	b, err := ml.NewBackend(ml.BackendParams{AllocMemory: true})
	if err != nil {
		t.Skipf("no GGML backend available: %v", err)
	}
	return b
}

func TestGGML_MeasureOp_SILU(t *testing.T) {
	backend := skipIfNoBackend(t)
	defer backend.Close()

	cfg := BenchmarkConfig{WarmupReps: 2, MeasureReps: 10, TrimPercent: 0.1}
	pt := measureOp(backend, "SILU", []int64{65536}, "f32", cfg)

	assert.Greater(t, pt.LatencyUs, 0.0, "SILU should have measurable latency")
	assert.Greater(t, pt.Reps, 0)
	t.Logf("SILU N=65536: %.2f us (stddev: %.2f us)", pt.LatencyUs, pt.StddevUs)
}

func TestGGML_MeasureOp_SILU_Monotonic(t *testing.T) {
	backend := skipIfNoBackend(t)
	defer backend.Close()

	cfg := BenchmarkConfig{WarmupReps: 2, MeasureReps: 20, TrimPercent: 0.1}
	sizes := []int64{1024, 16384, 262144, 4194304}
	var prev float64
	for _, N := range sizes {
		pt := measureOp(backend, "SILU", []int64{N}, "f32", cfg)
		t.Logf("SILU N=%d: %.2f us", N, pt.LatencyUs)
		if prev > 0 {
			// Latency should generally increase with size (not necessarily strictly,
			// but the overall trend should be clear)
			assert.Greater(t, pt.LatencyUs, prev*0.5,
				"SILU N=%d should not be much faster than N=%d", N, N/16)
		}
		prev = pt.LatencyUs
	}
}

func TestGGML_MeasureOp_MulMat(t *testing.T) {
	backend := skipIfNoBackend(t)
	defer backend.Close()

	cfg := BenchmarkConfig{WarmupReps: 2, MeasureReps: 10, TrimPercent: 0.1}
	pt := measureOp(backend, "MUL_MAT", []int64{4096, 4096, 1}, "f16", cfg)

	assert.Greater(t, pt.LatencyUs, 0.0)
	t.Logf("MUL_MAT [4096,4096,1] f16: %.2f us", pt.LatencyUs)
}

func TestGGML_AdaptiveSample_SILU(t *testing.T) {
	backend := skipIfNoBackend(t)
	defer backend.Close()

	cfg := BenchmarkConfig{
		WarmupReps: 2, MeasureReps: 20, TrimPercent: 0.1,
		ErrorThreshold: 0.05, MaxPointsPerOp: 15,
	}

	measure := func(shape []int64) LatencyPoint {
		return measureOp(backend, "SILU", shape, "f32", cfg)
	}

	points := AdaptiveSample1D(measure, 1024, 16*1024*1024, 6, cfg)

	assert.GreaterOrEqual(t, len(points), 6)
	assert.LessOrEqual(t, len(points), 15)

	// Points should be sorted
	for i := 1; i < len(points); i++ {
		assert.Greater(t, points[i].Shape[0], points[i-1].Shape[0])
	}

	t.Logf("Adaptive SILU: %d points", len(points))
	for _, pt := range points {
		t.Logf("  N=%d: %.2f us", pt.Shape[0], pt.LatencyUs)
	}
}

func TestGGML_HardwareCharacterization(t *testing.T) {
	backend := skipIfNoBackend(t)
	defer backend.Close()

	cfg := BenchmarkConfig{WarmupReps: 2, MeasureReps: 20, TrimPercent: 0.1}
	result, err := CharacterizeHardware(backend, cfg)
	require.NoError(t, err)

	assert.Greater(t, result.PeakBW, 0.0, "should measure non-zero bandwidth")
	assert.NotEmpty(t, result.PeakTOPS, "should measure at least one dtype TOPS")

	t.Logf("Peak BW: %.1f GB/s", result.PeakBW/1e9)
	for dtype, tops := range result.PeakTOPS {
		t.Logf("Peak TOPS (%s): %.1f TOPS", dtype, tops/1e12)
		bp := result.BalancePoint[dtype]
		t.Logf("Balance point (%s): %.1f FLOP/byte", dtype, bp)
	}
}

func TestGGML_FullPipeline_SILU(t *testing.T) {
	backend := skipIfNoBackend(t)
	defer backend.Close()

	cfg := BenchmarkConfig{
		WarmupReps: 2, MeasureReps: 20, TrimPercent: 0.1,
		ErrorThreshold: 0.10, MaxPointsPerOp: 10,
	}

	// Run benchmark for SILU only
	profile, err := RunBenchmark(backend, []string{"SILU"}, []string{"f32"}, cfg)
	require.NoError(t, err)

	assert.Equal(t, 2, profile.Version)
	assert.NotEmpty(t, profile.Operators)

	// Find SILU curve
	var siluCurve *OperatorCurve
	for i := range profile.Operators {
		if profile.Operators[i].Op == "SILU" {
			siluCurve = &profile.Operators[i]
			break
		}
	}
	require.NotNil(t, siluCurve)
	assert.Equal(t, []string{"N"}, siluCurve.Dimensions)
	assert.GreaterOrEqual(t, len(siluCurve.Points), 6)

	// Test interpolation on the real curve
	lat := Interpolate1D(siluCurve.Points, 100000)
	assert.Greater(t, lat, 0.0)

	// Save and reload
	dir := t.TempDir()
	path := filepath.Join(dir, "profile.json")
	require.NoError(t, WriteProfile(path, profile))
	loaded, err := LoadProfile(path)
	require.NoError(t, err)
	assert.Equal(t, len(profile.Operators), len(loaded.Operators))

	// Generate HTML viewer
	htmlPath := filepath.Join(dir, "viewer.html")
	require.NoError(t, GenerateHTMLViewer(profile, htmlPath))
	info, err := os.Stat(htmlPath)
	require.NoError(t, err)
	assert.Greater(t, info.Size(), int64(1000))

	t.Logf("Full SILU pipeline: %d curves, %d total points",
		len(profile.Operators),
		func() int {
			n := 0
			for _, c := range profile.Operators {
				n += len(c.Points)
			}
			return n
		}())
}
```

- [ ] **Step 2: Run integration tests (with GGML backend)**

Run: `DAOP_INTEGRATION=1 go test ./perf/ -run "TestGGML" -v -timeout 300s`

Without GGML: tests are skipped. With GGML: all should PASS.

Run regular tests to make sure nothing is broken:

Run: `go test ./perf/ -v -count=1`

Expected: All non-GGML tests PASS, GGML tests are skipped.

- [ ] **Step 3: Commit**

```bash
git add perf/ggml_integration_test.go
git commit -m "perf: add GGML integration tests for real-backend benchmarking pipeline"
```

---

## Summary

| Task | Component | Tests | Key Risk |
|------|-----------|-------|----------|
| 1 | types.go rewrite | 7 JSON round-trip | Foundation for everything |
| 2 | ops.go trim | 3 utility | Dead code removal |
| 3 | registry.go | 11 structure + shape | Op API correctness |
| 4 | interpolate.go | 30+ math | **Most critical** — numerical accuracy |
| 5 | hwchar.go | 1 pure + integration | Backend API usage |
| 6 | adaptive.go | 9 with mock | Convergence guarantees |
| 7 | bench.go rewrite | 8 structure | Glue between registry+adaptive+hw |
| 8 | profile.go rewrite | 7 I/O | Data persistence |
| 9 | estimate.go (graph) | 8 shape extraction | GraphNode parsing |
| 10 | estimate.go (pipeline) | 11 lookup+phase | End-to-end estimation |
| 11 | viewer.go update | compile check | v2 type compatibility |
| 12 | viewer_html.go | 6 HTML gen | Template correctness |
| 13 | cleanup + roofline delete | full suite | No regressions |
| 14 | GGML integration | 6 real backend | Real hardware validation |
| 15 | CLI commands (cmd.go) | compile check | Cobra wiring |

**Total: ~120+ test cases across 15 tasks**

**Execution order is strict:** Tasks 1→15 must be executed sequentially because each builds on the previous.

**After all tasks complete:** Use `superpowers:finishing-a-development-branch` to merge or create a PR.

---

## Self-Review Fixes

The following gaps were identified during self-review and addressed:

1. **CLI commands (`cmd.go`)** — Added as Task 15 below
2. **OpRunner vs OpRunnerML confusion** — Clarified in types.go that OpRunner is documentation-only; all runtime code uses OpRunnerML from registry.go
3. **Missing edge case tests** — Key gaps noted below for implementers to add during TDD:
   - `Interpolate1D` with flat function (all latencies identical)
   - `InterpolateFlashAttn` with empty decode/prefill points
   - `AdaptiveSample1D` with shapeMin == shapeMax
   - `lookupLatency` with invalid shape length (MUL_MAT with <3 dims) — already tested via `TestNodeToQueryShape_MulMat_InsufficientInputShapes`
   - `buildModelGraphNodes` error cases (only testable with real GGML, deferred to integration tests)
4. **HTML viewer browser opening** — `OpenHTMLViewer` intentionally only prints the path; actual opening varies by platform and is handled by the CLI command in Task 15

---

## Task 15: CLI Commands (`cmd.go`)

**Files:**
- Create: `perf/cmd.go`

The spec defines three CLI commands: `daop-bench`, `daop-estimate`, `daop-viewer`. These are registered by the main Ollama CLI as subcommands.

**NOTE:** The existing v1 `cmd.go` was previously committed but may have been lost when the file was not found. This task creates it fresh.

- [ ] **Step 1: Implement `cmd.go`**

Create `perf/cmd.go`:

```go
package perf

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"runtime"
	"strings"

	"github.com/ollama/ollama/ml"
)

// RunBenchmarkCLI is the entry point for `ollama daop-bench`.
func RunBenchmarkCLI(backend ml.Backend, opts BenchmarkCLIOptions) error {
	cfg := DefaultBenchmarkConfig()

	ops := []string{"SILU", "MUL_MAT", "FLASH_ATTN_EXT"}
	if opts.Ops != "" {
		ops = strings.Split(opts.Ops, ",")
	}

	dtypes := Phase1Dtypes()
	if opts.Dtypes != "" {
		dtypes = strings.Split(opts.Dtypes, ",")
	}

	slog.Info("starting calibration", "ops", ops, "dtypes", dtypes)

	profile, err := RunBenchmark(backend, ops, dtypes, cfg)
	if err != nil {
		return fmt.Errorf("benchmark failed: %w", err)
	}

	outputPath := opts.Output
	if outputPath == "" {
		outputPath = ProfilePath()
	}

	if err := WriteProfile(outputPath, profile); err != nil {
		return fmt.Errorf("write profile: %w", err)
	}
	fmt.Fprintf(os.Stderr, "Profile saved to %s\n", outputPath)

	if opts.Verbose {
		PrintProfile(os.Stdout, profile, true)
	} else {
		PrintProfile(os.Stdout, profile, false)
	}

	if opts.Viewer {
		htmlPath := outputPath + ".html"
		if err := GenerateHTMLViewer(profile, htmlPath); err != nil {
			return fmt.Errorf("generate viewer: %w", err)
		}
		fmt.Fprintf(os.Stderr, "HTML viewer saved to %s\n", htmlPath)
		openBrowser(htmlPath)
	}

	return nil
}

// RunEstimateCLI is the entry point for `ollama daop-estimate`.
func RunEstimateCLI(modelRef string, opts EstimateCLIOptions) error {
	result, err := RunEstimate(modelRef, opts.Profile)
	if err != nil {
		return err
	}

	if opts.JSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(result)
	}

	PrintEstimateResult(os.Stdout, result, opts.Verbose)
	return nil
}

// RunViewerCLI is the entry point for `ollama daop-viewer`.
func RunViewerCLI(opts ViewerCLIOptions) error {
	profilePath := opts.Profile
	if profilePath == "" {
		profilePath = ProfilePath()
	}

	profile, err := LoadProfile(profilePath)
	if err != nil {
		return fmt.Errorf("load profile: %w (have you run 'ollama daop-bench'?)", err)
	}

	outputPath := opts.Output
	if outputPath == "" {
		outputPath = profilePath + ".html"
	}

	if err := GenerateHTMLViewer(profile, outputPath); err != nil {
		return fmt.Errorf("generate viewer: %w", err)
	}

	fmt.Fprintf(os.Stderr, "HTML viewer saved to %s\n", outputPath)
	if opts.Output == "" {
		openBrowser(outputPath)
	}
	return nil
}

// BenchmarkCLIOptions controls `ollama daop-bench`.
type BenchmarkCLIOptions struct {
	Output  string // --output: profile output path
	Ops     string // --ops: comma-separated op list
	Dtypes  string // --dtypes: comma-separated dtype list
	Viewer  bool   // --viewer: generate HTML viewer after benchmarking
	Verbose bool   // --verbose: show per-point results
}

// EstimateCLIOptions controls `ollama daop-estimate`.
type EstimateCLIOptions struct {
	Profile string // --profile: profile path
	JSON    bool   // --json: output as JSON
	Verbose bool   // --verbose: show per-op breakdown
}

// ViewerCLIOptions controls `ollama daop-viewer`.
type ViewerCLIOptions struct {
	Profile string // --profile: profile path
	Output  string // --output: save HTML instead of opening browser
}

// openBrowser opens a file in the system default browser.
func openBrowser(path string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", path)
	case "linux":
		cmd = exec.Command("xdg-open", path)
	case "windows":
		cmd = exec.Command("cmd", "/c", "start", path)
	default:
		fmt.Fprintf(os.Stderr, "Open in browser: %s\n", path)
		return
	}
	if err := cmd.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "Could not open browser: %v\nOpen manually: %s\n", err, path)
	}
}
```

- [ ] **Step 2: Verify compilation**

Run: `go build ./perf/`

Expected: Compiles without errors.

- [ ] **Step 3: Commit**

```bash
git add perf/cmd.go
git commit -m "perf: add CLI command entry points (daop-bench, daop-estimate, daop-viewer)"
```

**NOTE FOR COBRA INTEGRATION:** The actual Cobra command registration (in `cmd/` package) depends on how Ollama's CLI is structured. The `perf/cmd.go` file provides the entry points. The Cobra wiring in `cmd/cmd.go` is a separate concern and may need adaptation based on the current CLI structure. If a previous commit already registered these commands, update it to call the new v2 functions.
