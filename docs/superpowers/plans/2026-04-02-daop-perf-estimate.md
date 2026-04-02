# DAOP Performance Estimation System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ollama daop-bench` and `ollama daop-estimate` commands that benchmark local hardware and estimate LLM inference performance (prefill/decode tok/s) before loading model weights.

**Architecture:** Roofline model + microbenchmark calibration. Three modules: Benchmark (hardware profiling + operator calibration), Profile (storage + viewer), Estimate (graph-based performance estimation). New `perf/` package with CGo graph traversal API additions to `ml/backend/ggml/`.

**Tech Stack:** Go 1.24, CGo (GGML C API), Cobra CLI, testify

**Spec:** `docs/superpowers/specs/2026-04-02-perf-predict-design.md` — 14 sections, 完整设计文档

---

## Agent Context (必读)

执行此计划前请确认以下背景知识。如果从新 session 恢复，先读此节。

### 核心概念
- **Roofline 模型**: `T = max(FLOPs/peak_FLOPS, bytes/peak_BW)`，实际延迟用校准因子 η 修正: `T_actual = T_predicted / η`
- **三模块**: Benchmark (硬件 profiling + 算子校准 → raw.json) → Profile (处理 raw → profile.json，存 η) → Estimate (构图 → 遍历节点 → Roofline 估算 tok/s)
- **两层校准**: Layer 1 用预定义算子集 (无需模型)，Layer 2 从模型计算图发现缺失算子 (`--update`)
- **GGUF header**: 构图需要 GGUF 文件头 (几十 KB，含 tensor name/shape/dtype)，不需要权重数据 (GB 级)

### 关键 codebase 文件
| 文件 | 用途 | 关键行 |
|------|------|--------|
| `ml/backend.go` | Backend/Context/Tensor 接口定义 | Context:94-128, Tensor:130-241 |
| `ml/backend/ggml/ggml.go` | GGML backend 实现，CGo 调用 | initDevices:47, NewContextSize:667, Reserve:845, ComputeWithNotify:814 |
| `model/model.go` | model.New() 入口 | New:113, modelForArch:161 |
| `runner/ollamarunner/runner.go` | reserveWorstCaseGraph 参考 | :1069-1168 |
| `types/model/name.go` | Model ID 解析 | ParseName() |
| `manifest/manifest.go` | Manifest → GGUF blob 路径 | ParseNamedManifest() |
| `manifest/paths.go` | Digest → 文件路径 | BlobsPath() |
| `fs/ggml/gguf.go` | GGUF 文件解析 | Decode() |
| `ggml/include/ggml-backend.h` | CGo 可用的 C API | :337 ggml_backend_sched_get_tensor_backend |

### 已验证的 CGo API
| 函数 | 存在? | 头文件 |
|------|-------|--------|
| `ggml_backend_sched_get_tensor_backend(sched, node)` | **YES** | ggml-backend.h:337 |
| `ggml_backend_sched_get_node_backend_id(sched, i)` | **NO** | — |
| `ggml_graph_node(cgraph, i)` / `ggml_graph_n_nodes(cgraph)` | **YES** | ggml.h:2594-2596 |
| `ggml_op_name(op)` / `ggml_type_name(type)` / `ggml_get_name(tensor)` | **YES** | ggml.h:731-846 |
| `ggml_backend_name(backend)` | **YES** | ggml-backend.h:78 |

### Backend 初始化约束
`ml.NewBackend(modelPath, params)` 内部调 `os.Open(modelPath)` + `fsggml.Decode()` — **必须传有效 GGUF 文件**。
`daop-bench` 不针对特定模型，需要独立的 backend 初始化路径。方案见 Task 5a。

---

## File Structure

### New files to create

| File | Responsibility |
|------|---------------|
| `perf/types.go` | All shared types: Profile, RawData, EstimateResult, OpSpec, etc. |
| `perf/ops.go` | Per-op FLOPs/bytes computation rules (Section 3.3 of spec) |
| `perf/ops_test.go` | Unit tests for FLOPs/bytes calculations |
| `perf/roofline.go` | Roofline model: T_compute, T_memory, η lookup, bound classification |
| `perf/roofline_test.go` | Unit tests for Roofline calculations |
| `perf/profile.go` | Profile read/write/merge, raw data processing |
| `perf/profile_test.go` | Unit tests for profile I/O and merging |
| `perf/bench.go` | Hardware characterization + operator calibration orchestration |
| `perf/bench_test.go` | Unit tests for benchmark point selection, adaptive logic |
| `perf/estimate.go` | Graph-based estimation: build graph, traverse, estimate per-node |
| `perf/estimate_test.go` | Unit tests for estimation logic |
| `perf/viewer.go` | Profile viewer (human-readable output) |
| `perf/resolve.go` | Model ID → local GGUF path resolution |
| `perf/resolve_test.go` | Tests for model resolution |

### Existing files to modify

| File | Change |
|------|--------|
| `ml/backend.go` | Add `GraphNode` type and `GraphNodes() []GraphNode` to Context interface |
| `ml/backend/ggml/ggml.go` | Implement `GraphNodes()` + `NewForBench()` |
| `cmd/cmd.go` | Register `daop-bench` and `daop-estimate` subcommands |

---

## Tasks

### Task 1: Shared Types (`perf/types.go`)

**Files:**
- Create: `perf/types.go`

All types used across the perf package. No dependencies on other perf files.

- [ ] **Step 1: Create `perf/types.go` with all shared types**

```go
package perf

import "time"

// --- Profile types ---

// Profile is the processed hardware profile loaded by the estimate module.
type Profile struct {
	Version       int                `json:"version"`
	GeneratedFrom []string           `json:"generated_from"`
	GeneratedAt   time.Time          `json:"generated_at"`
	Hardware      HardwareProfile    `json:"hardware"`
	Operators     []OperatorProfile  `json:"operators"`
	Interconnects []InterconnectInfo `json:"interconnects"`
}

type HardwareProfile struct {
	Backends []BackendProfile `json:"backends"`
}

type BackendProfile struct {
	Name          string             `json:"name"`
	Device        string             `json:"device"`
	PeakFLOPS     map[string]float64 `json:"peak_flops"`      // dtype -> FLOPS
	PeakBandwidth float64            `json:"peak_bandwidth"`   // bytes/sec
	BalancePoints map[string]float64 `json:"balance_points"`   // dtype -> FLOP/byte
}

type OperatorProfile struct {
	Op           string  `json:"op"`
	Backend      string  `json:"backend"`
	ComputeDtype string  `json:"compute_dtype"`
	WeightDtype  string  `json:"weight_dtype,omitempty"`
	Eta          float64 `json:"eta"`
	EtaVariance  float64 `json:"eta_variance"`
	NumPoints    int     `json:"num_points"`
}

type InterconnectInfo struct {
	From      string  `json:"from"`
	To        string  `json:"to"`
	Bandwidth float64 `json:"bandwidth"` // bytes/sec
}

// --- Raw benchmark data types ---

type RawData struct {
	Version                int                    `json:"version"`
	Timestamp              time.Time              `json:"timestamp"`
	Hardware               RawHardware            `json:"hardware"`
	HardwareBenchmarks     []HardwareBenchmark    `json:"hardware_benchmarks"`
	OperatorBenchmarks     []OperatorBenchmark    `json:"operator_benchmarks"`
	InterconnectBenchmarks []InterconnectBenchmark `json:"interconnect_benchmarks"`
}

type RawHardware struct {
	Backends []RawBackendInfo `json:"backends"`
}

type RawBackendInfo struct {
	Name               string `json:"name"`
	Device             string `json:"device"`
	Driver             string `json:"driver,omitempty"`
	ComputeCapability  string `json:"compute_capability,omitempty"`
	VRAMBytes          uint64 `json:"vram_bytes,omitempty"`
	Cores              int    `json:"cores,omitempty"`
	RAMBytes           uint64 `json:"ram_bytes,omitempty"`
}

type HardwareBenchmark struct {
	Backend string  `json:"backend"`
	Dtype   string  `json:"dtype,omitempty"`
	Test    string  `json:"test"`  // "peak_flops" or "peak_bandwidth"
	Value   float64 `json:"value"`
	Unit    string  `json:"unit"`
}

type OperatorBenchmark struct {
	Op           string           `json:"op"`
	Backend      string           `json:"backend"`
	ComputeDtype string           `json:"compute_dtype"`
	WeightDtype  string           `json:"weight_dtype,omitempty"`
	Points       []BenchmarkPoint `json:"points"`
}

type BenchmarkPoint struct {
	InputShapes [][]int64 `json:"input_shapes"`
	OutputShape []int64   `json:"output_shape"`
	FLOPs       float64   `json:"flops"`
	BytesMoved  float64   `json:"bytes_moved"`
	Intensity   float64   `json:"intensity"`
	LatencyUs   float64   `json:"latency_us"`
	Reps        int       `json:"reps"`
	StddevUs    float64   `json:"stddev_us"`
}

type InterconnectBenchmark struct {
	From          string  `json:"from"`
	To            string  `json:"to"`
	Bandwidth     float64 `json:"bandwidth"`
	LatencyUs     float64 `json:"latency_us"`
	TestSizeBytes int64   `json:"test_size_bytes"`
}

// --- Estimate output types ---

type EstimateResult struct {
	Model        string          `json:"model"`
	Backends     []BackendInfo   `json:"backends"`
	InputLength  int             `json:"input_length"`
	OutputLength int             `json:"output_length"`
	MaxBatchSize int             `json:"max_batch_size"`
	Prefill      PhaseEstimation `json:"prefill"`
	Decode       PhaseEstimation `json:"decode"`
	Warnings     []string        `json:"warnings,omitempty"`
	Summary      string          `json:"summary"`
}

type BackendInfo struct {
	Name          string  `json:"name"`
	Device        string  `json:"device"`
	PeakFLOPS     float64 `json:"peak_flops"`
	PeakBandwidth float64 `json:"peak_bandwidth"`
	BalancePoint  float64 `json:"balance_point"`
}

type PhaseEstimation struct {
	TotalLatencyMs float64       `json:"total_latency_ms"`
	TokensPerSec   float64       `json:"tokens_per_sec"`
	TTFTMs         float64       `json:"ttft_ms,omitempty"`
	NumBatches     int           `json:"num_batches,omitempty"`
	Bottleneck     string        `json:"bottleneck"`
	TopOps         []OpBreakdown `json:"top_ops"`
}

type OpBreakdown struct {
	Op             string  `json:"op"`
	Backend        string  `json:"backend"`
	ComputeDtype   string  `json:"compute_dtype"`
	WeightDtype    string  `json:"weight_dtype,omitempty"`
	Count          int     `json:"count"`
	TotalMs        float64 `json:"total_ms"`
	Percentage     float64 `json:"percentage"`
	BoundBreakdown string  `json:"bound_breakdown"` // e.g. "72x mem + 24x compute"
}

// --- Internal helper types ---

// OpKey uniquely identifies an operator configuration for η lookup.
type OpKey struct {
	Op           string
	Backend      string
	ComputeDtype string
	WeightDtype  string
}

// OpCost holds the estimated cost for a single graph node.
type OpCost struct {
	FLOPs        float64
	BytesMoved   float64
	Intensity    float64
	TCompute     float64 // seconds
	TMemory      float64 // seconds
	TActual      float64 // seconds (after η)
	Bound        string  // "compute" or "memory"
	Eta          float64
	Uncalibrated bool
}
```

- [ ] **Step 2: Verify the file compiles**

Run: `cd /c/workspace/daop-ollama && go build ./perf/`
Expected: BUILD SUCCESS (no errors)

- [ ] **Step 3: Commit**

```bash
git add perf/types.go
git commit -m "perf: add shared type definitions for DAOP performance estimation"
```

---

### Task 2: Per-Op FLOPs/Bytes Computation (`perf/ops.go`)

**Files:**
- Create: `perf/ops.go`
- Create: `perf/ops_test.go`

Implements the FLOPs and bytes-moved formulas from spec Section 3.3. Pure math, no external dependencies.

- [ ] **Step 1: Write failing tests for key ops**

```go
package perf

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestComputeFLOPs_MulMat(t *testing.T) {
	// MUL_MAT [M,K]×[K,N]: FLOPs = 2*M*K*N
	f := ComputeFLOPs("MUL_MAT", [][]int64{{4096, 4096}, {4096, 1}})
	assert.InDelta(t, 2*4096*4096*1, f, 1)
}

func TestComputeFLOPs_FlashAttn(t *testing.T) {
	// FLASH_ATTN_EXT Q[B,H,Sq,D] × K,V[B,H,Skv,D]: FLOPs = 2*B*H*Sq*Skv*D
	f := ComputeFLOPs("FLASH_ATTN_EXT", [][]int64{{1, 32, 512, 128}, {1, 32, 512, 128}})
	assert.InDelta(t, 2.0*1*32*512*512*128, f, 1)
}

func TestComputeFLOPs_RMSNorm(t *testing.T) {
	// RMS_NORM [N,M]: 3*N*M
	f := ComputeFLOPs("RMS_NORM", [][]int64{{4096, 512}})
	assert.InDelta(t, 3*4096*512, f, 1)
}

func TestComputeFLOPs_Add(t *testing.T) {
	// ADD [N]: N
	f := ComputeFLOPs("ADD", [][]int64{{4096}})
	assert.InDelta(t, 4096, f, 1)
}

func TestComputeFLOPs_View(t *testing.T) {
	// VIEW: 0 FLOPs
	f := ComputeFLOPs("VIEW", [][]int64{{4096}})
	assert.Equal(t, float64(0), f)
}

func TestComputeBytes_MulMat(t *testing.T) {
	// MUL_MAT: elem_A*M*K + elem_B*K*N + 4*M*N
	// fp16 weight (2 bytes), fp32 activation (4 bytes)
	b := ComputeBytes("MUL_MAT", [][]int64{{4096, 4096}, {4096, 1}}, "f32", "f16")
	// A=f16: 2*4096*4096, B=f32: 4*4096*1, out=f32: 4*4096*1
	expected := 2.0*4096*4096 + 4.0*4096*1 + 4.0*4096*1
	assert.InDelta(t, expected, b, 1)
}

func TestComputeBytes_MulMat_Q4_0(t *testing.T) {
	// q4_0: 0.5625 bytes/element (18 bytes per 32 elements)
	b := ComputeBytes("MUL_MAT", [][]int64{{4096, 4096}, {4096, 1}}, "f32", "q4_0")
	expected := 0.5625*4096*4096 + 4.0*4096*1 + 4.0*4096*1
	assert.InDelta(t, expected, b, 1)
}

func TestComputeBytes_RMSNorm(t *testing.T) {
	// RMS_NORM [N,M]: 2*N*M*elem + weight
	b := ComputeBytes("RMS_NORM", [][]int64{{4096, 512}}, "f32", "")
	expected := 2.0*4096*512*4 + 4096*4 // read+write + weight
	assert.InDelta(t, expected, b, 1)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /c/workspace/daop-ollama && go test ./perf/ -run TestCompute -v`
Expected: FAIL — `ComputeFLOPs` and `ComputeBytes` not defined

- [ ] **Step 3: Implement ops.go**

```go
package perf

import "math"

// elemSize returns bytes per element for a given dtype string.
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
		return 0.5625 // 18 bytes per 32 elements
	case "q4_1":
		return 0.625 // 20/32
	case "q5_0":
		return 0.6875 // 22/32
	case "q5_1":
		return 0.75 // 24/32
	case "q8_0":
		return 1.0625 // 34/32
	case "q4_K":
		return 0.5625 // ~18/32, K-quant block=256
	case "q5_K":
		return 0.6875
	case "q6_K":
		return 0.8125
	case "q3_K":
		return 0.4375
	case "iq4_nl":
		return 0.5625
	default:
		return 4 // default to f32
	}
}

// product returns the product of all elements in a shape.
func product(shape []int64) float64 {
	p := float64(1)
	for _, v := range shape {
		p *= float64(v)
	}
	return p
}

// ComputeFLOPs returns the estimated FLOPs for an op given input shapes.
func ComputeFLOPs(op string, shapes [][]int64) float64 {
	switch op {
	case "MUL_MAT":
		// [M,K]×[K,N] -> 2*M*K*N
		if len(shapes) < 2 || len(shapes[0]) < 2 || len(shapes[1]) < 2 {
			return 0
		}
		M := float64(shapes[0][0])
		K := float64(shapes[0][1])
		N := float64(shapes[1][1])
		// Handle higher dims (batch)
		batch := float64(1)
		if len(shapes[1]) > 2 {
			for i := 2; i < len(shapes[1]); i++ {
				batch *= float64(shapes[1][i])
			}
		}
		return 2 * M * K * N * batch

	case "MUL_MAT_ID":
		if len(shapes) < 2 {
			return 0
		}
		base := ComputeFLOPs("MUL_MAT", shapes[:2])
		experts := float64(1)
		if len(shapes) > 2 && len(shapes[2]) > 0 {
			experts = float64(shapes[2][0])
		}
		return base * experts

	case "FLASH_ATTN_EXT":
		// Q[B,H,Sq,D] × K,V[B,H,Skv,D]: 2*B*H*Sq*Skv*D
		if len(shapes) < 2 || len(shapes[0]) < 4 || len(shapes[1]) < 4 {
			return 0
		}
		B := float64(shapes[0][0])
		H := float64(shapes[0][1])
		Sq := float64(shapes[0][2])
		D := float64(shapes[0][3])
		Skv := float64(shapes[1][2])
		return 2 * B * H * Sq * Skv * D

	case "RMS_NORM":
		// [N,M]: 3*N*M
		if len(shapes) < 1 {
			return 0
		}
		return 3 * product(shapes[0])

	case "LAYER_NORM":
		if len(shapes) < 1 {
			return 0
		}
		return 5 * product(shapes[0])

	case "SOFTMAX":
		if len(shapes) < 1 {
			return 0
		}
		return 4 * product(shapes[0])

	case "ADD", "MUL", "DIV", "NEG":
		if len(shapes) < 1 {
			return 0
		}
		return product(shapes[0])

	case "SILU", "GELU", "SIGMOID", "TANH":
		if len(shapes) < 1 {
			return 0
		}
		return 5 * product(shapes[0])

	case "GLU", "SWIGLU", "GEGLU":
		if len(shapes) < 1 {
			return 0
		}
		return 6 * product(shapes[0])

	case "ROPE":
		// [B,H,S,D]: 6*B*H*S*(D/2)
		if len(shapes) < 1 || len(shapes[0]) < 4 {
			return 0
		}
		B := float64(shapes[0][0])
		H := float64(shapes[0][1])
		S := float64(shapes[0][2])
		D := float64(shapes[0][3])
		return 6 * B * H * S * (D / 2)

	case "GET_ROWS":
		return 0 // pure memory

	case "CONT", "CPY", "CONCAT":
		return 0 // pure memory

	case "CONV_2D":
		// [Cout,Cin,Kh,Kw,Oh,Ow,B]: 2*Cout*Cin*Kh*Kw*Oh*Ow*B
		if len(shapes) < 1 || len(shapes[0]) < 7 {
			return 0
		}
		p := float64(2)
		for _, v := range shapes[0] {
			p *= float64(v)
		}
		return p

	case "EXP", "SQRT", "SQR", "SIN", "COS":
		if len(shapes) < 1 {
			return 0
		}
		return product(shapes[0])

	case "SUM_ROWS":
		if len(shapes) < 1 || len(shapes[0]) < 2 {
			return 0
		}
		return float64(shapes[0][0]) * float64(shapes[0][1])

	case "TOP_K":
		if len(shapes) < 1 || len(shapes[0]) < 2 {
			return 0
		}
		N := float64(shapes[0][0])
		K := float64(shapes[0][1])
		return N * math.Log2(K)

	case "L2_NORM":
		if len(shapes) < 1 {
			return 0
		}
		return 3 * product(shapes[0])

	case "SOFTPLUS":
		if len(shapes) < 1 {
			return 0
		}
		return 3 * product(shapes[0])

	case "CUM_SUM":
		if len(shapes) < 1 {
			return 0
		}
		return product(shapes[0])

	case "VIEW", "RESHAPE", "PERMUTE":
		return 0 // zero cost

	default:
		return 0 // unknown op — will be handled as memory-bound fallback
	}
}

// ComputeBytes returns the estimated bytes moved for an op.
// computeDtype is the compute data type (e.g. "f32"), weightDtype is the weight data type (e.g. "q4_0").
func ComputeBytes(op string, shapes [][]int64, computeDtype, weightDtype string) float64 {
	es := elemSize(computeDtype)
	ws := elemSize(weightDtype)
	if weightDtype == "" {
		ws = es
	}

	switch op {
	case "MUL_MAT":
		if len(shapes) < 2 || len(shapes[0]) < 2 || len(shapes[1]) < 2 {
			return 0
		}
		M := float64(shapes[0][0])
		K := float64(shapes[0][1])
		N := float64(shapes[1][1])
		batch := float64(1)
		if len(shapes[1]) > 2 {
			for i := 2; i < len(shapes[1]); i++ {
				batch *= float64(shapes[1][i])
			}
		}
		return (ws*M*K + es*K*N + 4*M*N) * batch

	case "MUL_MAT_ID":
		if len(shapes) < 2 {
			return 0
		}
		base := ComputeBytes("MUL_MAT", shapes[:2], computeDtype, weightDtype)
		experts := float64(1)
		if len(shapes) > 2 && len(shapes[2]) > 0 {
			experts = float64(shapes[2][0])
		}
		return base * experts

	case "FLASH_ATTN_EXT":
		if len(shapes) < 2 || len(shapes[0]) < 4 || len(shapes[1]) < 4 {
			return 0
		}
		B := float64(shapes[0][0])
		H := float64(shapes[0][1])
		Sq := float64(shapes[0][2])
		D := float64(shapes[0][3])
		Skv := float64(shapes[1][2])
		return B * H * (Sq*D + 2*Skv*D + Sq*D) * es

	case "RMS_NORM":
		if len(shapes) < 1 {
			return 0
		}
		total := product(shapes[0])
		N := float64(shapes[0][0])
		return 2*total*es + N*es // read+write + weight

	case "LAYER_NORM":
		if len(shapes) < 1 {
			return 0
		}
		total := product(shapes[0])
		N := float64(shapes[0][0])
		return 2*total*es + N*es + N*es // + bias

	case "SOFTMAX":
		if len(shapes) < 1 {
			return 0
		}
		return 2 * product(shapes[0]) * es

	case "ADD", "MUL", "DIV":
		if len(shapes) < 1 {
			return 0
		}
		return 3 * product(shapes[0]) * es // 2 read + 1 write

	case "NEG":
		if len(shapes) < 1 {
			return 0
		}
		return 2 * product(shapes[0]) * es // 1 read + 1 write

	case "SILU", "GELU", "SIGMOID", "TANH":
		if len(shapes) < 1 {
			return 0
		}
		return 2 * product(shapes[0]) * es

	case "GLU", "SWIGLU", "GEGLU":
		if len(shapes) < 1 {
			return 0
		}
		return 3 * product(shapes[0]) * es // gate+up read, out write

	case "GET_ROWS":
		if len(shapes) < 1 || len(shapes[0]) < 2 {
			return 0
		}
		Nidx := float64(shapes[0][0])
		D := float64(shapes[0][1])
		return Nidx*D*ws + Nidx*D*4 // random read + f32 write

	case "ROPE":
		if len(shapes) < 1 {
			return 0
		}
		return 2 * product(shapes[0]) * es

	case "CONT", "CPY", "CONCAT":
		if len(shapes) < 1 {
			return 0
		}
		return 2 * product(shapes[0]) * es

	case "CONV_2D":
		// Approximate: kernel + input + output
		if len(shapes) < 1 {
			return 0
		}
		return 2 * product(shapes[0]) * es

	case "VIEW", "RESHAPE", "PERMUTE":
		return 0

	default:
		// Fallback: estimate as 2 * total_elements * elem_size (read+write)
		if len(shapes) < 1 {
			return 0
		}
		return 2 * product(shapes[0]) * es
	}
}

// IsZeroCostOp returns true for ops that have no runtime cost (views, reshapes).
func IsZeroCostOp(op string) bool {
	switch op {
	case "VIEW", "RESHAPE", "PERMUTE":
		return true
	default:
		return false
	}
}

// CanComputeFLOPs returns true if we have a FLOPs formula for this op.
func CanComputeFLOPs(op string) bool {
	switch op {
	case "MUL_MAT", "MUL_MAT_ID", "FLASH_ATTN_EXT",
		"RMS_NORM", "LAYER_NORM", "SOFTMAX",
		"ADD", "MUL", "DIV", "NEG",
		"SILU", "GELU", "SIGMOID", "TANH",
		"GLU", "SWIGLU", "GEGLU",
		"ROPE", "CONV_2D",
		"EXP", "SQRT", "SQR", "SIN", "COS",
		"SUM_ROWS", "TOP_K", "L2_NORM", "SOFTPLUS", "CUM_SUM":
		return true
	default:
		return false
	}
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/workspace/daop-ollama && go test ./perf/ -run TestCompute -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add perf/ops.go perf/ops_test.go
git commit -m "perf: add per-op FLOPs and bytes computation rules"
```

---

### Task 3: Roofline Model Engine (`perf/roofline.go`)

**Files:**
- Create: `perf/roofline.go`
- Create: `perf/roofline_test.go`

Core Roofline calculation: given an op's FLOPs/bytes and a backend profile, compute estimated latency with η calibration.

- [ ] **Step 1: Write failing tests**

```go
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
	// Large MUL_MAT — should be compute-bound
	cost, err := EstimateOpCost(p, OpKey{"MUL_MAT", "cuda", "f16", "q4_0"},
		1e12,   // 1 TFLOP
		1e6,    // 1 MB
	)
	require.NoError(t, err)
	assert.Equal(t, "compute", cost.Bound)
	assert.InDelta(t, 0.62, cost.Eta, 0.001)
	// T_compute = 1e12 / 82.6e12 ≈ 0.0121s
	// T_actual = T_compute / 0.62 ≈ 0.0195s
	assert.InDelta(t, 0.0195, cost.TActual, 0.001)
}

func TestEstimateOpCost_MemoryBound(t *testing.T) {
	p := newTestProfile()
	// Small MUL_MAT — should be memory-bound
	cost, err := EstimateOpCost(p, OpKey{"MUL_MAT", "cuda", "f16", "q4_0"},
		1e6,    // 1 MFLOP
		1e9,    // 1 GB
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /c/workspace/daop-ollama && go test ./perf/ -run "TestEstimateOpCost|TestLookupEta" -v`
Expected: FAIL

- [ ] **Step 3: Implement roofline.go**

```go
package perf

import (
	"fmt"
	"math"
)

// LookupBackend finds a backend profile by name.
func LookupBackend(p *Profile, backendName string) (*BackendProfile, error) {
	for i := range p.Hardware.Backends {
		if p.Hardware.Backends[i].Name == backendName {
			return &p.Hardware.Backends[i], nil
		}
	}
	return nil, fmt.Errorf("backend %q not found in profile", backendName)
}

// LookupEta finds the calibration factor for a given op key.
func LookupEta(p *Profile, key OpKey) (float64, bool) {
	for _, op := range p.Operators {
		if op.Op == key.Op && op.Backend == key.Backend &&
			op.ComputeDtype == key.ComputeDtype && op.WeightDtype == key.WeightDtype {
			return op.Eta, true
		}
	}
	return 0, false
}

// LookupInterconnectBW finds the bandwidth between two backends.
func LookupInterconnectBW(p *Profile, from, to string) (float64, bool) {
	for _, ic := range p.Interconnects {
		if ic.From == from && ic.To == to {
			return ic.Bandwidth, true
		}
	}
	return 0, false
}

// EstimateOpCost computes the estimated latency for a single operation
// using the Roofline model with η calibration.
func EstimateOpCost(p *Profile, key OpKey, flops, bytesMoved float64) (OpCost, error) {
	bp, err := LookupBackend(p, key.Backend)
	if err != nil {
		return OpCost{}, err
	}

	peakFLOPS, ok := bp.PeakFLOPS[key.ComputeDtype]
	if !ok {
		// Fallback to f32 if specific dtype not found
		peakFLOPS, ok = bp.PeakFLOPS["f32"]
		if !ok {
			return OpCost{}, fmt.Errorf("no peak FLOPS for dtype %q on backend %q", key.ComputeDtype, key.Backend)
		}
	}
	peakBW := bp.PeakBandwidth

	var intensity float64
	if bytesMoved > 0 {
		intensity = flops / bytesMoved
	} else {
		intensity = math.Inf(1)
	}

	balancePoint := peakFLOPS / peakBW

	tCompute := flops / peakFLOPS
	tMemory := bytesMoved / peakBW
	tPredicted := math.Max(tCompute, tMemory)

	bound := "memory"
	if intensity > balancePoint {
		bound = "compute"
	}

	eta, found := LookupEta(p, key)
	uncalibrated := !found
	if !found {
		eta = 1.0 // optimistic fallback
	}

	tActual := tPredicted / eta

	return OpCost{
		FLOPs:        flops,
		BytesMoved:   bytesMoved,
		Intensity:    intensity,
		TCompute:     tCompute,
		TMemory:      tMemory,
		TActual:      tActual,
		Bound:        bound,
		Eta:          eta,
		Uncalibrated: uncalibrated,
	}, nil
}

// EstimateTransferCost estimates the cost of transferring data between backends.
func EstimateTransferCost(p *Profile, from, to string, bytes float64) float64 {
	if from == to || bytes == 0 {
		return 0
	}
	bw, found := LookupInterconnectBW(p, from, to)
	if !found || bw == 0 {
		return 0 // no interconnect data — assume co-located
	}
	return bytes / bw
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/workspace/daop-ollama && go test ./perf/ -run "TestEstimateOpCost|TestLookupEta" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add perf/roofline.go perf/roofline_test.go
git commit -m "perf: add Roofline model engine with eta calibration"
```

---

### Task 4: Profile Storage (`perf/profile.go`)

**Files:**
- Create: `perf/profile.go`
- Create: `perf/profile_test.go`

Read/write profile and raw data JSON files. Process raw benchmark data into profile (compute η from benchmark points).

- [ ] **Step 1: Write failing tests**

```go
package perf

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProfileRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "profile.json")

	p := newTestProfile()
	err := WriteProfile(path, p)
	require.NoError(t, err)

	loaded, err := LoadProfile(path)
	require.NoError(t, err)
	assert.Equal(t, len(p.Hardware.Backends), len(loaded.Hardware.Backends))
	assert.Equal(t, p.Hardware.Backends[0].Name, loaded.Hardware.Backends[0].Name)
	assert.InDelta(t, p.Operators[0].Eta, loaded.Operators[0].Eta, 0.001)
}

func TestRawDataRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "raw.json")

	raw := &RawData{
		Version: 1,
		HardwareBenchmarks: []HardwareBenchmark{
			{Backend: "cuda", Dtype: "f16", Test: "peak_flops", Value: 82.6e12, Unit: "FLOPS"},
		},
		OperatorBenchmarks: []OperatorBenchmark{
			{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Points: []BenchmarkPoint{
					{LatencyUs: 12.5, FLOPs: 33554432, BytesMoved: 8396800, Intensity: 4.0, Reps: 200},
				},
			},
		},
	}
	err := WriteRawData(path, raw)
	require.NoError(t, err)

	loaded, err := LoadRawData(path)
	require.NoError(t, err)
	assert.Equal(t, 1, len(loaded.HardwareBenchmarks))
}

func TestComputeEta(t *testing.T) {
	peakFLOPS := 82.6e12
	peakBW := 1008e9

	points := []BenchmarkPoint{
		{FLOPs: 1e9, BytesMoved: 1e7, LatencyUs: 15.0},  // T_pred = max(1e9/82.6e12, 1e7/1008e9)
		{FLOPs: 1e10, BytesMoved: 1e8, LatencyUs: 130.0},
	}
	eta, variance := ComputeEtaFromPoints(points, peakFLOPS, peakBW)
	assert.Greater(t, eta, 0.0)
	assert.LessOrEqual(t, eta, 1.0)
	_ = variance
}

func TestBenchDir(t *testing.T) {
	dir := BenchDir()
	assert.Contains(t, dir, "bench")
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /c/workspace/daop-ollama && go test ./perf/ -run "TestProfile|TestRawData|TestComputeEta|TestBenchDir" -v`
Expected: FAIL

- [ ] **Step 3: Implement profile.go**

```go
package perf

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// BenchDir returns the default benchmark data directory (~/.ollama/bench/).
func BenchDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, ".ollama", "bench")
}

// ProfilePath returns the default profile.json path.
func ProfilePath() string {
	return filepath.Join(BenchDir(), "profile.json")
}

// LoadProfile reads a profile from disk.
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

// WriteProfile writes a profile to disk.
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

// LoadRawData reads raw benchmark data from disk.
func LoadRawData(path string) (*RawData, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading raw data: %w", err)
	}
	var r RawData
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, fmt.Errorf("parsing raw data: %w", err)
	}
	return &r, nil
}

// WriteRawData writes raw benchmark data to disk.
func WriteRawData(path string, r *RawData) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// RawDataPath returns a timestamped filename for raw data.
func RawDataPath() string {
	ts := time.Now().Format("20060102-150405")
	return filepath.Join(BenchDir(), fmt.Sprintf("raw-%s.json", ts))
}

// ComputeEtaFromPoints computes η (calibration factor) from benchmark points.
// Returns median η and variance.
func ComputeEtaFromPoints(points []BenchmarkPoint, peakFLOPS, peakBW float64) (float64, float64) {
	if len(points) == 0 {
		return 1.0, 0
	}

	etas := make([]float64, 0, len(points))
	for _, pt := range points {
		tCompute := pt.FLOPs / peakFLOPS
		tMemory := pt.BytesMoved / peakBW
		tPredicted := math.Max(tCompute, tMemory)
		tMeasured := pt.LatencyUs * 1e-6 // convert to seconds

		if tMeasured <= 0 || tPredicted <= 0 {
			continue
		}
		eta := tPredicted / tMeasured
		if eta > 0 && eta <= 2.0 { // sanity: η > 2 means measurement error
			etas = append(etas, eta)
		}
	}

	if len(etas) == 0 {
		return 1.0, 0
	}

	sort.Float64s(etas)
	median := etas[len(etas)/2]

	// Compute variance
	mean := 0.0
	for _, e := range etas {
		mean += e
	}
	mean /= float64(len(etas))

	variance := 0.0
	for _, e := range etas {
		d := e - mean
		variance += d * d
	}
	variance /= float64(len(etas))

	return median, variance
}

// ProcessRawToProfile converts raw benchmark data into a Profile.
func ProcessRawToProfile(rawFiles []string) (*Profile, error) {
	p := &Profile{
		Version:       1,
		GeneratedFrom: rawFiles,
		GeneratedAt:   time.Now(),
	}

	for _, path := range rawFiles {
		raw, err := LoadRawData(path)
		if err != nil {
			return nil, err
		}

		// Merge hardware info (use latest)
		if len(raw.Hardware.Backends) > 0 {
			p.Hardware.Backends = make([]BackendProfile, 0, len(raw.Hardware.Backends))
			for _, rb := range raw.Hardware.Backends {
				p.Hardware.Backends = append(p.Hardware.Backends, BackendProfile{
					Name:          rb.Name,
					Device:        rb.Device,
					PeakFLOPS:     make(map[string]float64),
					BalancePoints: make(map[string]float64),
				})
			}
		}

		// Process hardware benchmarks
		for _, hb := range raw.HardwareBenchmarks {
			bp := findOrCreateBackend(&p.Hardware, hb.Backend)
			switch hb.Test {
			case "peak_flops":
				bp.PeakFLOPS[hb.Dtype] = hb.Value
			case "peak_bandwidth":
				bp.PeakBandwidth = hb.Value
			}
		}

		// Compute balance points
		for i := range p.Hardware.Backends {
			bp := &p.Hardware.Backends[i]
			for dtype, flops := range bp.PeakFLOPS {
				if bp.PeakBandwidth > 0 {
					bp.BalancePoints[dtype] = flops / bp.PeakBandwidth
				}
			}
		}

		// Process operator benchmarks
		for _, ob := range raw.OperatorBenchmarks {
			bp := findBackend(&p.Hardware, ob.Backend)
			if bp == nil {
				continue
			}
			peakFLOPS := bp.PeakFLOPS[ob.ComputeDtype]
			if peakFLOPS == 0 {
				peakFLOPS = bp.PeakFLOPS["f32"]
			}
			eta, variance := ComputeEtaFromPoints(ob.Points, peakFLOPS, bp.PeakBandwidth)
			p.Operators = append(p.Operators, OperatorProfile{
				Op:           ob.Op,
				Backend:      ob.Backend,
				ComputeDtype: ob.ComputeDtype,
				WeightDtype:  ob.WeightDtype,
				Eta:          eta,
				EtaVariance:  variance,
				NumPoints:    len(ob.Points),
			})
		}

		// Process interconnects
		for _, ic := range raw.InterconnectBenchmarks {
			p.Interconnects = append(p.Interconnects, InterconnectInfo{
				From:      ic.From,
				To:        ic.To,
				Bandwidth: ic.Bandwidth,
			})
		}
	}

	return p, nil
}

// MergeProfile merges new operator data into an existing profile.
func MergeProfile(existing *Profile, update *Profile) *Profile {
	merged := *existing
	merged.GeneratedFrom = append(merged.GeneratedFrom, update.GeneratedFrom...)
	merged.GeneratedAt = time.Now()

	// Add new operators (skip duplicates)
	existingKeys := make(map[OpKey]bool)
	for _, op := range existing.Operators {
		existingKeys[OpKey{op.Op, op.Backend, op.ComputeDtype, op.WeightDtype}] = true
	}
	for _, op := range update.Operators {
		key := OpKey{op.Op, op.Backend, op.ComputeDtype, op.WeightDtype}
		if !existingKeys[key] {
			merged.Operators = append(merged.Operators, op)
		}
	}

	// Add new interconnects
	for _, ic := range update.Interconnects {
		found := false
		for _, eic := range merged.Interconnects {
			if eic.From == ic.From && eic.To == ic.To {
				found = true
				break
			}
		}
		if !found {
			merged.Interconnects = append(merged.Interconnects, ic)
		}
	}

	return &merged
}

func findBackend(hw *HardwareProfile, name string) *BackendProfile {
	for i := range hw.Backends {
		if hw.Backends[i].Name == name {
			return &hw.Backends[i]
		}
	}
	return nil
}

func findOrCreateBackend(hw *HardwareProfile, name string) *BackendProfile {
	if bp := findBackend(hw, name); bp != nil {
		return bp
	}
	hw.Backends = append(hw.Backends, BackendProfile{
		Name:          name,
		PeakFLOPS:     make(map[string]float64),
		BalancePoints: make(map[string]float64),
	})
	return &hw.Backends[len(hw.Backends)-1]
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/workspace/daop-ollama && go test ./perf/ -run "TestProfile|TestRawData|TestComputeEta|TestBenchDir" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add perf/profile.go perf/profile_test.go
git commit -m "perf: add profile storage, raw data I/O, and eta computation"
```

---

### Task 5: Graph Traversal CGo API (`ml/backend.go`, `ml/backend/ggml/ggml.go`)

**Files:**
- Modify: `ml/backend.go:94-128` (Context interface)
- Modify: `ml/backend/ggml/ggml.go:741-870` (Context struct + Reserve)

Add `GraphNode` type to `ml/backend.go` and `GraphNodes()` method to the Context interface. Implement in the GGML backend by capturing node info during Reserve before reset clears it. This is the **Scheme A** from spec Section 6.4.

- [ ] **Step 1: Add GraphNode type and update Context interface in `ml/backend.go`**

Add after the existing `Context` interface closing brace (after line ~128), and update the interface:

In `ml/backend.go`, add the `GraphNode` type (before the Context interface):

```go
// GraphNode represents a single node in a GGML computation graph.
// Used by the performance estimation system to traverse fused graphs.
type GraphNode struct {
	Op           string   // GGML op name: "MUL_MAT", "RMS_NORM", etc.
	Name         string   // Tensor name: "blk.0.attn_q", etc.
	Backend      string   // Assigned backend: "cuda", "cpu", etc.
	Shape        [4]int64 // Output tensor shape (ne[0..3])
	ComputeDtype string   // Compute data type: "f32", "f16", etc.
	WeightDtype  string   // Weight data type (for MUL_MAT): "q4_0", etc.
	InputShapes  [][]int64 // Shapes of input tensors (src[0..])
}
```

Add `GraphNodes() []GraphNode` to the Context interface:

```go
type Context interface {
	// ... existing methods ...
	Reserve()
	GraphNodes() []GraphNode // Returns graph nodes captured during Reserve
	MaxGraphNodes() int
	Close()
	// ... rest ...
}
```

- [ ] **Step 2: Implement `GraphNodes()` in GGML backend (`ml/backend/ggml/ggml.go`)**

Add a field to the Context struct to store captured nodes:

```go
type Context struct {
	// ... existing fields ...
	graphNodes []ml.GraphNode // captured during Reserve
}
```

Add the `GraphNodes()` method:

```go
func (c *Context) GraphNodes() []ml.GraphNode {
	return c.graphNodes
}
```

Modify the `Reserve()` method to capture node info before reset. After `ggml_backend_sched_reserve` returns but **before** the memory tracking loop (which is before the `if !reserved` check), insert:

```go
func (c *Context) Reserve() {
	if c.batchSize > 0 {
		C.ggml_backend_sched_set_batch_size(c.b.sched, C.int(c.batchSize))
	}

	reserved := C.ggml_backend_sched_reserve(c.b.sched, c.graph)

	// Capture graph nodes BEFORE reset clears node_backend_ids
	c.captureGraphNodes()

	slog.Debug("compute graph", "nodes", C.ggml_graph_n_nodes(c.graph),
		"splits", C.ggml_backend_sched_get_n_splits(c.b.sched))

	// ... rest of existing Reserve code unchanged ...
}
```

Implement `captureGraphNodes`:

```go
func (c *Context) captureGraphNodes() {
	nNodes := int(C.ggml_graph_n_nodes(c.graph))
	c.graphNodes = make([]ml.GraphNode, 0, nNodes)

	for i := 0; i < nNodes; i++ {
		node := C.ggml_graph_node(c.graph, C.int(i))

		opName := C.GoString(C.ggml_op_name(node.op))
		tensorName := C.GoString(C.ggml_get_name(node))
		typeName := C.GoString(C.ggml_type_name(node._type))

		shape := [4]int64{
			int64(node.ne[0]), int64(node.ne[1]),
			int64(node.ne[2]), int64(node.ne[3]),
		}

		// Determine backend via public API (ggml-backend.h:337)
		backendName := "unknown"
		backendPtr := C.ggml_backend_sched_get_tensor_backend(c.b.sched, node)
		if backendPtr != nil {
			backendName = C.GoString(C.ggml_backend_name(backendPtr))
		}

		// Collect input shapes
		var inputShapes [][]int64
		for j := 0; j < C.GGML_MAX_SRC; j++ {
			src := node.src[j]
			if src == nil {
				break
			}
			inputShapes = append(inputShapes, []int64{
				int64(src.ne[0]), int64(src.ne[1]),
				int64(src.ne[2]), int64(src.ne[3]),
			})
		}

		// Determine weight dtype (for MUL_MAT, src[0] is weight)
		weightDtype := ""
		if opName == "MUL_MAT" || opName == "MUL_MAT_ID" {
			if node.src[0] != nil {
				weightDtype = C.GoString(C.ggml_type_name(node.src[0]._type))
			}
		}

		c.graphNodes = append(c.graphNodes, ml.GraphNode{
			Op:           opName,
			Name:         tensorName,
			Backend:      backendName,
			Shape:        shape,
			ComputeDtype: typeName,
			WeightDtype:  weightDtype,
			InputShapes:  inputShapes,
		})
	}
}
```

**Note**: Uses `ggml_backend_sched_get_tensor_backend()` (ggml-backend.h:337) — verified to exist in vendored headers. `ggml_backend_sched_get_node_backend_id()` does NOT exist as public API.

- [ ] **Step 3: Verify compilation**

Run: `cd /c/workspace/daop-ollama && go build ./ml/...`
Expected: BUILD SUCCESS

Note: Full compilation requires CGo + GGML headers. If building without GPU support:
Run: `cd /c/workspace/daop-ollama && go build -tags cpu ./ml/...`

- [ ] **Step 4: Commit**

```bash
git add ml/backend.go ml/backend/ggml/ggml.go
git commit -m "ml: add GraphNodes() API for graph traversal without weights"
```

---

### Task 5a: Benchmark Backend Initialization (`ml/backend/ggml/ggml.go`)

**Files:**
- Modify: `ml/backend/ggml/ggml.go`
- Modify: `ml/backend.go`

`ml.NewBackend()` 需要 GGUF 文件。`daop-bench` 不针对特定模型，需要独立的初始化路径。
方案：新增 `NewBackendForBench()` 函数，复用 `initDevices()` 发现的设备，
创建 `ggml_backend_sched` 但不解析任何模型。

- [ ] **Step 1: 在 `ml/backend.go` 添加 `NewBackendForBench` 函数声明**

在 `NewBackend` 函数附近添加：

```go
// NewBackendForBench initializes a backend for benchmarking without requiring a model.
// It discovers available devices and creates a scheduler, but loads no model tensors.
// Used by daop-bench to run standalone operator benchmarks.
func NewBackendForBench(params BackendParams) (Backend, error) {
	return newBackendForBench(params)
}
```

`newBackendForBench` 是包级变量（同 `newBackend` 的模式），由 ggml 包注册。

- [ ] **Step 2: 在 `ml/backend/ggml/ggml.go` 实现**

在 `init()` 函数中注册（参考现有 `ml.newBackend = ggml.New` 模式）：

```go
func init() {
	ml.SetBackendForBenchInit(NewForBench)
}

// NewForBench creates a Backend with device discovery and scheduler but no model.
func NewForBench(params ml.BackendParams) (ml.Backend, error) {
	initDevices()

	if len(gpus) == 0 && len(cpus) == 0 {
		return nil, fmt.Errorf("no compute devices found")
	}

	// Collect backends and buffer types (same logic as New() after GGUF parsing)
	var schedBackends []C.ggml_backend_t
	var schedBufts []C.ggml_backend_buffer_type_t

	// Add GPU backends first
	for dev, be := range backends {
		devType := C.ggml_backend_dev_type(dev)
		if devType == C.GGML_BACKEND_DEVICE_TYPE_GPU ||
			devType == C.GGML_BACKEND_DEVICE_TYPE_ACCEL {
			schedBackends = append(schedBackends, be)
			schedBufts = append(schedBufts, C.ggml_backend_get_default_buffer_type(be))
		}
	}
	// Always add CPU backend last
	for dev, be := range backends {
		if C.ggml_backend_dev_type(dev) == C.GGML_BACKEND_DEVICE_TYPE_CPU {
			schedBackends = append(schedBackends, be)
			schedBufts = append(schedBufts, C.ggml_backend_get_default_buffer_type(be))
			break
		}
	}

	if len(schedBackends) == 0 {
		return nil, fmt.Errorf("no backends available")
	}

	// Create scheduler
	sched := C.ggml_backend_sched_new(
		&schedBackends[0],
		&schedBufts[0],
		C.int(len(schedBackends)),
		C.size_t(8192), // default graph size
		false,           // no parallel
	)

	b := &Backend{
		sched:         sched,
		schedBackends: schedBackends,
		schedBufts:    schedBufts,
	}

	return b, nil
}
```

**注意**：这段代码依赖 `Backend` 结构体的内部字段。实现时需要检查 `Backend` 的完整结构，
确保必要字段都被初始化。可能需要初始化 `btDeviceMemory`, `schedMu` 等字段。
参考 `New()` 函数中 scheduler 创建部分的代码（约 line 350-400）。

- [ ] **Step 3: 验证编译**

Run: `cd /c/workspace/daop-ollama && go build ./ml/...`
Expected: BUILD SUCCESS

- [ ] **Step 4: Commit**

```bash
git add ml/backend.go ml/backend/ggml/ggml.go
git commit -m "ml: add NewBackendForBench() for model-free backend initialization"
```

---

### Task 6a: Model ID Resolution Helper (`perf/resolve.go`)

**Files:**
- Create: `perf/resolve.go`
- Create: `perf/resolve_test.go`

将 Model ID (如 `qwen3:8b-q4_0`) 解析为本地 GGUF 文件路径。不需要 Ollama server。

- [ ] **Step 1: Write failing test**

```go
package perf

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIsGGUFPath(t *testing.T) {
	assert.True(t, IsGGUFPath("./model.gguf"))
	assert.True(t, IsGGUFPath("/tmp/custom-model.gguf"))
	assert.True(t, IsGGUFPath("C:\\models\\test.gguf"))
	assert.False(t, IsGGUFPath("qwen3:8b-q4_0"))
	assert.False(t, IsGGUFPath("llama3"))
}

func TestResolveModelPath_GGUFFile(t *testing.T) {
	// Create a temp GGUF file
	dir := t.TempDir()
	ggufPath := filepath.Join(dir, "test.gguf")
	os.WriteFile(ggufPath, []byte("dummy"), 0o644)

	resolved, err := ResolveModelPath(ggufPath)
	assert.NoError(t, err)
	assert.Equal(t, ggufPath, resolved)
}

func TestResolveModelPath_GGUFNotFound(t *testing.T) {
	_, err := ResolveModelPath("/nonexistent/model.gguf")
	assert.Error(t, err)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /c/workspace/daop-ollama && go test ./perf/ -run "TestIsGGUF|TestResolveModel" -v`
Expected: FAIL

- [ ] **Step 3: Implement resolve.go**

```go
package perf

import (
	"fmt"
	"os"
	"strings"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// IsGGUFPath returns true if the ref looks like a file path rather than a model ID.
func IsGGUFPath(ref string) bool {
	return strings.HasSuffix(ref, ".gguf") ||
		strings.Contains(ref, "/") ||
		strings.Contains(ref, "\\")
}

// ResolveModelPath resolves a model reference to a local GGUF file path.
//
// If ref is a file path (contains / or \ or ends with .gguf), it's used directly.
// Otherwise, it's treated as an Ollama model ID and resolved via the local manifest:
//   1. Parse model ID → fully qualified name
//   2. Load manifest from ~/.ollama/models/manifests/...
//   3. Find the model layer (mediaType "application/vnd.ollama.image.model")
//   4. Resolve digest → blob path
//
// Returns error if the model is not found locally.
// Does NOT download — caller should handle the "not found" case.
func ResolveModelPath(ref string) (string, error) {
	if IsGGUFPath(ref) {
		if _, err := os.Stat(ref); err != nil {
			return "", fmt.Errorf("GGUF file not found: %s", ref)
		}
		return ref, nil
	}

	// Parse as Ollama model ID
	name := model.ParseName(ref)
	if !name.IsValid() {
		return "", fmt.Errorf("invalid model reference: %s", ref)
	}

	// Load manifest
	m, err := manifest.ParseNamedManifest(name)
	if err != nil {
		return "", fmt.Errorf("model %q not found locally: %w (run 'ollama pull %s' first)", ref, err, ref)
	}

	// Find model layer
	for _, layer := range m.Layers {
		if layer.MediaType == "application/vnd.ollama.image.model" {
			blobPath, err := manifest.BlobsPath(layer.Digest)
			if err != nil {
				return "", fmt.Errorf("cannot resolve blob path: %w", err)
			}
			if _, err := os.Stat(blobPath); err != nil {
				return "", fmt.Errorf("model blob missing: %s", blobPath)
			}
			return blobPath, nil
		}
	}

	return "", fmt.Errorf("no GGUF model layer found in manifest for %q", ref)
}
```

**注意**: `manifest.ParseNamedManifest` 和 `manifest.BlobsPath` 的签名需要在实现时验证。
如果 API 不完全匹配（例如 `ParseNamedManifest` 需要额外参数），参考 `server/` 中的调用方式调整。

- [ ] **Step 4: Run tests**

Run: `cd /c/workspace/daop-ollama && go test ./perf/ -run "TestIsGGUF|TestResolveModel" -v`
Expected: PASS (at least the GGUF path tests; model ID tests may need a real manifest)

- [ ] **Step 5: Commit**

```bash
git add perf/resolve.go perf/resolve_test.go
git commit -m "perf: add model ID to GGUF path resolution"
```

---

### Task 6: Benchmark Module (`perf/bench.go`)

**Files:**
- Create: `perf/bench.go`
- Create: `perf/bench_test.go`

Orchestrates hardware characterization (peak FLOPS/BW), interconnect measurement, and operator calibration (Layer 1 predefined set + Layer 2 graph-driven). Depends on Tasks 1-4.

- [ ] **Step 1: Write failing tests for benchmark point selection and adaptive logic**

```go
package perf

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSelectBenchmarkShapes_MulMat(t *testing.T) {
	// Balance point ~82 FLOP/byte (like RTX 4090 fp16)
	shapes := SelectBenchmarkShapes("MUL_MAT", 82.0, "f16", "q4_0")
	assert.Equal(t, 5, len(shapes))
	// First shape should have low intensity (decode-like, N=1)
	assert.Equal(t, int64(1), shapes[0][1][1]) // B shape N=1
	// Last shape should have high intensity (N=4096)
	assert.Equal(t, int64(4096), shapes[4][1][1])
}

func TestSelectBenchmarkShapes_MemoryBound(t *testing.T) {
	// ADD has fixed intensity, should vary tensor size
	shapes := SelectBenchmarkShapes("ADD", 82.0, "f32", "")
	assert.Equal(t, 5, len(shapes))
	// Sizes should increase
	size0 := shapes[0][0][0]
	size4 := shapes[4][0][0]
	assert.Greater(t, size4, size0)
}

func TestShouldAdaptiveExtend(t *testing.T) {
	// Low variance — no extension needed
	assert.False(t, ShouldAdaptiveExtend([]float64{0.61, 0.62, 0.63, 0.62, 0.61}))
	// High variance — extension needed
	assert.True(t, ShouldAdaptiveExtend([]float64{0.4, 0.6, 0.8, 0.5, 0.9}))
}

func TestPredefinedOps(t *testing.T) {
	ops := PredefinedOps()
	// Should contain MUL_MAT, FLASH_ATTN_EXT, RMS_NORM, etc.
	found := false
	for _, op := range ops {
		if op.Op == "MUL_MAT" {
			found = true
			assert.Contains(t, op.Dtypes, "q4_0")
		}
	}
	assert.True(t, found, "MUL_MAT should be in predefined ops")
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /c/workspace/daop-ollama && go test ./perf/ -run "TestSelect|TestShouldAdaptive|TestPredefined" -v`
Expected: FAIL

- [ ] **Step 3: Implement bench.go**

```go
package perf

import (
	"fmt"
	"log/slog"
	"math"
	"time"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
)

// OpSpec defines an operator to benchmark with its dtype combinations.
type OpSpec struct {
	Op     string
	Dtypes []string // compute_dtype values to test
}

// PredefinedOps returns the Layer 1 predefined operator set (no GGUF needed).
func PredefinedOps() []OpSpec {
	return []OpSpec{
		{Op: "MUL_MAT", Dtypes: []string{"f16", "f32", "bf16", "q4_0", "q4_K", "q5_K", "q6_K", "q8_0"}},
		{Op: "FLASH_ATTN_EXT", Dtypes: []string{"f16", "bf16"}},
		{Op: "RMS_NORM", Dtypes: []string{"f32"}},
		{Op: "SOFTMAX", Dtypes: []string{"f32"}},
		{Op: "SILU", Dtypes: []string{"f32"}},
		{Op: "GELU", Dtypes: []string{"f32"}},
		{Op: "GLU", Dtypes: []string{"f32"}},
		{Op: "ADD", Dtypes: []string{"f32"}},
		{Op: "MUL", Dtypes: []string{"f32"}},
		{Op: "ROPE", Dtypes: []string{"f32", "f16"}},
		{Op: "GET_ROWS", Dtypes: []string{"f16", "q4_0", "q8_0"}},
		{Op: "CONT", Dtypes: []string{"f32", "f16"}},
		{Op: "CONV_2D", Dtypes: []string{"f32", "f16"}},
	}
}

// isFixedIntensityOp returns true for ops whose arithmetic intensity
// doesn't change with tensor size (memory-bound ops).
func isFixedIntensityOp(op string) bool {
	switch op {
	case "ADD", "MUL", "DIV", "NEG", "SILU", "GELU", "SIGMOID", "TANH",
		"RMS_NORM", "LAYER_NORM", "SOFTMAX", "ROPE", "CONT", "CPY",
		"GET_ROWS", "GLU", "SWIGLU", "GEGLU":
		return true
	default:
		return false
	}
}

// SelectBenchmarkShapes returns 5 shape configurations for benchmarking an op.
// For MUL_MAT: varies N to sweep arithmetic intensity across the balance point.
// For fixed-intensity ops: varies tensor size.
func SelectBenchmarkShapes(op string, balancePoint float64, computeDtype, weightDtype string) [][][]int64 {
	K := int64(4096) // standard hidden dim

	if op == "MUL_MAT" || op == "MUL_MAT_ID" {
		// Vary N: 1, 32, 256, 1024, 4096
		Ns := []int64{1, 32, 256, 1024, 4096}
		shapes := make([][][]int64, len(Ns))
		for i, N := range Ns {
			shapes[i] = [][]int64{{K, K}, {K, N}} // [M=K, K] × [K, N]
		}
		return shapes
	}

	if op == "FLASH_ATTN_EXT" {
		// Q[B,H,Sq,D] × K,V[B,H,Skv,D]
		seqLens := []int64{1, 64, 256, 512, 2048}
		shapes := make([][][]int64, len(seqLens))
		for i, S := range seqLens {
			shapes[i] = [][]int64{{1, 32, S, 128}, {1, 32, S, 128}}
		}
		return shapes
	}

	if op == "CONV_2D" {
		sizes := []int64{16, 32, 64, 128, 256}
		shapes := make([][][]int64, len(sizes))
		for i, s := range sizes {
			// [Cout, Cin, Kh, Kw, Oh, Ow, B]
			shapes[i] = [][]int64{{64, 3, 3, 3, s, s, 1}}
		}
		return shapes
	}

	// Fixed-intensity ops: vary tensor size
	sizes := []int64{1024, 65536, 1048576, 16777216, 67108864}
	shapes := make([][][]int64, len(sizes))
	for i, size := range sizes {
		shapes[i] = [][]int64{{size}}
	}
	return shapes
}

// ShouldAdaptiveExtend returns true if η values have high variance (CV > 10%).
func ShouldAdaptiveExtend(etas []float64) bool {
	if len(etas) < 3 {
		return false
	}
	mean := 0.0
	for _, e := range etas {
		mean += e
	}
	mean /= float64(len(etas))
	if mean == 0 {
		return false
	}

	variance := 0.0
	for _, e := range etas {
		d := e - mean
		variance += d * d
	}
	variance /= float64(len(etas))
	cv := math.Sqrt(variance) / mean
	return cv > 0.10
}

// BenchmarkConfig controls benchmark behavior.
type BenchmarkConfig struct {
	Backends      []string // empty = all available
	WarmupReps    int      // default 3
	MeasureReps   int      // default 50
	MaxAdaptive   int      // max additional points (default 5)
}

// DefaultBenchmarkConfig returns sensible defaults.
func DefaultBenchmarkConfig() BenchmarkConfig {
	return BenchmarkConfig{
		WarmupReps:  3,
		MeasureReps: 50,
		MaxAdaptive: 5,
	}
}

// RunFullBenchmark executes the complete Layer 1 benchmark:
// 1. Hardware characterization (peak FLOPS + bandwidth per backend)
// 2. Interconnect bandwidth
// 3. Predefined operator calibration
//
// Returns raw data ready to be written to disk and processed into a profile.
func RunFullBenchmark(backend ml.Backend, cfg BenchmarkConfig) (*RawData, error) {
	raw := &RawData{
		Version:   1,
		Timestamp: time.Now(),
	}

	devices := backend.BackendDevices()
	if len(devices) == 0 {
		return nil, fmt.Errorf("no backend devices found")
	}

	// Populate hardware info
	for _, dev := range devices {
		raw.Hardware.Backends = append(raw.Hardware.Backends, RawBackendInfo{
			Name:   dev.Library,
			Device: dev.Name,
		})
	}

	slog.Info("starting hardware characterization", "devices", len(devices))

	// --- Hardware characterization ---
	// Peak FLOPS: large MUL_MAT per dtype
	// Peak Bandwidth: large CONT (pure memory copy)
	dtypes := []ml.DType{ml.DTypeF16, ml.DTypeF32}
	for _, dev := range devices {
		for _, dtype := range dtypes {
			flops, err := benchPeakFLOPS(backend, dtype, cfg)
			if err != nil {
				slog.Warn("peak FLOPS benchmark failed", "device", dev.Name, "dtype", dtype, "error", err)
				continue
			}
			raw.HardwareBenchmarks = append(raw.HardwareBenchmarks, HardwareBenchmark{
				Backend: dev.Library, Dtype: dtype.String(), Test: "peak_flops", Value: flops, Unit: "FLOPS",
			})
		}
		bw, err := benchPeakBandwidth(backend, cfg)
		if err != nil {
			slog.Warn("peak bandwidth benchmark failed", "device", dev.Name, "error", err)
			continue
		}
		raw.HardwareBenchmarks = append(raw.HardwareBenchmarks, HardwareBenchmark{
			Backend: dev.Library, Test: "peak_bandwidth", Value: bw, Unit: "bytes/sec",
		})
	}

	slog.Info("starting operator calibration (Layer 1)")

	// --- Operator calibration ---
	// Compute balance points from hardware benchmarks
	hwProfile := buildHWProfile(raw.HardwareBenchmarks)
	for _, opSpec := range PredefinedOps() {
		for _, dtypeStr := range opSpec.Dtypes {
			for _, bp := range hwProfile {
				balancePoint := bp.BalancePoints["f32"]
				if v, ok := bp.BalancePoints[dtypeStr]; ok {
					balancePoint = v
				}
				shapes := SelectBenchmarkShapes(opSpec.Op, balancePoint, dtypeStr, dtypeStr)
				ob := OperatorBenchmark{
					Op: opSpec.Op, Backend: bp.Name, ComputeDtype: dtypeStr,
				}
				if opSpec.Op == "MUL_MAT" || opSpec.Op == "MUL_MAT_ID" {
					ob.WeightDtype = dtypeStr
				}
				for _, shape := range shapes {
					pt, err := benchSingleOp(backend, opSpec.Op, shape, dtypeStr, cfg)
					if err != nil {
						slog.Warn("op benchmark failed", "op", opSpec.Op, "error", err)
						continue
					}
					ob.Points = append(ob.Points, pt)
				}
				// Adaptive extension
				if len(ob.Points) >= 3 {
					etas := computePointEtas(ob.Points, bp, dtypeStr)
					if ShouldAdaptiveExtend(etas) {
						// Insert additional points in highest-variance region
						for extra := 0; extra < cfg.MaxAdaptive && len(ob.Points) < 10; extra++ {
							midShape := interpolateShapes(shapes, etas)
							pt, err := benchSingleOp(backend, opSpec.Op, midShape, dtypeStr, cfg)
							if err != nil {
								break
							}
							ob.Points = append(ob.Points, pt)
						}
					}
				}
				raw.OperatorBenchmarks = append(raw.OperatorBenchmarks, ob)
			}
		}
	}

	return raw, nil
}

// benchPeakFLOPS measures peak FLOPS via large MUL_MAT.
func benchPeakFLOPS(backend ml.Backend, dtype ml.DType, cfg BenchmarkConfig) (float64, error) {
	const M, K, N = 4096, 4096, 4096
	ctx := backend.NewContext()
	defer ctx.Close()

	a := ctx.Zeros(dtype, M, K)
	b := ctx.Zeros(dtype, K, N)
	out := a.Mulmat(b)
	ctx.Forward(out)

	// Warmup
	for i := 0; i < cfg.WarmupReps; i++ {
		ctx.Compute(out)
	}
	// Measure
	start := time.Now()
	for i := 0; i < cfg.MeasureReps; i++ {
		ctx.Compute(out)
	}
	elapsed := time.Since(start)

	latencySec := elapsed.Seconds() / float64(cfg.MeasureReps)
	flops := 2.0 * M * K * N
	return flops / latencySec, nil
}

// benchPeakBandwidth measures peak memory bandwidth via large CONT (copy).
func benchPeakBandwidth(backend ml.Backend, cfg BenchmarkConfig) (float64, error) {
	const size = 64 * 1024 * 1024 // 64M elements
	ctx := backend.NewContext()
	defer ctx.Close()

	src := ctx.Zeros(ml.DTypeF32, size)
	dst := src.Contiguous(ctx)
	ctx.Forward(dst)

	for i := 0; i < cfg.WarmupReps; i++ {
		ctx.Compute(dst)
	}
	start := time.Now()
	for i := 0; i < cfg.MeasureReps; i++ {
		ctx.Compute(dst)
	}
	elapsed := time.Since(start)

	latencySec := elapsed.Seconds() / float64(cfg.MeasureReps)
	bytesTotal := 2.0 * size * 4 // read + write, f32 = 4 bytes
	return bytesTotal / latencySec, nil
}

// benchSingleOp benchmarks one op at one shape.
func benchSingleOp(backend ml.Backend, op string, shapes [][]int64, dtype string, cfg BenchmarkConfig) (BenchmarkPoint, error) {
	ctx := backend.NewContext()
	defer ctx.Close()

	dt := parseDType(dtype)

	// Build single-op graph based on op type
	var out ml.Tensor
	switch op {
	case "MUL_MAT":
		a := ctx.Zeros(dt, int(shapes[0][0]), int(shapes[0][1]))
		b := ctx.Zeros(dt, int(shapes[1][0]), int(shapes[1][1]))
		out = a.Mulmat(b)
	case "ADD":
		a := ctx.Zeros(dt, int(shapes[0][0]))
		b := ctx.Zeros(dt, int(shapes[0][0]))
		out = a.Add(ctx, b)
	case "SILU":
		a := ctx.Zeros(dt, int(shapes[0][0]))
		out = a.SILU(ctx)
	case "GELU":
		a := ctx.Zeros(dt, int(shapes[0][0]))
		out = a.GELU(ctx)
	case "RMS_NORM":
		a := ctx.Zeros(dt, int(shapes[0][0]))
		out = a.RMSNorm(ctx, nil, 1e-5)
	case "SOFTMAX":
		a := ctx.Zeros(dt, int(shapes[0][0]))
		out = a.Softmax(ctx)
	case "ROPE":
		a := ctx.Zeros(dt, int(shapes[0][0]))
		// ROPE needs position info — use basic rotation
		out = a.RoPE(ctx, nil, nil, nil, 0, 0)
	case "CONT":
		a := ctx.Zeros(dt, int(shapes[0][0]))
		out = a.Contiguous(ctx)
	default:
		return BenchmarkPoint{}, fmt.Errorf("benchmark not implemented for op %s", op)
	}

	ctx.Forward(out)

	for i := 0; i < cfg.WarmupReps; i++ {
		ctx.Compute(out)
	}

	var latencies []float64
	for i := 0; i < cfg.MeasureReps; i++ {
		start := time.Now()
		ctx.Compute(out)
		latencies = append(latencies, float64(time.Since(start).Microseconds()))
	}

	mean := 0.0
	for _, l := range latencies {
		mean += l
	}
	mean /= float64(len(latencies))

	stddev := 0.0
	for _, l := range latencies {
		d := l - mean
		stddev += d * d
	}
	stddev = math.Sqrt(stddev / float64(len(latencies)))

	flops := ComputeFLOPs(op, shapes)
	bytes := ComputeBytes(op, shapes, dtype, dtype)
	intensity := 0.0
	if bytes > 0 {
		intensity = flops / bytes
	}

	outputShape := shapes[0] // simplified
	return BenchmarkPoint{
		InputShapes: shapes,
		OutputShape: outputShape,
		FLOPs:       flops,
		BytesMoved:  bytes,
		Intensity:   intensity,
		LatencyUs:   mean,
		Reps:        cfg.MeasureReps,
		StddevUs:    stddev,
	}, nil
}

// parseDType converts string dtype to ml.DType.
func parseDType(s string) ml.DType {
	switch s {
	case "f16":
		return ml.DTypeF16
	case "f32":
		return ml.DTypeF32
	case "bf16":
		return ml.DTypeBF16
	default:
		return ml.DTypeF32
	}
}

// buildHWProfile extracts backend profiles from hardware benchmarks for balance point calculation.
func buildHWProfile(hbs []HardwareBenchmark) []BackendProfile {
	byBackend := make(map[string]*BackendProfile)
	for _, hb := range hbs {
		bp, ok := byBackend[hb.Backend]
		if !ok {
			bp = &BackendProfile{
				Name:          hb.Backend,
				PeakFLOPS:     make(map[string]float64),
				BalancePoints: make(map[string]float64),
			}
			byBackend[hb.Backend] = bp
		}
		switch hb.Test {
		case "peak_flops":
			bp.PeakFLOPS[hb.Dtype] = hb.Value
		case "peak_bandwidth":
			bp.PeakBandwidth = hb.Value
		}
	}
	var result []BackendProfile
	for _, bp := range byBackend {
		for dtype, flops := range bp.PeakFLOPS {
			if bp.PeakBandwidth > 0 {
				bp.BalancePoints[dtype] = flops / bp.PeakBandwidth
			}
		}
		result = append(result, *bp)
	}
	return result
}

// computePointEtas computes per-point η values from benchmark results.
func computePointEtas(points []BenchmarkPoint, bp BackendProfile, dtype string) []float64 {
	peakFLOPS := bp.PeakFLOPS[dtype]
	if peakFLOPS == 0 {
		peakFLOPS = bp.PeakFLOPS["f32"]
	}
	peakBW := bp.PeakBandwidth
	etas := make([]float64, 0, len(points))
	for _, pt := range points {
		tComp := pt.FLOPs / peakFLOPS
		tMem := pt.BytesMoved / peakBW
		tPred := math.Max(tComp, tMem)
		tMeas := pt.LatencyUs * 1e-6
		if tMeas > 0 && tPred > 0 {
			eta := tPred / tMeas
			if eta > 0 && eta <= 2.0 {
				etas = append(etas, eta)
			}
		}
	}
	return etas
}

// interpolateShapes picks a shape between existing shapes where η variance is highest.
// Simplified: returns the midpoint between first and last shape.
func interpolateShapes(shapes [][][]int64, etas []float64) [][]int64 {
	if len(shapes) < 2 {
		return shapes[0]
	}
	mid := len(shapes) / 2
	return shapes[mid]
}

// RunUpdateBenchmark executes Layer 2 graph-driven discovery.
// Builds graphs for specified models, finds uncalibrated ops, and benchmarks them.
func RunUpdateBenchmark(backend ml.Backend, existingProfile *Profile,
	modelPaths []string, cfg BenchmarkConfig) (*RawData, error) {

	raw := &RawData{
		Version:   1,
		Timestamp: time.Now(),
	}

	// Collect all (op, backend, compute_dtype, weight_dtype) from model graphs
	needed := make(map[OpKey]bool)
	for _, modelPath := range modelPaths {
		nodes, err := buildModelGraphNodes(modelPath)
		if err != nil {
			slog.Warn("failed to build graph for model", "path", modelPath, "error", err)
			continue
		}
		for _, node := range nodes {
			if IsZeroCostOp(node.Op) {
				continue
			}
			needed[OpKey{node.Op, node.Backend, node.ComputeDtype, node.WeightDtype}] = true
		}
	}

	// Filter out already-calibrated ops
	for _, op := range existingProfile.Operators {
		delete(needed, OpKey{op.Op, op.Backend, op.ComputeDtype, op.WeightDtype})
	}

	if len(needed) == 0 {
		slog.Info("all operators already calibrated")
		return raw, nil
	}

	slog.Info("operator calibration (Layer 2)", "missing_ops", len(needed))

	// Benchmark missing ops
	hwProfile := buildHWProfile(raw.HardwareBenchmarks)
	for key := range needed {
		bp := findBackendProfile(hwProfile, key.Backend)
		if bp == nil {
			// Use existing profile's hardware data
			bp2, _ := LookupBackend(existingProfile, key.Backend)
			if bp2 != nil {
				bp = bp2
			} else {
				continue
			}
		}
		balancePoint := bp.BalancePoints[key.ComputeDtype]
		shapes := SelectBenchmarkShapes(key.Op, balancePoint, key.ComputeDtype, key.WeightDtype)
		ob := OperatorBenchmark{
			Op: key.Op, Backend: key.Backend,
			ComputeDtype: key.ComputeDtype, WeightDtype: key.WeightDtype,
		}
		for _, shape := range shapes {
			pt, err := benchSingleOp(backend, key.Op, shape, key.ComputeDtype, cfg)
			if err != nil {
				continue
			}
			ob.Points = append(ob.Points, pt)
		}
		if len(ob.Points) > 0 {
			raw.OperatorBenchmarks = append(raw.OperatorBenchmarks, ob)
		}
	}

	return raw, nil
}

// buildModelGraphNodes loads a model, builds prefill+decode graphs, returns all graph nodes.
func buildModelGraphNodes(modelPath string) ([]ml.GraphNode, error) {
	m, err := model.New(modelPath, ml.BackendParams{})
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	defer m.Backend().Close()

	var allNodes []ml.GraphNode
	for _, batchSize := range []int{512, 1} { // prefill + decode
		ctx := m.Backend().NewContext()
		batch := createFakeBatch(ctx, batchSize, batchSize)
		output, err := m.Forward(ctx, batch)
		if err != nil {
			ctx.Close()
			continue
		}
		ctx.Forward(output)
		ctx.Reserve()
		allNodes = append(allNodes, ctx.GraphNodes()...)
		ctx.Close()
	}
	return allNodes, nil
}

func findBackendProfile(profiles []BackendProfile, name string) *BackendProfile {
	for i := range profiles {
		if profiles[i].Name == name {
			return &profiles[i]
		}
	}
	return nil
}
```

**实现注意**: `benchSingleOp` 中的 Tensor API 调用 (`a.Mulmat(b)`, `a.SILU(ctx)` 等) 需要验证
方法签名是否匹配 `ml/backend.go:130-241` 的 Tensor 接口。某些 op（如 ROPE、FLASH_ATTN_EXT）
参数较多，`benchSingleOp` 的 switch 可能需要扩展。同时 `ctx.Compute(out)` 可能需要替换为
`ctx.ComputeWithNotify(nil, out)` 以确保 GPU 同步。参考 `ggml.go:814`。

`createFakeBatch` 函数需要实现。参考 `runner/ollamarunner/runner.go:1069-1168` 中
`reserveWorstCaseGraph` 的 fake batch 构造方式。

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/workspace/daop-ollama && go test ./perf/ -run "TestSelect|TestShouldAdaptive|TestPredefined" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add perf/bench.go perf/bench_test.go
git commit -m "perf: add benchmark module with point selection and adaptive logic"
```

---

### Task 7: Estimate Module (`perf/estimate.go`)

**Files:**
- Create: `perf/estimate.go`
- Create: `perf/estimate_test.go`

Core estimation logic: build graph from model metadata, traverse fused graph, estimate per-node latency using Roofline + η. Depends on Tasks 1-5.

- [ ] **Step 1: Write failing tests**

```go
package perf

import (
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEstimateGraph(t *testing.T) {
	p := newTestProfile()

	// Simulate a simple graph: 2 MUL_MAT nodes + 1 RMS_NORM
	nodes := []ml.GraphNode{
		{
			Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			Shape:       [4]int64{4096, 1, 1, 1},
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}},
		},
		{
			Op: "RMS_NORM", Backend: "cuda", ComputeDtype: "f32",
			Shape:       [4]int64{4096, 1, 1, 1},
			InputShapes: [][]int64{{4096, 1}},
		},
		{
			Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			Shape:       [4]int64{4096, 1, 1, 1},
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}},
		},
	}

	latency, stats, warnings := EstimateGraphLatency(p, nodes)

	assert.Greater(t, latency, 0.0)
	assert.Greater(t, len(stats), 0)
	_ = warnings
}

func TestEstimateGraph_SkipsViews(t *testing.T) {
	p := newTestProfile()

	nodes := []ml.GraphNode{
		{Op: "VIEW", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}},
		{Op: "RESHAPE", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}},
		{
			Op: "ADD", Backend: "cuda", ComputeDtype: "f32",
			Shape:       [4]int64{4096, 1, 1, 1},
			InputShapes: [][]int64{{4096}},
		},
	}

	latency, stats, _ := EstimateGraphLatency(p, nodes)
	assert.Greater(t, latency, 0.0)
	// Only ADD should contribute
	assert.Equal(t, 1, len(stats))
}

func TestBuildEstimateResult(t *testing.T) {
	p := newTestProfile()

	result := &EstimateResult{
		Model:        "qwen3:8b-q4_0",
		InputLength:  1024,
		OutputLength: 256,
		MaxBatchSize: 512,
	}

	// Test that summary is generated
	BuildSummary(result)
	assert.NotEmpty(t, result.Summary)
}

func TestComputePhaseEstimation(t *testing.T) {
	p := newTestProfile()
	nodes := []ml.GraphNode{
		{
			Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			Shape:       [4]int64{4096, 512, 1, 1},
			InputShapes: [][]int64{{4096, 4096}, {4096, 512}},
		},
	}

	phase := ComputePhaseEstimation(p, nodes, 1024, 512)
	require.NotNil(t, phase)
	assert.Greater(t, phase.TokensPerSec, 0.0)
	assert.Greater(t, phase.TotalLatencyMs, 0.0)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /c/workspace/daop-ollama && go test ./perf/ -run "TestEstimateGraph|TestBuildEstimate|TestComputePhase" -v`
Expected: FAIL

- [ ] **Step 3: Implement estimate.go**

```go
package perf

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
)

// EstimateConfig controls estimation behavior.
type EstimateConfig struct {
	InputLength  int
	OutputLength int
	MaxBatchSize int
	ProfilePath  string
	JSON         bool
	Detail       bool
}

// DefaultEstimateConfig returns sensible defaults.
func DefaultEstimateConfig() EstimateConfig {
	return EstimateConfig{
		InputLength:  512,
		OutputLength: 128,
		MaxBatchSize: 512,
	}
}

// OpStats tracks aggregate stats for an op key across graph nodes.
type OpStats struct {
	Key        OpKey
	Count      int
	TotalSec   float64
	MemCount   int
	CompCount  int
}

// EstimateGraphLatency estimates total latency for a set of graph nodes.
// Returns total latency (seconds), per-op stats, and warnings.
func EstimateGraphLatency(p *Profile, nodes []ml.GraphNode) (float64, map[OpKey]*OpStats, []string) {
	totalLatency := 0.0
	stats := make(map[OpKey]*OpStats)
	var warnings []string
	seenWarnings := make(map[string]bool)

	var prevBackend string
	for _, node := range nodes {
		if IsZeroCostOp(node.Op) {
			continue
		}

		// Cross-backend transfer cost
		if prevBackend != "" && node.Backend != prevBackend {
			// Estimate transfer bytes from first input shape
			transferBytes := float64(0)
			if len(node.InputShapes) > 0 {
				transferBytes = product(node.InputShapes[0]) * elemSize(node.ComputeDtype)
			}
			transferTime := EstimateTransferCost(p, prevBackend, node.Backend, transferBytes)
			totalLatency += transferTime
		}
		prevBackend = node.Backend

		flops := ComputeFLOPs(node.Op, node.InputShapes)
		bytes := ComputeBytes(node.Op, node.InputShapes, node.ComputeDtype, node.WeightDtype)

		key := OpKey{
			Op:           node.Op,
			Backend:      node.Backend,
			ComputeDtype: node.ComputeDtype,
			WeightDtype:  node.WeightDtype,
		}

		var cost OpCost
		var err error

		if flops == 0 && bytes == 0 {
			continue
		}

		if !CanComputeFLOPs(node.Op) && bytes > 0 {
			// Unknown op: memory-bound fallback
			bp, bpErr := LookupBackend(p, node.Backend)
			if bpErr != nil {
				continue
			}
			cost = OpCost{
				BytesMoved:   bytes,
				TMemory:      bytes / bp.PeakBandwidth,
				TActual:      bytes / bp.PeakBandwidth,
				Bound:        "memory",
				Eta:          1.0,
				Uncalibrated: true,
			}
			warnKey := fmt.Sprintf("unknown op: %s(%s,%s)", node.Op, node.Backend, node.ComputeDtype)
			if !seenWarnings[warnKey] {
				warnings = append(warnings, warnKey)
				seenWarnings[warnKey] = true
			}
		} else {
			cost, err = EstimateOpCost(p, key, flops, bytes)
			if err != nil {
				continue
			}
			if cost.Uncalibrated {
				warnKey := fmt.Sprintf("uncalibrated: %s(%s,%s,%s)", key.Op, key.Backend, key.ComputeDtype, key.WeightDtype)
				if !seenWarnings[warnKey] {
					warnings = append(warnings, warnKey)
					seenWarnings[warnKey] = true
				}
			}
		}

		totalLatency += cost.TActual

		if _, ok := stats[key]; !ok {
			stats[key] = &OpStats{Key: key}
		}
		s := stats[key]
		s.Count++
		s.TotalSec += cost.TActual
		if cost.Bound == "memory" {
			s.MemCount++
		} else {
			s.CompCount++
		}
	}

	return totalLatency, stats, warnings
}

// ComputePhaseEstimation wraps EstimateGraphLatency for a phase (prefill or decode).
func ComputePhaseEstimation(p *Profile, nodes []ml.GraphNode, tokenCount, batchSize int) *PhaseEstimation {
	latencySec, stats, _ := EstimateGraphLatency(p, nodes)

	tokPerSec := 0.0
	if latencySec > 0 {
		tokPerSec = float64(tokenCount) / latencySec
	}

	// Determine overall bottleneck
	memTotal, compTotal := 0.0, 0.0
	for _, s := range stats {
		memFrac := float64(s.MemCount) / float64(max(s.Count, 1))
		compFrac := float64(s.CompCount) / float64(max(s.Count, 1))
		memTotal += s.TotalSec * memFrac
		compTotal += s.TotalSec * compFrac
	}
	bottleneck := "memory"
	if compTotal > memTotal {
		bottleneck = "compute"
	}

	// Top ops by total time
	topOps := buildTopOps(stats, latencySec)

	nBatches := int(math.Ceil(float64(tokenCount) / float64(batchSize)))

	return &PhaseEstimation{
		TotalLatencyMs: latencySec * 1000,
		TokensPerSec:   tokPerSec,
		TTFTMs:         latencySec * 1000, // TTFT = total prefill latency (set by caller for decode)
		NumBatches:     nBatches,
		Bottleneck:     bottleneck,
		TopOps:         topOps,
	}
}

func buildTopOps(stats map[OpKey]*OpStats, totalSec float64) []OpBreakdown {
	var ops []OpBreakdown
	for _, s := range stats {
		pct := 0.0
		if totalSec > 0 {
			pct = (s.TotalSec / totalSec) * 100
		}
		bound := fmt.Sprintf("%dx mem + %dx compute", s.MemCount, s.CompCount)
		if s.MemCount == 0 {
			bound = fmt.Sprintf("%dx compute", s.CompCount)
		} else if s.CompCount == 0 {
			bound = fmt.Sprintf("%dx memory", s.MemCount)
		}

		ops = append(ops, OpBreakdown{
			Op:             s.Key.Op,
			Backend:        s.Key.Backend,
			ComputeDtype:   s.Key.ComputeDtype,
			WeightDtype:    s.Key.WeightDtype,
			Count:          s.Count,
			TotalMs:        s.TotalSec * 1000,
			Percentage:     pct,
			BoundBreakdown: bound,
		})
	}

	sort.Slice(ops, func(i, j int) bool {
		return ops[i].TotalMs > ops[j].TotalMs
	})

	// Return top 10
	if len(ops) > 10 {
		ops = ops[:10]
	}
	return ops
}

// BuildSummary generates the human-readable summary line for an EstimateResult.
func BuildSummary(r *EstimateResult) {
	var parts []string
	parts = append(parts, fmt.Sprintf("%s | input=%d | output=%d", r.Model, r.InputLength, r.OutputLength))

	if r.Prefill.TokensPerSec > 0 {
		parts = append(parts, fmt.Sprintf("Prefill: ~%.0f tok/s (batch=%d, %d batches, TTFT ≈ %.0fms)",
			r.Prefill.TokensPerSec, r.MaxBatchSize, r.Prefill.NumBatches, r.Prefill.TTFTMs))
	}
	if r.Decode.TokensPerSec > 0 {
		parts = append(parts, fmt.Sprintf("Decode: ~%.0f tok/s (%s-bound)",
			r.Decode.TokensPerSec, r.Decode.Bottleneck))
	}

	r.Summary = strings.Join(parts, "\n  ")
}

// RunEstimate is the main entry point for the estimate command.
// It loads the model, builds graphs for prefill and decode phases,
// and estimates performance using the loaded profile.
func RunEstimate(modelPath string, cfg EstimateConfig) (*EstimateResult, error) {
	// 1. Load profile
	profilePath := cfg.ProfilePath
	if profilePath == "" {
		profilePath = ProfilePath()
	}
	profile, err := LoadProfile(profilePath)
	if err != nil {
		return nil, fmt.Errorf("load profile: %w (have you run 'ollama daop-bench'?)", err)
	}

	result := &EstimateResult{
		Model:        modelPath,
		InputLength:  cfg.InputLength,
		OutputLength: cfg.OutputLength,
		MaxBatchSize: cfg.MaxBatchSize,
	}

	// Populate backend info from profile
	for _, bp := range profile.Hardware.Backends {
		primaryDtype := "f16"
		peakFLOPS := bp.PeakFLOPS[primaryDtype]
		balancePoint := bp.BalancePoints[primaryDtype]
		if peakFLOPS == 0 {
			primaryDtype = "f32"
			peakFLOPS = bp.PeakFLOPS[primaryDtype]
			balancePoint = bp.BalancePoints[primaryDtype]
		}
		result.Backends = append(result.Backends, BackendInfo{
			Name:          bp.Name,
			Device:        bp.Device,
			PeakFLOPS:     peakFLOPS,
			PeakBandwidth: bp.PeakBandwidth,
			BalancePoint:  balancePoint,
		})
	}

	// 2. Resolve and load model (GGUF header only, no weights)
	ggufPath, err := ResolveModelPath(modelPath)
	if err != nil {
		return nil, err
	}
	m, err := model.New(ggufPath, ml.BackendParams{})
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	defer m.Backend().Close()

	// 3. Build prefill graphs
	maxBatch := cfg.MaxBatchSize
	nBatches := int(math.Ceil(float64(cfg.InputLength) / float64(maxBatch)))
	var allPrefillNodes []ml.GraphNode
	for i := 0; i < nBatches; i++ {
		batchLen := min(maxBatch, cfg.InputLength-i*maxBatch)
		kvLen := min((i+1)*maxBatch, cfg.InputLength)

		ctx := m.Backend().NewContext()
		batch := createFakeBatch(ctx, batchLen, kvLen)
		output, err := m.Forward(ctx, batch)
		if err != nil {
			ctx.Close()
			return nil, fmt.Errorf("forward (prefill batch %d): %w", i, err)
		}
		ctx.Forward(output)
		ctx.Reserve()
		allPrefillNodes = append(allPrefillNodes, ctx.GraphNodes()...)
		ctx.Close()
	}

	prefill := ComputePhaseEstimation(profile, allPrefillNodes, cfg.InputLength, maxBatch)
	prefill.TTFTMs = prefill.TotalLatencyMs
	result.Prefill = *prefill

	// 4. Build decode graphs (sample 3 KV positions for FA cost scaling)
	positions := []int{
		cfg.InputLength,
		cfg.InputLength + cfg.OutputLength/2,
		cfg.InputLength + cfg.OutputLength,
	}
	var decodeLatencies []float64
	for _, pos := range positions {
		ctx := m.Backend().NewContext()
		batch := createFakeBatch(ctx, 1, pos)
		output, err := m.Forward(ctx, batch)
		if err != nil {
			ctx.Close()
			continue
		}
		ctx.Forward(output)
		ctx.Reserve()
		latency, _, warnings := EstimateGraphLatency(profile, ctx.GraphNodes())
		decodeLatencies = append(decodeLatencies, latency)
		result.Warnings = append(result.Warnings, warnings...)
		ctx.Close()
	}

	if len(decodeLatencies) > 0 {
		avgDecode := 0.0
		for _, l := range decodeLatencies {
			avgDecode += l
		}
		avgDecode /= float64(len(decodeLatencies))

		decodePhase := ComputePhaseEstimation(profile, allPrefillNodes[:0], 1, 1)
		decodePhase.TotalLatencyMs = avgDecode * 1000
		decodePhase.TokensPerSec = 1.0 / avgDecode
		decodePhase.TTFTMs = 0 // not applicable for decode
		result.Decode = *decodePhase
	}

	// 5. Deduplicate warnings
	result.Warnings = deduplicateStrings(result.Warnings)

	// 6. Build summary
	BuildSummary(result)

	return result, nil
}

// createFakeBatch builds a fake input batch for graph construction.
// Mirrors the approach in runner/ollamarunner/runner.go:reserveWorstCaseGraph.
func createFakeBatch(ctx ml.Context, batchSize, kvCacheLen int) input.Batch {
	tokens := make([]int32, batchSize)
	positions := make([]int32, batchSize)
	sequences := make([]int, batchSize)
	for i := range tokens {
		positions[i] = int32(kvCacheLen - batchSize + i)
	}
	return input.Batch{
		Inputs:    ctx.Input().FromInts(tokens, len(tokens)),
		Positions: positions,
		Sequences: sequences,
	}
}

func deduplicateStrings(ss []string) []string {
	seen := make(map[string]bool)
	var result []string
	for _, s := range ss {
		if !seen[s] {
			seen[s] = true
			result = append(result, s)
		}
	}
	return result
}
```

**实现注意**:
- `createFakeBatch` 需要 `input` 包的 `Batch` 类型。添加 import: `"github.com/ollama/ollama/model/input"`
- `input.Batch` 的字段可能与实际 API 不完全匹配。参考 `runner/ollamarunner/runner.go:1069-1168`
  中 `reserveWorstCaseGraph` 的实际 batch 构造方式，特别是 `Outputs` 字段和 multimodal 处理。
- KV cache 初始化: `model.New()` 返回的 model 需要 cache 初始化才能 Forward。
  参考 `runner/ollamarunner/cache.go:34` 的 `NewInputCache()`，可能需要在 Forward 前调用。
- 如果 `model.New()` 的 `BackendParams` 不支持 "header only" 模式，可能会尝试加载完整权重。
  需要验证 `BackendParams{AllocMemory: false}` 或类似参数是否跳过权重加载。

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/workspace/daop-ollama && go test ./perf/ -run "TestEstimateGraph|TestBuildEstimate|TestComputePhase" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add perf/estimate.go perf/estimate_test.go
git commit -m "perf: add estimate module with graph-based latency estimation"
```

---

### Task 8: Profile Viewer (`perf/viewer.go`)

**Files:**
- Create: `perf/viewer.go`

Human-readable output for profile data and estimate results. Matches the format in spec Section 7.2.

- [ ] **Step 1: Implement viewer.go**

```go
package perf

import (
	"fmt"
	"io"
	"sort"
	"strings"
)

// PrintProfile prints a human-readable profile summary to w.
func PrintProfile(w io.Writer, p *Profile, detail bool) {
	fmt.Fprintln(w, "Hardware Profile")
	fmt.Fprintln(w, strings.Repeat("─", 60))
	fmt.Fprintf(w, "%-10s %-16s %-6s %-14s %-12s %s\n",
		"Backend", "Device", "Dtype", "Peak FLOPS", "Peak BW", "Balance Point")

	for _, bp := range p.Hardware.Backends {
		for dtype, flops := range bp.PeakFLOPS {
			balPt := bp.BalancePoints[dtype]
			fmt.Fprintf(w, "%-10s %-16s %-6s %-14s %-12s %.1f FLOP/byte\n",
				bp.Name, truncate(bp.Device, 16), dtype,
				formatSI(flops, "FLOPS"), formatSI(bp.PeakBandwidth, "B/s"), balPt)
		}
	}

	if len(p.Interconnects) > 0 {
		fmt.Fprintln(w)
		fmt.Fprintln(w, "Interconnects")
		fmt.Fprintln(w, strings.Repeat("─", 40))
		for _, ic := range p.Interconnects {
			fmt.Fprintf(w, "  %s → %s: %s\n", ic.From, ic.To, formatSI(ic.Bandwidth, "B/s"))
		}
	}

	if detail {
		fmt.Fprintln(w)
		fmt.Fprintln(w, "Operator Calibration (η)")
		fmt.Fprintln(w, strings.Repeat("─", 80))
		fmt.Fprintf(w, "%-16s %-10s %-8s %-8s %8s %10s %6s\n",
			"Op", "Backend", "Compute", "Weight", "η", "Variance", "Pts")

		sorted := make([]OperatorProfile, len(p.Operators))
		copy(sorted, p.Operators)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i].Op < sorted[j].Op })

		for _, op := range sorted {
			fmt.Fprintf(w, "%-16s %-10s %-8s %-8s %8.4f %10.6f %6d\n",
				op.Op, op.Backend, op.ComputeDtype, op.WeightDtype,
				op.Eta, op.EtaVariance, op.NumPoints)
		}
	}
}

// PrintEstimateResult prints a human-readable estimate result to w.
func PrintEstimateResult(w io.Writer, r *EstimateResult, detail bool) {
	// Header
	backends := make([]string, len(r.Backends))
	for i, b := range r.Backends {
		backends[i] = fmt.Sprintf("%s (%s)", b.Name, b.Device)
	}
	fmt.Fprintf(w, "Model: %s | Backend: %s\n", r.Model, strings.Join(backends, " + "))
	fmt.Fprintf(w, "Input: %d tokens | Output: %d tokens | Max batch: %d\n\n",
		r.InputLength, r.OutputLength, r.MaxBatchSize)

	// Prefill
	fmt.Fprintf(w, "Prefill (%d tokens", r.InputLength)
	if r.Prefill.NumBatches > 1 {
		fmt.Fprintf(w, ", %d batches of %d", r.Prefill.NumBatches, r.MaxBatchSize)
	}
	fmt.Fprintln(w, ")")
	fmt.Fprintln(w, strings.Repeat("─", 50))
	fmt.Fprintf(w, "  Estimated: %.0fms total, %.0f tok/s",
		r.Prefill.TotalLatencyMs, r.Prefill.TokensPerSec)
	if r.Prefill.TTFTMs > 0 {
		fmt.Fprintf(w, ", TTFT ≈ %.0fms", r.Prefill.TTFTMs)
	}
	fmt.Fprintln(w)
	fmt.Fprintf(w, "  Bottleneck: %s-bound\n", r.Prefill.Bottleneck)
	printTopOps(w, r.Prefill.TopOps, detail)

	fmt.Fprintln(w)

	// Decode
	fmt.Fprintf(w, "Decode (avg over %d positions)\n", r.OutputLength)
	fmt.Fprintln(w, strings.Repeat("─", 50))
	fmt.Fprintf(w, "  Estimated: %.1fms/tok, %.0f tok/s\n",
		r.Decode.TotalLatencyMs, r.Decode.TokensPerSec)
	fmt.Fprintf(w, "  Bottleneck: %s-bound\n", r.Decode.Bottleneck)
	printTopOps(w, r.Decode.TopOps, detail)

	// Warnings
	if len(r.Warnings) > 0 {
		fmt.Fprintln(w)
		fmt.Fprintln(w, "Warnings:")
		for _, w2 := range r.Warnings {
			fmt.Fprintf(w, "  ⚠ %s\n", w2)
		}
	}

	// Summary
	fmt.Fprintln(w)
	fmt.Fprintf(w, "Summary: %s\n", r.Summary)
}

func printTopOps(w io.Writer, ops []OpBreakdown, detail bool) {
	if len(ops) == 0 {
		return
	}
	fmt.Fprintln(w, "  Top ops:")
	fmt.Fprintf(w, "    %-16s %-8s %6s %10s %8s  %s\n",
		"Op", "Dtype", "Count", "Total ms", "%", "Bound breakdown")
	for _, op := range ops {
		dtype := op.ComputeDtype
		if op.WeightDtype != "" {
			dtype = op.WeightDtype
		}
		fmt.Fprintf(w, "    %-16s %-8s %6d %10.1fms %7.1f%%  %s\n",
			op.Op, dtype, op.Count, op.TotalMs, op.Percentage, op.BoundBreakdown)
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
	return s[:maxLen-1] + "…"
}
```

- [ ] **Step 2: Verify compilation**

Run: `cd /c/workspace/daop-ollama && go build ./perf/`
Expected: BUILD SUCCESS

- [ ] **Step 3: Commit**

```bash
git add perf/viewer.go
git commit -m "perf: add profile viewer and estimate result printer"
```

---

### Task 9: CLI Command Registration (`cmd/cmd.go`)

**Files:**
- Modify: `cmd/cmd.go:2078-2345` (NewCLI function)

Register `daop-bench` and `daop-estimate` as Cobra subcommands. The handler functions call into the `perf` package.

- [ ] **Step 1: Add import for perf package in `cmd/cmd.go`**

Add to the import block:

```go
"encoding/json"

"github.com/ollama/ollama/ml"
"github.com/ollama/ollama/perf"
```

- [ ] **Step 2: Add `daop-bench` command handler and registration**

Add a new function before `NewCLI()`:

```go
func daopBenchHandler(cmd *cobra.Command, args []string) error {
	backends, _ := cmd.Flags().GetStringSlice("backends")
	update, _ := cmd.Flags().GetBool("update")
	modelName, _ := cmd.Flags().GetString("model")
	viewMode := len(args) > 0 && args[0] == "view"
	detail, _ := cmd.Flags().GetBool("detail")

	if viewMode {
		profilePath := perf.ProfilePath()
		p, err := perf.LoadProfile(profilePath)
		if err != nil {
			return fmt.Errorf("cannot load profile: %w\nHave you run 'ollama daop-bench'?", err)
		}
		perf.PrintProfile(os.Stdout, p, detail)
		return nil
	}

	// Initialize backend without model (Task 5a: NewBackendForBench)
	backend, err := ml.NewBackendForBench(ml.BackendParams{})
	if err != nil {
		return fmt.Errorf("backend initialization failed: %w", err)
	}
	defer backend.Close()

	cfg := perf.DefaultBenchmarkConfig()
	cfg.Backends = backends

	if update {
		existing, err := perf.LoadProfile(perf.ProfilePath())
		if err != nil {
			return fmt.Errorf("cannot load existing profile for update: %w", err)
		}
		var modelPaths []string
		if modelName != "" {
			resolved, err := perf.ResolveModelPath(modelName)
			if err != nil {
				return err
			}
			modelPaths = append(modelPaths, resolved)
		} else {
			// Scan all local models (~/.ollama/models/)
			modelPaths, err = perf.ListLocalModels()
			if err != nil {
				return fmt.Errorf("cannot scan local models: %w", err)
			}
			if len(modelPaths) == 0 {
				return fmt.Errorf("no local models found; specify --model or run 'ollama pull <model>' first")
			}
		}
		raw, err := perf.RunUpdateBenchmark(backend, existing, modelPaths, cfg)
		if err != nil {
			return err
		}
		rawPath := perf.RawDataPath()
		if err := perf.WriteRawData(rawPath, raw); err != nil {
			return err
		}
		updated := perf.MergeProfile(existing, mustProcessRaw(rawPath))
		if err := perf.WriteProfile(perf.ProfilePath(), updated); err != nil {
			return err
		}
		fmt.Printf("Profile updated: %s\n", perf.ProfilePath())
		return nil
	}

	// Run full Layer 1 benchmark
	fmt.Println("Running hardware characterization + operator calibration...")
	fmt.Println("This may take 1-5 minutes.")
	raw, err := perf.RunFullBenchmark(backend, cfg)
	if err != nil {
		return err
	}
	rawPath := perf.RawDataPath()
	if err := perf.WriteRawData(rawPath, raw); err != nil {
		return err
	}
	profile, err := perf.ProcessRawToProfile([]string{rawPath})
	if err != nil {
		return err
	}
	if err := perf.WriteProfile(perf.ProfilePath(), profile); err != nil {
		return err
	}
	fmt.Printf("Profile saved to %s\n", perf.ProfilePath())
	perf.PrintProfile(os.Stdout, profile, false)
	return nil
}

func mustProcessRaw(rawPath string) *perf.Profile {
	p, err := perf.ProcessRawToProfile([]string{rawPath})
	if err != nil {
		panic(err)
	}
	return p
}
```

Add another function for the estimate handler:

```go
func daopEstimateHandler(cmd *cobra.Command, args []string) error {
	modelRef := args[0]

	inputLen, _ := cmd.Flags().GetInt("input-length")
	outputLen, _ := cmd.Flags().GetInt("output-length")
	batchSize, _ := cmd.Flags().GetInt("batch-size")
	jsonOutput, _ := cmd.Flags().GetBool("json")
	detail, _ := cmd.Flags().GetBool("detail")

	cfg := perf.DefaultEstimateConfig()
	cfg.InputLength = inputLen
	cfg.OutputLength = outputLen
	cfg.MaxBatchSize = batchSize
	cfg.JSON = jsonOutput
	cfg.Detail = detail

	result, err := perf.RunEstimate(modelRef, cfg)
	if err != nil {
		return err
	}

	if jsonOutput {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(result)
	}
	perf.PrintEstimateResult(os.Stdout, result, detail)
	return nil
}
```

- [ ] **Step 3: Register commands inside `NewCLI()`**

In `cmd/cmd.go`, inside the `NewCLI()` function, add before the final `rootCmd.AddCommand(...)` call:

```go
	daopBenchCmd := &cobra.Command{
		Use:   "daop-bench [view]",
		Short: "Run DAOP hardware benchmark and operator calibration",
		Long:  "Benchmark local hardware (peak FLOPS, bandwidth, operator calibration) for DAOP performance estimation.",
		Args:  cobra.MaximumNArgs(1),
		RunE:  daopBenchHandler,
	}
	daopBenchCmd.Flags().StringSlice("backends", nil, "Only benchmark specified backends (e.g. cuda,cpu)")
	daopBenchCmd.Flags().Bool("update", false, "Graph-driven discovery: scan local models for uncalibrated ops")
	daopBenchCmd.Flags().String("model", "", "Only scan specified model for --update")
	daopBenchCmd.Flags().Bool("detail", false, "Show detailed operator η values")

	daopEstimateCmd := &cobra.Command{
		Use:   "daop-estimate MODEL [flags]",
		Short: "Estimate LLM inference performance before loading",
		Long:  "Estimate prefill and decode tokens/sec for a model using DAOP Roofline model + calibration.",
		Args:  cobra.ExactArgs(1),
		RunE:  daopEstimateHandler,
	}
	daopEstimateCmd.Flags().Int("input-length", 512, "Input prompt length in tokens")
	daopEstimateCmd.Flags().Int("output-length", 128, "Output generation length in tokens")
	daopEstimateCmd.Flags().Int("batch-size", 512, "Max batch size for prefill")
	daopEstimateCmd.Flags().Bool("json", false, "Output structured JSON")
	daopEstimateCmd.Flags().Bool("detail", false, "Show per-op instance details")
```

Then add them to the rootCmd.AddCommand call:

```go
	rootCmd.AddCommand(
		// ... existing commands ...
		daopBenchCmd,
		daopEstimateCmd,
	)
```

- [ ] **Step 4: Verify compilation**

Run: `cd /c/workspace/daop-ollama && go build ./cmd/`
Expected: BUILD SUCCESS

- [ ] **Step 5: Verify CLI help output**

Run: `cd /c/workspace/daop-ollama && go run . daop-bench --help`
Expected: Shows usage for daop-bench with flags

Run: `cd /c/workspace/daop-ollama && go run . daop-estimate --help`
Expected: Shows usage for daop-estimate with flags

- [ ] **Step 6: Commit**

```bash
git add cmd/cmd.go
git commit -m "cmd: register daop-bench and daop-estimate CLI commands"
```

---

### Task 10: End-to-End Integration Test

**Files:**
- Create: `perf/integration_test.go`

Tests the full estimation pipeline with a mock profile: types → ops → roofline → estimate → viewer output.

- [ ] **Step 1: Write integration test**

```go
package perf

import (
	"bytes"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEndToEndEstimation(t *testing.T) { //nolint: uses fmt from package imports above
	// 1. Create a realistic profile
	profile := &Profile{
		Version:       1,
		GeneratedFrom: []string{"test"},
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
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0", Eta: 0.62, NumPoints: 7},
			{Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16", Eta: 0.75, NumPoints: 5},
			{Op: "RMS_NORM", Backend: "cuda", ComputeDtype: "f32", Eta: 0.85, NumPoints: 5},
			{Op: "ADD", Backend: "cuda", ComputeDtype: "f32", Eta: 0.90, NumPoints: 5},
			{Op: "SILU", Backend: "cuda", ComputeDtype: "f32", Eta: 0.88, NumPoints: 5},
		},
	}

	// 2. Simulate a simplified Llama-like transformer block graph for decode (batch=1)
	//    Real graph: embed → [block × N: RMS → QKV_MUL_MAT → ROPE → FA → O_PROJ → RMS → FFN_GATE → SILU → FFN_UP → MUL → FFN_DOWN] → RMS → LM_HEAD
	numLayers := 32
	var nodes []ml.GraphNode

	for l := 0; l < numLayers; l++ {
		// Attention: Q/K/V projections (3 MUL_MATs)
		for _, proj := range []string{"q", "k", "v"} {
			nodes = append(nodes, ml.GraphNode{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Name:        fmt.Sprintf("blk.%d.attn_%s", l, proj),
				Shape:       [4]int64{4096, 1, 1, 1},
				InputShapes: [][]int64{{4096, 4096}, {4096, 1}},
			})
		}
		// RMS Norm (pre-attention)
		nodes = append(nodes, ml.GraphNode{
			Op: "RMS_NORM", Backend: "cuda", ComputeDtype: "f32",
			Shape: [4]int64{4096, 1, 1, 1}, InputShapes: [][]int64{{4096, 1}},
		})
		// Flash Attention
		nodes = append(nodes, ml.GraphNode{
			Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
			Shape:       [4]int64{1, 32, 1, 128},
			InputShapes: [][]int64{{1, 32, 1, 128}, {1, 32, 1024, 128}},
		})
		// Output projection
		nodes = append(nodes, ml.GraphNode{
			Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			Shape: [4]int64{4096, 1, 1, 1}, InputShapes: [][]int64{{4096, 4096}, {4096, 1}},
		})
		// FFN: gate + up + down (3 MUL_MATs) + SILU + ADD
		for _, ffn := range []string{"gate", "up", "down"} {
			dim := int64(11008)
			if ffn == "down" {
				dim = 4096
			}
			nodes = append(nodes, ml.GraphNode{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Name:  fmt.Sprintf("blk.%d.ffn_%s", l, ffn),
				Shape: [4]int64{dim, 1, 1, 1},
				InputShapes: func() [][]int64 {
					if ffn == "down" {
						return [][]int64{{4096, 11008}, {11008, 1}}
					}
					return [][]int64{{11008, 4096}, {4096, 1}}
				}(),
			})
		}
		nodes = append(nodes, ml.GraphNode{
			Op: "SILU", Backend: "cuda", ComputeDtype: "f32",
			Shape: [4]int64{11008, 1, 1, 1}, InputShapes: [][]int64{{11008}},
		})
	}

	// LM head
	nodes = append(nodes, ml.GraphNode{
		Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
		Name: "lm_head", Shape: [4]int64{128256, 1, 1, 1},
		InputShapes: [][]int64{{128256, 4096}, {4096, 1}},
	})

	// 3. Estimate
	latency, stats, warnings := EstimateGraphLatency(profile, nodes)

	// Sanity checks
	require.Greater(t, latency, 0.0)
	assert.Greater(t, len(stats), 0)

	// MUL_MAT should dominate
	mulmatKey := OpKey{"MUL_MAT", "cuda", "f16", "q4_0"}
	mulmatStats, ok := stats[mulmatKey]
	require.True(t, ok)
	assert.Equal(t, 32*7+1, mulmatStats.Count) // 7 per layer × 32 + 1 lm_head
	assert.Greater(t, mulmatStats.TotalSec/latency, 0.5) // >50% of total

	// Should be memory-bound for decode (batch=1)
	assert.Greater(t, mulmatStats.MemCount, mulmatStats.CompCount)

	// 4. Build full result and print
	phase := ComputePhaseEstimation(profile, nodes, 1, 1)
	require.NotNil(t, phase)

	result := &EstimateResult{
		Model:        "llama3:8b-q4_0",
		InputLength:  1024,
		OutputLength: 256,
		MaxBatchSize: 512,
		Decode:       *phase,
		Warnings:     warnings,
		Backends: []BackendInfo{
			{Name: "cuda", Device: "RTX 4090", PeakFLOPS: 82.6e12, PeakBandwidth: 1008e9, BalancePoint: 81.9},
		},
	}
	BuildSummary(result)

	var buf bytes.Buffer
	PrintEstimateResult(&buf, result, false)
	output := buf.String()

	assert.Contains(t, output, "llama3:8b-q4_0")
	assert.Contains(t, output, "RTX 4090")
	assert.Contains(t, output, "MUL_MAT")
	assert.Contains(t, output, "tok/s")

	// Decode tok/s sanity: for 8B q4_0 on RTX 4090, expect ~50-200 tok/s
	assert.Greater(t, phase.TokensPerSec, 10.0, "decode too slow — check roofline math")
	assert.Less(t, phase.TokensPerSec, 1000.0, "decode too fast — check roofline math")

	t.Logf("Decode estimate: %.1f tok/s (%.2f ms/tok)", phase.TokensPerSec, phase.TotalLatencyMs)
	t.Log(output)
}

func TestProfileRoundTripIntegration(t *testing.T) {
	dir := t.TempDir()

	// Write raw data
	raw := &RawData{
		Version: 1,
		HardwareBenchmarks: []HardwareBenchmark{
			{Backend: "cuda", Dtype: "f16", Test: "peak_flops", Value: 82.6e12, Unit: "FLOPS"},
			{Backend: "cuda", Test: "peak_bandwidth", Value: 1008e9, Unit: "bytes/sec"},
		},
		OperatorBenchmarks: []OperatorBenchmark{
			{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Points: []BenchmarkPoint{
					{FLOPs: 33554432, BytesMoved: 8396800, Intensity: 4.0, LatencyUs: 12.5, Reps: 200},
					{FLOPs: 33554432e3, BytesMoved: 8396800e2, Intensity: 40.0, LatencyUs: 500, Reps: 200},
				},
			},
		},
	}

	rawPath := filepath.Join(dir, "raw.json")
	require.NoError(t, WriteRawData(rawPath, raw))

	// Process to profile
	profile, err := ProcessRawToProfile([]string{rawPath})
	require.NoError(t, err)

	assert.Equal(t, 1, len(profile.Hardware.Backends))
	assert.InDelta(t, 82.6e12, profile.Hardware.Backends[0].PeakFLOPS["f16"], 1e6)
	assert.Equal(t, 1, len(profile.Operators))
	assert.Greater(t, profile.Operators[0].Eta, 0.0)
	assert.LessOrEqual(t, profile.Operators[0].Eta, 1.5) // reasonable range

	// Write and re-read profile
	profilePath := filepath.Join(dir, "profile.json")
	require.NoError(t, WriteProfile(profilePath, profile))

	loaded, err := LoadProfile(profilePath)
	require.NoError(t, err)
	assert.Equal(t, profile.Operators[0].Eta, loaded.Operators[0].Eta)
}
```

- [ ] **Step 2: Run all tests**

Run: `cd /c/workspace/daop-ollama && go test ./perf/ -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add perf/integration_test.go
git commit -m "perf: add end-to-end integration tests for estimation pipeline"
```

---

## Task Dependency Graph

```
Task 1  (types)          ── no deps
Task 2  (ops)            ── depends on 1
Task 3  (roofline)       ── depends on 1, 2
Task 4  (profile)        ── depends on 1
Task 5  (GraphNodes CGo) ── independent of perf/, can parallel with 2-4
Task 5a (BackendForBench)── depends on 5 (same files)
Task 6  (bench)          ── depends on 1-4, 5a
Task 6a (model resolve)  ── depends on 1
Task 7  (estimate)       ── depends on 1-5, 6a
Task 8  (viewer)         ── depends on 1
Task 9  (CLI)            ── depends on 6, 6a, 7, 8
Task 10 (integration)    ── depends on all above
```

**Recommended execution order**:
1. Tasks 1→2→3→4 (sequential, pure Go, fast)
2. Tasks 5→5a (CGo, can parallel with 2-4)
3. Task 6a (model resolution, fast)
4. Tasks 6→7→8 (core modules)
5. Task 9 (CLI wiring)
6. Task 10 (integration test)

---

## Implementation Notes

### CGo Compilation
Tasks 5, 5a 及之后需要 CGo + GGML 头文件。无 GPU 环境:
- `go build -tags cpu` 可编译 CPU-only 版本
- GGML C 头文件位于 `ml/backend/ggml/ggml/include/`

### Tensor API 方法签名验证
bench.go 的 `benchSingleOp` 和 estimate.go 的 `createFakeBatch` 使用了 `ml.Tensor` 和
`input.Batch` 接口。实现时需要验证:
1. `a.Mulmat(b)` — 检查是否需要 Context 参数: `a.Mulmat(ctx, b)` vs `a.Mulmat(b)`
2. `a.Contiguous(ctx)` — 检查是否存在，可能是 `a.Copy(ctx)` 或 `a.Cont(ctx)`
3. `a.RoPE(...)` — 参数列表可能与简化版不同，参考 `ml/backend.go` Tensor 接口
4. `input.Batch` 字段 — 参考 `runner/ollamarunner/runner.go:1069` 中的实际构造
5. `ctx.Compute(out)` — 可能需要 `ctx.ComputeWithNotify(nil, out)` 以确保 GPU 同步

### KV Cache 初始化
`model.Forward()` 可能要求 KV cache 已初始化。estimate 模块在调 Forward 前可能需要:
```go
cache := model.Config().Cache
if cache != nil {
    cache.Init(m.Backend(), ml.DTypeF16, 1, kvCacheLen, batchSize)
}
```
参考 `runner/ollamarunner/cache.go:34` 的 `NewInputCache()`。

### Edge Cases
以下 guard clause 应在 `RunEstimate()` 入口和 CLI handler 中实现:
- **No profile**: `LoadProfile` 失败 → 返回 "请先运行 `ollama daop-bench`"（已实现）
- **Unsupported arch**: `model.New()` 返回 `ErrUnsupportedModel` → 返回 "架构 X 暂不支持"
- **Missing ops**: 未校准 op 用 η=1.0 + warning（已在 `EstimateGraphLatency` 中实现）
- **Multi-GPU split**: 由 `GraphNodes()` 的 backend 字段处理（已在 Task 5 中实现）
- **High η variance**: 在 viewer 输出中标记 `EtaVariance / Eta > 0.1` 的 op

### ListLocalModels 辅助函数
Task 9 CLI 的 `--update` 不指定 model 时需扫描所有本地模型。需要在 `perf/resolve.go` 中
添加 `ListLocalModels()` 函数，遍历 `~/.ollama/models/manifests/` 目录结构，
对每个 manifest 解析出 GGUF blob 路径。
