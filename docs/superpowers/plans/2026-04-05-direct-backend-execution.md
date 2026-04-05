# Direct Backend Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix benchmark running on CPU instead of GPU by bypassing the `ggml_backend_sched` scheduler and executing directly on the target backend. This unblocks E2E validation (Task 11) for the DAOP accuracy redesign.

**Architecture:** Add `ComputeOnBackend(backendIdx)` to `ml.Context` interface. Implementation reallocates graph tensors from CPU buffer to GPU buffer type, then calls `ggml_backend_graph_compute` directly. Benchmark control flow is unified via a work plan pattern.

**Tech Stack:** Go, CGO, GGML C API (`ggml-alloc.h`, `ggml-backend.h`)

**Design Spec:** `docs/superpowers/specs/2026-04-05-daop-accuracy-redesign.md` Section 2B

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `ml/backend.go` | `ml.Context` interface | Modify: add `ComputeOnBackend(int, ...Tensor)` |
| `ml/backend/ggml/ggml.go` | CGO GGML backend | Modify: implement `ComputeOnBackend`, add C helpers, clean debug logs |
| `perf/bench.go` | Benchmark orchestration | Modify: refactor to work plan pattern, use `ComputeOnBackend` |
| `perf/hwchar.go` | Hardware characterization | Modify: use `ComputeOnBackend` for peak TOPS/BW |
| `perf/types.go` | Benchmark types | Modify: add `BenchmarkStep`/`BenchmarkPlan`, remove `SkipHWChar` |
| `perf/cmd.go` | CLI wiring | Modify: remove `SkipHWChar` |
| `perf/estimate.go` | Estimation | Modify: remove debug logging |
| `cmd/cmd.go` | Cobra CLI | Modify: remove `--skip-hwchar` flag |
| `perf/bench_test.go` | Benchmark tests | Modify: add plan builder tests |
| `perf/direct_compute_test.go` | ComputeOnBackend tests | Create: integration tests |

---

### Task 1: ComputeOnBackend — CGO Method + ml.Context Interface

**Files:**
- Modify: `ml/backend.go:131` (Context interface)
- Modify: `ml/backend/ggml/ggml.go:1-50` (CGO preamble) and `:903-1008` (Context methods)
- Create: `perf/direct_compute_test.go`

**Problem Context:**
Currently, benchmark tensors are created via `ctx.Input().FromFloats()` which uses CPU buffer type. The `ggml_backend_sched` scheduler sees tensors on CPU and keeps ops on CPU. We need to:
1. Reallocate tensors from CPU buffer to GPU buffer
2. Execute `ggml_backend_graph_compute` directly (bypass scheduler)

**Key API semantics:**
- `ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)`: Allocates ALL unallocated tensors in a `ggml_context` on the specified buffer type. Only allocates tensors where `t->data == NULL`.
- `ggml_backend_graph_compute(backend, graph)`: Executes graph directly on one backend. No scheduler heuristics.
- After freeing old CPU buffers and resetting tensor `data`/`buffer` pointers to NULL, `alloc_ctx_tensors_from_buft` can re-allocate them on GPU.

**Implementation approach:**
1. On first `ComputeOnBackend` call: free existing CPU-allocated buffers, reset tensor pointers to NULL, allocate all tensors on GPU via `ggml_backend_alloc_ctx_tensors_from_buft`
2. On subsequent calls: tensors already on GPU, just call `ggml_backend_graph_compute` directly
3. Track state via `directBackendIdx` field on Context (-1 = not yet allocated)

- [ ] **Step 1: Add C helper functions to CGO preamble**

Add to `ml/backend/ggml/ggml.go` CGO preamble, after the existing `call_vk_*` functions and before `import "C"`:

```c
// #include "ggml-alloc.h"
//
// // Reset all tensor allocations in a ggml_context so they can be re-allocated.
// // Frees old_buffers[0..n_old-1], then sets data=NULL and buffer=NULL on all tensors.
// static void reset_ctx_tensor_allocs(
//     struct ggml_context * ctx,
//     ggml_backend_buffer_t * old_buffers,
//     int n_old) {
//     for (int i = 0; i < n_old; i++) {
//         ggml_backend_buffer_free(old_buffers[i]);
//     }
//     for (struct ggml_tensor * t = ggml_get_first_tensor(ctx);
//          t != NULL;
//          t = ggml_get_next_tensor(ctx, t)) {
//         t->data   = NULL;
//         t->buffer = NULL;
//     }
// }
```

Also add the `#include "ggml-alloc.h"` line near the top of the CGO preamble (after `#include "ggml-backend.h"`).

- [ ] **Step 2: Run build to verify C includes compile**

Run: `cd /c/workspace/daop-ollama && go build ./ml/backend/ggml/...`
Expected: Compiles without errors (or only pre-existing warnings).

- [ ] **Step 3: Add `directBackendIdx` and `directBuffer` fields to Context struct**

In `ml/backend/ggml/ggml.go`, modify the `Context` struct (around line 903):

```go
type Context struct {
	b *Backend

	ctx   *C.struct_ggml_context
	graph *C.struct_ggml_cgraph

	batchSize int
	buft      C.ggml_backend_buffer_type_t

	allocatedBuffers *[]C.ggml_backend_buffer_t

	maxGraphNodes int
	layer         int
	graphNodes    []ml.GraphNode

	// directBackendIdx tracks which backend tensors have been allocated on
	// for direct (non-scheduler) compute. -1 means not yet allocated.
	directBackendIdx int
	// directBuffer holds the GPU buffer allocated by ComputeOnBackend,
	// freed on Context.Close().
	directBuffer C.ggml_backend_buffer_t
}
```

Update `NewContextSize` to initialize `directBackendIdx: -1`.

Update `Context.Close()` to free `directBuffer`:

```go
func (c *Context) Close() {
	if c.directBuffer != nil {
		C.ggml_backend_buffer_free(c.directBuffer)
		c.directBuffer = nil
	}
	// ... existing cleanup ...
}
```

- [ ] **Step 4: Implement `ComputeOnBackend` on ggml Context**

```go
// ComputeOnBackend bypasses the ggml_backend_sched scheduler and executes
// the computation graph directly on the specified backend.
//
// backendIdx indexes into schedBackends (0 = first GPU, typically Vulkan/CUDA).
//
// On the first call, all tensors in this context are reallocated from their
// current buffer (typically CPU) onto the target backend's buffer type.
// Subsequent calls reuse the GPU allocation and just re-execute the graph.
//
// This is used by benchmark to ensure ops run on the intended hardware,
// bypassing the scheduler's heuristic that keeps small ops on CPU.
func (c *Context) ComputeOnBackend(backendIdx int, tensors ...ml.Tensor) {
	c.b.schedMu.Lock()
	defer c.b.schedMu.Unlock()

	if backendIdx < 0 || backendIdx >= len(c.b.schedBackends) {
		panic(fmt.Errorf("ComputeOnBackend: backendIdx %d out of range [0, %d)",
			backendIdx, len(c.b.schedBackends)))
	}

	backend := c.b.schedBackends[backendIdx]
	buft := c.b.schedBufts[backendIdx]

	// First call: reallocate all context tensors on the target backend
	if c.directBackendIdx != backendIdx {
		// Free old CPU-allocated buffers and reset tensor pointers
		if c.allocatedBuffers != nil && len(*c.allocatedBuffers) > 0 {
			C.reset_ctx_tensor_allocs(
				c.ctx,
				&(*c.allocatedBuffers)[0],
				C.int(len(*c.allocatedBuffers)),
			)
			*c.allocatedBuffers = nil
		}

		// Free any previous direct buffer
		if c.directBuffer != nil {
			C.ggml_backend_buffer_free(c.directBuffer)
		}

		// Allocate all tensors on the target backend's buffer type
		c.directBuffer = C.ggml_backend_alloc_ctx_tensors_from_buft(c.ctx, buft)
		if c.directBuffer == nil {
			panic("ComputeOnBackend: failed to allocate tensors on target backend")
		}

		c.directBackendIdx = backendIdx
	}

	// Execute graph directly on the target backend (no scheduler)
	if status := C.ggml_backend_graph_compute(backend, c.graph); status != C.GGML_STATUS_SUCCESS {
		panic(fmt.Errorf("ComputeOnBackend: graph compute failed: %v", status))
	}

	C.ggml_backend_synchronize(backend)

	// Mark output tensors as already synchronized — data is on the backend
	// and benchmark doesn't need to read values back to CPU.
	for _, t := range tensors {
		if C.ggml_nbytes(t.(*Tensor).t) > 0 {
			t.(*Tensor).sync = func() {}
		}
	}
}
```

- [ ] **Step 5: Add `ComputeOnBackend` to `ml.Context` interface**

In `ml/backend.go`, add to the `Context` interface (around line 147, after `Compute`):

```go
	Compute(...Tensor)
	ComputeWithNotify(func(), ...Tensor)

	// ComputeOnBackend bypasses the scheduler and executes the graph
	// directly on the specified backend. backendIdx 0 = first GPU.
	// Used by benchmark to ensure ops run on the intended hardware.
	ComputeOnBackend(backendIdx int, tensors ...Tensor)
```

- [ ] **Step 6: Write integration tests for ComputeOnBackend**

Create `perf/direct_compute_test.go`. These tests use the real GGML backend to verify:
1. ComputeOnBackend doesn't panic for a simple graph
2. Repeated calls (warmup + measure pattern) work correctly
3. GPU timestamps are returned when using direct backend on Vulkan
4. Invalid backendIdx panics with clear error message
5. Interleaving normal Compute and ComputeOnBackend on different contexts doesn't interfere

```go
package perf

import (
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// setupBenchBackend creates a real GGML backend for testing.
// Skips the test if no backend is available (e.g., CI without GPU).
func setupBenchBackend(t *testing.T) ml.Backend {
	t.Helper()
	backend, err := ml.NewBackendForBench(ml.BackendParams{NumThreads: 1})
	if err != nil {
		t.Skipf("no backend available: %v", err)
	}
	t.Cleanup(func() { backend.Close() })
	return backend
}

func TestComputeOnBackend_SimpleAdd(t *testing.T) {
	backend := setupBenchBackend(t)
	ctx := backend.NewContext()
	defer ctx.Close()

	// Build a trivial ADD graph
	a := randomTensor(ctx, ml.DTypeF32, 1024)
	b := randomTensor(ctx, ml.DTypeF32, 1024)
	out := a.Add(ctx, b)
	ctx.Forward(out)

	// Should not panic — allocates on backend 0 and computes directly
	require.NotPanics(t, func() {
		ctx.ComputeOnBackend(0, out)
	})
}

func TestComputeOnBackend_RepeatedCalls(t *testing.T) {
	backend := setupBenchBackend(t)
	ctx := backend.NewContext()
	defer ctx.Close()

	a := randomTensor(ctx, ml.DTypeF32, 4096)
	b := randomTensor(ctx, ml.DTypeF32, 4096)
	out := a.Add(ctx, b)
	ctx.Forward(out)

	// Simulate warmup + measurement: multiple calls should not re-allocate
	for i := 0; i < 5; i++ {
		require.NotPanics(t, func() {
			ctx.ComputeOnBackend(0, out)
		})
	}
}

func TestComputeOnBackend_MulMat(t *testing.T) {
	backend := setupBenchBackend(t)
	ctx := backend.NewContext()
	defer ctx.Close()

	// MUL_MAT: weight [K, M] * activation [K, N] = result [M, N]
	M, K, N := 256, 256, 4
	weight := randomTensor(ctx, ml.DTypeF32, K, M)
	activation := randomTensor(ctx, ml.DTypeF32, K, N)
	out := weight.Mulmat(ctx, activation)
	ctx.Forward(out)

	require.NotPanics(t, func() {
		ctx.ComputeOnBackend(0, out)
	})
}

func TestComputeOnBackend_InvalidIndex(t *testing.T) {
	backend := setupBenchBackend(t)
	ctx := backend.NewContext()
	defer ctx.Close()

	a := randomTensor(ctx, ml.DTypeF32, 64)
	b := randomTensor(ctx, ml.DTypeF32, 64)
	out := a.Add(ctx, b)
	ctx.Forward(out)

	assert.Panics(t, func() {
		ctx.ComputeOnBackend(999, out)
	}, "should panic on out-of-range backend index")

	assert.Panics(t, func() {
		ctx.ComputeOnBackend(-1, out)
	}, "should panic on negative backend index")
}

func TestComputeOnBackend_GPUTimestamps(t *testing.T) {
	backend := setupBenchBackend(t)

	// Check if we have a GPU backend (Vulkan/CUDA)
	caps := DiscoverBackend(backend)
	if !caps.HasGPUTimestamp {
		t.Skip("no GPU timestamp support (CPU-only backend)")
	}

	backend.EnableGPUTimestamps(true)
	defer backend.EnableGPUTimestamps(false)

	ctx := backend.NewContext()
	defer ctx.Close()

	// Build a non-trivial graph that should produce measurable GPU time
	a := randomTensor(ctx, ml.DTypeF32, 65536)
	b := randomTensor(ctx, ml.DTypeF32, 65536)
	out := a.Add(ctx, b)
	ctx.Forward(out)

	// Warmup
	ctx.ComputeOnBackend(0, out)

	// Measure — should get GPU timings back
	ctx.ComputeOnBackend(0, out)
	timings := backend.GetOpTimings()
	require.NotEmpty(t, timings, "GPU timings should be non-empty when computing directly on GPU backend")
	assert.Greater(t, timings[0].GPUTimeUs, 0.0, "GPU time should be positive")
}

func TestComputeOnBackend_LargeMulMat(t *testing.T) {
	backend := setupBenchBackend(t)

	// Test with a realistically-sized MUL_MAT to verify GPU allocation handles larger buffers
	ctx := backend.NewContext()
	defer ctx.Close()

	M, K, N := 4096, 4096, 1
	weight := randomTensor(ctx, ml.DTypeF32, K, M)
	activation := randomTensor(ctx, ml.DTypeF32, K, N)
	out := weight.Mulmat(ctx, activation)
	ctx.Forward(out)

	require.NotPanics(t, func() {
		ctx.ComputeOnBackend(0, out)
	})
}

func TestComputeOnBackend_MultipleContexts(t *testing.T) {
	backend := setupBenchBackend(t)

	// Two independent contexts should not interfere
	ctx1 := backend.NewContext()
	defer ctx1.Close()

	ctx2 := backend.NewContext()
	defer ctx2.Close()

	a1 := randomTensor(ctx1, ml.DTypeF32, 512)
	b1 := randomTensor(ctx1, ml.DTypeF32, 512)
	out1 := a1.Add(ctx1, b1)
	ctx1.Forward(out1)

	a2 := randomTensor(ctx2, ml.DTypeF32, 1024)
	b2 := randomTensor(ctx2, ml.DTypeF32, 1024)
	out2 := a2.Mul(ctx2, b2)
	ctx2.Forward(out2)

	require.NotPanics(t, func() {
		ctx1.ComputeOnBackend(0, out1)
		ctx2.ComputeOnBackend(0, out2)
	})
}

func TestComputeOnBackend_CPUBackendFallback(t *testing.T) {
	backend := setupBenchBackend(t)
	devices := backend.BackendDevices()

	// Find CPU backend index (usually last)
	cpuIdx := len(devices) // heuristic: CPU is the last schedBackend
	// For NewForBench, order is: GPUs first, then CPU
	// If only CPU exists, index 0 is CPU

	ctx := backend.NewContext()
	defer ctx.Close()

	a := randomTensor(ctx, ml.DTypeF32, 256)
	b := randomTensor(ctx, ml.DTypeF32, 256)
	out := a.Add(ctx, b)
	ctx.Forward(out)

	// Even if we specify backendIdx=0 on a CPU-only system, it should work
	// (schedBackends[0] would be CPU)
	require.NotPanics(t, func() {
		ctx.ComputeOnBackend(0, out)
	})
	_ = cpuIdx
}

func TestComputeOnBackend_QuantizedTensors(t *testing.T) {
	backend := setupBenchBackend(t)
	ctx := backend.NewContext()
	defer ctx.Close()

	// MUL_MAT with q4_0 weight — tests that quantized tensor allocation works on GPU
	M, K, N := 256, 256, 1
	// q4_0 requires K to be a multiple of 32 (block size)
	weight := randomTensor(ctx, ml.DTypeQ40, K, M)
	activation := randomTensor(ctx, ml.DTypeF32, K, N)
	out := weight.Mulmat(ctx, activation)
	ctx.Forward(out)

	require.NotPanics(t, func() {
		ctx.ComputeOnBackend(0, out)
	})
}
```

- [ ] **Step 7: Run tests**

Run: `cd /c/workspace/daop-ollama && go test ./perf/ -run TestComputeOnBackend -v -count=1`
Expected: All tests pass. The GPUTimestamps test may skip if no Vulkan backend.

- [ ] **Step 8: Commit**

```bash
git add ml/backend.go ml/backend/ggml/ggml.go perf/direct_compute_test.go
git commit -m "perf: add ComputeOnBackend to bypass scheduler for direct GPU execution"
```

---

### Task 2: Refactor measureOpGPU and measureOp to Use Direct Backend

**Files:**
- Modify: `perf/bench.go:51-270` (measureOp, measureOpGPU, measureOpForBackend)
- Modify: `perf/bench.go:720-783` (benchmarkElementwise, benchmarkMulMat, benchmarkFlashAttn)
- Modify: `perf/bench.go:272-314` (benchOrchestrationOverhead)
- Test: `perf/bench_test.go` (existing + new tests)

**Problem:**
`measureOp` and `measureOpGPU` both use `ctx.Compute(out)` which goes through the scheduler. With Task 1's `ComputeOnBackend`, we can ensure GPU ops run on GPU.

**Strategy:**
- `measureOpGPU`: Replace `ctx.Compute(out)` with `ctx.ComputeOnBackend(0, out)` for both warmup and measurement loops
- `measureOp` (wall-clock): Also use `ctx.ComputeOnBackend(0, out)` for GPU backends — wall-clock measurement on GPU is still useful for orchestration overhead, but individual op measurement should be on the correct backend
- `benchOrchestrationOverhead`: Keep using `ctx.Compute(out)` (scheduler) — this is intentional because we're measuring scheduler/dispatch overhead
- Add a `backendIdx` parameter to `measureOpForBackend` to pass through

**Key insight:** `measureOpGPU` already enables GPU timestamps. With `ComputeOnBackend`, the Vulkan backend will actually execute the ops and collect timestamps. This should fix the "nTimings=0" issue that blocked Task 11.

- [ ] **Step 1: Write failing test — measureOpGPU returns non-zero GPU timings**

Add to `perf/bench_test.go`:

```go
func TestMeasureOpGPU_ReturnsGPUTimings(t *testing.T) {
	backend := setupBenchBackend(t)
	caps := DiscoverBackend(backend)
	if !caps.HasGPUTimestamp {
		t.Skip("no GPU timestamp support")
	}

	cfg := DefaultBenchmarkConfig()
	cfg.MeasureReps = 5
	cfg.MinReps = 3

	// Measure a simple ADD op — should get GPU-based latency
	pt := measureOpGPU(backend, "ADD", []int64{65536}, "f32", cfg)
	assert.Greater(t, pt.LatencyUs, 0.0, "GPU-measured latency should be positive")
	assert.Greater(t, pt.Reps, 0, "should have measured at least 1 rep")
}

func TestMeasureOpForBackend_GPUDispatch(t *testing.T) {
	backend := setupBenchBackend(t)
	caps := DiscoverBackend(backend)

	cfg := DefaultBenchmarkConfig()
	cfg.MeasureReps = 5
	cfg.MinReps = 3

	// measureOpForBackend should dispatch to GPU measurement for Vulkan
	pt := measureOpForBackend(backend, caps, "ADD", []int64{65536}, "f32", cfg)
	assert.Greater(t, pt.LatencyUs, 0.0, "latency should be positive")
	assert.Greater(t, pt.Reps, 0, "should have measured at least 1 rep")
}

func TestMeasureOpGPU_FallbackToWallClock(t *testing.T) {
	backend := setupBenchBackend(t)
	caps := DiscoverBackend(backend)
	if caps.HasGPUTimestamp {
		t.Skip("this test is for backends WITHOUT GPU timestamps (CPU-only)")
	}

	cfg := DefaultBenchmarkConfig()
	cfg.MeasureReps = 5
	cfg.MinReps = 3

	// On CPU backend, measureOpForBackend should fall back to wall-clock
	pt := measureOpForBackend(backend, caps, "ADD", []int64{65536}, "f32", cfg)
	assert.Greater(t, pt.LatencyUs, 0.0, "wall-clock fallback should still produce positive latency")
	assert.Greater(t, pt.Reps, 0, "should have measured at least 1 rep")
}

func TestMeasureOp_SchedulerPath(t *testing.T) {
	backend := setupBenchBackend(t)

	cfg := DefaultBenchmarkConfig()
	cfg.MeasureReps = 5
	cfg.MinReps = 3

	// backendIdx=-1 uses scheduler (ctx.Compute) — the legacy path
	pt := measureOp(backend, "ADD", []int64{4096}, "f32", cfg, -1)
	assert.Greater(t, pt.LatencyUs, 0.0, "scheduler path should produce positive latency")

	// backendIdx=0 uses ComputeOnBackend — the direct path
	ptDirect := measureOp(backend, "ADD", []int64{4096}, "f32", cfg, 0)
	assert.Greater(t, ptDirect.LatencyUs, 0.0, "direct path should produce positive latency")

	// Both should produce similar order-of-magnitude results
	t.Logf("scheduler latency: %.1f us, direct latency: %.1f us", pt.LatencyUs, ptDirect.LatencyUs)
}

func TestMeasureOpGPU_MulMatVec(t *testing.T) {
	backend := setupBenchBackend(t)
	caps := DiscoverBackend(backend)
	if !caps.HasGPUTimestamp {
		t.Skip("no GPU timestamp support")
	}

	cfg := DefaultBenchmarkConfig()
	cfg.MeasureReps = 5
	cfg.MinReps = 3

	// MUL_MAT with N=1 (MUL_MAT_VEC kernel path) — verify it works with direct backend
	pt := measureOpGPU(backend, "MUL_MAT", []int64{256, 256, 1}, "f32", cfg)
	assert.Greater(t, pt.LatencyUs, 0.0, "MUL_MAT_VEC should produce positive GPU latency")
	t.Logf("MUL_MAT_VEC (256x256, N=1) GPU latency: %.1f us", pt.LatencyUs)
}

func TestMeasureOpGPU_ConvergenceEarlyExit(t *testing.T) {
	backend := setupBenchBackend(t)
	caps := DiscoverBackend(backend)
	if !caps.HasGPUTimestamp {
		t.Skip("no GPU timestamp support")
	}

	cfg := DefaultBenchmarkConfig()
	cfg.MeasureReps = 50 // high max to test early exit
	cfg.MinReps = 3
	cfg.ConvergenceCV = 0.5 // very lenient — should converge quickly

	pt := measureOpGPU(backend, "ADD", []int64{65536}, "f32", cfg)
	assert.Greater(t, pt.LatencyUs, 0.0, "should produce positive latency")
	// With CV=0.5 and a stable op, should converge well before 50 reps
	assert.Less(t, pt.Reps, 50, "should converge before max reps with lenient CV")
	t.Logf("converged in %d reps, latency: %.1f ± %.1f us", pt.Reps, pt.LatencyUs, pt.StddevUs)
}
```

- [ ] **Step 2: Run test to verify they fail (or give suspicious results)**

Run: `go test ./perf/ -run "TestMeasureOp" -v -count=1`
Expected: GPU timestamp tests fail (0 timings) before our fix. Scheduler/wallclock tests may pass.

- [ ] **Step 3: Modify `measureOp` to accept and use `backendIdx`**

The wall-clock measurement path also needs to run on the correct backend. Add a `backendIdx` parameter:

```go
// measureOp benchmarks an operator at one shape point using wall-clock timing.
// backendIdx specifies which backend to execute on (-1 = use scheduler).
func measureOp(backend ml.Backend, op string, gridPoint []int64,
	computeDtype string, cfg BenchmarkConfig, backendIdx int) LatencyPoint {
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
		slog.Warn("op runner returned nil", "op", op)
		return LatencyPoint{Shape: gridPoint}
	}
	ctx.Forward(out)

	// Choose compute function based on backendIdx
	compute := func() { ctx.Compute(out) }
	if backendIdx >= 0 {
		compute = func() { ctx.ComputeOnBackend(backendIdx, out) }
	}

	// Adaptive warmup
	warmupStart := time.Now()
	for i := 0; i < cfg.WarmupReps; i++ {
		compute()
		if i == 0 {
			elapsed := float64(time.Since(warmupStart).Microseconds())
			if elapsed > 5e6 {
				break
			} else if elapsed > 1e6 {
				compute()
				break
			}
		}
	}

	med, sd, actualReps := convergentMeasure(func() float64 {
		start := time.Now()
		compute()
		return float64(time.Since(start).Microseconds())
	}, cfg)

	return LatencyPoint{
		Shape:     gridPoint,
		LatencyUs: med,
		StddevUs:  sd,
		Reps:      actualReps,
	}
}
```

- [ ] **Step 4: Modify `measureOpGPU` to use `ComputeOnBackend`**

```go
// measureOpGPU benchmarks an operator using GPU timestamps on direct backend execution.
// This eliminates both scheduler heuristics and Vulkan dispatch overhead from measurements.
func measureOpGPU(backend ml.Backend, op string, gridPoint []int64,
	computeDtype string, cfg BenchmarkConfig) LatencyPoint {
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
		slog.Warn("op runner returned nil", "op", op)
		return LatencyPoint{Shape: gridPoint}
	}
	ctx.Forward(out)

	// Warmup — use direct backend execution
	for range 2 {
		ctx.ComputeOnBackend(0, out)
	}

	// Measure using GPU timestamps
	samples := make([]float64, 0, cfg.MeasureReps)
	for i := 0; i < cfg.MeasureReps; i++ {
		ctx.ComputeOnBackend(0, out)
		timings := backend.GetOpTimings()
		if len(timings) == 0 {
			slog.Warn("no GPU timings returned, falling back to wall-clock",
				"op", op, "shape", gridPoint)
			// Fall back to direct-backend wall-clock (still on GPU, just wall-clock timing)
			return measureOp(backend, op, gridPoint, computeDtype, cfg, 0)
		}
		var gpuUs float64
		for _, t := range timings {
			gpuUs += t.GPUTimeUs
		}
		samples = append(samples, gpuUs)

		if len(samples) >= cfg.MinReps {
			med, sd := trimmedStats(samples, cfg.TrimPercent)
			if med > 0 && sd/med < cfg.ConvergenceCV {
				return LatencyPoint{
					Shape:     gridPoint,
					LatencyUs: med,
					StddevUs:  sd,
					Reps:      len(samples),
				}
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

- [ ] **Step 5: Update `measureOpForBackend` to pass `backendIdx=0` for wall-clock path**

```go
func measureOpForBackend(backend ml.Backend, caps BackendCapabilities,
	op string, gridPoint []int64, computeDtype string, cfg BenchmarkConfig) LatencyPoint {
	if caps.HasGPUTimestamp {
		return measureOpGPU(backend, op, gridPoint, computeDtype, cfg)
	}
	// Wall-clock on direct backend (still bypasses scheduler for consistent backend)
	return measureOp(backend, op, gridPoint, computeDtype, cfg, 0)
}
```

- [ ] **Step 6: Update `benchOrchestrationOverhead` — keep scheduler (intentional)**

`benchOrchestrationOverhead` intentionally measures scheduler/dispatch overhead, so it should continue using `ctx.Compute(out)`. Add a comment to make this explicit:

```go
// benchOrchestrationOverhead measures CPU orchestration overhead for different graph sizes.
// IMPORTANT: This intentionally uses ctx.Compute (scheduler path) because we're measuring
// the scheduler's dispatch overhead itself, not GPU kernel time.
func benchOrchestrationOverhead(backend ml.Backend, cfg BenchmarkConfig) []LatencyPoint {
```

No functional change needed — just the clarifying comment.

- [ ] **Step 7: Run tests to verify GPU timings**

Run: `go test ./perf/ -run TestMeasureOp -v -count=1`
Expected: `TestMeasureOpGPU_ReturnsGPUTimings` passes with positive latency. If Vulkan is available, GPU timestamps should be non-zero.

- [ ] **Step 8: Commit**

```bash
git add perf/bench.go perf/bench_test.go
git commit -m "perf: use ComputeOnBackend in measureOp/measureOpGPU for direct GPU execution"
```

---

### Task 3: Refactor CharacterizeHardware for Direct Backend Execution

**Files:**
- Modify: `perf/hwchar.go:60-135` (benchPeakTOPS, benchPeakBandwidth)
- Test: `perf/hwchar_test.go` (existing + new tests)

**Problem:**
`benchPeakTOPS` and `benchPeakBandwidth` use `ctx.Compute(out)` which goes through the scheduler. For peak TOPS, a 4096³ MUL_MAT *might* get offloaded to Vulkan by the scheduler, but it's not guaranteed — the scheduler could keep it on CPU. For peak bandwidth, `CONT` on a large tensor might also stay on CPU.

This means our measured "44 GFLOPS" peak might actually be CPU performance, not GPU. With `ComputeOnBackend`, we guarantee execution on GPU and should see results much closer to the Intel Iris Xe theoretical ~1.69 TFLOPS.

- [ ] **Step 1: Write failing test — peak TOPS should be significantly higher than CPU**

Add to `perf/hwchar_test.go` (create if needed):

```go
package perf

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBenchPeakTOPS_DirectBackend(t *testing.T) {
	backend := setupBenchBackend(t)
	caps := DiscoverBackend(backend)
	if caps.Name == "CPU" {
		t.Skip("no GPU backend available")
	}

	cfg := DefaultBenchmarkConfig()
	cfg.MeasureReps = 10
	cfg.MinReps = 5

	tops, err := benchPeakTOPS(backend, caps, parseDTypeUnsafe("f32"), cfg)
	require.NoError(t, err)

	// For any discrete/integrated GPU, peak TOPS should be > 100 GFLOPS.
	// CPU might measure ~10-50 GFLOPS. Intel Iris Xe should be ~1.69 TFLOPS.
	// This threshold is conservative — mainly checks we're on GPU, not CPU.
	t.Logf("measured peak TOPS: %.2f GFLOPS (theoretical Iris Xe: ~1690 GFLOPS)", tops/1e9)
	assert.Greater(t, tops, 100e9,
		"peak TOPS should be > 100 GFLOPS — if lower, likely still running on CPU")
}

func TestBenchPeakBandwidth_DirectBackend(t *testing.T) {
	backend := setupBenchBackend(t)
	caps := DiscoverBackend(backend)
	if caps.Name == "CPU" {
		t.Skip("no GPU backend available")
	}

	cfg := DefaultBenchmarkConfig()
	cfg.MeasureReps = 10
	cfg.MinReps = 5

	bw, err := benchPeakBandwidth(backend, caps, cfg)
	require.NoError(t, err)

	// Peak bandwidth should be > 10 GB/s for any modern GPU/iGPU.
	// DDR4-3200 dual-channel theoretical is 51.2 GB/s.
	t.Logf("measured peak BW: %.2f GB/s (theoretical DDR4-3200: 51.2 GB/s)", bw/1e9)
	assert.Greater(t, bw, 10e9,
		"peak bandwidth should be > 10 GB/s")
}

// parseDTypeUnsafe is a test helper that panics if dtype is invalid.
func parseDTypeUnsafe(s string) ml.DType {
	dt, ok := parseDType(s)
	if !ok {
		panic("invalid dtype: " + s)
	}
	return dt
}
```

Note: The import for `ml` needs to be added: `"github.com/ollama/ollama/ml"`.

- [ ] **Step 2: Run test to verify baseline (likely fails threshold)**

Run: `go test ./perf/ -run TestBenchPeakTOPS_DirectBackend -v -count=1`
Expected: Likely fails because current code measures ~44 GFLOPS (probably CPU).

- [ ] **Step 3: Modify `benchPeakTOPS` to accept `BackendCapabilities` and use `ComputeOnBackend`**

```go
// benchPeakTOPS measures peak TOPS via large MUL_MAT (M=K=N=4096).
// Uses direct backend execution to ensure computation runs on GPU, not CPU.
func benchPeakTOPS(backend ml.Backend, caps BackendCapabilities,
	dtype ml.DType, cfg BenchmarkConfig) (float64, error) {
	const M, K, N = 4096, 4096, 4096
	ctx := backend.NewContext()
	defer ctx.Close()

	a := ctx.Input().Zeros(dtype, M, K)
	b := ctx.Input().Zeros(dtype, K, N)
	out := a.Mulmat(ctx, b)
	ctx.Forward(out)

	// Determine backend index (0 = first GPU for GPU backends)
	backendIdx := 0

	// Adaptive warmup
	warmupStart := time.Now()
	for i := 0; i < cfg.WarmupReps; i++ {
		ctx.ComputeOnBackend(backendIdx, out)
		if i == 0 {
			elapsed := float64(time.Since(warmupStart).Microseconds())
			if elapsed > 5e6 {
				break
			} else if elapsed > 1e6 {
				ctx.ComputeOnBackend(backendIdx, out)
				break
			}
		}
	}

	med, _, _ := convergentMeasure(func() float64 {
		start := time.Now()
		ctx.ComputeOnBackend(backendIdx, out)
		return float64(time.Since(start).Microseconds())
	}, cfg)

	median := med / 1e6
	flops := 2.0 * M * K * N
	return flops / median, nil
}
```

- [ ] **Step 4: Modify `benchPeakBandwidth` similarly**

```go
// benchPeakBandwidth measures peak memory bandwidth via large CONT (copy).
// Uses direct backend execution to ensure computation runs on GPU, not CPU.
func benchPeakBandwidth(backend ml.Backend, caps BackendCapabilities,
	cfg BenchmarkConfig) (float64, error) {
	const size = 64 * 1024 * 1024 // 64M elements
	ctx := backend.NewContext()
	defer ctx.Close()

	src := ctx.Input().Zeros(ml.DTypeF32, size)
	dst := src.Contiguous(ctx)
	ctx.Forward(dst)

	backendIdx := 0

	warmupStart := time.Now()
	for i := 0; i < cfg.WarmupReps; i++ {
		ctx.ComputeOnBackend(backendIdx, dst)
		if i == 0 {
			elapsed := float64(time.Since(warmupStart).Microseconds())
			if elapsed > 5e6 {
				break
			} else if elapsed > 1e6 {
				ctx.ComputeOnBackend(backendIdx, dst)
				break
			}
		}
	}

	med, _, _ := convergentMeasure(func() float64 {
		start := time.Now()
		ctx.ComputeOnBackend(backendIdx, dst)
		return float64(time.Since(start).Microseconds())
	}, cfg)

	median := med / 1e6
	bytesTotal := 2.0 * size * 4
	return bytesTotal / median, nil
}
```

- [ ] **Step 5: Update `CharacterizeHardware` to pass `caps`**

The `CharacterizeHardware` function currently doesn't have access to `BackendCapabilities`. Since `benchPeakTOPS` and `benchPeakBandwidth` now accept `caps`, update the caller:

```go
func CharacterizeHardware(backend ml.Backend, cfg BenchmarkConfig) (*HWCharResult, error) {
	devices := backend.BackendDevices()
	caps := DiscoverBackend(backend)

	result := &HWCharResult{
		PeakTOPS:     make(map[string]float64),
		BalancePoint: make(map[string]float64),
	}

	slog.Info("hardware characterization", "devices", len(devices), "backend", caps.Name)

	for _, dtypeStr := range []string{"f16", "f32"} {
		dt, ok := parseDType(dtypeStr)
		if !ok {
			continue
		}
		slog.Info("measuring peak TOPS", "dtype", dtypeStr)
		tops, err := benchPeakTOPS(backend, caps, dt, cfg)
		if err != nil {
			slog.Warn("peak TOPS failed", "dtype", dtypeStr, "error", err)
			continue
		}
		result.PeakTOPS[dtypeStr] = tops
		slog.Info("peak TOPS", "dtype", dtypeStr, "TOPS", tops)
	}

	slog.Info("measuring peak bandwidth")
	bw, err := benchPeakBandwidth(backend, caps, cfg)
	if err != nil {
		return nil, fmt.Errorf("peak bandwidth failed: %w", err)
	}
	result.PeakBW = bw
	slog.Info("peak bandwidth", "bytes_per_sec", bw)

	for dtype, tops := range result.PeakTOPS {
		if result.PeakBW > 0 {
			result.BalancePoint[dtype] = tops / result.PeakBW
		}
	}

	return result, nil
}
```

- [ ] **Step 6: Run tests**

Run: `go test ./perf/ -run "TestBenchPeak" -v -count=1`
Expected: Both pass. TOPS should be significantly higher than before (closer to GPU theoretical).

- [ ] **Step 7: Commit**

```bash
git add perf/hwchar.go perf/hwchar_test.go
git commit -m "perf: use ComputeOnBackend in CharacterizeHardware for accurate GPU peak measurement"
```

---

### Task 4: Benchmark Control Flow Unification (Work Plan Pattern)

**Files:**
- Modify: `perf/bench.go:316-595` (RunBenchmark, countGrids)
- Modify: `perf/types.go:107-130` (BenchmarkConfig — remove SkipHWChar)
- Create: `perf/plan.go` (BenchmarkStep, BenchmarkPlan, buildBenchmarkPlan)
- Test: `perf/plan_test.go`

**Problem:**
Current `RunBenchmark` has scattered conditional logic: `if cfg.SkipHWChar { ... }`, `if fusedOps[op] { continue }`, `if caps.HasGPUTimestamp && !cfg.SkipHWChar { ... }`. User feedback: "到处 patch，不干净。你应该上来就有一个准备跑的列表，根据参数确定这个列表，然后挨个跑。"

**Solution:** Extract a `buildBenchmarkPlan()` function that produces a flat list of `BenchmarkStep` entries based on (ops, dtypes, caps, cfg). `RunBenchmark` iterates the plan uniformly.

- [ ] **Step 1: Write failing tests for `buildBenchmarkPlan`**

Create `perf/plan_test.go`:

```go
package perf

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildBenchmarkPlan_FullPipeline(t *testing.T) {
	caps := GetBackendCapabilities("Vulkan")
	ops := DefaultBenchmarkOps()
	dtypes := Phase1Dtypes()

	plan := buildBenchmarkPlan(ops, dtypes, caps)
	require.NotEmpty(t, plan)

	// Must contain exactly one HWChar step at the beginning
	assert.Equal(t, StepHWChar, plan[0].Type, "first step should be hardware characterization")

	hwCharCount := 0
	opCount := 0
	fusedCount := 0
	overheadCount := 0
	mulMatRefCount := 0
	for _, s := range plan {
		switch s.Type {
		case StepHWChar:
			hwCharCount++
		case StepOperator:
			opCount++
		case StepMulMatRef:
			mulMatRefCount++
		case StepFusedOp:
			fusedCount++
		case StepOverhead:
			overheadCount++
		}
	}

	assert.Equal(t, 1, hwCharCount, "exactly one HWChar step")
	assert.Greater(t, opCount, 0, "should have operator steps")
	assert.Greater(t, mulMatRefCount, 0, "should have MUL_MAT reference curve steps")
	assert.Greater(t, fusedCount, 0, "should have fused op steps (Vulkan has fusion rules)")
	assert.Equal(t, 1, overheadCount, "exactly one orchestration overhead step (Vulkan has GPU timestamps)")
}

func TestBuildBenchmarkPlan_SpecificOps(t *testing.T) {
	caps := GetBackendCapabilities("Vulkan")
	ops := []string{"ADD", "SILU"}
	dtypes := []string{"f32"}

	plan := buildBenchmarkPlan(ops, dtypes, caps)
	require.NotEmpty(t, plan)

	// Should have: HWChar + ADD + SILU (no MUL_MAT, no fused, but still overhead)
	stepOps := make(map[string]int)
	for _, s := range plan {
		if s.Type == StepOperator {
			stepOps[s.Op]++
		}
	}
	assert.Equal(t, 1, stepOps["ADD"])
	assert.Equal(t, 1, stepOps["SILU"])
	assert.Zero(t, stepOps["MUL_MAT"], "MUL_MAT not requested")
}

func TestBuildBenchmarkPlan_NoFusionOnCPU(t *testing.T) {
	caps := GetBackendCapabilities("CPU")
	ops := DefaultBenchmarkOps()
	dtypes := Phase1Dtypes()

	plan := buildBenchmarkPlan(ops, dtypes, caps)

	fusedCount := 0
	overheadCount := 0
	for _, s := range plan {
		if s.Type == StepFusedOp {
			fusedCount++
		}
		if s.Type == StepOverhead {
			overheadCount++
		}
	}
	assert.Zero(t, fusedCount, "CPU has no fusion rules — no fused op steps")
	assert.Zero(t, overheadCount, "CPU has no GPU timestamps — no overhead step")
}

func TestBuildBenchmarkPlan_MulMatGeneratesRefCurves(t *testing.T) {
	caps := GetBackendCapabilities("Vulkan")
	ops := []string{"MUL_MAT"}
	dtypes := Phase1Dtypes()

	plan := buildBenchmarkPlan(ops, dtypes, caps)

	refCount := 0
	for _, s := range plan {
		if s.Type == StepMulMatRef {
			refCount++
		}
	}
	// One reference curve per weight dtype
	assert.Equal(t, len(Phase1Dtypes()), refCount,
		"MUL_MAT should produce one reference curve per weight dtype")
}

func TestBuildBenchmarkPlan_1DOpsUseF32Only(t *testing.T) {
	caps := GetBackendCapabilities("Vulkan")
	ops := []string{"ADD", "SILU", "RMS_NORM"}
	dtypes := []string{"f32", "f16", "q4_0"}

	plan := buildBenchmarkPlan(ops, dtypes, caps)

	for _, s := range plan {
		if s.Type == StepOperator {
			// 1D ops should only benchmark f32
			assert.Equal(t, "f32", s.Dtype,
				"1D op %s should only use f32, got %s", s.Op, s.Dtype)
		}
	}
}

func TestBuildBenchmarkPlan_FusedOpsExcludedFromMainLoop(t *testing.T) {
	caps := GetBackendCapabilities("Vulkan")
	ops := DefaultBenchmarkOps()
	dtypes := Phase1Dtypes()

	plan := buildBenchmarkPlan(ops, dtypes, caps)

	// Fused ops should appear as StepFusedOp, NOT as StepOperator
	for _, s := range plan {
		if s.Type == StepOperator {
			assert.NotContains(t, []string{"RMS_NORM_MUL", "RMS_NORM_MUL_ROPE", "MUL_MAT_ADD"},
				s.Op, "fused op %s should not appear as StepOperator", s.Op)
		}
	}
}

func TestBuildBenchmarkPlan_StepCount(t *testing.T) {
	caps := GetBackendCapabilities("Vulkan")
	ops := []string{"ADD"}
	dtypes := []string{"f32"}

	plan := buildBenchmarkPlan(ops, dtypes, caps)

	// Expected: HWChar(1) + ADD/f32(1) + FusedOps(3: RMS_NORM_MUL, RMS_NORM_MUL_ROPE, MUL_MAT_ADD) + Overhead(1)
	// But fused ops need their entries to be in the registry... MUL_MAT_ADD needs MUL_MAT benchmarks.
	// For this minimal plan: HWChar + ADD + 2 fused (RMS_NORM_MUL, RMS_NORM_MUL_ROPE) + MUL_MAT_ADD(per dtype) + Overhead
	// Actually fused ops are always included regardless of ops filter
	assert.GreaterOrEqual(t, len(plan), 3,
		"plan should have at least HWChar + ADD + Overhead")
}

func TestBuildBenchmarkPlan_EmptyOps(t *testing.T) {
	caps := GetBackendCapabilities("Vulkan")
	plan := buildBenchmarkPlan(nil, nil, caps)

	// With empty ops, should still have HWChar + fused + overhead
	hwCount := 0
	for _, s := range plan {
		if s.Type == StepHWChar {
			hwCount++
		}
	}
	assert.Equal(t, 1, hwCount, "always include HWChar")
}
```

- [ ] **Step 2: Run tests to verify they fail (types don't exist yet)**

Run: `go test ./perf/ -run TestBuildBenchmarkPlan -v -count=1`
Expected: Compilation error — `StepHWChar`, `buildBenchmarkPlan` not defined.

- [ ] **Step 3: Create `perf/plan.go` with types and `buildBenchmarkPlan`**

```go
package perf

import "log/slog"

// StepType identifies the kind of benchmark step in a BenchmarkPlan.
type StepType int

const (
	StepHWChar    StepType = iota // Hardware characterization (peak TOPS, BW)
	StepMulMatRef                 // MUL_MAT reference curve for one weight dtype
	StepOperator                  // Single op benchmark (1D or multi-dim)
	StepFusedOp                   // Fused op benchmark (RMS_NORM_MUL, etc.)
	StepOverhead                  // Orchestration overhead benchmark
)

// BenchmarkStep is one unit of work in a BenchmarkPlan.
type BenchmarkStep struct {
	Type        StepType
	Op          string           // for StepOperator/StepFusedOp/StepMulMatRef
	Dtype       string           // compute dtype
	WeightDtype string           // for MUL_MAT: weight dtype
	FixedDims   map[string]int64 // for multi-dim ops
}

// BenchmarkPlan is an ordered list of steps that RunBenchmark executes uniformly.
type BenchmarkPlan []BenchmarkStep

// buildBenchmarkPlan creates the complete list of benchmark steps based on parameters.
// This is the single point where filtering and ordering decisions are made.
// RunBenchmark simply iterates this list — no scattered conditionals.
func buildBenchmarkPlan(ops []string, dtypes []string, caps BackendCapabilities) BenchmarkPlan {
	var plan BenchmarkPlan

	// 1. Hardware characterization — always first
	plan = append(plan, BenchmarkStep{Type: StepHWChar})

	// Known fused ops — these are benchmarked separately, not in the main op loop
	fusedOps := map[string]bool{
		"RMS_NORM_MUL":      true,
		"RMS_NORM_MUL_ROPE": true,
		"MUL_MAT_ADD":       true,
	}

	// 2. Main operator benchmarks
	for _, op := range ops {
		if fusedOps[op] {
			continue // handled in step 3
		}

		if op == "MUL_MAT" {
			// MUL_MAT: one reference curve per weight dtype
			for _, wdt := range Phase1Dtypes() {
				plan = append(plan, BenchmarkStep{
					Type:        StepMulMatRef,
					Op:          "MUL_MAT",
					WeightDtype: wdt,
					FixedDims:   map[string]int64{"M": 4096, "K": 4096},
				})
			}
			continue
		}

		// Determine which dtypes to benchmark for this op
		opDtypes := dtypes
		if op == "FLASH_ATTN_EXT" {
			opDtypes = []string{"f16"}
		}
		runner, ok := LookupRegistry(op)
		if !ok {
			slog.Warn("skipping unknown op in plan", "op", op)
			continue
		}
		if len(runner.Dimensions) == 1 && runner.Dimensions[0] == "N" {
			opDtypes = []string{"f32"} // 1D ops only need f32
		}

		for _, dtype := range opDtypes {
			grids := buildSamplingGrids(op, dtype, "")
			for _, grid := range grids {
				plan = append(plan, BenchmarkStep{
					Type:      StepOperator,
					Op:        op,
					Dtype:     dtype,
					FixedDims: grid.FixedDims,
				})
			}
		}
	}

	// 3. Fused op benchmarks — if backend supports fusion
	if len(caps.FusionRules) > 0 {
		fusedList := []string{"RMS_NORM_MUL", "RMS_NORM_MUL_ROPE", "MUL_MAT_ADD"}
		for _, fop := range fusedList {
			if _, ok := LookupRegistry(fop); !ok {
				continue
			}
			if fop == "MUL_MAT_ADD" {
				// MUL_MAT_ADD: one per weight dtype
				for _, wdt := range Phase1Dtypes() {
					plan = append(plan, BenchmarkStep{
						Type:        StepFusedOp,
						Op:          fop,
						Dtype:       "f32",
						WeightDtype: wdt,
						FixedDims:   map[string]int64{"M": 4096, "K": 4096},
					})
				}
			} else {
				plan = append(plan, BenchmarkStep{
					Type:  StepFusedOp,
					Op:    fop,
					Dtype: "f32",
				})
			}
		}
	}

	// 4. Orchestration overhead — for GPU backends with timestamp support
	if caps.HasGPUTimestamp {
		plan = append(plan, BenchmarkStep{Type: StepOverhead})
	}

	return plan
}
```

- [ ] **Step 4: Run plan tests**

Run: `go test ./perf/ -run TestBuildBenchmarkPlan -v -count=1`
Expected: All tests pass.

- [ ] **Step 5: Rewrite `RunBenchmark` to iterate the plan**

Replace the existing `RunBenchmark` function body:

```go
func RunBenchmark(backend ml.Backend, ops []string, dtypes []string, cfg BenchmarkConfig) (*Profile, error) {
	benchStart := time.Now()

	caps := DiscoverBackend(backend)
	slog.Info("backend capabilities", "name", caps.Name,
		"gpu_timestamp", caps.HasGPUTimestamp, "fusion_rules", len(caps.FusionRules),
		"mul_mat_vec", caps.HasMulMatVec)

	plan := buildBenchmarkPlan(ops, dtypes, caps)
	slog.Info("benchmark plan", "steps", len(plan))

	profile := &Profile{
		Version:   3,
		Timestamp: time.Now(),
		BackendCaps: map[string]BackendCapabilitiesJSON{
			caps.Name: caps.ToJSON(),
		},
	}

	for i, step := range plan {
		elapsed := time.Since(benchStart).Round(time.Second)
		progress := fmt.Sprintf("[%d/%d]", i+1, len(plan))

		switch step.Type {
		case StepHWChar:
			slog.Info("hardware characterization", "progress", progress, "elapsed", elapsed)
			hwStart := time.Now()
			hwResult, err := CharacterizeHardware(backend, cfg)
			if err != nil {
				return nil, fmt.Errorf("hardware characterization: %w", err)
			}
			profile.Hardware = HWCharResultToHardwareProfile(hwResult, backend)
			slog.Info("hardware characterization complete",
				"progress", progress, "duration", time.Since(hwStart).Round(time.Second))

		case StepMulMatRef:
			slog.Info("benchmarking MUL_MAT reference curve",
				"progress", progress, "weight_dtype", step.WeightDtype, "elapsed", elapsed)
			gridStart := time.Now()

			refPoints := benchmarkMulMat(backend, caps, step.WeightDtype, step.FixedDims, cfg)
			if len(refPoints) == 0 {
				slog.Warn("MUL_MAT reference curve produced no points", "weight_dtype", step.WeightDtype)
				continue
			}

			refCurve := OperatorCurve{
				Op: "MUL_MAT", ComputeDtype: step.WeightDtype, WeightDtype: step.WeightDtype,
				Dimensions: []string{"N"}, FixedDims: step.FixedDims, Points: refPoints,
			}
			if devices := backend.BackendDevices(); len(devices) > 0 {
				refCurve.Backend = devices[0].Library
			}
			profile.Operators = append(profile.Operators, refCurve)

			// Extract per-dtype efficiency constants
			peakTOPS := profile.Hardware.PeakTOPS["f32"]
			if tops, ok := profile.Hardware.PeakTOPS[step.WeightDtype]; ok {
				peakTOPS = tops
			}
			eff := extractEfficiencyConstants(refPoints, 4096, 4096, peakTOPS,
				profile.Hardware.PeakBandwidthBytesPerSec, step.WeightDtype)
			if profile.Hardware.EfficiencyConstants == nil {
				profile.Hardware.EfficiencyConstants = make(map[string]OpEfficiency)
			}
			profile.Hardware.EfficiencyConstants["MUL_MAT_"+step.WeightDtype] = eff

			slog.Info("completed MUL_MAT reference", "progress", progress,
				"weight_dtype", step.WeightDtype, "points", len(refPoints),
				"eff_compute", fmt.Sprintf("%.2f", eff.ComputeEff),
				"eff_bw", fmt.Sprintf("%.2f", eff.BWEff),
				"duration", time.Since(gridStart).Round(time.Second))

		case StepOperator:
			slog.Info("benchmarking", "progress", progress,
				"op", step.Op, "dtype", step.Dtype, "fixed", step.FixedDims, "elapsed", elapsed)
			gridStart := time.Now()

			var curve OperatorCurve
			curve.Op = step.Op
			curve.ComputeDtype = step.Dtype
			curve.Dimensions = sweepDimensions(step.Op)
			curve.FixedDims = step.FixedDims
			if devices := backend.BackendDevices(); len(devices) > 0 {
				curve.Backend = devices[0].Library
			}

			switch step.Op {
			case "FLASH_ATTN_EXT":
				curve.Points = benchmarkFlashAttn(backend, caps, step.Dtype, step.FixedDims, cfg)
			default:
				curve.Points = benchmarkElementwise(backend, caps, step.Op, step.Dtype, cfg)
			}

			if len(curve.Points) > 0 {
				slog.Info("completed", "progress", progress,
					"op", step.Op, "dtype", step.Dtype, "points", len(curve.Points),
					"duration", time.Since(gridStart).Round(time.Second))
				profile.Operators = append(profile.Operators, curve)
			} else {
				slog.Warn("no points collected", "op", step.Op, "dtype", step.Dtype)
			}

		case StepFusedOp:
			slog.Info("benchmarking fused op", "progress", progress,
				"op", step.Op, "weight_dtype", step.WeightDtype, "elapsed", elapsed)

			switch step.Op {
			case "MUL_MAT_ADD":
				points := benchmarkMulMat(backend, caps, step.WeightDtype, step.FixedDims, cfg)
				var vecPoints []LatencyPoint
				for _, p := range points {
					if len(p.Shape) > 0 && p.Shape[0] <= 8 {
						vecPoints = append(vecPoints, p)
					}
				}
				if len(vecPoints) > 0 {
					profile.Operators = append(profile.Operators, OperatorCurve{
						Op: step.Op, Backend: caps.Name, ComputeDtype: "f32",
						WeightDtype: step.WeightDtype, Dimensions: []string{"N"},
						FixedDims: step.FixedDims, Points: vecPoints,
					})
				}
			default:
				measure := func(shape []int64) LatencyPoint {
					return measureOpForBackend(backend, caps, step.Op, shape, "f32", cfg)
				}
				points := AdaptiveSample1D(measure, 1024, 8*1024*1024, 8, cfg)
				if len(points) > 0 {
					profile.Operators = append(profile.Operators, OperatorCurve{
						Op: step.Op, Backend: caps.Name, ComputeDtype: "f32",
						Dimensions: []string{"N"}, Points: points,
					})
				}
			}

		case StepOverhead:
			slog.Info("benchmarking orchestration overhead", "progress", progress, "elapsed", elapsed)
			ohPoints := benchOrchestrationOverhead(backend, cfg)
			if len(ohPoints) > 0 {
				profile.Operators = append(profile.Operators, OperatorCurve{
					Op: "ORCHESTRATION_OVERHEAD", Backend: caps.Name,
					ComputeDtype: "f32", Dimensions: []string{"num_nodes"},
					Points: ohPoints,
				})
			}
		}
	}

	totalDuration := time.Since(benchStart).Round(time.Second)
	slog.Info("calibration complete", "operators", len(profile.Operators), "total_duration", totalDuration)

	return profile, nil
}
```

- [ ] **Step 6: Remove `countGrids` function**

The `countGrids` function is no longer needed — the plan length IS the count. Delete the function.

- [ ] **Step 7: Run existing benchmark tests to verify no regression**

Run: `go test ./perf/ -run Test -v -count=1 -short`
Expected: All existing tests pass. The plan produces the same set of benchmarks as before.

- [ ] **Step 8: Commit**

```bash
git add perf/plan.go perf/plan_test.go perf/bench.go
git commit -m "perf: unify RunBenchmark control flow with work plan pattern"
```

---

### Task 5: Cleanup — Remove Debug Logging and Hack Flags

**Files:**
- Modify: `ml/backend/ggml/ggml.go:1-50` (CGO preamble: remove `#include <stdio.h>`, `fprintf` in `resolve_vk_timestamps`)
- Modify: `ml/backend/ggml/ggml.go:866-901` (remove `slog.Info("DEBUG ...")` in `EnableGPUTimestamps` and `GetOpTimings`)
- Modify: `perf/estimate.go:384-392` (remove DEBUG slog prints)
- Modify: `perf/types.go:115` (remove `SkipHWChar` field from `BenchmarkConfig`)
- Modify: `perf/cmd.go:111-118` (remove `SkipHWChar` from `BenchmarkCLIOptions`, remove `cfg.SkipHWChar` assignment)
- Modify: `cmd/cmd.go:2096,2378` (remove `--skip-hwchar` flag)
- Test: compile and run existing tests

**Cleanup inventory (from compact-snapshot.md):**

| Location | What | Action |
|----------|------|--------|
| `ggml.go` CGO preamble | `#include <stdio.h>` | Remove (only needed for fprintf debug) |
| `ggml.go` CGO `resolve_vk_timestamps` | `fprintf(stderr, "DEBUG: ...")` (3 lines) | Remove |
| `ggml.go` `EnableGPUTimestamps` | `slog.Info("DEBUG EnableGPUTimestamps", ...)` (2 lines) | Remove |
| `ggml.go` `GetOpTimings` | `slog.Info("DEBUG GetOpTimings", ...)` | Remove |
| `estimate.go:384-392` | DEBUG slog prints for MUL_MAT and 1D ops | Remove |
| `types.go:115` | `SkipHWChar bool` field | Remove |
| `cmd.go:31-32` | `cfg.SkipHWChar = opts.SkipHWChar` | Remove |
| `cmd.go:117` | `SkipHWChar bool` in `BenchmarkCLIOptions` | Remove |
| `cmd/cmd.go:2096` | `skipHWChar, _ := cmd.Flags().GetBool("skip-hwchar")` | Remove |
| `cmd/cmd.go:2378` | `daopBenchCmd.Flags().Bool("skip-hwchar", ...)` | Remove |

- [ ] **Step 1: Remove debug logging from `ggml.go` CGO preamble**

In the `resolve_vk_timestamps` C function, remove the 3 `fprintf` lines. Also remove `#include <stdio.h>` if it's not needed by other code.

Before:
```c
// static void resolve_vk_timestamps(ggml_backend_dev_t dev) {
//     if (_vk_is_vk) return;
//     ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
//     if (!reg) { fprintf(stderr, "DEBUG: resolve_vk_timestamps: reg is NULL\n"); return; }
//     const char * reg_name = ggml_backend_reg_name(reg);
//     fprintf(stderr, "DEBUG: resolve_vk_timestamps: reg=%p reg_name=%s\n", (void*)reg, reg_name ? reg_name : "(null)");
//     ...
//     fprintf(stderr, "DEBUG: resolve_vk_timestamps: is_vk=%p enable=%p get_timings=%p\n", ...);
// }
```

After:
```c
// static void resolve_vk_timestamps(ggml_backend_dev_t dev) {
//     if (_vk_is_vk) return;
//     ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
//     if (!reg) return;
//     _vk_is_vk = (vk_is_vk_fn)(uintptr_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_is_vk");
//     _vk_enable_timestamps = ...;
//     _vk_get_op_timings = ...;
// }
```

- [ ] **Step 2: Remove debug logging from `EnableGPUTimestamps` and `GetOpTimings`**

In `EnableGPUTimestamps` (line ~866):
```go
// Remove these lines:
slog.Info("DEBUG EnableGPUTimestamps", "enable", enable, "num_backends", len(b.schedBackends))
slog.Info("DEBUG EnableGPUTimestamps backend", "idx", i, "is_vk", isVk)
```

In `GetOpTimings` (line ~882):
```go
// Remove this line:
slog.Info("DEBUG GetOpTimings", "backend_idx", i, "nTimings", int(nTimings), "timings_nil", timings == nil)
```

- [ ] **Step 3: Remove debug logging from `estimate.go`**

Remove the DEBUG slog prints around lines 384-392:
```go
// Remove these blocks:
if op == "MUL_MAT" && wdt == "f16" {
    slog.Info("DEBUG f16 MUL_MAT", ...)
}
if (op == "MUL" || op == "RMS_NORM" || op == "ADD" || op == "ROPE") && lat > 500 {
    slog.Info("DEBUG 1D op", ...)
}
```

- [ ] **Step 4: Remove `SkipHWChar` from types and CLI**

From `perf/types.go`, remove:
```go
SkipHWChar     bool    // skip hardware characterization (for debugging)
```

From `perf/cmd.go`, remove:
```go
SkipHWChar bool   // --skip-hwchar: skip hardware characterization (for debugging)
```
And remove:
```go
cfg.SkipHWChar = opts.SkipHWChar
```

From `cmd/cmd.go`, remove:
```go
skipHWChar, _ := cmd.Flags().GetBool("skip-hwchar")
```
And remove from the opts struct literal. And remove:
```go
daopBenchCmd.Flags().Bool("skip-hwchar", false, "Skip hardware characterization (for debugging)")
```

- [ ] **Step 5: Verify compilation**

Run: `go build ./...`
Expected: Clean compilation with no errors.

- [ ] **Step 6: Run all perf tests**

Run: `go test ./perf/ -v -count=1 -short`
Expected: All tests pass. No references to `SkipHWChar` or DEBUG logs remain.

- [ ] **Step 7: Verify no remaining debug artifacts**

Run grep to confirm cleanup is complete:
```bash
grep -rn "DEBUG" perf/ ml/backend/ggml/ggml.go | grep -v "_test.go" | grep -v "vendor"
grep -rn "SkipHWChar" perf/ cmd/ ml/
grep -rn "skip-hwchar" perf/ cmd/ ml/
grep -rn "fprintf.*stderr.*DEBUG" ml/backend/ggml/ggml.go
```
Expected: No matches (or only legitimate non-debug uses).

- [ ] **Step 8: Commit**

```bash
git add ml/backend/ggml/ggml.go perf/estimate.go perf/types.go perf/cmd.go cmd/cmd.go
git commit -m "perf: remove debug logging and temporary hack flags"
```

---

## Execution Order and Dependencies

```
Task 1 (ComputeOnBackend CGO) ──┐
                                 ├── Task 2 (measureOp refactor) ──┐
                                 ├── Task 3 (hwchar refactor)      ├── Task 5 (cleanup)
                                 │                                 │
Task 4 (work plan) ──────────────┘─────────────────────────────────┘
```

- **Task 1** is the foundation — Tasks 2 and 3 depend on it.
- **Task 4** (work plan) is independent of Tasks 2/3 and can be done in parallel, but logically fits after Tasks 2/3 since it restructures `RunBenchmark` which uses the refactored functions.
- **Task 5** (cleanup) should be last — it removes temporary code that earlier tasks may still reference during development.

## Post-Implementation: Resume E2E Validation (Task 11)

After all 5 tasks are complete:
1. Run `ollama daop-bench` — verify GPU timestamps are returned, peak TOPS is realistic
2. Run `ollama daop-estimate qwen3:1.7b` — compare with actual 75ms/tok
3. Target: estimate error < 2x
