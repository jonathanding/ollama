# Benchmark Clean Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除 benchmark graph 中数据准备 ops（Cast/CPY）对 GPU timing 测量的污染，使非 f32 dtype 的 benchmark 结果准确反映目标 op 的真实 kernel 时间。

**Architecture:** 两阶段数据准备 — 在独立 context 中 materialize quantized 数据（Cast + ComputeOnBackend + Bytes），然后在 benchmark context 中用 FromBytes 创建 leaf tensor，确保 graph 只包含目标 op。

**Tech Stack:** Go, GGML C API (via CGO), Vulkan GPU timestamps

**Spec:** `docs/superpowers/specs/2026-04-06-benchmark-clean-graph.md`

**Testing conventions:**
- Backend 创建使用 `setupBenchBackend(t)`（定义在 `perf/direct_compute_test.go:13`），自动 skip 无 GPU 环境，Cleanup 自动 Close
- GPU timestamp 测试需检查 `caps.HasGPUTimestamp`，无则 skip
- 使用 `require` 做前置条件检查，`assert` 做验证断言
- 集成测试（需真实 GPU）放在 `_test.go` 中用 `setupBenchBackend` skip

---

## Phase 1: PoC — MUL_MAT clean graph 验证

### File Structure

| File | Action | Responsibility |
|---|---|---|
| `perf/registry.go` | Modify | 添加 `materializeTensor`；修改 `CreateInputs` 签名增加 backend 参数；修改 MUL_MAT/MUL_MAT_ADD CreateInputs 实现 |
| `perf/bench.go` | Modify | `measureOpGPU` 传递 backend 给 CreateInputs |
| `perf/registry_test.go` | Modify | 纯单元测试（不需要 GPU） |
| `perf/direct_compute_test.go` | Modify | GPU 集成测试：materializeTensor、clean graph 验证 |

---

### Task 1: 添加 `materializeTensor` 函数

**Files:**
- Modify: `perf/registry.go`
- Test: `perf/direct_compute_test.go`

- [ ] **Step 1: 写 `materializeTensor` 的测试**

在 `perf/direct_compute_test.go` 中添加。需要真实 GPU backend 来执行 Cast 和 ComputeOnBackend。

```go
func TestMaterializeTensor_Basic(t *testing.T) {
	backend := setupBenchBackend(t)

	// q4_0 block size is 32 elements — dimensions must be multiples of 32
	bytes := materializeTensor(backend, ml.DTypeQ40, 64, 32)

	require.NotNil(t, bytes, "should return non-nil bytes")
	assert.Greater(t, len(bytes), 0, "should return non-empty bytes")
	// q4_0: 32 elements per block, block = 2 bytes (f16 scale) + 16 bytes (data) = 18 bytes
	// 64*32 = 2048 elements, 2048/32 = 64 blocks, 64 * 18 = 1152 bytes
	assert.Equal(t, 1152, len(bytes), "byte count should match q4_0 format: 64*32 elements = 64 blocks * 18 bytes/block")
}

func TestMaterializeTensor_RoundTrip(t *testing.T) {
	backend := setupBenchBackend(t)

	// Materialize, then create leaf tensor from bytes and use it in a graph
	weightBytes := materializeTensor(backend, ml.DTypeQ40, 64, 32)

	ctx := backend.NewContext()
	defer ctx.Close()

	// Create leaf tensor from materialized bytes — no Cast in graph
	weight := ctx.Input().FromBytes(ml.DTypeQ40, weightBytes, 64, 32)
	activation := randomTensor(ctx, ml.DTypeF32, 64, 1)
	out := weight.Mulmat(ctx, activation)
	ctx.Forward(out)

	// Should compute without panic
	require.NotPanics(t, func() {
		ctx.ComputeOnBackend(0, out)
	})

	// Output shape should be [M=32, N=1] = 32 elements
	result := out.Floats()
	require.Len(t, result, 32, "MUL_MAT output should have M=32 elements")
}

func TestMaterializeTensor_MultipleDtypes(t *testing.T) {
	backend := setupBenchBackend(t)

	dtypes := []struct {
		dt   ml.DType
		name string
	}{
		{ml.DTypeF16, "f16"},
		{ml.DTypeQ40, "q4_0"},
		{ml.DTypeQ80, "q8_0"},
	}

	for _, tc := range dtypes {
		t.Run(tc.name, func(t *testing.T) {
			bytes := materializeTensor(backend, tc.dt, 64, 32)
			require.NotNil(t, bytes, "%s should return non-nil bytes", tc.name)
			assert.Greater(t, len(bytes), 0, "%s should return non-empty bytes", tc.name)
		})
	}
}

func TestMaterializeTensor_PrepContextDoesNotLeakIntoGraph(t *testing.T) {
	backend := setupBenchBackend(t)

	caps := DiscoverBackend(backend)
	if !caps.HasGPUTimestamp {
		t.Skip("no GPU timestamp support")
	}

	// Materialize weight outside graph
	weightBytes := materializeTensor(backend, ml.DTypeQ40, 256, 256)

	// Build a benchmark graph using materialized bytes
	backend.EnableGPUTimestamps(true)
	defer backend.EnableGPUTimestamps(false)

	ctx := backend.NewContext()
	defer ctx.Close()

	weight := ctx.Input().FromBytes(ml.DTypeQ40, weightBytes, 256, 256)
	activation := randomTensor(ctx, ml.DTypeF32, 256, 1)
	out := weight.Mulmat(ctx, activation)
	ctx.Forward(out)

	// Warmup
	ctx.ComputeOnBackend(0, out)
	// Measure
	ctx.ComputeOnBackend(0, out)
	timings := backend.GetOpTimings()

	require.NotEmpty(t, timings, "should have timing entries")
	for _, timing := range timings {
		assert.NotEqual(t, "CPY", timing.OpName,
			"graph should not contain CPY ops from prep context — found CPY at node %d", timing.NodeIdx)
	}
	// Should only have MUL_MAT or MUL_MAT_VEC
	assert.Len(t, timings, 1,
		"graph should have exactly 1 op (MUL_MAT/MUL_MAT_VEC), got %d", len(timings))
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `go test ./perf/ -run TestMaterializeTensor -v -count=1`
Expected: FAIL — `materializeTensor` undefined

- [ ] **Step 3: 实现 `materializeTensor`**

在 `perf/registry.go` 中添加，放在 `randomTensor` 函数之后：

```go
// materializeTensor creates quantized tensor data by executing a Cast op in a
// separate prep context, then reading back the bytes to CPU. The returned bytes
// can be passed to ctx.Input().FromBytes() to create a clean leaf tensor that
// does not inject any Cast/CPY ops into the benchmark graph.
//
// This implements the "two-phase data preparation" pattern:
//   Phase 1 (here): f32 random data → Cast to target dtype → ComputeOnBackend → Bytes()
//   Phase 2 (caller): FromBytes() creates a leaf tensor in the benchmark context
func materializeTensor(backend ml.Backend, dt ml.DType, shape ...int) []byte {
	prepCtx := backend.NewContext()
	defer prepCtx.Close()

	f32Tensor := randomTensor(prepCtx, ml.DTypeF32, shape...)
	castTensor := f32Tensor.Cast(prepCtx, dt)
	prepCtx.Forward(castTensor)
	prepCtx.ComputeOnBackend(0, castTensor)
	return castTensor.Bytes()
}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `go test ./perf/ -run TestMaterializeTensor -v -count=1`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add perf/registry.go perf/direct_compute_test.go
git commit -m "perf: add materializeTensor for clean graph data preparation"
```

---

### Task 2: 修改 CreateInputs 签名和 MUL_MAT 实现

**Files:**
- Modify: `perf/registry.go` — CreateInputs 签名 + 所有 ops 的签名更新 + MUL_MAT/MUL_MAT_ADD 实现
- Modify: `perf/bench.go` — measureOpGPU 传递 backend
- Test: `perf/direct_compute_test.go` — GPU 集成测试
注意：Step 3-6 必须在同一次编辑中完成，否则中间态无法编译（签名改了但调用点没改）。
注意：`perf/registry_test.go` 无需修改 — 现有测试只检查 `CreateInputs != nil`，不调用它，不受签名变更影响。

- [ ] **Step 1: 写测试验证 MUL_MAT graph 无 CPY op**

在 `perf/direct_compute_test.go` 中添加：

```go
func TestMulMatCleanGraph_Q40(t *testing.T) {
	backend := setupBenchBackend(t)

	caps := DiscoverBackend(backend)
	if !caps.HasGPUTimestamp {
		t.Skip("no GPU timestamp support")
	}

	runner, ok := LookupRegistry("MUL_MAT")
	require.True(t, ok)

	backend.EnableGPUTimestamps(true)
	defer backend.EnableGPUTimestamps(false)

	ctx := backend.NewContext()
	defer ctx.Close()

	inputs := runner.CreateInputs(ctx, backend, "q4_0", []int64{512, 512, 1})
	require.Len(t, inputs, 2, "MUL_MAT needs weight + activation")

	out := runner.Run(ctx, inputs)
	require.NotNil(t, out)
	ctx.Forward(out)

	// Warmup
	ctx.ComputeOnBackend(0, out)
	// Measure
	ctx.ComputeOnBackend(0, out)
	timings := backend.GetOpTimings()

	require.NotEmpty(t, timings)
	for _, timing := range timings {
		assert.NotEqual(t, "CPY", timing.OpName,
			"MUL_MAT(q4_0) graph should not contain CPY ops — found at node %d", timing.NodeIdx)
	}
}

func TestMulMatCleanGraph_F32_NoCast(t *testing.T) {
	backend := setupBenchBackend(t)

	caps := DiscoverBackend(backend)
	if !caps.HasGPUTimestamp {
		t.Skip("no GPU timestamp support")
	}

	runner, ok := LookupRegistry("MUL_MAT")
	require.True(t, ok)

	backend.EnableGPUTimestamps(true)
	defer backend.EnableGPUTimestamps(false)

	ctx := backend.NewContext()
	defer ctx.Close()

	// f32 path should also have no CPY (never had the bug, but verify regression)
	inputs := runner.CreateInputs(ctx, backend, "f32", []int64{256, 256, 1})
	out := runner.Run(ctx, inputs)
	ctx.Forward(out)

	ctx.ComputeOnBackend(0, out)
	ctx.ComputeOnBackend(0, out)
	timings := backend.GetOpTimings()

	for _, timing := range timings {
		assert.NotEqual(t, "CPY", timing.OpName, "f32 path should never have CPY")
	}
}

func TestMulMatAddCleanGraph_Q80(t *testing.T) {
	backend := setupBenchBackend(t)

	caps := DiscoverBackend(backend)
	if !caps.HasGPUTimestamp {
		t.Skip("no GPU timestamp support")
	}

	runner, ok := LookupRegistry("MUL_MAT_ADD")
	require.True(t, ok)

	backend.EnableGPUTimestamps(true)
	defer backend.EnableGPUTimestamps(false)

	ctx := backend.NewContext()
	defer ctx.Close()

	inputs := runner.CreateInputs(ctx, backend, "q8_0", []int64{512, 512, 1})
	require.Len(t, inputs, 3, "MUL_MAT_ADD needs weight + activation + bias")

	out := runner.Run(ctx, inputs)
	require.NotNil(t, out)
	ctx.Forward(out)

	ctx.ComputeOnBackend(0, out)
	ctx.ComputeOnBackend(0, out)
	timings := backend.GetOpTimings()

	require.NotEmpty(t, timings)
	for _, timing := range timings {
		assert.NotEqual(t, "CPY", timing.OpName,
			"MUL_MAT_ADD(q8_0) graph should not contain CPY ops")
	}
}

// TestMulMatCleanGraph_TimingAccuracy verifies the key fix:
// q4_0 MUL_MAT timing should be in the same order of magnitude as f32,
// not 4-5x higher (which was the bug: Cast time was being included).
func TestMulMatCleanGraph_TimingAccuracy(t *testing.T) {
	backend := setupBenchBackend(t)

	caps := DiscoverBackend(backend)
	if !caps.HasGPUTimestamp {
		t.Skip("no GPU timestamp support")
	}

	measureSingleOp := func(dtype string, M, K, N int64) float64 {
		runner, _ := LookupRegistry("MUL_MAT")

		backend.EnableGPUTimestamps(true)
		defer backend.EnableGPUTimestamps(false)

		ctx := backend.NewContext()
		defer ctx.Close()

		inputs := runner.CreateInputs(ctx, backend, dtype, []int64{M, K, N})
		out := runner.Run(ctx, inputs)
		ctx.Forward(out)

		// Warmup
		for range 3 {
			ctx.ComputeOnBackend(0, out)
		}
		// Measure
		ctx.ComputeOnBackend(0, out)
		timings := backend.GetOpTimings()
		var total float64
		for _, timing := range timings {
			total += timing.GPUTimeUs
		}
		return total
	}

	f32Time := measureSingleOp("f32", 2048, 2048, 1)
	q40Time := measureSingleOp("q4_0", 2048, 2048, 1)

	t.Logf("f32 MUL_MAT 2048x2048 N=1: %.1f us", f32Time)
	t.Logf("q4_0 MUL_MAT 2048x2048 N=1: %.1f us", q40Time)

	// q4_0 should be FASTER than f32 (less data to load), not 4-5x slower.
	// With the old bug, q4_0 was ~560us vs f32 ~275us.
	// After fix, q4_0 should be ~100-150us vs f32 ~275us.
	assert.Less(t, q40Time, f32Time*2,
		"q4_0 should not be much slower than f32 — if it is, Cast timing may still be included")
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `go test ./perf/ -run TestMulMatCleanGraph -v -count=1`
Expected: FAIL — `CreateInputs` 签名不匹配

- [ ] **Step 3: 修改 `OpRunnerML.CreateInputs` 签名并更新所有 ops**

在 `perf/registry.go` 中，一次性完成以下修改：

1) 修改 struct 定义：
```go
type OpRunnerML struct {
	Dimensions   []string
	CreateInputs func(ctx ml.Context, backend ml.Backend, dtypeStr string, gridPoint []int64) []ml.Tensor
	Run          func(ctx ml.Context, inputs []ml.Tensor) ml.Tensor
}
```

2) 更新所有 ops 的 CreateInputs 签名（添加 `backend ml.Backend` 参数）：
- `MUL_MAT` — 修改实现（见 step 4）
- `MUL_MAT_ADD` — 修改实现（见 step 4）
- `FLASH_ATTN_EXT` — 只加参数，实现不变
- `ROPE` — 只加参数，实现不变（Phase 2 修改）
- `RMS_NORM_MUL` — 只加参数，实现不变
- `RMS_NORM_MUL_ROPE` — 只加参数，实现不变

3) 修改 `perf/bench.go` 中 `measureOpGPU` 的 CreateInputs 调用（约第 199 行）：
```go
// 原代码：
inputs = runner.CreateInputs(ctx, computeDtype, gridPoint)
// 改为：
inputs = runner.CreateInputs(ctx, backend, computeDtype, gridPoint)
```

- [ ] **Step 4: 修改 MUL_MAT 和 MUL_MAT_ADD 的 CreateInputs 实现**

MUL_MAT：
```go
"MUL_MAT": {
	Dimensions: []string{"M", "K", "N"},
	CreateInputs: func(ctx ml.Context, backend ml.Backend, dtypeStr string, gridPoint []int64) []ml.Tensor {
		dt, _ := parseDType(dtypeStr)
		wShape, aShape := mulMatInputShapes(gridPoint)

		var weight ml.Tensor
		if dt != ml.DTypeF32 {
			weightBytes := materializeTensor(backend, dt, wShape...)
			weight = ctx.Input().FromBytes(dt, weightBytes, wShape...)
		} else {
			weight = randomTensor(ctx, dt, wShape...)
		}
		activation := randomTensor(ctx, ml.DTypeF32, aShape...)
		return []ml.Tensor{weight, activation}
	},
	Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
		return in[0].Mulmat(ctx, in[1])
	},
},
```

MUL_MAT_ADD：
```go
"MUL_MAT_ADD": {
	Dimensions: []string{"M", "K", "N"},
	CreateInputs: func(ctx ml.Context, backend ml.Backend, dtypeStr string, gridPoint []int64) []ml.Tensor {
		dt, _ := parseDType(dtypeStr)
		wShape, aShape := mulMatInputShapes(gridPoint)

		var weight ml.Tensor
		if dt != ml.DTypeF32 {
			weightBytes := materializeTensor(backend, dt, wShape...)
			weight = ctx.Input().FromBytes(dt, weightBytes, wShape...)
		} else {
			weight = randomTensor(ctx, dt, wShape...)
		}
		activation := randomTensor(ctx, ml.DTypeF32, aShape...)
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

- [ ] **Step 5: 运行 clean graph 测试确认通过**

Run: `go test ./perf/ -run TestMulMatCleanGraph -v -count=1`
Expected: 全部 PASS

- [ ] **Step 6: 运行全量测试确认无 regression**

Run: `go test ./perf/ -v -count=1`
Expected: 全部 PASS。注意检查：
- `TestRegistryCustomCreateInputs` — 仍应通过（检查 MUL_MAT 有 CreateInputs）
- `TestFusedOpRegistryEntries` — CreateInputs != nil 仍成立
- `TestBenchmarkMulMat_OutputShapeContract` — bench.go 调用签名已更新
- `TestMeasureOpForBackend_*` — 如果 mock 了 CreateInputs，需更新签名

- [ ] **Step 7: Commit**

```bash
git add perf/registry.go perf/bench.go perf/direct_compute_test.go
git commit -m "perf: clean graph for MUL_MAT benchmark — materialize inputs outside graph"
```

---

### Task 3: E2E 验证

**目标：** 运行完整 benchmark + estimate 流程，验证预测误差从 3.9x 降到 <2x。

- [ ] **Step 1: 运行 MUL_MAT benchmark**

```bash
go run . daop-bench --ops MUL_MAT
```

观察输出：
- q4_0 M=2048 K=2048 N=1 的 latency 应从 ~560μs 降到 ~100-150μs
- f32 的 latency 应不变（~275μs）
- 如果 q4_0 latency 仍然远高于 f32，说明 materializeTensor 或 FromBytes 路径有问题

- [ ] **Step 2: 用 GGML_VK_PERF_LOGGER 验证 graph 纯净**

```bash
GGML_VK_PERF_LOGGER=1 go run . daop-bench --ops MUL_MAT --dtypes q4_0
```

检查 stderr 输出：应该只有 MUL_MAT_VEC 条目，没有 CPY 条目。

- [ ] **Step 3: 运行 estimate**

```bash
go run . daop-estimate qwen3:1.7b
```

记录新的预测值和 tok/s。

- [ ] **Step 4: 对比结果**

| 指标 | 修改前 | 修改后 | 目标 |
|---|---|---|---|
| q4_0 2048×2048 N=1 benchmark | ~560 μs | ? (expect ~100-150 μs) | — |
| 预测 decode latency | 294 ms/tok | ? | <150 ms/tok |
| 实际 decode latency | ~75 ms/tok | ~75 ms/tok | — |
| 预测误差 | 3.9x | ? | <2x |

- [ ] **Step 5: 如果验证通过，记录结果**

在 spec 文件中记录 Phase 1 验证结果。

---

## Phase 2: 全面推广（Phase 1 验证通过后）

### Task 4: 添加 BytesCache 和 `randomLeafTensor`

**Files:**
- Modify: `perf/registry.go`
- Test: `perf/direct_compute_test.go`

- [ ] **Step 1: 写 BytesCache 测试**

```go
func TestMaterializeTensorCached_HitAndMiss(t *testing.T) {
	backend := setupBenchBackend(t)

	cache := make(BytesCache)

	// Miss: first call creates bytes
	b1 := materializeTensorCached(backend, cache, ml.DTypeQ40, 64, 32)
	require.NotNil(t, b1)
	assert.Len(t, cache, 1, "first call should add one entry")

	// Hit: same dtype+shape returns cached bytes (same slice)
	b2 := materializeTensorCached(backend, cache, ml.DTypeQ40, 64, 32)
	assert.Equal(t, len(b1), len(b2), "cached bytes should have same length")
	assert.Len(t, cache, 1, "second call should not add new entry")

	// Miss: different shape
	b3 := materializeTensorCached(backend, cache, ml.DTypeQ40, 128, 32)
	assert.Len(t, cache, 2, "different shape should add new entry")
	assert.NotEqual(t, len(b1), len(b3), "different shapes should produce different byte lengths")

	// Miss: different dtype, same shape
	b4 := materializeTensorCached(backend, cache, ml.DTypeF16, 64, 32)
	assert.Len(t, cache, 3, "different dtype should add new entry")
	assert.NotEqual(t, len(b1), len(b4), "different dtypes should produce different byte lengths")
}

func TestRandomLeafTensor_F32_NoPrep(t *testing.T) {
	backend := setupBenchBackend(t)

	cache := make(BytesCache)
	ctx := backend.NewContext()
	defer ctx.Close()

	// f32 should use randomTensor directly, not materialize
	leaf := randomLeafTensor(ctx, backend, cache, ml.DTypeF32, 256)
	require.NotNil(t, leaf)
	assert.Len(t, cache, 0, "f32 should not add to cache — no materialization needed")
}

func TestRandomLeafTensor_Q40_UsesCache(t *testing.T) {
	backend := setupBenchBackend(t)

	cache := make(BytesCache)
	ctx := backend.NewContext()
	defer ctx.Close()

	leaf := randomLeafTensor(ctx, backend, cache, ml.DTypeQ40, 64, 32)
	require.NotNil(t, leaf)
	assert.Len(t, cache, 1, "q4_0 should add to cache")

	// Second call with same shape should hit cache
	leaf2 := randomLeafTensor(ctx, backend, cache, ml.DTypeQ40, 64, 32)
	require.NotNil(t, leaf2)
	assert.Len(t, cache, 1, "same shape should hit cache")
}

func TestRandomLeafTensor_CleanGraph(t *testing.T) {
	backend := setupBenchBackend(t)

	caps := DiscoverBackend(backend)
	if !caps.HasGPUTimestamp {
		t.Skip("no GPU timestamp support")
	}

	cache := make(BytesCache)

	backend.EnableGPUTimestamps(true)
	defer backend.EnableGPUTimestamps(false)

	ctx := backend.NewContext()
	defer ctx.Close()

	// Build graph: two q4_0 leaf tensors → ADD
	a := randomLeafTensor(ctx, backend, cache, ml.DTypeQ40, 64, 32)
	b := randomLeafTensor(ctx, backend, cache, ml.DTypeQ40, 64, 32)
	// Cast both to f32 for ADD (ADD requires same dtype)
	// Actually, let's just use f16 for a simpler test
	a16 := randomLeafTensor(ctx, backend, cache, ml.DTypeF16, 256)
	b16 := randomLeafTensor(ctx, backend, cache, ml.DTypeF16, 256)
	out := a16.Add(ctx, b16)
	ctx.Forward(out)

	ctx.ComputeOnBackend(0, out)
	ctx.ComputeOnBackend(0, out)
	timings := backend.GetOpTimings()

	for _, timing := range timings {
		assert.NotEqual(t, "CPY", timing.OpName,
			"randomLeafTensor graph should not contain CPY ops")
	}
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `go test ./perf/ -run "TestMaterializeTensorCached|TestRandomLeafTensor" -v -count=1`
Expected: FAIL — `BytesCache` undefined

- [ ] **Step 3: 实现 BytesCache, materializeTensorCached, randomLeafTensor**

在 `perf/registry.go` 中添加（注意：需在 import 中添加 `"fmt"`）：

```go
// prepKey identifies a unique (dtype, shape) combination for caching materialized tensor bytes.
type prepKey struct {
	dtype string
	shape string
}

// BytesCache caches materialized tensor bytes to avoid redundant prep work.
type BytesCache map[prepKey][]byte

// materializeTensorCached is like materializeTensor but caches results.
// Same (dtype, shape) combination returns the same bytes without re-executing Cast.
func materializeTensorCached(backend ml.Backend, cache BytesCache, dt ml.DType, shape ...int) []byte {
	key := prepKey{dtypeToString(dt), fmt.Sprint(shape)}
	if b, ok := cache[key]; ok {
		return b
	}
	b := materializeTensor(backend, dt, shape...)
	cache[key] = b
	return b
}

// randomLeafTensor creates a tensor with random data as a graph leaf node.
// For f32: uses FromFloats directly (no Cast, no prep context).
// For non-f32: uses materializeTensorCached + FromBytes to avoid injecting Cast ops.
func randomLeafTensor(ctx ml.Context, backend ml.Backend, cache BytesCache, dt ml.DType, shape ...int) ml.Tensor {
	if dt == ml.DTypeF32 {
		return randomTensor(ctx, ml.DTypeF32, shape...)
	}
	bytes := materializeTensorCached(backend, cache, dt, shape...)
	return ctx.Input().FromBytes(dt, bytes, shape...)
}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `go test ./perf/ -run "TestMaterializeTensorCached|TestRandomLeafTensor" -v -count=1`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add perf/registry.go perf/direct_compute_test.go
git commit -m "perf: add BytesCache and randomLeafTensor for reusable clean inputs"
```

---

### Task 5: 推广到所有受影响的 ops

**Files:**
- Modify: `perf/registry.go` — ROPE + FLASH_ATTN_EXT CreateInputs 使用 materializeTensor
- Modify: `perf/bench.go` — measureOpGPU 默认路径使用 randomLeafTensor；BytesCache 传递
- Test: `perf/direct_compute_test.go`

- [ ] **Step 1: 写测试验证所有 non-f32 ops 的 graph 无 CPY**

```go
func TestAllOps_CleanGraph_NonF32(t *testing.T) {
	backend := setupBenchBackend(t)

	caps := DiscoverBackend(backend)
	if !caps.HasGPUTimestamp {
		t.Skip("no GPU timestamp support")
	}

	testCases := []struct {
		op    string
		dtype string
		shape []int64
	}{
		{"MUL_MAT", "q4_0", []int64{512, 512, 1}},
		{"MUL_MAT", "f16", []int64{512, 512, 1}},
		{"MUL_MAT", "q8_0", []int64{512, 512, 1}},
		{"MUL_MAT_ADD", "q4_0", []int64{512, 512, 1}},
		{"FLASH_ATTN_EXT", "f16", []int64{1, 512}},  // K/V are f16, affected by Cast bug
		{"ROPE", "f16", []int64{1024}},
		// 1D ops via default path
		{"ADD", "f16", []int64{256}},
		{"SILU", "f16", []int64{256}},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%s_%s", tc.op, tc.dtype), func(t *testing.T) {
			runner, ok := LookupRegistry(tc.op)
			require.True(t, ok)

			backend.EnableGPUTimestamps(true)
			defer backend.EnableGPUTimestamps(false)

			ctx := backend.NewContext()
			defer ctx.Close()

			var inputs []ml.Tensor
			if runner.CreateInputs != nil {
				inputs = runner.CreateInputs(ctx, backend, tc.dtype, tc.shape)
			} else {
				// Simulate default path with randomLeafTensor
				dt, ok := parseDType(tc.dtype)
				require.True(t, ok)
				cache := make(BytesCache)
				tensorShapes := expandShapes(tc.op, tc.shape)
				inputs = make([]ml.Tensor, len(tensorShapes))
				for i, shape := range tensorShapes {
					intShape := make([]int, len(shape))
					for j, s := range shape {
						intShape[j] = int(s)
					}
					inputs[i] = randomLeafTensor(ctx, backend, cache, dt, intShape...)
				}
			}

			out := runner.Run(ctx, inputs)
			require.NotNil(t, out)
			ctx.Forward(out)

			ctx.ComputeOnBackend(0, out)
			ctx.ComputeOnBackend(0, out)
			timings := backend.GetOpTimings()

			for _, timing := range timings {
				assert.NotEqual(t, "CPY", timing.OpName,
					"%s(%s): graph should not contain CPY ops", tc.op, tc.dtype)
			}
		})
	}
}
```

- [ ] **Step 2: 运行测试确认 MUL_MAT/MUL_MAT_ADD 通过，ROPE 和 1D ops 失败**

Run: `go test ./perf/ -run TestAllOps_CleanGraph_NonF32 -v -count=1`
Expected: MUL_MAT/MUL_MAT_ADD PASS, FLASH_ATTN_EXT/ROPE/ADD/SILU FAIL

- [ ] **Step 3: 修改 FLASH_ATTN_EXT 的 CreateInputs**

```go
"FLASH_ATTN_EXT": {
	Dimensions: []string{"seq_q", "seq_kv"},
	CreateInputs: func(ctx ml.Context, backend ml.Backend, dtypeStr string, gridPoint []int64) []ml.Tensor {
		// Q is f32 (matmul output in real inference), K/V are f16 (KV cache)
		seqQ, seqKV := gridPoint[0], gridPoint[1]
		q := randomTensor(ctx, ml.DTypeF32, 128, 32, int(seqQ), 1)
		// K/V are f16 — use materializeTensor to avoid Cast/CPY in graph
		kBytes := materializeTensor(backend, ml.DTypeF16, 128, 32, int(seqKV), 1)
		vBytes := materializeTensor(backend, ml.DTypeF16, 128, 32, int(seqKV), 1)
		k := ctx.Input().FromBytes(ml.DTypeF16, kBytes, 128, 32, int(seqKV), 1)
		v := ctx.Input().FromBytes(ml.DTypeF16, vBytes, 128, 32, int(seqKV), 1)
		return []ml.Tensor{q, k, v}
	},
	Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
		sdpa, ok := in[0].(ml.ScaledDotProductAttention)
		if !ok {
			return nil
		}
		headDim := in[0].Shape()[0]
		scale := 1.0 / math.Sqrt(float64(headDim))
		return sdpa.ScaledDotProductAttention(ctx, in[1], in[2], nil, nil, nil, scale, false)
	},
},
```

- [ ] **Step 4: 修改 ROPE 的 CreateInputs**

```go
"ROPE": {
	Dimensions: []string{"N"},
	CreateInputs: func(ctx ml.Context, backend ml.Backend, dtypeStr string, gridPoint []int64) []ml.Tensor {
		dt, _ := parseDType(dtypeStr)
		shape, seqLen := ropeInputParams(gridPoint[0])

		var input ml.Tensor
		if dt != ml.DTypeF32 {
			inputBytes := materializeTensor(backend, dt, shape...)
			input = ctx.Input().FromBytes(dt, inputBytes, shape...)
		} else {
			input = randomTensor(ctx, dt, shape...)
		}
		pos := ropePositions(seqLen)
		posTensor := ctx.Input().FromInts(pos, int(seqLen))
		return []ml.Tensor{input, posTensor}
	},
	Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
		type roper interface {
			RoPE(ctx ml.Context, positions ml.Tensor, dim int, base, scale float32, options ...func(*rope.Options)) ml.Tensor
		}
		if t, ok := in[0].(roper); ok {
			return t.RoPE(ctx, in[1], 128, 10000.0, 1.0)
		}
		return nil
	},
},
```

- [ ] **Step 5: 修改 `measureOpGPU` 默认路径使用 `randomLeafTensor`**

在 `perf/bench.go` 的 `measureOpGPU` 中：

1) 在函数开头创建 BytesCache：
```go
func measureOpGPU(backend ml.Backend, op string, gridPoint []int64, computeDtype string, cfg BenchmarkConfig) LatencyPoint {
	runner, ok := LookupRegistry(op)
	// ...
	dt, ok := parseDType(computeDtype)
	// ...

	cache := make(BytesCache) // local cache per measurement call
	// ...
```

2) 修改默认路径（约第 201-210 行）：
```go
} else {
	tensorShapes := expandShapes(op, gridPoint)
	inputs = make([]ml.Tensor, len(tensorShapes))
	for i, shape := range tensorShapes {
		intShape := make([]int, len(shape))
		for j, s := range shape {
			intShape[j] = int(s)
		}
		inputs[i] = randomLeafTensor(ctx, backend, cache, dt, intShape...)
	}
}
```

- [ ] **Step 6: 运行测试确认全部通过**

Run: `go test ./perf/ -run TestAllOps_CleanGraph_NonF32 -v -count=1`
Expected: 全部 PASS

- [ ] **Step 7: 运行全量测试**

Run: `go test ./perf/ -v -count=1`
Expected: 全部 PASS

- [ ] **Step 8: Commit**

```bash
git add perf/registry.go perf/bench.go perf/direct_compute_test.go
git commit -m "perf: extend clean graph to all ops and non-f32 dtypes"
```

---

### Task 6: Phase 2 E2E 验证

- [ ] **Step 1: 运行完整 benchmark（所有 ops + 所有 dtypes）**

```bash
go run . daop-bench
```

- [ ] **Step 2: 用 GGML_VK_PERF_LOGGER 验证无 CPY**

```bash
GGML_VK_PERF_LOGGER=1 go run . daop-bench --ops ROPE --dtypes f16
```

检查 stderr 无 CPY 条目。

- [ ] **Step 3: 运行 estimate**

```bash
go run . daop-estimate qwen3:1.7b
```

- [ ] **Step 4: 记录结果并更新 spec**

---

## Phase 3: 清理

### Task 7: 清理 randomTensor 和文档

- [ ] **Step 1: 检查 `randomTensor` 的 Cast 路径调用者**

```bash
# 查找所有 randomTensor 调用，确认是否还有传入非 f32 dtype 的
```

已知调用者：
- `materializeTensor` — 只传 `ml.DTypeF32`
- `benchOrchestrationOverhead`（bench.go:290）— 只传 `ml.DTypeF32`
- 各 op 的 CreateInputs 中 — 应已改用 `materializeTensor` / `randomLeafTensor`

如果确认 Cast 路径无调用者，简化 `randomTensor`：

```go
// randomTensor creates an f32 tensor with random data in [-1, 1].
// For non-f32 leaf tensors, use randomLeafTensor instead.
func randomTensor(ctx ml.Context, dt ml.DType, shape ...int) ml.Tensor {
	n := 1
	for _, s := range shape {
		n *= s
	}
	data := randomFloat32Slice(n)
	return ctx.Input().FromFloats(data, shape...)
}
```

注意：移除 Cast 路径后，`dt` 参数不再使用。可以保留参数以减少调用点改动，或删除参数并更新所有调用点。选择保留参数更安全（YAGNI 反向 — 删参数的 churn 不值得）。

- [ ] **Step 2: 运行全量测试确认无 regression**

Run: `go test ./perf/ -v -count=1`

- [ ] **Step 3: 更新 compact-snapshot.md**

- [ ] **Step 4: Commit**

```bash
git add perf/registry.go docs/
git commit -m "perf: remove Cast path from randomTensor, update docs"
```
