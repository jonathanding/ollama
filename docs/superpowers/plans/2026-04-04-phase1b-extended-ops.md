# Phase 1B: Extended Operator Coverage + Random Initialization

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend benchmark/estimation to all common LLM operators, eliminating "uncalibrated" warnings for standard architectures (Llama, Gemma, Qwen, Mistral).

**Architecture:** Extend `OpRunnerML` with an optional `CreateInputs` function for non-standard tensor creation. Replace zero-initialized tensors with random data. Add ~10 new ops to registry. Unify `measureMulMat` into `measureOp` via custom `CreateInputs`.

**Tech Stack:** Go 1.24, `github.com/ollama/ollama/ml`, `github.com/stretchr/testify`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `perf/registry.go` | Modify | Extend `OpRunnerML` struct, add new op entries, add `randomTensor` helper |
| `perf/registry_test.go` | Modify | Tests for new ops, expanded `TestRegistryContainsAllOps`, `expandShapes` for 2-input ops |
| `perf/bench.go` | Modify | Update `measureOp` for random init + `CreateInputs`, remove `measureMulMat`, update `countGrids` and `RunBenchmark` |
| `perf/bench_test.go` | Modify | Update tests for new `measureOp` signature, remove `measureMulMat` references |
| `perf/ops.go` | Modify | Add `TRANSPOSE` to `IsZeroCostOp` |
| `perf/ops_test.go` | Modify | Add `TRANSPOSE` test case |
| `perf/cmd.go` | Modify | Expand default ops list |
| `perf/estimate_test.go` | Modify | Update `makeTestProfileForEstimation` with new op curves |

---

### Task 1: Random Tensor Helper + Update `measureOp`

Replace `ctx.Input().Zeros()` with random data for all benchmark tensor creation. Add `randomTensor` helper to `perf/registry.go`.

**Files:**
- Modify: `perf/registry.go` — add `randomTensor` function
- Modify: `perf/bench.go:72-81` — replace `Zeros` with `randomTensor`
- Test: `perf/registry_test.go` — add `TestRandomTensor`

- [ ] **Step 1: Write the test for randomTensor**

Add to `perf/registry_test.go`:

```go
func TestRandomFloat32Slice_Range(t *testing.T) {
	data := randomFloat32Slice(10000)
	for i, v := range data {
		assert.GreaterOrEqual(t, v, float32(-1.0), "index %d out of range low", i)
		assert.LessOrEqual(t, v, float32(1.0), "index %d out of range high", i)
	}
}

func TestRandomFloat32Slice_NotAllZeros(t *testing.T) {
	data := randomFloat32Slice(1024)
	allZero := true
	for _, v := range data {
		if v != 0 {
			allZero = false
			break
		}
	}
	assert.False(t, allZero, "random data should not be all zeros")
}

func TestRandomFloat32Slice_MeanNearZero(t *testing.T) {
	// Uniform[-1,1] has expected mean=0. With 10k samples, mean should be within ~0.05.
	data := randomFloat32Slice(10000)
	var sum float64
	for _, v := range data {
		sum += float64(v)
	}
	mean := sum / float64(len(data))
	assert.InDelta(t, 0.0, mean, 0.1, "mean of uniform[-1,1] should be near zero")
}

func TestRandomFloat32Slice_SpreadAcrossRange(t *testing.T) {
	// Verify values span the full [-1, 1] range — at least some negative and positive
	data := randomFloat32Slice(1000)
	var hasNeg, hasPos bool
	for _, v := range data {
		if v < -0.5 {
			hasNeg = true
		}
		if v > 0.5 {
			hasPos = true
		}
	}
	assert.True(t, hasNeg, "should have values < -0.5")
	assert.True(t, hasPos, "should have values > 0.5")
}

func TestRandomFloat32Slice_Length(t *testing.T) {
	for _, n := range []int{0, 1, 100, 10000} {
		data := randomFloat32Slice(n)
		assert.Len(t, data, n)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./perf/ -run TestRandomTensor -count=1 -v`
Expected: FAIL — `randomTensor` and `randomFloat32Slice` undefined

- [ ] **Step 3: Implement randomTensor and randomFloat32Slice**

Add to `perf/registry.go` (after the `import` block, before `opRegistry`):

```go
import (
	"math"
	"math/rand/v2"

	"github.com/ollama/ollama/ml"
)

// randomFloat32Slice generates a slice of n random float32 values in [-1, 1].
func randomFloat32Slice(n int) []float32 {
	data := make([]float32, n)
	for i := range data {
		data[i] = rand.Float32()*2 - 1
	}
	return data
}

// randomTensor creates a tensor with random float32 data, then casts to the target dtype.
// This avoids benchmarking with all-zero tensors which could trigger special-case fast paths.
func randomTensor(ctx ml.Context, dt ml.DType, shape ...int) ml.Tensor {
	n := 1
	for _, s := range shape {
		n *= s
	}
	data := randomFloat32Slice(n)
	t := ctx.Input().FromFloats(data, shape...)
	if dt != ml.DTypeF32 {
		t = t.Cast(ctx, dt)
	}
	return t
}
```

- [ ] **Step 4: Update measureOp to use randomTensor**

In `perf/bench.go`, replace lines 72-81 (the tensor creation loop):

Old:
```go
	// Create input tensors
	inputs := make([]ml.Tensor, runner.NumInputs)
	for i := 0; i < runner.NumInputs; i++ {
		if i < len(tensorShapes) {
			shape := tensorShapes[i]
			intShape := make([]int, len(shape))
			for j, s := range shape {
				intShape[j] = int(s)
			}
			inputs[i] = ctx.Input().Zeros(dt, intShape...)
		}
	}
```

New:
```go
	// Create input tensors with random data
	inputs := make([]ml.Tensor, len(tensorShapes))
	for i, shape := range tensorShapes {
		intShape := make([]int, len(shape))
		for j, s := range shape {
			intShape[j] = int(s)
		}
		inputs[i] = randomTensor(ctx, dt, intShape...)
	}
```

Note: `runner.NumInputs` is replaced by `len(tensorShapes)` — this prepares for removing `NumInputs` in Task 2.

- [ ] **Step 5: Run all tests**

Run: `go test ./perf/ -count=1 -v`
Expected: All existing tests pass + new random tensor tests pass

- [ ] **Step 6: Commit**

```bash
git add perf/registry.go perf/registry_test.go perf/bench.go
git commit -m "perf: add randomTensor helper and replace Zeros with random init in measureOp"
```

---

### Task 2: Extend OpRunnerML with CreateInputs + Remove NumInputs

Refactor `OpRunnerML` to support an optional `CreateInputs` function. Remove `NumInputs` field.

**Files:**
- Modify: `perf/registry.go:9-16` — update struct, update existing entries
- Modify: `perf/registry_test.go` — update `TestRegistryNumInputs` → `TestRegistryCreateInputs`
- Modify: `perf/bench.go:53-82` — update `measureOp` to use `CreateInputs`

- [ ] **Step 1: Update tests**

In `perf/registry_test.go`, replace `TestRegistryNumInputs`:

```go
func TestRegistryCreateInputsOrExpandShapes(t *testing.T) {
	// Every op must have Run and Dimensions.
	// Ops with complex tensor needs must have CreateInputs.
	// Ops using default (single random tensor) should NOT have CreateInputs.
	for name, runner := range opRegistry {
		t.Run(name, func(t *testing.T) {
			assert.NotNil(t, runner.Run, "op %q must have a Run function", name)
			assert.NotEmpty(t, runner.Dimensions, "op %q must have Dimensions", name)
		})
	}
}

func TestRegistryCustomCreateInputs(t *testing.T) {
	// These ops MUST have CreateInputs because they need non-standard tensor creation
	needCustom := []string{"MUL_MAT", "ROPE", "GET_ROWS"}
	for _, op := range needCustom {
		t.Run(op, func(t *testing.T) {
			runner, ok := opRegistry[op]
			require.True(t, ok)
			assert.NotNil(t, runner.CreateInputs, "%s requires custom CreateInputs", op)
		})
	}
}

func TestRegistryDefaultCreateInputs(t *testing.T) {
	// These 1D ops should use the default path (nil CreateInputs)
	defaultOps := []string{"SILU", "GELU", "RELU", "SOFT_MAX", "CONT", "RMS_NORM", "ADD", "MUL"}
	for _, op := range defaultOps {
		t.Run(op, func(t *testing.T) {
			runner, ok := opRegistry[op]
			require.True(t, ok)
			assert.Nil(t, runner.CreateInputs, "%s should use default tensor creation", op)
		})
	}
}
```

Also update `TestRegistryContainsPhase1Ops` — remove the `NumInputs > 0` assertion:

```go
func TestRegistryContainsPhase1Ops(t *testing.T) {
	required := []string{"SILU", "MUL_MAT", "FLASH_ATTN_EXT"}
	for _, op := range required {
		t.Run(op, func(t *testing.T) {
			runner, ok := opRegistry[op]
			assert.True(t, ok, "op %q must be in registry", op)
			assert.NotEmpty(t, runner.Dimensions)
			assert.NotNil(t, runner.Run)
		})
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./perf/ -run TestRegistryCreateInputs -count=1 -v`
Expected: FAIL — `NumInputs` still exists, new test references don't match

- [ ] **Step 3: Update OpRunnerML struct**

In `perf/registry.go`, replace the struct definition:

Old:
```go
type OpRunnerML struct {
	NumInputs  int
	Dimensions []string
	Run        func(ctx ml.Context, inputs []ml.Tensor) ml.Tensor
}
```

New:
```go
// OpRunnerML is the concrete registry entry type using ml package types.
// OpRunner in types.go uses interface{} to avoid circular imports;
// this is the real implementation.
type OpRunnerML struct {
	// Dimensions lists the performance-relevant shape dimensions for this op.
	Dimensions []string

	// CreateInputs creates input tensors for benchmarking.
	// If nil, measureOp uses the default: randomTensor with shapes from expandShapes.
	// Set this for ops that need non-standard tensor creation (mixed dtypes, special shapes).
	// dtypeStr is the raw dtype string (e.g. "f32", "q4_0") — call parseDType() internally if needed.
	CreateInputs func(ctx ml.Context, dtypeStr string, gridPoint []int64) []ml.Tensor

	// Run invokes the operator and returns the output tensor.
	Run func(ctx ml.Context, inputs []ml.Tensor) ml.Tensor
}
```

Update existing registry entries — remove `NumInputs` field:

```go
var opRegistry = map[string]OpRunnerML{
	"SILU": {
		Dimensions: []string{"N"},
		Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
			return in[0].SILU(ctx)
		},
	},
	"MUL_MAT": {
		Dimensions: []string{"M", "K", "N"},
		Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
			return in[0].Mulmat(ctx, in[1])
		},
	},
	"FLASH_ATTN_EXT": {
		Dimensions: []string{"seq_q", "seq_kv"},
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
}
```

- [ ] **Step 4: Update measureOp to use CreateInputs**

In `perf/bench.go`, update the tensor creation section in `measureOp`:

Old (after Task 1):
```go
	tensorShapes := expandShapes(op, gridPoint)

	ctx := backend.NewContext()
	defer ctx.Close()

	// Create input tensors with random data
	inputs := make([]ml.Tensor, len(tensorShapes))
	for i, shape := range tensorShapes {
		intShape := make([]int, len(shape))
		for j, s := range shape {
			intShape[j] = int(s)
		}
		inputs[i] = randomTensor(ctx, dt, intShape...)
	}
```

New:
```go
	ctx := backend.NewContext()
	defer ctx.Close()

	// Create input tensors — use custom CreateInputs if provided, else default
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
```

- [ ] **Step 5: Run all tests**

Run: `go test ./perf/ -count=1`
Expected: PASS (all 177 tests)

- [ ] **Step 6: Commit**

```bash
git add perf/registry.go perf/registry_test.go perf/bench.go
git commit -m "perf: extend OpRunnerML with optional CreateInputs, remove NumInputs"
```

---

### Task 3: Unify measureMulMat into measureOp via CreateInputs

Remove `measureMulMat` function. Add `CreateInputs` to the MUL_MAT registry entry that creates weight at quantized dtype and activation at f32.

**Files:**
- Modify: `perf/registry.go` — add `CreateInputs` to MUL_MAT entry
- Modify: `perf/bench.go:411-464` — remove `measureMulMat`, update `benchmarkMulMat` to call `measureOp`
- Modify: `perf/bench_test.go` — remove/update `measureMulMat`-specific tests

- [ ] **Step 1: Write tests for MUL_MAT shape helper**

Add to `perf/registry_test.go`:

```go
func TestMulMatInputShapes(t *testing.T) {
	tests := []struct {
		name      string
		gridPoint []int64
		wantW     []int // weight shape [K, M]
		wantA     []int // activation shape [K, N]
	}{
		{"decode", []int64{4096, 4096, 1}, []int{4096, 4096}, []int{4096, 1}},
		{"prefill", []int64{4096, 4096, 512}, []int{4096, 4096}, []int{4096, 512}},
		{"non-square", []int64{14336, 4096, 32}, []int{4096, 14336}, []int{4096, 32}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wShape, aShape := mulMatInputShapes(tt.gridPoint)
			assert.Equal(t, tt.wantW, wShape, "weight shape")
			assert.Equal(t, tt.wantA, aShape, "activation shape")
		})
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./perf/ -run TestMulMatInputShapes -count=1 -v`
Expected: FAIL — `mulMatInputShapes` undefined

- [ ] **Step 3: Implement helper and update MUL_MAT registry entry**

Add to `perf/registry.go`:

```go
// mulMatInputShapes returns (weightShape, activationShape) for MUL_MAT benchmarking.
// gridPoint = [M, K, N]. Weight is [K, M], activation is [K, N].
func mulMatInputShapes(gridPoint []int64) (weightShape, activationShape []int) {
	M, K, N := gridPoint[0], gridPoint[1], gridPoint[2]
	return []int{int(K), int(M)}, []int{int(K), int(N)}
}
```

Update the MUL_MAT registry entry:

```go
	"MUL_MAT": {
		Dimensions: []string{"M", "K", "N"},
		CreateInputs: func(ctx ml.Context, dtypeStr string, gridPoint []int64) []ml.Tensor {
			dt := parseDType(dtypeStr)
			wShape, aShape := mulMatInputShapes(gridPoint)
			weight := randomTensor(ctx, dt, wShape...)
			activation := randomTensor(ctx, ml.DTypeF32, aShape...)
			return []ml.Tensor{weight, activation}
		},
		Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
			return in[0].Mulmat(ctx, in[1])
		},
	},
```

- [ ] **Step 2: Update benchmarkMulMat to use measureOp**

In `perf/bench.go`, update `benchmarkMulMat`:

Old:
```go
func benchmarkMulMat(backend ml.Backend, weightDtype string, fixedDims map[string]int64, cfg BenchmarkConfig) []LatencyPoint {
	M := fixedDims["M"]
	K := fixedDims["K"]
	measure := func(shape []int64) LatencyPoint {
		N := shape[0]
		pt := measureMulMat(backend, M, K, N, weightDtype, cfg)
		pt.Shape = []int64{N} // 1D for AdaptiveSample1D
		return pt
	}
	return AdaptiveSample1D(measure, 1, 4096, 8, cfg)
}
```

New:
```go
func benchmarkMulMat(backend ml.Backend, weightDtype string, fixedDims map[string]int64, cfg BenchmarkConfig) []LatencyPoint {
	M := fixedDims["M"]
	K := fixedDims["K"]
	measure := func(shape []int64) LatencyPoint {
		N := shape[0]
		pt := measureOp(backend, "MUL_MAT", []int64{M, K, N}, weightDtype, cfg)
		pt.Shape = []int64{N} // 1D for AdaptiveSample1D
		return pt
	}
	return AdaptiveSample1D(measure, 1, 4096, 8, cfg)
}
```

- [ ] **Step 3: Remove measureMulMat function**

Delete `measureMulMat` function entirely from `perf/bench.go` (lines 411-464).

- [ ] **Step 4: Run all tests**

Run: `go test ./perf/ -count=1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add perf/registry.go perf/bench.go perf/bench_test.go
git commit -m "perf: unify measureMulMat into measureOp via MUL_MAT CreateInputs"
```

---

### Task 4: Add Simple 1-Input Ops (GELU, RELU, SOFT_MAX, CONT, RMS_NORM)

Add 5 new element-wise ops to the registry. All follow the same 1D pattern as SILU. NORM (LayerNorm) is excluded — no model currently uses it.

**Files:**
- Modify: `perf/registry.go` — add 5 new entries to `opRegistry`
- Modify: `perf/registry_test.go` — add tests for new ops

- [ ] **Step 1: Write tests for new ops**

Add to `perf/registry_test.go`:

```go
func TestRegistryContainsPhase1BOps(t *testing.T) {
	newOps := []string{"GELU", "RELU", "SOFT_MAX", "CONT", "RMS_NORM"}
	for _, op := range newOps {
		t.Run(op, func(t *testing.T) {
			runner, ok := opRegistry[op]
			assert.True(t, ok, "op %q must be in registry", op)
			assert.Equal(t, []string{"N"}, runner.Dimensions, "1D ops should have Dimensions=[N]")
			assert.NotNil(t, runner.Run)
			assert.Nil(t, runner.CreateInputs, "simple 1-input ops should use default tensor creation")
		})
	}
}

func TestExpandShapes_SimpleOps(t *testing.T) {
	for _, op := range []string{"GELU", "RELU", "SOFT_MAX", "CONT", "RMS_NORM"} {
		t.Run(op, func(t *testing.T) {
			shapes := expandShapes(op, []int64{65536})
			require.Len(t, shapes, 1, "%s needs 1 input tensor", op)
			assert.Equal(t, []int64{65536}, shapes[0])
		})
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./perf/ -run TestRegistryContainsPhase1BOps -count=1 -v`
Expected: FAIL — ops not in registry

- [ ] **Step 3: Add new ops to registry**

Add to `opRegistry` in `perf/registry.go`:

```go
	"GELU": {
		Dimensions: []string{"N"},
		Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
			return in[0].GELU(ctx)
		},
	},
	"RELU": {
		Dimensions: []string{"N"},
		Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
			return in[0].RELU(ctx)
		},
	},
	"SOFT_MAX": {
		Dimensions: []string{"N"},
		Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
			return in[0].Softmax(ctx)
		},
	},
	"CONT": {
		Dimensions: []string{"N"},
		Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
			return in[0].Contiguous(ctx)
		},
	},
	"RMS_NORM": {
		Dimensions: []string{"N"},
		Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
			// nil weight → only ggml_rms_norm, no subsequent ggml_mul
			return in[0].RMSNorm(ctx, nil, 1e-5)
		},
	},
```

- [ ] **Step 4: Run all tests**

Run: `go test ./perf/ -count=1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add perf/registry.go perf/registry_test.go
git commit -m "perf: add GELU, RELU, SOFT_MAX, CONT, RMS_NORM to operator registry"
```

---

### Task 5: Add 2-Input Ops (ADD, MUL) + Update expandShapes

Add ADD and MUL ops. These need `expandShapes` to return 2 same-shape tensors.

**Files:**
- Modify: `perf/registry.go:56-81` — update `expandShapes` for 2-input ops, add registry entries
- Modify: `perf/registry_test.go` — add tests

- [ ] **Step 1: Write tests**

Add to `perf/registry_test.go`:

```go
func TestRegistryContains2InputOps(t *testing.T) {
	for _, op := range []string{"ADD", "MUL"} {
		t.Run(op, func(t *testing.T) {
			runner, ok := opRegistry[op]
			assert.True(t, ok, "op %q must be in registry", op)
			assert.Equal(t, []string{"N"}, runner.Dimensions)
			assert.NotNil(t, runner.Run)
		})
	}
}

func TestExpandShapes_ADD(t *testing.T) {
	shapes := expandShapes("ADD", []int64{65536})
	require.Len(t, shapes, 2, "ADD needs 2 input tensors")
	assert.Equal(t, []int64{65536}, shapes[0])
	assert.Equal(t, []int64{65536}, shapes[1])
}

func TestExpandShapes_MUL(t *testing.T) {
	shapes := expandShapes("MUL", []int64{65536})
	require.Len(t, shapes, 2, "MUL needs 2 input tensors")
	assert.Equal(t, []int64{65536}, shapes[0])
	assert.Equal(t, []int64{65536}, shapes[1])
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./perf/ -run "TestRegistryContains2InputOps|TestExpandShapes_ADD|TestExpandShapes_MUL" -count=1 -v`
Expected: FAIL

- [ ] **Step 3: Update expandShapes and add registry entries**

In `perf/registry.go`, update `expandShapes`:

```go
func expandShapes(op string, gridPoint []int64) [][]int64 {
	switch op {
	case "FLASH_ATTN_EXT":
		seqQ, seqKV := gridPoint[0], gridPoint[1]
		return [][]int64{
			{128, 32, seqQ, 1},
			{128, 32, seqKV, 1},
			{128, 32, seqKV, 1},
		}
	case "MUL_MAT":
		_, K, N := gridPoint[0], gridPoint[1], gridPoint[2]
		M := gridPoint[0]
		return [][]int64{
			{K, M},
			{K, N},
		}
	case "ADD", "MUL":
		return [][]int64{gridPoint, gridPoint}
	default:
		return [][]int64{gridPoint}
	}
}
```

Add registry entries:

```go
	"ADD": {
		Dimensions: []string{"N"},
		Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
			return in[0].Add(ctx, in[1])
		},
	},
	"MUL": {
		Dimensions: []string{"N"},
		Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
			return in[0].Mul(ctx, in[1])
		},
	},
```

- [ ] **Step 4: Run all tests**

Run: `go test ./perf/ -count=1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add perf/registry.go perf/registry_test.go
git commit -m "perf: add ADD, MUL ops with 2-input expandShapes support"
```

---

### Task 6: Add ROPE (Custom CreateInputs + Type Assertion)

ROPE requires a 4D input tensor and an i32 positions tensor. `RoPE` is not on `ml.Tensor` — requires type assertion.

**Files:**
- Modify: `perf/registry.go` — add ROPE entry with CreateInputs
- Modify: `perf/registry_test.go` — add tests

- [ ] **Step 1: Write tests for ROPE shape helper**

Add to `perf/registry_test.go`:

```go
func TestRopeInputParams(t *testing.T) {
	tests := []struct {
		name       string
		totalN     int64
		wantShape  []int   // input tensor shape [headDim, 1, seqLen, 1]
		wantSeqLen int64
	}{
		{"single_token", 128, []int{128, 1, 1, 1}, 1},
		{"batch_8", 1024, []int{128, 1, 8, 1}, 8},
		{"large_seq", 65536, []int{128, 1, 512, 1}, 512},
		{"below_head_dim", 64, []int{128, 1, 1, 1}, 1}, // clamp to 1
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			shape, seqLen := ropeInputParams(tt.totalN)
			assert.Equal(t, tt.wantShape, shape, "input tensor shape")
			assert.Equal(t, tt.wantSeqLen, seqLen, "seqLen")
		})
	}
}

func TestRopeInputParams_PositionsAreSequential(t *testing.T) {
	_, seqLen := ropeInputParams(1024)
	pos := ropePositions(seqLen)
	require.Len(t, pos, int(seqLen))
	for i, v := range pos {
		assert.Equal(t, int32(i), v, "position %d", i)
	}
}

func TestRegistryROPE(t *testing.T) {
	runner, ok := opRegistry["ROPE"]
	assert.True(t, ok, "ROPE must be in registry")
	assert.Equal(t, []string{"N"}, runner.Dimensions)
	assert.NotNil(t, runner.CreateInputs, "ROPE needs custom tensor creation")
	assert.NotNil(t, runner.Run)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./perf/ -run "TestRopeInput|TestRegistryROPE" -count=1 -v`
Expected: FAIL — `ropeInputParams` and `ropePositions` undefined

- [ ] **Step 3: Add ROPE helpers and registry entry**

Add helper functions to `perf/registry.go`:

```go
// ropeInputParams computes the 4D input tensor shape and seqLen for ROPE benchmarking.
// gridPoint[0] = N (total elements). Input shape is [headDim=128, 1, seqLen, 1].
func ropeInputParams(totalN int64) (shape []int, seqLen int64) {
	const headDim = 128
	seqLen = totalN / headDim
	if seqLen < 1 {
		seqLen = 1
	}
	return []int{headDim, 1, int(seqLen), 1}, seqLen
}

// ropePositions returns sequential position indices [0, 1, ..., seqLen-1] as int32.
func ropePositions(seqLen int64) []int32 {
	pos := make([]int32, seqLen)
	for i := range pos {
		pos[i] = int32(i)
	}
	return pos
}
```

Add import for the rope package at top of `perf/registry.go`:

```go
import (
	"math"
	"math/rand/v2"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn/rope"
)
```

Add to `opRegistry`:

```go
	"ROPE": {
		Dimensions: []string{"N"},
		CreateInputs: func(ctx ml.Context, dtypeStr string, gridPoint []int64) []ml.Tensor {
			dt := parseDType(dtypeStr)
			shape, seqLen := ropeInputParams(gridPoint[0])
			input := randomTensor(ctx, dt, shape...)
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
			return nil // unsupported backend
		},
	},
```

- [ ] **Step 4: Run all tests**

Run: `go test ./perf/ -count=1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add perf/registry.go perf/registry_test.go
git commit -m "perf: add ROPE op with custom CreateInputs and type assertion"
```

---

### Task 7: Add GET_ROWS (Custom CreateInputs with i32 Indices)

GET_ROWS does embedding lookup: reads rows from a table using i32 indices.

**Files:**
- Modify: `perf/registry.go` — add GET_ROWS entry
- Modify: `perf/registry_test.go` — add tests

- [ ] **Step 1: Write tests for GET_ROWS shape helper**

Add to `perf/registry_test.go`:

```go
func TestGetRowsInputParams(t *testing.T) {
	tests := []struct {
		name       string
		numRows    int64
		wantTable  []int // [hiddenDim, vocabSize]
		wantNumIdx int
	}{
		{"single_row", 1, []int{4096, 32000}, 1},
		{"batch_32", 32, []int{4096, 32000}, 32},
		{"large_batch", 4096, []int{4096, 32000}, 4096},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tableShape, numIdx := getRowsInputParams(tt.numRows)
			assert.Equal(t, tt.wantTable, tableShape, "table shape")
			assert.Equal(t, tt.wantNumIdx, numIdx, "num indices")
		})
	}
}

func TestGetRowsIndices_InRange(t *testing.T) {
	indices := getRowsIndices(100)
	require.Len(t, indices, 100)
	for i, v := range indices {
		assert.GreaterOrEqual(t, v, int32(0), "index %d", i)
		assert.Less(t, v, int32(32000), "index %d should be < vocabSize", i)
	}
}

func TestGetRowsIndices_NotAllSame(t *testing.T) {
	indices := getRowsIndices(100)
	allSame := true
	for _, v := range indices {
		if v != indices[0] {
			allSame = false
			break
		}
	}
	assert.False(t, allSame, "random indices should not all be the same value")
}

func TestRegistryGET_ROWS(t *testing.T) {
	runner, ok := opRegistry["GET_ROWS"]
	assert.True(t, ok, "GET_ROWS must be in registry")
	assert.Equal(t, []string{"N"}, runner.Dimensions)
	assert.NotNil(t, runner.CreateInputs, "GET_ROWS needs custom tensor creation")
	assert.NotNil(t, runner.Run)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./perf/ -run "TestGetRows|TestRegistryGET_ROWS" -count=1 -v`
Expected: FAIL — `getRowsInputParams` and `getRowsIndices` undefined

- [ ] **Step 3: Add GET_ROWS helpers and registry entry**

Add helper functions to `perf/registry.go`:

```go
const (
	getRowsHiddenDim = 4096
	getRowsVocabSize = 32000
)

// getRowsInputParams returns the embedding table shape and number of indices for GET_ROWS.
func getRowsInputParams(numRows int64) (tableShape []int, numIdx int) {
	return []int{getRowsHiddenDim, getRowsVocabSize}, int(numRows)
}

// getRowsIndices returns random int32 indices in [0, vocabSize).
func getRowsIndices(n int) []int32 {
	indices := make([]int32, n)
	for i := range indices {
		indices[i] = int32(rand.IntN(getRowsVocabSize))
	}
	return indices
}
```

Add to `opRegistry`:

```go
	"GET_ROWS": {
		Dimensions: []string{"N"},
		CreateInputs: func(ctx ml.Context, dtypeStr string, gridPoint []int64) []ml.Tensor {
			dt := parseDType(dtypeStr)
			tableShape, numIdx := getRowsInputParams(gridPoint[0])
			table := randomTensor(ctx, dt, tableShape...)
			indices := getRowsIndices(numIdx)
			idxTensor := ctx.Input().FromInts(indices, numIdx)
			return []ml.Tensor{table, idxTensor}
		},
		Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
			return in[0].Rows(ctx, in[1])
		},
	},
```

- [ ] **Step 4: Run all tests**

Run: `go test ./perf/ -count=1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add perf/registry.go perf/registry_test.go
git commit -m "perf: add GET_ROWS op with i32 index tensor creation"
```

---

### Task 8: Update IsZeroCostOp, Default Ops List, and countGrids

Add TRANSPOSE to zero-cost ops. Update the default ops list in `cmd.go` to include all registered ops. Update `countGrids` to handle 1D ops correctly.

**Files:**
- Modify: `perf/ops.go:50-59` — add TRANSPOSE
- Modify: `perf/ops_test.go` — add TRANSPOSE test case
- Modify: `perf/cmd.go:19` — expand default ops list
- Modify: `perf/bench.go` — update `countGrids` for new 1D ops

- [ ] **Step 1: Update IsZeroCostOp**

In `perf/ops.go`:

```go
func IsZeroCostOp(op string) bool {
	switch op {
	case "VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE":
		return true
	default:
		return false
	}
}
```

- [ ] **Step 2: Add TRANSPOSE test case**

In `perf/ops_test.go`, add to the `TestIsZeroCostOp` test table:

```go
		{"TRANSPOSE", true},
```

- [ ] **Step 3: Update default ops list**

In `perf/cmd.go`, replace line 19:

Old:
```go
	ops := []string{"SILU", "MUL_MAT", "FLASH_ATTN_EXT"}
```

New:
```go
	ops := DefaultBenchmarkOps()
```

Add to `perf/registry.go`:

```go
// DefaultBenchmarkOps returns the list of ops to benchmark by default.
// This includes all registered ops.
func DefaultBenchmarkOps() []string {
	ops := make([]string, 0, len(opRegistry))
	for name := range opRegistry {
		ops = append(ops, name)
	}
	// Sort for deterministic order
	sort.Strings(ops)
	return ops
}
```

Add `"sort"` to the import block in `perf/registry.go`.

- [ ] **Step 4: Update countGrids for new 1D ops**

The current `countGrids` in `perf/bench.go` already handles 1D ops correctly via the registry lookup. Verify that `RunBenchmark`'s loop handles new ops — the `default` case in the `switch op` block at line 246 routes to `benchmarkElementwise`, which is correct for all new 1D ops (ADD, MUL, GELU, etc.).

Check that the dtype selection logic is correct:

In `perf/bench.go`, the existing code at lines 220-222:
```go
if len(runner.Dimensions) == 1 && runner.Dimensions[0] == "N" {
    opDtypes = []string{"f32"}
}
```

This already forces 1D ops to f32 only. New ops (ADD, MUL, GELU, etc.) all have `Dimensions: ["N"]`, so they'll correctly use f32 only. ROPE and GET_ROWS also have `Dimensions: ["N"]` so they'll use f32. No changes needed here.

- [ ] **Step 5: Add tests for DefaultBenchmarkOps and update countGrids test**

In `perf/registry_test.go`, add:

```go
func TestDefaultBenchmarkOps_ContainsAllRegistered(t *testing.T) {
	ops := DefaultBenchmarkOps()
	assert.Len(t, ops, len(opRegistry), "should contain all registered ops")
	// Verify sorted
	for i := 1; i < len(ops); i++ {
		assert.Less(t, ops[i-1], ops[i], "ops should be sorted alphabetically")
	}
}

func TestDefaultBenchmarkOps_ContainsExpectedOps(t *testing.T) {
	ops := DefaultBenchmarkOps()
	expected := []string{"SILU", "MUL_MAT", "FLASH_ATTN_EXT", "ADD", "MUL", "GELU", "ROPE", "GET_ROWS", "RMS_NORM", "SOFT_MAX", "CONT", "RELU"}
	for _, e := range expected {
		assert.Contains(t, ops, e, "should contain %s", e)
	}
}
```

In `perf/bench_test.go`, update `TestCountGrids_MulMatIsFour`:

```go
func TestCountGrids_AllOps(t *testing.T) {
	ops := DefaultBenchmarkOps()
	dtypes := []string{"f32", "f16", "q4_0", "q8_0"}
	count := countGrids(ops, dtypes)
	// MUL_MAT: 4 (one per weight dtype)
	// FLASH_ATTN_EXT: 1 (f16 only)
	// All other ops: 1 each (f32 only, 1D)
	numOtherOps := len(ops) - 2 // minus MUL_MAT and FLASH_ATTN_EXT
	expected := numOtherOps + 4 + 1
	assert.Equal(t, expected, count)
}

func TestCountGrids_SubsetOps(t *testing.T) {
	// Verify countGrids with a known small subset
	dtypes := []string{"f32", "f16", "q4_0", "q8_0"}
	count := countGrids([]string{"SILU"}, dtypes)
	assert.Equal(t, 1, count, "SILU is 1D, f32 only → 1 grid")

	count = countGrids([]string{"MUL_MAT"}, dtypes)
	assert.Equal(t, 4, count, "MUL_MAT → 4 grids (one per weight dtype)")

	count = countGrids([]string{"SILU", "ADD", "MUL_MAT"}, dtypes)
	assert.Equal(t, 6, count, "2 × 1D + 4 MUL_MAT = 6")
}
```

- [ ] **Step 6: Run all tests**

Run: `go test ./perf/ -count=1`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add perf/ops.go perf/ops_test.go perf/cmd.go perf/registry.go perf/bench.go perf/bench_test.go
git commit -m "perf: add TRANSPOSE to zero-cost ops, expand default benchmark ops list"
```

---

### Task 9: Update Estimation Tests with New Op Curves

Add curves for newly-supported ops to the test profile so estimation tests can verify no "uncalibrated" warnings for common ops.

**Files:**
- Modify: `perf/estimate_test.go` — add new op curves to `makeTestProfileForEstimation`, add test for mixed-op graph

- [ ] **Step 1: Update makeTestProfileForEstimation**

In `perf/estimate_test.go`, add curves for the new ops to `makeTestProfileForEstimation()`:

```go
func makeTestProfileForEstimation() *Profile {
	return &Profile{
		Version: 2,
		Hardware: HardwareProfile{
			Backends:                 []BackendInfo{{Name: "cuda", Device: "RTX 4090"}},
			PeakTOPS:                 map[string]float64{"f16": 330e12, "f32": 82.6e12},
			PeakBandwidthBytesPerSec: 1008e9,
			BalancePoints:            map[string]float64{"f16": 327.38},
			EfficiencyConstants: map[string]OpEfficiency{
				"MUL_MAT_f16":  {ComputeEff: 0.95, BWEff: 0.80, OverheadUs: 5},
				"MUL_MAT_q4_0": {ComputeEff: 0.90, BWEff: 0.70, OverheadUs: 10},
			},
		},
		Operators: []OperatorCurve{
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
			{
				Op: "ADD", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 2.0},
					{Shape: []int64{65536}, LatencyUs: 12.0},
					{Shape: []int64{1048576}, LatencyUs: 180.0},
					{Shape: []int64{16777216}, LatencyUs: 2800.0},
				},
			},
			{
				Op: "MUL", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 2.0},
					{Shape: []int64{65536}, LatencyUs: 12.0},
					{Shape: []int64{1048576}, LatencyUs: 180.0},
					{Shape: []int64{16777216}, LatencyUs: 2800.0},
				},
			},
			{
				Op: "GELU", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 2.5},
					{Shape: []int64{65536}, LatencyUs: 15.0},
					{Shape: []int64{1048576}, LatencyUs: 200.0},
					{Shape: []int64{16777216}, LatencyUs: 3000.0},
				},
			},
			{
				Op: "SOFT_MAX", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 3.0},
					{Shape: []int64{65536}, LatencyUs: 16.0},
					{Shape: []int64{1048576}, LatencyUs: 210.0},
					{Shape: []int64{16777216}, LatencyUs: 3100.0},
				},
			},
			{
				Op: "CONT", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 1.5},
					{Shape: []int64{65536}, LatencyUs: 8.0},
					{Shape: []int64{1048576}, LatencyUs: 120.0},
					{Shape: []int64{16777216}, LatencyUs: 1800.0},
				},
			},
			{
				Op: "RMS_NORM", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 3.0},
					{Shape: []int64{65536}, LatencyUs: 18.0},
					{Shape: []int64{1048576}, LatencyUs: 220.0},
					{Shape: []int64{16777216}, LatencyUs: 3200.0},
				},
			},
			{
				Op: "GET_ROWS", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 5.0},
					{Shape: []int64{32}, LatencyUs: 8.0},
					{Shape: []int64{512}, LatencyUs: 30.0},
					{Shape: []int64{4096}, LatencyUs: 200.0},
				},
			},
			{
				Op: "ROPE", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 3.5},
					{Shape: []int64{65536}, LatencyUs: 20.0},
					{Shape: []int64{1048576}, LatencyUs: 250.0},
					{Shape: []int64{16777216}, LatencyUs: 3500.0},
				},
			},
			{
				Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
				Dimensions: []string{"seq_q", "seq_kv"},
				FixedDims:  map[string]int64{"num_heads": 32, "head_dim": 128},
				Points: []LatencyPoint{
					{Shape: []int64{1, 128}, LatencyUs: 5.0},
					{Shape: []int64{1, 512}, LatencyUs: 15.0},
					{Shape: []int64{1, 2048}, LatencyUs: 55.0},
					{Shape: []int64{128, 128}, LatencyUs: 20.0},
					{Shape: []int64{512, 512}, LatencyUs: 100.0},
					{Shape: []int64{2048, 2048}, LatencyUs: 500.0},
				},
			},
		},
	}
}
```

- [ ] **Step 2: Add test for Llama-like layer with new ops**

Add to `perf/estimate_test.go`:

```go
func TestEstimatePhase_LlamaDecodeLayerNoWarnings(t *testing.T) {
	p := makeTestProfileForEstimation()
	// A realistic Llama decode layer including ops from Phase 1B
	nodes := []ml.GraphNode{
		// RMS norm (pre-attention)
		{Op: "RMS_NORM", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "MUL", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		// Q/K/V projections
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		// RoPE
		{Op: "ROPE", Backend: "cuda", Shape: [4]int64{128, 32, 1, 1}, ComputeDtype: "f32"},
		// Flash attention
		{Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
			InputShapes: [][]int64{{128, 32, 1, 1}, {128, 32, 2048, 1}}},
		// Output projection
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		// Residual add
		{Op: "ADD", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		// RMS norm (pre-FFN)
		{Op: "RMS_NORM", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "MUL", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		// FFN
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{14336, 4096}, {4096, 1}}},
		{Op: "SILU", Backend: "cuda", Shape: [4]int64{14336, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 14336}, {14336, 1}}},
		// Residual add
		{Op: "ADD", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		// Zero-cost ops (should be skipped)
		{Op: "VIEW", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "TRANSPOSE", Backend: "cuda", Shape: [4]int64{4096, 32, 1, 1}, ComputeDtype: "f32"},
	}

	var warnings []string
	result := estimatePhase(p, nodes, &warnings)

	assert.Empty(t, warnings, "should have no uncalibrated warnings for standard Llama ops")
	assert.Greater(t, result.TotalLatencyMs, 0.0)
	assert.Equal(t, "MUL_MAT", result.TopOps[0].Op, "MUL_MAT should dominate")
}

func TestEstimatePhase_GemmaDecodeLayerNoWarnings(t *testing.T) {
	p := makeTestProfileForEstimation()
	// Gemma-like layer: uses GELU instead of SILU, SOFT_MAX for attention routing
	nodes := []ml.GraphNode{
		{Op: "RMS_NORM", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
		{Op: "ROPE", Backend: "cuda", Shape: [4]int64{128, 32, 1, 1}, ComputeDtype: "f32"},
		{Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
			InputShapes: [][]int64{{128, 32, 1, 1}, {128, 32, 2048, 1}}},
		{Op: "SOFT_MAX", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "ADD", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{14336, 4096}, {4096, 1}}},
		{Op: "GELU", Backend: "cuda", Shape: [4]int64{14336, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 14336}, {14336, 1}}},
		{Op: "ADD", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
		{Op: "CONT", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}, ComputeDtype: "f32"},
	}

	var warnings []string
	result := estimatePhase(p, nodes, &warnings)

	assert.Empty(t, warnings, "should have no uncalibrated warnings for Gemma-like ops")
	assert.Greater(t, result.TotalLatencyMs, 0.0)
}
```

- [ ] **Step 3: Add lookupLatency tests for individual new ops**

Add to `perf/estimate_test.go`:

```go
func TestLookupLatency_NewOps(t *testing.T) {
	p := makeTestProfileForEstimation()
	tests := []struct {
		op    string
		shape []int64
	}{
		{"ADD", []int64{65536}},
		{"MUL", []int64{65536}},
		{"GELU", []int64{65536}},
		{"SOFT_MAX", []int64{65536}},
		{"CONT", []int64{65536}},
		{"RMS_NORM", []int64{65536}},
		{"ROPE", []int64{4096}},
		{"GET_ROWS", []int64{32}},
	}
	for _, tt := range tests {
		t.Run(tt.op, func(t *testing.T) {
			lat, err := lookupLatency(p, tt.op, tt.shape, "f32", "", "cuda")
			require.NoError(t, err, "op %s should not be uncalibrated", tt.op)
			assert.Greater(t, lat, 0.0, "op %s should return positive latency", tt.op)
		})
	}
}

func TestLookupLatency_NewOps_Interpolated(t *testing.T) {
	p := makeTestProfileForEstimation()
	// Verify interpolation works — mid-range query should return mid-range latency
	lat, err := lookupLatency(p, "ADD", []int64{500000}, "f32", "", "cuda")
	require.NoError(t, err)
	assert.Greater(t, lat, 12.0, "should be > latency at 65536")
	assert.Less(t, lat, 180.0, "should be < latency at 1048576")
}
```

- [ ] **Step 4: Run tests**

Run: `go test ./perf/ -run "TestEstimatePhase_LlamaDecodeLayer|TestEstimatePhase_Gemma|TestLookupLatency_NewOps" -count=1 -v`
Expected: PASS

Run: `go test ./perf/ -count=1`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add perf/estimate_test.go
git commit -m "perf: add estimation tests for Phase 1B ops, verify no uncalibrated warnings"
```

---

## Verification

After all tasks are complete:

1. **Run full test suite**: `go test ./perf/ -count=1 -v` — all tests pass
2. **Run benchmark** (if GPU available): `go run . daop-bench` — should show ~14 op grids instead of 6, total time ~8 min
3. **Check profile.json**: Should contain curves for all new ops (ADD, MUL, GELU, etc.)
4. **Check no regression**: MUL_MAT efficiency constants should be similar to Phase 1A results (random init may cause minor differences)
