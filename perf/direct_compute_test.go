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

	a := randomTensor(ctx, ml.DTypeF32, 1024)
	b := randomTensor(ctx, ml.DTypeF32, 1024)
	out := a.Add(ctx, b)
	ctx.Forward(out)

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

	caps := DiscoverBackend(backend)
	if !caps.HasGPUTimestamp {
		t.Skip("no GPU timestamp support (CPU-only backend)")
	}

	backend.EnableGPUTimestamps(true)
	defer backend.EnableGPUTimestamps(false)

	ctx := backend.NewContext()
	defer ctx.Close()

	a := randomTensor(ctx, ml.DTypeF32, 65536)
	b := randomTensor(ctx, ml.DTypeF32, 65536)
	out := a.Add(ctx, b)
	ctx.Forward(out)

	ctx.ComputeOnBackend(0, out)
	ctx.ComputeOnBackend(0, out)
	timings := backend.GetOpTimings()
	require.NotEmpty(t, timings, "GPU timings should be non-empty when computing directly on GPU backend")
	assert.Greater(t, timings[0].GPUTimeUs, 0.0, "GPU time should be positive")
}

func TestComputeOnBackend_LargeMulMat(t *testing.T) {
	backend := setupBenchBackend(t)
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

	ctx := backend.NewContext()
	defer ctx.Close()

	a := randomTensor(ctx, ml.DTypeF32, 256)
	b := randomTensor(ctx, ml.DTypeF32, 256)
	out := a.Add(ctx, b)
	ctx.Forward(out)

	require.NotPanics(t, func() {
		ctx.ComputeOnBackend(0, out)
	})
}

func TestComputeOnBackend_QuantizedTensors(t *testing.T) {
	backend := setupBenchBackend(t)
	ctx := backend.NewContext()
	defer ctx.Close()

	M, K, N := 256, 256, 1
	weight := randomTensor(ctx, ml.DTypeQ40, K, M)
	activation := randomTensor(ctx, ml.DTypeF32, K, N)
	out := weight.Mulmat(ctx, activation)
	ctx.Forward(out)

	require.NotPanics(t, func() {
		ctx.ComputeOnBackend(0, out)
	})
}

// TestComputeOnBackend_NumericalCorrectness verifies that ComputeOnBackend
// transfers input data to the target backend and produces correct results.
// This catches the bug where tensor data was lost during buffer reallocation.
func TestComputeOnBackend_NumericalCorrectness(t *testing.T) {
	backend := setupBenchBackend(t)
	ctx := backend.NewContext()
	defer ctx.Close()

	// Create tensors with known values: a=[1,2,3,4], b=[10,20,30,40]
	a := ctx.Input().FromFloats([]float32{1, 2, 3, 4}, 4)
	b := ctx.Input().FromFloats([]float32{10, 20, 30, 40}, 4)
	out := a.Add(ctx, b)
	ctx.Forward(out)

	ctx.ComputeOnBackend(0, out)
	result := out.Floats()

	require.Len(t, result, 4, "output should have 4 elements")
	expected := []float32{11, 22, 33, 44}
	for i, exp := range expected {
		assert.InDelta(t, exp, result[i], 0.01,
			"element %d: expected %.1f, got %.1f", i, exp, result[i])
	}
}

// TestComputeOnBackend_NumericalMulMat verifies MUL_MAT correctness with known inputs.
// ggml Mulmat computes: out[m][n] = sum_k(weight[m][k] * activation[n][k])
// Weight shape (ne[0]=K, ne[1]=M), stored row-major: row m has K elements.
func TestComputeOnBackend_NumericalMulMat(t *testing.T) {
	backend := setupBenchBackend(t)
	ctx := backend.NewContext()
	defer ctx.Close()

	// Weight (K=2, M=2): row 0=[1,3], row 1=[2,4]
	weight := ctx.Input().FromFloats([]float32{1, 3, 2, 4}, 2, 2)
	// Activation (K=2, N=1): row 0=[5,6]
	activation := ctx.Input().FromFloats([]float32{5, 6}, 2, 1)
	out := weight.Mulmat(ctx, activation)
	ctx.Forward(out)

	ctx.ComputeOnBackend(0, out)
	result := out.Floats()

	require.Len(t, result, 2, "output should have 2 elements")
	// out[0] = 1*5 + 3*6 = 23
	// out[1] = 2*5 + 4*6 = 34
	assert.InDelta(t, 23.0, result[0], 0.1, "first element")
	assert.InDelta(t, 34.0, result[1], 0.1, "second element")
}

// TestComputeOnBackend_RepeatedCallsPreserveData verifies that data is preserved
// across multiple ComputeOnBackend calls (only first call triggers reallocation).
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

func TestComputeOnBackend_RepeatedCallsPreserveData(t *testing.T) {
	backend := setupBenchBackend(t)
	ctx := backend.NewContext()
	defer ctx.Close()

	a := ctx.Input().FromFloats([]float32{100, 200, 300}, 3)
	b := ctx.Input().FromFloats([]float32{1, 2, 3}, 3)
	out := a.Add(ctx, b)
	ctx.Forward(out)

	// Multiple calls should produce identical results
	for i := 0; i < 3; i++ {
		ctx.ComputeOnBackend(0, out)
		result := out.Floats()
		require.Len(t, result, 3)
		assert.InDelta(t, 101.0, result[0], 0.01, "call %d, elem 0", i)
		assert.InDelta(t, 202.0, result[1], 0.01, "call %d, elem 1", i)
		assert.InDelta(t, 303.0, result[2], 0.01, "call %d, elem 2", i)
	}
}

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
	ctx.ComputeOnBackend(0, out)
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
		for range 3 {
			ctx.ComputeOnBackend(0, out)
		}
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
	assert.Less(t, q40Time, f32Time*2,
		"q4_0 should not be much slower than f32 — if it is, Cast timing may still be included")
}
