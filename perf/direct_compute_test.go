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
