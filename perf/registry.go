package perf

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn/rope"
)

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
	CreateInputs func(ctx ml.Context, backend ml.Backend, dtypeStr string, gridPoint []int64) []ml.Tensor

	// Run invokes the operator and returns the output tensor.
	Run func(ctx ml.Context, inputs []ml.Tensor) ml.Tensor
}

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

// materializeTensor creates quantized tensor data by executing a Cast op in a
// separate prep context, then reading back the bytes to CPU. The returned bytes
// can be passed to ctx.Input().FromBytes() to create a clean leaf tensor that
// does not inject any Cast/CPY ops into the benchmark graph.
//
// This implements the "two-phase data preparation" pattern:
//
//	Phase 1 (here): f32 random data → Cast to target dtype → Compute → Bytes()
//	Phase 2 (caller): FromBytes() creates a leaf tensor in the benchmark context
//
// Uses Compute (scheduler) instead of ComputeOnBackend so the scheduler can
// route Cast to CPU when the GPU backend lacks a quantization kernel (e.g.
// Vulkan does not support f32→q4_K Cast).
func materializeTensor(backend ml.Backend, dt ml.DType, shape ...int) []byte {
	prepCtx := backend.NewContext()
	defer prepCtx.Close()

	f32Tensor := randomTensor(prepCtx, ml.DTypeF32, shape...)
	castTensor := f32Tensor.Cast(prepCtx, dt)
	prepCtx.Forward(castTensor)
	prepCtx.Compute(castTensor)
	return castTensor.Bytes()
}

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

// opRegistry maps GGML op names to their benchmark definitions.
// To add a new operator:
//  1. Add an entry with Dimensions and Run (and optionally CreateInputs)
//  2. Add shape expansion in expandShapes() if using default path
//  3. Add tests in registry_test.go
var opRegistry = map[string]OpRunnerML{
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
	"SILU": {
		Dimensions: []string{"N"},
		Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
			return in[0].SILU(ctx)
		},
	},
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
	"FLASH_ATTN_EXT": {
		Dimensions: []string{"seq_q", "seq_kv"},
		CreateInputs: func(ctx ml.Context, backend ml.Backend, dtypeStr string, gridPoint []int64) []ml.Tensor {
			// Q is f32 (matmul output in real inference), K/V are f16 (KV cache)
			seqQ, seqKV := gridPoint[0], gridPoint[1]
			numHeads := 32
			if len(gridPoint) >= 3 {
				numHeads = int(gridPoint[2])
			}
			q := randomTensor(ctx, ml.DTypeF32, 128, numHeads, int(seqQ), 1)
			// K/V are f16 — use materializeTensor to avoid Cast/CPY in graph
			kBytes := materializeTensor(backend, ml.DTypeF16, 128, numHeads, int(seqKV), 1)
			vBytes := materializeTensor(backend, ml.DTypeF16, 128, numHeads, int(seqKV), 1)
			k := ctx.Input().FromBytes(ml.DTypeF16, kBytes, 128, numHeads, int(seqKV), 1)
			v := ctx.Input().FromBytes(ml.DTypeF16, vBytes, 128, numHeads, int(seqKV), 1)
			return []ml.Tensor{q, k, v}
		},
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
			return in[0].RMSNorm(ctx, nil, 1e-5)
		},
	},
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

	// Fused ops: these construct small graphs containing fusable patterns.
	// When Vulkan processes them with GPU timestamps, the backend fuses them
	// and returns fused kernel timing.

	"RMS_NORM_MUL": {
		Dimensions: []string{"N"},
		CreateInputs: func(ctx ml.Context, backend ml.Backend, dtypeStr string, gridPoint []int64) []ml.Tensor {
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
		CreateInputs: func(ctx ml.Context, backend ml.Backend, dtypeStr string, gridPoint []int64) []ml.Tensor {
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
}

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

// mulMatInputShapes returns (weightShape, activationShape) for MUL_MAT benchmarking.
// gridPoint = [M, K, N]. Weight is [K, M], activation is [K, N].
func mulMatInputShapes(gridPoint []int64) (weightShape, activationShape []int) {
	M, K, N := gridPoint[0], gridPoint[1], gridPoint[2]
	return []int{int(K), int(M)}, []int{int(K), int(N)}
}

// expandShapes converts grid dimensions to full tensor shapes per op.
// gridPoint contains values for OpRunner.Dimensions in order.
func expandShapes(op string, gridPoint []int64) [][]int64 {
	switch op {
	case "FLASH_ATTN_EXT":
		// gridPoint = [seq_q, seq_kv] or [seq_q, seq_kv, num_heads]
		seqQ, seqKV := gridPoint[0], gridPoint[1]
		numHeads := int64(32)
		if len(gridPoint) >= 3 {
			numHeads = gridPoint[2]
		}
		return [][]int64{
			{128, numHeads, seqQ, 1},  // Q
			{128, numHeads, seqKV, 1}, // K
			{128, numHeads, seqKV, 1}, // V
		}
	case "ADD", "MUL":
		return [][]int64{gridPoint, gridPoint}
	case "MUL_MAT", "MUL_MAT_ADD":
		// gridPoint = [M, K, N]
		// GGML mul_mat computes weight^T * activation, so weight ne[0] must equal activation ne[0].
		// weight: {K, M} (ne[0]=K, ne[1]=M), activation: {K, N} (ne[0]=K, ne[1]=N)
		// Result: {M, N}
		_, K, N := gridPoint[0], gridPoint[1], gridPoint[2]
		M := gridPoint[0]
		return [][]int64{
			{K, M}, // weight (ne[0]=K must match activation ne[0])
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
	case "q4_K":
		return ml.DTypeQ4K, true
	case "q6_K":
		return ml.DTypeQ6K, true
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
	case ml.DTypeQ4K:
		return "q4_K"
	case ml.DTypeQ6K:
		return "q6_K"
	default:
		return "unknown"
	}
}

// Phase1Dtypes returns the dtypes supported in Phase 1 benchmarks.
func Phase1Dtypes() []string {
	return []string{"f32", "f16", "q4_0", "q8_0", "q4_K", "q6_K"}
}

// Phase1MulMatFixedDims returns the 3×3 log-spaced (M, K) grid for MUL_MAT benchmarks.
// Grid values: {512, 2048, 8192} — factor of 4 between steps.
// Covers any transformer architecture with dimensions in [512, 8192].
// Total: 9 (M,K) pairs × 6 dtypes × 3 N values = 162 measurements.
func Phase1MulMatFixedDims() [][2]int64 {
	gridValues := []int64{512, 2048, 8192}
	var pairs [][2]int64
	for _, m := range gridValues {
		for _, k := range gridValues {
			pairs = append(pairs, [2]int64{m, k})
		}
	}
	return pairs
}

// Phase1FlashAttnHeads returns the num_heads values for FLASH_ATTN_EXT benchmarks.
// Covers common transformer configurations from small (4) to large (32) models.
func Phase1FlashAttnHeads() []int64 {
	return []int64{4, 8, 16, 32}
}

// DefaultBenchmarkOps returns the list of ops to benchmark by default.
// This includes all registered ops plus ORCHESTRATION_OVERHEAD (a pseudo-op
// that triggers orchestration overhead measurement in the benchmark plan).
func DefaultBenchmarkOps() []string {
	ops := make([]string, 0, len(opRegistry)+1)
	for name := range opRegistry {
		ops = append(ops, name)
	}
	ops = append(ops, "ORCHESTRATION_OVERHEAD")
	sort.Strings(ops)
	return ops
}

// LookupRegistry returns the OpRunnerML for a given op name.
// Returns (runner, true) if found, (zero, false) otherwise.
func LookupRegistry(op string) (OpRunnerML, bool) {
	r, ok := opRegistry[op]
	return r, ok
}
