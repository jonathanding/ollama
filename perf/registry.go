package perf

import (
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
	CreateInputs func(ctx ml.Context, dtypeStr string, gridPoint []int64) []ml.Tensor

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
		CreateInputs: func(ctx ml.Context, dtypeStr string, gridPoint []int64) []ml.Tensor {
			dt, _ := parseDType(dtypeStr)
			wShape, aShape := mulMatInputShapes(gridPoint)
			weight := randomTensor(ctx, dt, wShape...)
			activation := randomTensor(ctx, ml.DTypeF32, aShape...)
			return []ml.Tensor{weight, activation}
		},
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
		CreateInputs: func(ctx ml.Context, dtypeStr string, gridPoint []int64) []ml.Tensor {
			dt, _ := parseDType(dtypeStr)
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
			return nil
		},
	},
	"GET_ROWS": {
		Dimensions: []string{"N"},
		CreateInputs: func(ctx ml.Context, dtypeStr string, gridPoint []int64) []ml.Tensor {
			dt, _ := parseDType(dtypeStr)
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
}

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
		// gridPoint = [seq_q, seq_kv], fixed head_dim=128, num_heads=32
		seqQ, seqKV := gridPoint[0], gridPoint[1]
		return [][]int64{
			{128, 32, seqQ, 1},  // Q
			{128, 32, seqKV, 1}, // K
			{128, 32, seqKV, 1}, // V
		}
	case "ADD", "MUL":
		return [][]int64{gridPoint, gridPoint}
	case "MUL_MAT":
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

// DefaultBenchmarkOps returns the list of ops to benchmark by default.
// This includes all registered ops.
func DefaultBenchmarkOps() []string {
	ops := make([]string, 0, len(opRegistry))
	for name := range opRegistry {
		ops = append(ops, name)
	}
	sort.Strings(ops)
	return ops
}

// LookupRegistry returns the OpRunnerML for a given op name.
// Returns (runner, true) if found, (zero, false) otherwise.
func LookupRegistry(op string) (OpRunnerML, bool) {
	r, ok := opRegistry[op]
	return r, ok
}
