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
