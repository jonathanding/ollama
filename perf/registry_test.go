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
	assert.Equal(t, []int64{4096, 4096}, shapes[0]) // weight [K, M] (ne[0]=K must match activation)
	assert.Equal(t, []int64{4096, 32}, shapes[1])    // activation [K, N]
}

func TestExpandShapes_MulMat_Rectangular(t *testing.T) {
	// M != K: weight must be {K, M} not {M, K} for GGML mul_mat (requires ne[0] match)
	shapes := expandShapes("MUL_MAT", []int64{14336, 4096, 1})
	require.Len(t, shapes, 2)
	assert.Equal(t, []int64{4096, 14336}, shapes[0]) // weight [K=4096, M=14336]
	assert.Equal(t, []int64{4096, 1}, shapes[1])      // activation [K=4096, N=1]
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
