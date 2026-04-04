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
			assert.NotNil(t, runner.Run)
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

func TestRegistryCreateInputsOrExpandShapes(t *testing.T) {
	for name, runner := range opRegistry {
		t.Run(name, func(t *testing.T) {
			assert.NotNil(t, runner.Run, "op %q must have a Run function", name)
			assert.NotEmpty(t, runner.Dimensions, "op %q must have Dimensions", name)
		})
	}
}

func TestRegistryCustomCreateInputs(t *testing.T) {
	// MUL_MAT needs CreateInputs for mixed-dtype tensor creation
	runner, ok := opRegistry["MUL_MAT"]
	require.True(t, ok)
	assert.NotNil(t, runner.CreateInputs, "MUL_MAT requires custom CreateInputs")
}

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

func TestRegistryDefaultCreateInputs(t *testing.T) {
	// SILU should use the default path (nil CreateInputs)
	runner, ok := opRegistry["SILU"]
	require.True(t, ok)
	assert.Nil(t, runner.CreateInputs, "SILU should use default tensor creation")
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
	data := randomFloat32Slice(10000)
	var sum float64
	for _, v := range data {
		sum += float64(v)
	}
	mean := sum / float64(len(data))
	assert.InDelta(t, 0.0, mean, 0.1, "mean of uniform[-1,1] should be near zero")
}

func TestRandomFloat32Slice_SpreadAcrossRange(t *testing.T) {
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

func TestRopeInputParams(t *testing.T) {
	tests := []struct {
		name       string
		totalN     int64
		wantShape  []int
		wantSeqLen int64
	}{
		{"single_token", 128, []int{128, 1, 1, 1}, 1},
		{"batch_8", 1024, []int{128, 1, 8, 1}, 8},
		{"large_seq", 65536, []int{128, 1, 512, 1}, 512},
		{"below_head_dim", 64, []int{128, 1, 1, 1}, 1},
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
