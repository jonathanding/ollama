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

func TestDefaultBenchmarkOps_ContainsAllRegistered(t *testing.T) {
	ops := DefaultBenchmarkOps()
	assert.Len(t, ops, len(opRegistry)+1, "should contain all registered ops + ORCHESTRATION_OVERHEAD")
	for i := 1; i < len(ops); i++ {
		assert.Less(t, ops[i-1], ops[i], "ops should be sorted alphabetically")
	}
}

func TestDefaultBenchmarkOps_ContainsExpectedOps(t *testing.T) {
	ops := DefaultBenchmarkOps()
	expected := []string{"SILU", "MUL_MAT", "FLASH_ATTN_EXT", "ADD", "MUL", "GELU", "ROPE", "RMS_NORM", "SOFT_MAX", "CONT", "RELU", "RMS_NORM_MUL", "RMS_NORM_MUL_ROPE", "MUL_MAT_ADD"}
	for _, e := range expected {
		assert.Contains(t, ops, e, "should contain %s", e)
	}
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

// --- Fused op tests ---

func TestFusedOpRegistryEntries(t *testing.T) {
	fusedOps := []struct {
		name string
		dims []string
	}{
		{"RMS_NORM_MUL", []string{"N"}},
		{"RMS_NORM_MUL_ROPE", []string{"N"}},
		{"MUL_MAT_ADD", []string{"M", "K", "N"}},
	}
	for _, tt := range fusedOps {
		t.Run(tt.name, func(t *testing.T) {
			runner, ok := opRegistry[tt.name]
			require.True(t, ok, "op %q must be in registry", tt.name)
			assert.Equal(t, tt.dims, runner.Dimensions)
			assert.NotNil(t, runner.Run, "Run must not be nil")
			assert.NotNil(t, runner.CreateInputs, "CreateInputs must not be nil for fused ops")
		})
	}
}

func TestFusedOpRMSNormMulCreateInputs(t *testing.T) {
	runner, ok := opRegistry["RMS_NORM_MUL"]
	require.True(t, ok)
	assert.NotNil(t, runner.CreateInputs)

	// Verify the function signature expectations: gridPoint[0] = N
	// We can't call CreateInputs without a real context, but we can verify
	// it exists and that the op uses 1D dimensions
	assert.Equal(t, []string{"N"}, runner.Dimensions,
		"RMS_NORM_MUL should be a 1D op (input and scale have the same N)")
}

func TestFusedOpRMSNormMulRopeCreateInputs(t *testing.T) {
	runner, ok := opRegistry["RMS_NORM_MUL_ROPE"]
	require.True(t, ok)
	assert.NotNil(t, runner.CreateInputs)

	// This op needs 3 tensors: input, scale, positions
	// It reuses ropeInputParams for shape computation, same as ROPE
	assert.Equal(t, []string{"N"}, runner.Dimensions,
		"RMS_NORM_MUL_ROPE should be a 1D op like ROPE")
}

func TestFusedOpMulMatAddCreateInputs(t *testing.T) {
	runner, ok := opRegistry["MUL_MAT_ADD"]
	require.True(t, ok)
	assert.NotNil(t, runner.CreateInputs)

	// MUL_MAT_ADD needs 3 tensors: weight, activation, bias
	// It uses mulMatInputShapes for weight/activation, plus a bias vector of size M
	assert.Equal(t, []string{"M", "K", "N"}, runner.Dimensions,
		"MUL_MAT_ADD should have the same 3D dimensions as MUL_MAT")
}

func TestFusedOpMulMatAddDimensions(t *testing.T) {
	// MUL_MAT_ADD should share the same dimension convention as MUL_MAT
	mulMatRunner, ok := opRegistry["MUL_MAT"]
	require.True(t, ok)
	mulMatAddRunner, ok := opRegistry["MUL_MAT_ADD"]
	require.True(t, ok)

	assert.Equal(t, mulMatRunner.Dimensions, mulMatAddRunner.Dimensions,
		"MUL_MAT_ADD must use the same dimension names as MUL_MAT")
}

func TestFusedOpBuildSamplingGrids(t *testing.T) {
	t.Run("RMS_NORM_MUL_gets_single_grid", func(t *testing.T) {
		grids := buildSamplingGrids("RMS_NORM_MUL", "f32", "")
		require.Len(t, grids, 1, "1D fused ops should get a single grid")
		assert.Equal(t, "RMS_NORM_MUL", grids[0].Op)
		assert.Equal(t, "f32", grids[0].Dtype)
		assert.Nil(t, grids[0].FixedDims, "1D ops should not have fixed dims")
	})

	t.Run("RMS_NORM_MUL_ROPE_gets_single_grid", func(t *testing.T) {
		grids := buildSamplingGrids("RMS_NORM_MUL_ROPE", "f32", "")
		require.Len(t, grids, 1, "1D fused ops should get a single grid")
		assert.Equal(t, "RMS_NORM_MUL_ROPE", grids[0].Op)
		assert.Nil(t, grids[0].FixedDims)
	})

	t.Run("MUL_MAT_ADD_gets_per_MK_grids", func(t *testing.T) {
		grids := buildSamplingGrids("MUL_MAT_ADD", "f16", "q4_0")
		pairs := Phase1MulMatFixedDims()
		require.Len(t, grids, len(pairs),
			"MUL_MAT_ADD should get one grid per (M,K) pair, same as MUL_MAT")

		for i, g := range grids {
			assert.Equal(t, "MUL_MAT_ADD", g.Op)
			assert.Equal(t, "q4_0", g.WeightDtype)
			assert.NotNil(t, g.FixedDims)
			assert.Contains(t, g.FixedDims, "M")
			assert.Contains(t, g.FixedDims, "K")
			assert.Equal(t, pairs[i][0], g.FixedDims["M"])
			assert.Equal(t, pairs[i][1], g.FixedDims["K"])
		}
	})

	t.Run("MUL_MAT_ADD_grid_count_matches_MUL_MAT", func(t *testing.T) {
		mulMatGrids := buildSamplingGrids("MUL_MAT", "f16", "q4_0")
		mulMatAddGrids := buildSamplingGrids("MUL_MAT_ADD", "f16", "q4_0")
		assert.Equal(t, len(mulMatGrids), len(mulMatAddGrids),
			"MUL_MAT_ADD should have the same number of grids as MUL_MAT")
	})
}
