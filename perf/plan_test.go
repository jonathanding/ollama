package perf

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildBenchmarkPlan_FullPipeline(t *testing.T) {
	caps := GetBackendCapabilities("Vulkan")
	ops := DefaultBenchmarkOps()
	dtypes := Phase1Dtypes()
	plan := buildBenchmarkPlan(ops, dtypes, caps, DefaultBenchmarkConfig())
	require.NotEmpty(t, plan)
	assert.Equal(t, StepHWChar, plan[0].Type)
	hwCharCount, opCount, fusedCount, overheadCount, mulMatRefCount := 0, 0, 0, 0, 0
	for _, s := range plan {
		switch s.Type {
		case StepHWChar:
			hwCharCount++
		case StepOperator:
			opCount++
		case StepMulMatRef:
			mulMatRefCount++
		case StepFusedOp:
			fusedCount++
		case StepOverhead:
			overheadCount++
		}
	}
	assert.Equal(t, 1, hwCharCount)
	assert.Greater(t, opCount, 0)
	assert.Greater(t, mulMatRefCount, 0)
	assert.Greater(t, fusedCount, 0)
	assert.Equal(t, 1, overheadCount)
}

func TestBuildBenchmarkPlan_SpecificOps(t *testing.T) {
	caps := GetBackendCapabilities("Vulkan")
	plan := buildBenchmarkPlan([]string{"ADD", "SILU"}, []string{"f32"}, caps, DefaultBenchmarkConfig())
	require.NotEmpty(t, plan)
	stepOps := make(map[string]int)
	for _, s := range plan {
		if s.Type == StepOperator {
			stepOps[s.Op]++
		}
	}
	assert.Equal(t, 1, stepOps["ADD"])
	assert.Equal(t, 1, stepOps["SILU"])
	assert.Zero(t, stepOps["MUL_MAT"])
}

func TestBuildBenchmarkPlan_NoFusionOnCPU(t *testing.T) {
	caps := GetBackendCapabilities("CPU")
	plan := buildBenchmarkPlan(DefaultBenchmarkOps(), Phase1Dtypes(), caps, DefaultBenchmarkConfig())
	fusedCount, overheadCount := 0, 0
	for _, s := range plan {
		if s.Type == StepFusedOp {
			fusedCount++
		}
		if s.Type == StepOverhead {
			overheadCount++
		}
	}
	assert.Zero(t, fusedCount)
	assert.Zero(t, overheadCount)
}

func TestBuildBenchmarkPlan_MulMatGeneratesRefCurves(t *testing.T) {
	caps := GetBackendCapabilities("Vulkan")
	plan := buildBenchmarkPlan([]string{"MUL_MAT"}, Phase1Dtypes(), caps, DefaultBenchmarkConfig())
	refCount := 0
	for _, s := range plan {
		if s.Type == StepMulMatRef {
			refCount++
		}
	}
	assert.Equal(t, len(Phase1Dtypes()), refCount)
}

func TestBuildBenchmarkPlan_1DOpsUseF32Only(t *testing.T) {
	caps := GetBackendCapabilities("Vulkan")
	plan := buildBenchmarkPlan([]string{"ADD", "SILU", "RMS_NORM"}, []string{"f32", "f16", "q4_0"}, caps, DefaultBenchmarkConfig())
	for _, s := range plan {
		if s.Type == StepOperator {
			assert.Equal(t, "f32", s.Dtype, "1D op %s should only use f32", s.Op)
		}
	}
}

func TestBuildBenchmarkPlan_FusedOpsExcludedFromMainLoop(t *testing.T) {
	caps := GetBackendCapabilities("Vulkan")
	plan := buildBenchmarkPlan(DefaultBenchmarkOps(), Phase1Dtypes(), caps, DefaultBenchmarkConfig())
	for _, s := range plan {
		if s.Type == StepOperator {
			assert.NotContains(t, []string{"RMS_NORM_MUL", "RMS_NORM_MUL_ROPE", "MUL_MAT_ADD"}, s.Op)
		}
	}
}

func TestBuildBenchmarkPlan_StepCount(t *testing.T) {
	caps := GetBackendCapabilities("Vulkan")
	plan := buildBenchmarkPlan([]string{"ADD"}, []string{"f32"}, caps, DefaultBenchmarkConfig())
	assert.Equal(t, 2, len(plan)) // HWChar + ADD (no overhead unless explicitly requested)
}

func TestBuildBenchmarkPlan_EmptyOps(t *testing.T) {
	caps := GetBackendCapabilities("Vulkan")
	plan := buildBenchmarkPlan(nil, nil, caps, DefaultBenchmarkConfig())
	hwCount := 0
	for _, s := range plan {
		if s.Type == StepHWChar {
			hwCount++
		}
	}
	assert.Equal(t, 1, hwCount)
}

func TestBuildBenchmarkPlan_SkipHWChar(t *testing.T) {
	caps := GetBackendCapabilities("Vulkan")
	cfg := DefaultBenchmarkConfig()
	cfg.SkipHWChar = true
	plan := buildBenchmarkPlan(DefaultBenchmarkOps(), Phase1Dtypes(), caps, cfg)
	for _, s := range plan {
		assert.NotEqual(t, StepHWChar, s.Type, "plan should not contain StepHWChar when SkipHWChar=true")
	}
	assert.Greater(t, len(plan), 0, "plan should still have operator steps")
}
