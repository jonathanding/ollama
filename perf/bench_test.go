package perf

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSelectBenchmarkShapes_MulMat(t *testing.T) {
	shapes := SelectBenchmarkShapes("MUL_MAT", 82.0, "f16", "q4_0")
	assert.Equal(t, 5, len(shapes))
	assert.Equal(t, int64(1), shapes[0][1][1])
	assert.Equal(t, int64(4096), shapes[4][1][1])
}

func TestSelectBenchmarkShapes_MemoryBound(t *testing.T) {
	shapes := SelectBenchmarkShapes("ADD", 82.0, "f32", "")
	assert.Equal(t, 5, len(shapes))
	size0 := shapes[0][0][0]
	size4 := shapes[4][0][0]
	assert.Greater(t, size4, size0)
}

func TestShouldAdaptiveExtend(t *testing.T) {
	assert.False(t, ShouldAdaptiveExtend([]float64{0.61, 0.62, 0.63, 0.62, 0.61}))
	assert.True(t, ShouldAdaptiveExtend([]float64{0.4, 0.6, 0.8, 0.5, 0.9}))
}

func TestPredefinedOps(t *testing.T) {
	ops := PredefinedOps()
	found := false
	for _, op := range ops {
		if op.Op == "MUL_MAT" {
			found = true
			assert.Contains(t, op.Dtypes, "q4_0")
		}
	}
	assert.True(t, found, "MUL_MAT should be in predefined ops")
}
