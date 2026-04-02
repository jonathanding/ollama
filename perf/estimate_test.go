package perf

import (
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEstimateGraph(t *testing.T) {
	p := newTestProfile()

	nodes := []ml.GraphNode{
		{
			Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			Shape:       [4]int64{4096, 1, 1, 1},
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}},
		},
		{
			Op: "RMS_NORM", Backend: "cuda", ComputeDtype: "f32",
			Shape:       [4]int64{4096, 1, 1, 1},
			InputShapes: [][]int64{{4096, 1}},
		},
		{
			Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			Shape:       [4]int64{4096, 1, 1, 1},
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}},
		},
	}

	latency, stats, warnings := EstimateGraphLatency(p, nodes)

	assert.Greater(t, latency, 0.0)
	assert.Greater(t, len(stats), 0)
	_ = warnings
}

func TestEstimateGraph_SkipsViews(t *testing.T) {
	p := newTestProfile()

	nodes := []ml.GraphNode{
		{Op: "VIEW", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}},
		{Op: "RESHAPE", Backend: "cuda", Shape: [4]int64{4096, 1, 1, 1}},
		{
			Op: "ADD", Backend: "cuda", ComputeDtype: "f32",
			Shape:       [4]int64{4096, 1, 1, 1},
			InputShapes: [][]int64{{4096}},
		},
	}

	latency, stats, _ := EstimateGraphLatency(p, nodes)
	assert.Greater(t, latency, 0.0)
	assert.Equal(t, 1, len(stats))
}

func TestBuildEstimateResult(t *testing.T) {
	result := &EstimateResult{
		Model:        "qwen3:8b-q4_0",
		InputLength:  1024,
		OutputLength: 256,
		MaxBatchSize: 512,
	}

	BuildSummary(result)
	assert.NotEmpty(t, result.Summary)
}

func TestComputePhaseEstimation(t *testing.T) {
	p := newTestProfile()
	nodes := []ml.GraphNode{
		{
			Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			Shape:       [4]int64{4096, 512, 1, 1},
			InputShapes: [][]int64{{4096, 4096}, {4096, 512}},
		},
	}

	phase := ComputePhaseEstimation(p, nodes, 1024, 512)
	require.NotNil(t, phase)
	assert.Greater(t, phase.TokensPerSec, 0.0)
	assert.Greater(t, phase.TotalLatencyMs, 0.0)
}
