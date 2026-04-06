package perf

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/ml"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestProfileRoundTripIntegration tests the full profile write/load cycle.
func TestProfileRoundTripIntegration(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "profile.json")

	original := &Profile{
		Version:   2,
		Timestamp: time.Now(),
		Hardware: HardwareProfile{
			Backends:                 []BackendInfo{{Name: "cuda", Device: "RTX 4090", VRAMBytes: 24_000_000_000}},
			PeakTOPS:                 map[string]float64{"f16": 330e12, "f32": 82.6e12},
			PeakBandwidthBytesPerSec: 1008e9,
			BalancePoints:            map[string]float64{"f16": 327.38},
		},
		Operators: []OperatorCurve{
			{
				Op: "SILU", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 2.5, StddevUs: 0.1, Reps: 100},
					{Shape: []int64{1048576}, LatencyUs: 200.0, StddevUs: 5.0, Reps: 100},
				},
			},
			{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Dimensions: []string{"N"},
				FixedDims:  map[string]int64{"M": 4096, "K": 4096},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 10.0, StddevUs: 0.5, Reps: 100},
					{Shape: []int64{4096}, LatencyUs: 3000.0, StddevUs: 50.0, Reps: 100},
				},
			},
			{
				Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
				Dimensions: []string{"seq_q", "seq_kv"},
				FixedDims:  map[string]int64{"num_heads": 32, "head_dim": 128},
				Points: []LatencyPoint{
					{Shape: []int64{1, 128}, LatencyUs: 5.0, Reps: 100},
					{Shape: []int64{1, 2048}, LatencyUs: 55.0, Reps: 100},
					{Shape: []int64{512, 512}, LatencyUs: 100.0, Reps: 100},
				},
			},
		},
	}

	err := WriteProfile(path, original)
	require.NoError(t, err)

	loaded, err := LoadProfile(path)
	require.NoError(t, err)

	assert.Equal(t, 2, loaded.Version)
	assert.Len(t, loaded.Operators, 3)
	assert.Equal(t, "SILU", loaded.Operators[0].Op)
	assert.Equal(t, "MUL_MAT", loaded.Operators[1].Op)
	assert.Equal(t, "FLASH_ATTN_EXT", loaded.Operators[2].Op)
	assert.Equal(t, int64(4096), loaded.Operators[1].FixedDims["M"])
}

// TestEndToEndEstimation_Synthetic tests the full estimation pipeline
// with a synthetic profile and manually constructed graph nodes.
func TestEndToEndEstimation_Synthetic(t *testing.T) {
	p := makeTestProfileForEstimation()

	// Simulate a 32-layer Llama-8B decode pass
	var nodes []ml.GraphNode
	for layer := 0; layer < 32; layer++ {
		// 4 attention MUL_MATs: [4096, 4096] × [4096, 1]
		for i := 0; i < 4; i++ {
			nodes = append(nodes, ml.GraphNode{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				InputShapes: [][]int64{{4096, 4096}, {4096, 1}},
			})
		}
		// FLASH_ATTN decode
		nodes = append(nodes, ml.GraphNode{
			Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
			InputShapes: [][]int64{{128, 1, 32, 1}, {128, 2048, 32, 1}},
		})
		// FFN: up [14336, 4096] × [4096, 1]
		nodes = append(nodes, ml.GraphNode{
			Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{14336, 4096}, {4096, 1}},
		})
		// SILU
		nodes = append(nodes, ml.GraphNode{
			Op: "SILU", Backend: "cuda", Shape: [4]int64{14336, 1, 1, 1}, ComputeDtype: "f32",
		})
		// FFN: down [4096, 14336] × [14336, 1]
		nodes = append(nodes, ml.GraphNode{
			Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 14336}, {14336, 1}},
		})
		// VIEW, RESHAPE — should be skipped
		nodes = append(nodes, ml.GraphNode{Op: "VIEW", Backend: "cuda"})
		nodes = append(nodes, ml.GraphNode{Op: "RESHAPE", Backend: "cuda"})
	}

	var warnings []string
	result := estimatePhase(p, nodes, &warnings)

	assert.Greater(t, result.TotalLatencyMs, 0.0)

	require.NotEmpty(t, result.TopOps)
	assert.Equal(t, "MUL_MAT", result.TopOps[0].Op)
	assert.Greater(t, result.TopOps[0].Percentage, 0.5)

	assert.Greater(t, result.TokensPerSec, 0.0)

	t.Logf("Synthetic Llama-8B decode: %.2f ms/tok, %.0f tok/s",
		result.TotalLatencyMs, result.TokensPerSec)
	t.Logf("Top ops:")
	for _, op := range result.TopOps {
		t.Logf("  %-16s %4dx  %8.1fus  %.1f%%",
			op.Op, op.Count, op.TotalUs, op.Percentage*100)
	}
}

// TestHTMLViewerIntegration tests generating HTML viewer from a full profile.
func TestHTMLViewerIntegration(t *testing.T) {
	p := makeTestProfileForEstimation()
	dir := t.TempDir()
	path := filepath.Join(dir, "viewer.html")

	err := GenerateHTMLViewer(p, path)
	require.NoError(t, err)

	info, err := os.Stat(path)
	require.NoError(t, err)
	assert.Greater(t, info.Size(), int64(5000), "HTML viewer should have substantial content")
}
