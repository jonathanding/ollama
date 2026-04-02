package perf

import (
	"bytes"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEndToEndEstimation(t *testing.T) {
	// 1. Create a realistic profile
	profile := &Profile{
		Version:       1,
		GeneratedFrom: []string{"test"},
		Hardware: HardwareProfile{
			Backends: []BackendProfile{
				{
					Name:          "cuda",
					Device:        "RTX 4090",
					PeakFLOPS:     map[string]float64{"f16": 82.6e12, "f32": 41.3e12},
					PeakBandwidth: 1008e9,
					BalancePoints: map[string]float64{"f16": 81.9, "f32": 41.0},
				},
			},
		},
		Operators: []OperatorProfile{
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0", Eta: 0.62, NumPoints: 7},
			{Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16", Eta: 0.75, NumPoints: 5},
			{Op: "RMS_NORM", Backend: "cuda", ComputeDtype: "f32", Eta: 0.85, NumPoints: 5},
			{Op: "ADD", Backend: "cuda", ComputeDtype: "f32", Eta: 0.90, NumPoints: 5},
			{Op: "SILU", Backend: "cuda", ComputeDtype: "f32", Eta: 0.88, NumPoints: 5},
		},
	}

	// 2. Simulate a simplified Llama-like transformer block graph for decode (batch=1)
	numLayers := 32
	var nodes []ml.GraphNode

	for l := 0; l < numLayers; l++ {
		// Q/K/V projections (3 MUL_MATs)
		for _, proj := range []string{"q", "k", "v"} {
			nodes = append(nodes, ml.GraphNode{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Name:        fmt.Sprintf("blk.%d.attn_%s", l, proj),
				Shape:       [4]int64{4096, 1, 1, 1},
				InputShapes: [][]int64{{4096, 4096}, {4096, 1}},
			})
		}
		// RMS Norm
		nodes = append(nodes, ml.GraphNode{
			Op: "RMS_NORM", Backend: "cuda", ComputeDtype: "f32",
			Shape: [4]int64{4096, 1, 1, 1}, InputShapes: [][]int64{{4096, 1}},
		})
		// Flash Attention
		nodes = append(nodes, ml.GraphNode{
			Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
			Shape:       [4]int64{1, 32, 1, 128},
			InputShapes: [][]int64{{1, 32, 1, 128}, {1, 32, 1024, 128}},
		})
		// Output projection
		nodes = append(nodes, ml.GraphNode{
			Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			Shape: [4]int64{4096, 1, 1, 1}, InputShapes: [][]int64{{4096, 4096}, {4096, 1}},
		})
		// FFN: gate + up + down (3 MUL_MATs) + SILU
		for _, ffn := range []string{"gate", "up", "down"} {
			dim := int64(11008)
			if ffn == "down" {
				dim = 4096
			}
			var inputShapes [][]int64
			if ffn == "down" {
				inputShapes = [][]int64{{4096, 11008}, {11008, 1}}
			} else {
				inputShapes = [][]int64{{11008, 4096}, {4096, 1}}
			}
			nodes = append(nodes, ml.GraphNode{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Name:  fmt.Sprintf("blk.%d.ffn_%s", l, ffn),
				Shape: [4]int64{dim, 1, 1, 1},
				InputShapes: inputShapes,
			})
		}
		nodes = append(nodes, ml.GraphNode{
			Op: "SILU", Backend: "cuda", ComputeDtype: "f32",
			Shape: [4]int64{11008, 1, 1, 1}, InputShapes: [][]int64{{11008}},
		})
	}

	// LM head
	nodes = append(nodes, ml.GraphNode{
		Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
		Name: "lm_head", Shape: [4]int64{128256, 1, 1, 1},
		InputShapes: [][]int64{{128256, 4096}, {4096, 1}},
	})

	// 3. Estimate
	latency, stats, warnings := EstimateGraphLatency(profile, nodes)

	require.Greater(t, latency, 0.0)
	assert.Greater(t, len(stats), 0)

	// MUL_MAT should dominate
	mulmatKey := OpKey{"MUL_MAT", "cuda", "f16", "q4_0"}
	mulmatStats, ok := stats[mulmatKey]
	require.True(t, ok)
	assert.Equal(t, 32*7+1, mulmatStats.Count) // 7 per layer x 32 + 1 lm_head
	assert.Greater(t, mulmatStats.TotalSec/latency, 0.5) // >50% of total

	// Should be memory-bound for decode (batch=1)
	assert.Greater(t, mulmatStats.MemCount, mulmatStats.CompCount)

	// 4. Build full result and print
	phase := ComputePhaseEstimation(profile, nodes, 1, 1)
	require.NotNil(t, phase)

	result := &EstimateResult{
		Model:        "llama3:8b-q4_0",
		InputLength:  1024,
		OutputLength: 256,
		MaxBatchSize: 512,
		Decode:       *phase,
		Warnings:     warnings,
		Backends: []BackendInfo{
			{Name: "cuda", Device: "RTX 4090", PeakFLOPS: 82.6e12, PeakBandwidth: 1008e9, BalancePoint: 81.9},
		},
	}
	BuildSummary(result)

	var buf bytes.Buffer
	PrintEstimateResult(&buf, result, false)
	output := buf.String()

	assert.Contains(t, output, "llama3:8b-q4_0")
	assert.Contains(t, output, "RTX 4090")
	assert.Contains(t, output, "MUL_MAT")
	assert.Contains(t, output, "tok/s")

	// Decode tok/s sanity: for 8B q4_0 on RTX 4090, expect ~50-200 tok/s
	assert.Greater(t, phase.TokensPerSec, 10.0, "decode too slow")
	assert.Less(t, phase.TokensPerSec, 1000.0, "decode too fast")

	t.Logf("Decode estimate: %.1f tok/s (%.2f ms/tok)", phase.TokensPerSec, phase.TotalLatencyMs)
	t.Log(output)
}

func TestProfileRoundTripIntegration(t *testing.T) {
	dir := t.TempDir()

	raw := &RawData{
		Version: 1,
		HardwareBenchmarks: []HardwareBenchmark{
			{Backend: "cuda", Dtype: "f16", Test: "peak_flops", Value: 82.6e12, Unit: "FLOPS"},
			{Backend: "cuda", Test: "peak_bandwidth", Value: 1008e9, Unit: "bytes/sec"},
		},
		OperatorBenchmarks: []OperatorBenchmark{
			{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Points: []BenchmarkPoint{
					{FLOPs: 33554432, BytesMoved: 8396800, Intensity: 4.0, LatencyUs: 12.5, Reps: 200},
					{FLOPs: 33554432e3, BytesMoved: 8396800e2, Intensity: 40.0, LatencyUs: 500, Reps: 200},
				},
			},
		},
	}

	rawPath := filepath.Join(dir, "raw.json")
	require.NoError(t, WriteRawData(rawPath, raw))

	profile, err := ProcessRawToProfile([]string{rawPath})
	require.NoError(t, err)

	assert.Equal(t, 1, len(profile.Hardware.Backends))
	assert.InDelta(t, 82.6e12, profile.Hardware.Backends[0].PeakFLOPS["f16"], 1e6)
	assert.Equal(t, 1, len(profile.Operators))
	assert.Greater(t, profile.Operators[0].Eta, 0.0)
	assert.LessOrEqual(t, profile.Operators[0].Eta, 2.0)

	profilePath := filepath.Join(dir, "profile.json")
	require.NoError(t, WriteProfile(profilePath, profile))

	loaded, err := LoadProfile(profilePath)
	require.NoError(t, err)
	assert.Equal(t, profile.Operators[0].Eta, loaded.Operators[0].Eta)
}
