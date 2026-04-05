package perf

import (
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/stretchr/testify/assert"
)

func TestParseLayerIndex(t *testing.T) {
	assert.Equal(t, 0, parseLayerIndex("blk.0.attn_q.weight"))
	assert.Equal(t, 5, parseLayerIndex("blk.5.ffn_down.weight"))
	assert.Equal(t, 23, parseLayerIndex("blk.23.attn_output"))
	assert.Equal(t, -1, parseLayerIndex("output.weight"))
	assert.Equal(t, -1, parseLayerIndex("token_embd.weight"))
	assert.Equal(t, -1, parseLayerIndex(""))
}

func TestAssignBackends(t *testing.T) {
	// Simulate a small model with 2 blocks, all offloaded to "Vulkan"
	schedule := ml.GPULayersList{{
		DeviceID: ml.DeviceID{Library: "Vulkan"},
		Layers:   []int{0, 1, 2}, // blocks 0,1 + output layer 2
	}}
	blockCount := 2

	nodes := []ml.GraphNode{
		{Op: "GET_ROWS", Name: "embed", InputNames: []string{"token_embd.weight"}},
		{Op: "RMS_NORM", Name: "blk0_norm", InputNames: []string{"blk.0.attn_norm.weight"}},
		{Op: "MUL_MAT", Name: "blk0_q", InputNames: []string{"blk.0.attn_q.weight", "prev"}},
		{Op: "SOFTMAX", Name: "blk0_softmax", InputNames: []string{"blk0_q_out"}},
		{Op: "MUL_MAT", Name: "blk1_q", InputNames: []string{"blk.1.attn_q.weight", "prev"}},
		{Op: "ADD", Name: "blk1_add", InputNames: []string{"x", "y"}},
		{Op: "MUL_MAT", Name: "out", InputNames: []string{"output.weight", "prev"}},
	}

	assignBackends(nodes, schedule, blockCount, "CPU")

	// token_embd → CPU
	assert.Equal(t, "CPU", nodes[0].Backend, "token_embd should be CPU")
	// blk.0 ops → Vulkan
	assert.Equal(t, "Vulkan", nodes[1].Backend, "blk.0 norm")
	assert.Equal(t, "Vulkan", nodes[2].Backend, "blk.0 MUL_MAT")
	// SOFTMAX has no identifiable layer → should inherit from adjacent (Vulkan)
	assert.Equal(t, "Vulkan", nodes[3].Backend, "softmax adjacency")
	// blk.1 → Vulkan
	assert.Equal(t, "Vulkan", nodes[4].Backend, "blk.1 MUL_MAT")
	// ADD has no layer → adjacency → Vulkan
	assert.Equal(t, "Vulkan", nodes[5].Backend, "add adjacency")
	// output → Vulkan (layer = blockCount)
	assert.Equal(t, "Vulkan", nodes[6].Backend, "output layer")
}

func TestAssignBackendsPartialOffload(t *testing.T) {
	// Only block 0 on GPU, block 1 and output on CPU
	schedule := ml.GPULayersList{{
		DeviceID: ml.DeviceID{Library: "Vulkan"},
		Layers:   []int{0},
	}}
	blockCount := 2

	nodes := []ml.GraphNode{
		{Op: "RMS_NORM", Name: "blk0_norm", InputNames: []string{"blk.0.attn_norm.weight"}},
		{Op: "MUL_MAT", Name: "blk0_q", InputNames: []string{"blk.0.attn_q.weight", "x"}},
		{Op: "RMS_NORM", Name: "blk1_norm", InputNames: []string{"blk.1.attn_norm.weight"}},
		{Op: "MUL_MAT", Name: "blk1_q", InputNames: []string{"blk.1.attn_q.weight", "x"}},
		{Op: "MUL_MAT", Name: "out", InputNames: []string{"output.weight", "x"}},
	}

	assignBackends(nodes, schedule, blockCount, "CPU")

	assert.Equal(t, "Vulkan", nodes[0].Backend, "blk.0 on GPU")
	assert.Equal(t, "Vulkan", nodes[1].Backend, "blk.0 on GPU")
	assert.Equal(t, "CPU", nodes[2].Backend, "blk.1 on CPU")
	assert.Equal(t, "CPU", nodes[3].Backend, "blk.1 on CPU")
	assert.Equal(t, "CPU", nodes[4].Backend, "output on CPU")
}
