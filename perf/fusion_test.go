package perf

import (
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/stretchr/testify/assert"
)

func node(op, backend string) ml.GraphNode {
	return ml.GraphNode{Op: op, Backend: backend, ComputeDtype: "f32"}
}

func nodeWithInputShapes(op, backend string, inputShapes [][]int64) ml.GraphNode {
	return ml.GraphNode{Op: op, Backend: backend, ComputeDtype: "f32", InputShapes: inputShapes}
}

func TestApplyFusionEmpty(t *testing.T) {
	result := ApplyFusion(nil, VulkanFusionRules())
	assert.Nil(t, result)
}

func TestApplyFusionNoRules(t *testing.T) {
	nodes := []ml.GraphNode{node("RMS_NORM", "Vulkan"), node("MUL", "Vulkan")}
	result := ApplyFusion(nodes, nil)
	assert.Equal(t, nodes, result)
}

func TestApplyFusionRMSNormMul(t *testing.T) {
	nodes := []ml.GraphNode{
		node("RMS_NORM", "Vulkan"),
		node("MUL", "Vulkan"),
		node("ADD", "Vulkan"),
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 2)
	assert.Equal(t, "RMS_NORM_MUL", result[0].Op)
	assert.Equal(t, "ADD", result[1].Op)
}

func TestApplyFusionRMSNormMulRope(t *testing.T) {
	nodes := []ml.GraphNode{
		node("RMS_NORM", "Vulkan"),
		node("MUL", "Vulkan"),
		node("ROPE", "Vulkan"),
		node("ADD", "Vulkan"),
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 2)
	assert.Equal(t, "RMS_NORM_MUL_ROPE", result[0].Op)
	assert.Equal(t, "ADD", result[1].Op)
}

func TestApplyFusionMulMatAdd(t *testing.T) {
	// MUL_MAT with N=1 (mat-vec) followed by ADD -> fused
	nodes := []ml.GraphNode{
		nodeWithInputShapes("MUL_MAT", "Vulkan", [][]int64{{4096, 1024}, {4096, 1}}),
		node("ADD", "Vulkan"),
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 1)
	assert.Equal(t, "MUL_MAT_ADD", result[0].Op)
}

func TestApplyFusionMulMatAddNotMatVec(t *testing.T) {
	// MUL_MAT with N=512 (not mat-vec) -> NOT fused
	nodes := []ml.GraphNode{
		nodeWithInputShapes("MUL_MAT", "Vulkan", [][]int64{{4096, 1024}, {4096, 512}}),
		node("ADD", "Vulkan"),
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 2)
	assert.Equal(t, "MUL_MAT", result[0].Op)
	assert.Equal(t, "ADD", result[1].Op)
}

func TestApplyFusionDifferentBackends(t *testing.T) {
	// RMS_NORM on Vulkan + MUL on CPU -> NOT fused
	nodes := []ml.GraphNode{
		node("RMS_NORM", "Vulkan"),
		node("MUL", "CPU"),
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 2)
	assert.Equal(t, "RMS_NORM", result[0].Op)
}

func TestApplyFusion3OpPriorityOver2Op(t *testing.T) {
	// RMS_NORM + MUL + ROPE should match 3-op rule, not 2-op
	nodes := []ml.GraphNode{
		node("RMS_NORM", "Vulkan"),
		node("MUL", "Vulkan"),
		node("ROPE", "Vulkan"),
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 1)
	assert.Equal(t, "RMS_NORM_MUL_ROPE", result[0].Op)
}

func TestVulkanFusionRulesOrder(t *testing.T) {
	rules := VulkanFusionRules()
	assert.Equal(t, 3, len(rules))
	// 3-op rule must come before 2-op rules
	assert.Len(t, rules[0].Pattern, 3, "first rule should be 3-op pattern")
	assert.Len(t, rules[1].Pattern, 2)
	assert.Len(t, rules[2].Pattern, 2)
}

func TestCPUFusionRulesEmpty(t *testing.T) {
	assert.Nil(t, CPUFusionRules())
}

func TestApplyFusionMulMatAddBoundaryN8(t *testing.T) {
	// N=8 is exactly at the boundary — should be fused
	nodes := []ml.GraphNode{
		nodeWithInputShapes("MUL_MAT", "Vulkan", [][]int64{{4096, 1024}, {4096, 8}}),
		node("ADD", "Vulkan"),
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 1)
	assert.Equal(t, "MUL_MAT_ADD", result[0].Op)
}

func TestApplyFusionMulMatAddBoundaryN9(t *testing.T) {
	// N=9 exceeds boundary — should NOT be fused
	nodes := []ml.GraphNode{
		nodeWithInputShapes("MUL_MAT", "Vulkan", [][]int64{{4096, 1024}, {4096, 9}}),
		node("ADD", "Vulkan"),
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 2)
	assert.Equal(t, "MUL_MAT", result[0].Op)
	assert.Equal(t, "ADD", result[1].Op)
}

func TestApplyFusionMultipleSequences(t *testing.T) {
	// Two RMS_NORM+MUL sequences in one graph
	nodes := []ml.GraphNode{
		node("RMS_NORM", "Vulkan"),
		node("MUL", "Vulkan"),
		node("MUL_MAT", "Vulkan"),
		node("RMS_NORM", "Vulkan"),
		node("MUL", "Vulkan"),
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 3)
	assert.Equal(t, "RMS_NORM_MUL", result[0].Op)
	assert.Equal(t, "MUL_MAT", result[1].Op)
	assert.Equal(t, "RMS_NORM_MUL", result[2].Op)
}

func TestApplyFusionSingleNode(t *testing.T) {
	nodes := []ml.GraphNode{node("ADD", "Vulkan")}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 1)
	assert.Equal(t, "ADD", result[0].Op)
}

func TestApplyFusionPreservesMetadata(t *testing.T) {
	// Fused node should retain the first node's Backend and ComputeDtype
	nodes := []ml.GraphNode{
		{Op: "RMS_NORM", Backend: "Vulkan", ComputeDtype: "f32", Name: "blk.0.attn_norm"},
		{Op: "MUL", Backend: "Vulkan", ComputeDtype: "f32", Name: "blk.0.attn_mul"},
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 1)
	assert.Equal(t, "RMS_NORM_MUL", result[0].Op)
	assert.Equal(t, "Vulkan", result[0].Backend)
	assert.Equal(t, "f32", result[0].ComputeDtype)
	assert.Equal(t, "blk.0.attn_norm", result[0].Name, "fused node should keep first node's name")
}

func TestApplyFusionMulMatAddMissingInputShapes(t *testing.T) {
	// MUL_MAT with no InputShapes should not crash and should NOT fuse
	nodes := []ml.GraphNode{
		{Op: "MUL_MAT", Backend: "Vulkan", ComputeDtype: "f32"},
		node("ADD", "Vulkan"),
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 2)
	assert.Equal(t, "MUL_MAT", result[0].Op)
}

func TestApplyFusionMulMatAddInsufficientInputShape(t *testing.T) {
	// MUL_MAT with only 1 input shape (missing second input) — should NOT fuse
	nodes := []ml.GraphNode{
		nodeWithInputShapes("MUL_MAT", "Vulkan", [][]int64{{4096, 1024}}),
		node("ADD", "Vulkan"),
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 2)
	assert.Equal(t, "MUL_MAT", result[0].Op)
}

func TestApplyFusionMulMatAddShortSecondShape(t *testing.T) {
	// MUL_MAT where second input has only 1D shape — should NOT fuse
	nodes := []ml.GraphNode{
		nodeWithInputShapes("MUL_MAT", "Vulkan", [][]int64{{4096, 1024}, {4096}}),
		node("ADD", "Vulkan"),
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 2)
	assert.Equal(t, "MUL_MAT", result[0].Op)
}

func TestApplyFusionAtEndOfGraph(t *testing.T) {
	// Fusion pattern as the last ops in the graph
	nodes := []ml.GraphNode{
		node("ADD", "Vulkan"),
		node("RMS_NORM", "Vulkan"),
		node("MUL", "Vulkan"),
	}
	result := ApplyFusion(nodes, VulkanFusionRules())
	assert.Len(t, result, 2)
	assert.Equal(t, "ADD", result[0].Op)
	assert.Equal(t, "RMS_NORM_MUL", result[1].Op)
}
