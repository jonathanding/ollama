package perf

import "github.com/ollama/ollama/ml"

// FusionRule defines an op fusion pattern that a backend applies at dispatch time.
// The estimate uses these rules to simulate fusion before summing per-op latencies.
type FusionRule struct {
	Name    string   // Fused op name: "RMS_NORM_MUL", "MUL_MAT_ADD", etc.
	Pattern []string // Sequential op names to match: ["RMS_NORM", "MUL"]
	// Match checks whether nodes[start:start+len(Pattern)] satisfy fusion constraints.
	// Called only after op names match. Check backend consistency, shapes, etc.
	Match func(nodes []ml.GraphNode, start int) bool
}

// VulkanFusionRules returns the 3 core fusion rules for Vulkan LLM decode.
// Rules are ordered by pattern length descending: 3-op patterns must be tried
// before 2-op patterns to avoid partial matches (e.g., RMS_NORM+MUL matching
// the first two ops of a RMS_NORM+MUL+ROPE sequence).
func VulkanFusionRules() []FusionRule {
	return []FusionRule{
		{
			Name:    "RMS_NORM_MUL_ROPE",
			Pattern: []string{"RMS_NORM", "MUL", "ROPE"},
			Match: func(nodes []ml.GraphNode, i int) bool {
				return nodes[i].Backend == nodes[i+1].Backend &&
					nodes[i+1].Backend == nodes[i+2].Backend
			},
		},
		{
			Name:    "RMS_NORM_MUL",
			Pattern: []string{"RMS_NORM", "MUL"},
			Match: func(nodes []ml.GraphNode, i int) bool {
				return nodes[i].Backend == nodes[i+1].Backend
			},
		},
		{
			Name:    "MUL_MAT_ADD",
			Pattern: []string{"MUL_MAT", "ADD"},
			Match: func(nodes []ml.GraphNode, i int) bool {
				if len(nodes[i].InputShapes) < 2 || len(nodes[i].InputShapes[1]) < 2 {
					return false
				}
				N := nodes[i].InputShapes[1][1]
				return N <= 8 && nodes[i].Backend == nodes[i+1].Backend
			},
		},
	}
}

// CPUFusionRules returns fusion rules for CPU backend. CPU has no op fusion.
func CPUFusionRules() []FusionRule { return nil }

// ApplyFusion scans graph nodes and replaces fusable patterns with single fused nodes.
// Returns a new node list (potentially shorter than input).
//
// Limitation: only checks op name sequence and backend consistency, not data dependencies
// (i.e., does not verify MUL's src[0] comes from RMS_NORM's output). In LLM graphs,
// RMS_NORM is always followed by a MUL consuming its output, so false positives are
// extremely unlikely.
func ApplyFusion(nodes []ml.GraphNode, rules []FusionRule) []ml.GraphNode {
	if len(rules) == 0 || len(nodes) == 0 {
		return nodes
	}
	result := make([]ml.GraphNode, 0, len(nodes))
	i := 0
	for i < len(nodes) {
		fused := false
		for _, rule := range rules {
			pLen := len(rule.Pattern)
			if i+pLen > len(nodes) {
				continue
			}
			match := true
			for j, opName := range rule.Pattern {
				if nodes[i+j].Op != opName {
					match = false
					break
				}
			}
			if match && rule.Match(nodes, i) {
				fusedNode := nodes[i]
				fusedNode.Op = rule.Name
				result = append(result, fusedNode)
				i += pLen
				fused = true
				break
			}
		}
		if !fused {
			result = append(result, nodes[i])
			i++
		}
	}
	return result
}
