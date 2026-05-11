package daop

// DaopContext is sent by clients to provide routing hints.
type DaopContext struct {
	Subtask string `json:"subtask,omitempty"`
}

// DaopResult is the routing decision attached to chat responses.
type DaopResult struct {
	Decision        string   `json:"decision"`                      // "offload" or "fallback"
	FallbackReason  string   `json:"fallback_reason,omitempty"`     // "gate" or "threshold"
	Confidence      *float64 `json:"confidence"`                    // P(correct), null if gate blocked
	Threshold       float64  `json:"threshold"`                     // accuracy_threshold (θ)
	GatePassRate    *float64 `json:"gate_pass_rate,omitempty"`      // model's pass rate on this subtask
	GateThreshold   float64  `json:"gate_threshold"`                // gate threshold (G)
	Subtask         string   `json:"subtask,omitempty"`             // subtask name
	LatencyEstimate *float64 `json:"latency_estimate_ms,omitempty"` // estimated total latency
	LatencyActual   *float64 `json:"latency_actual_ms,omitempty"`   // actual latency (offload only)
	Model           string   `json:"model"`                         // target model name
}
