package perf

import "time"

// --- Profile (v2): empirical latency curves ---

// Profile stores calibrated operator latency curves for a specific hardware configuration.
// Version 2 replaces v1's Roofline-based eta coefficients with direct latency measurements.
type Profile struct {
	Version   int              `json:"version"`   // 2
	Timestamp time.Time        `json:"timestamp"`
	Hardware  HardwareProfile  `json:"hardware"`
	Operators []OperatorCurve  `json:"operators"`
}

// HardwareProfile captures hardware characteristics used for initial sampling grid placement.
type HardwareProfile struct {
	Backends                  []BackendInfo      `json:"backends"`
	PeakTOPS                  map[string]float64 `json:"peak_tops"`                    // dtype -> TOPS
	PeakBandwidthBytesPerSec  float64            `json:"peak_bandwidth_bytes_sec"`
	InterconnectBWBytesPerSec float64            `json:"interconnect_bandwidth_bytes_sec"` // Phase 2
	BalancePoints             map[string]float64 `json:"balance_points"`               // dtype -> FLOPs/byte
}

// BackendInfo identifies a compute backend (GPU, CPU, etc.).
type BackendInfo struct {
	Name      string `json:"name"`
	Device    string `json:"device"`
	VRAMBytes int64  `json:"vram_bytes"`
}

// OperatorCurve stores measured latency points for one operator configuration.
type OperatorCurve struct {
	Op           string           `json:"op"`
	Backend      string           `json:"backend"`
	ComputeDtype string           `json:"compute_dtype"`
	WeightDtype  string           `json:"weight_dtype,omitempty"`
	Dimensions   []string         `json:"dimensions"`
	FixedDims    map[string]int64 `json:"fixed_dims,omitempty"`
	Points       []LatencyPoint   `json:"points"`
}

// LatencyPoint is one measured (shape, latency) pair.
type LatencyPoint struct {
	Shape     []int64 `json:"shape"`
	LatencyUs float64 `json:"latency_us"`
	StddevUs  float64 `json:"stddev_us"`
	Reps      int     `json:"reps"`
}

// --- Operator Registry ---

// SamplingGrid defines the points to benchmark for one operator.
type SamplingGrid struct {
	Op          string
	Dtype       string
	WeightDtype string
	Dimensions  []string
	Points      [][]int64
}

// --- Estimation output ---

// EstimateResult is the output of EstimateModel().
type EstimateResult struct {
	Model                   string          `json:"model"`
	PrefillLatencyUs        float64         `json:"prefill_latency_us"`
	PrefillMs               float64         `json:"prefill_ms"`
	DecodeLatencyUsPerToken float64         `json:"decode_latency_us_per_token"`
	DecodeTokensPerSec      float64         `json:"decode_tokens_per_sec"`
	Prefill                 PhaseEstimation `json:"prefill"`
	Decode                  PhaseEstimation `json:"decode"`
	Warnings                []string        `json:"warnings,omitempty"`
}

// PhaseEstimation breaks down latency for one inference phase (prefill or decode).
type PhaseEstimation struct {
	TotalLatencyMs float64       `json:"total_latency_ms"`
	TokensPerSec   float64       `json:"tokens_per_sec"`
	TopOps         []OpBreakdown `json:"top_ops"`
}

// OpBreakdown shows the contribution of one operator type to total latency.
type OpBreakdown struct {
	Op           string  `json:"op"`
	Backend      string  `json:"backend"`
	ComputeDtype string  `json:"compute_dtype"`
	WeightDtype  string  `json:"weight_dtype,omitempty"`
	Count        int     `json:"count"`
	TotalUs      float64 `json:"total_us"`
	Percentage   float64 `json:"percentage"`
}

// --- Benchmarking configuration ---

// BenchmarkConfig controls adaptive sampling and measurement parameters.
type BenchmarkConfig struct {
	ErrorThreshold float64 // relative error threshold for adaptive refinement
	MaxPointsPerOp int     // maximum number of sample points per operator
	WarmupReps     int     // number of warmup iterations before measurement
	MeasureReps    int     // number of measurement iterations
	TrimPercent    float64 // percentage of outliers to trim (e.g., 0.1 = 10%)
}

// DefaultBenchmarkConfig returns sensible defaults for production benchmarking.
func DefaultBenchmarkConfig() BenchmarkConfig {
	return BenchmarkConfig{
		WarmupReps:     5,
		MeasureReps:    50,
		TrimPercent:    0.1,
		ErrorThreshold: 0.05,
		MaxPointsPerOp: 20,
	}
}

// --- Internal helper types ---

// OpKey uniquely identifies an operator configuration.
type OpKey struct {
	Op           string
	Backend      string
	ComputeDtype string
	WeightDtype  string
}

// --- Hardware characterization results ---

// HWCharResult holds hardware characterization results.
type HWCharResult struct {
	PeakTOPS     map[string]float64 // dtype -> TOPS
	PeakBW       float64            // bytes/sec
	BalancePoint map[string]float64 // dtype -> FLOPs/byte
}
