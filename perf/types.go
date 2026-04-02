package perf

import "time"

// --- Profile types ---

type Profile struct {
	Version       int                `json:"version"`
	GeneratedFrom []string           `json:"generated_from"`
	GeneratedAt   time.Time          `json:"generated_at"`
	Hardware      HardwareProfile    `json:"hardware"`
	Operators     []OperatorProfile  `json:"operators"`
	Interconnects []InterconnectInfo `json:"interconnects"`
}

type HardwareProfile struct {
	Backends []BackendProfile `json:"backends"`
}

type BackendProfile struct {
	Name          string             `json:"name"`
	Device        string             `json:"device"`
	PeakFLOPS     map[string]float64 `json:"peak_flops"`
	PeakBandwidth float64            `json:"peak_bandwidth"`
	BalancePoints map[string]float64 `json:"balance_points"`
}

type OperatorProfile struct {
	Op           string  `json:"op"`
	Backend      string  `json:"backend"`
	ComputeDtype string  `json:"compute_dtype"`
	WeightDtype  string  `json:"weight_dtype,omitempty"`
	Eta          float64 `json:"eta"`
	EtaVariance  float64 `json:"eta_variance"`
	NumPoints    int     `json:"num_points"`
}

type InterconnectInfo struct {
	From      string  `json:"from"`
	To        string  `json:"to"`
	Bandwidth float64 `json:"bandwidth"`
}

// --- Raw benchmark data types ---

type RawData struct {
	Version                int                     `json:"version"`
	Timestamp              time.Time               `json:"timestamp"`
	Hardware               RawHardware             `json:"hardware"`
	HardwareBenchmarks     []HardwareBenchmark     `json:"hardware_benchmarks"`
	OperatorBenchmarks     []OperatorBenchmark     `json:"operator_benchmarks"`
	InterconnectBenchmarks []InterconnectBenchmark `json:"interconnect_benchmarks"`
}

type RawHardware struct {
	Backends []RawBackendInfo `json:"backends"`
}

type RawBackendInfo struct {
	Name              string `json:"name"`
	Device            string `json:"device"`
	Driver            string `json:"driver,omitempty"`
	ComputeCapability string `json:"compute_capability,omitempty"`
	VRAMBytes         uint64 `json:"vram_bytes,omitempty"`
	Cores             int    `json:"cores,omitempty"`
	RAMBytes          uint64 `json:"ram_bytes,omitempty"`
}

type HardwareBenchmark struct {
	Backend string  `json:"backend"`
	Dtype   string  `json:"dtype,omitempty"`
	Test    string  `json:"test"`
	Value   float64 `json:"value"`
	Unit    string  `json:"unit"`
}

type OperatorBenchmark struct {
	Op           string           `json:"op"`
	Backend      string           `json:"backend"`
	ComputeDtype string           `json:"compute_dtype"`
	WeightDtype  string           `json:"weight_dtype,omitempty"`
	Points       []BenchmarkPoint `json:"points"`
}

type BenchmarkPoint struct {
	InputShapes [][]int64 `json:"input_shapes"`
	OutputShape []int64   `json:"output_shape"`
	FLOPs       float64   `json:"flops"`
	BytesMoved  float64   `json:"bytes_moved"`
	Intensity   float64   `json:"intensity"`
	LatencyUs   float64   `json:"latency_us"`
	Reps        int       `json:"reps"`
	StddevUs    float64   `json:"stddev_us"`
}

type InterconnectBenchmark struct {
	From          string  `json:"from"`
	To            string  `json:"to"`
	Bandwidth     float64 `json:"bandwidth"`
	LatencyUs     float64 `json:"latency_us"`
	TestSizeBytes int64   `json:"test_size_bytes"`
}

// --- Estimate output types ---

type EstimateResult struct {
	Model        string          `json:"model"`
	Backends     []BackendInfo   `json:"backends"`
	InputLength  int             `json:"input_length"`
	OutputLength int             `json:"output_length"`
	MaxBatchSize int             `json:"max_batch_size"`
	Prefill      PhaseEstimation `json:"prefill"`
	Decode       PhaseEstimation `json:"decode"`
	Warnings     []string        `json:"warnings,omitempty"`
	Summary      string          `json:"summary"`
}

type BackendInfo struct {
	Name          string  `json:"name"`
	Device        string  `json:"device"`
	PeakFLOPS     float64 `json:"peak_flops"`
	PeakBandwidth float64 `json:"peak_bandwidth"`
	BalancePoint  float64 `json:"balance_point"`
}

type PhaseEstimation struct {
	TotalLatencyMs float64       `json:"total_latency_ms"`
	TokensPerSec   float64       `json:"tokens_per_sec"`
	TTFTMs         float64       `json:"ttft_ms,omitempty"`
	NumBatches     int           `json:"num_batches,omitempty"`
	Bottleneck     string        `json:"bottleneck"`
	TopOps         []OpBreakdown `json:"top_ops"`
}

type OpBreakdown struct {
	Op             string  `json:"op"`
	Backend        string  `json:"backend"`
	ComputeDtype   string  `json:"compute_dtype"`
	WeightDtype    string  `json:"weight_dtype,omitempty"`
	Count          int     `json:"count"`
	TotalMs        float64 `json:"total_ms"`
	Percentage     float64 `json:"percentage"`
	BoundBreakdown string  `json:"bound_breakdown"`
}

// --- Internal helper types ---

type OpKey struct {
	Op           string
	Backend      string
	ComputeDtype string
	WeightDtype  string
}

type OpCost struct {
	FLOPs        float64
	BytesMoved   float64
	Intensity    float64
	TCompute     float64
	TMemory      float64
	TActual      float64
	Bound        string
	Eta          float64
	Uncalibrated bool
}
