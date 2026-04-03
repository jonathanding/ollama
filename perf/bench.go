package perf

import (
	"fmt"
	"log/slog"
	"math"
	"sort"
	"time"

	"github.com/ollama/ollama/ml"
)

// SamplingGridWithFixed extends SamplingGrid with fixed dimensions for multi-dim ops.
type SamplingGridWithFixed struct {
	Op          string
	Dtype       string
	WeightDtype string
	FixedDims   map[string]int64 // nil for 1D ops
}

// buildSamplingGrids creates the grid specifications for one operator + dtype combo.
// For 1D ops: one grid. For MUL_MAT: one grid per (M, K) pair. For FLASH_ATTN: one grid.
func buildSamplingGrids(op, computeDtype, weightDtype string) []SamplingGridWithFixed {
	switch op {
	case "MUL_MAT":
		pairs := Phase1MulMatFixedDims()
		grids := make([]SamplingGridWithFixed, len(pairs))
		for i, pair := range pairs {
			grids[i] = SamplingGridWithFixed{
				Op: op, Dtype: computeDtype, WeightDtype: weightDtype,
				FixedDims: map[string]int64{"M": pair[0], "K": pair[1]},
			}
		}
		return grids
	case "FLASH_ATTN_EXT":
		return []SamplingGridWithFixed{{
			Op: op, Dtype: computeDtype,
			FixedDims: map[string]int64{"num_heads": 32, "head_dim": 128},
		}}
	default:
		// Check if the op exists in the registry
		if _, ok := LookupRegistry(op); !ok {
			return nil
		}
		return []SamplingGridWithFixed{{
			Op: op, Dtype: computeDtype,
		}}
	}
}

// measureOp benchmarks an operator at one shape point using the GGML backend.
// It creates tensors, runs warmup+measurement, trims outliers, and returns the median latency.
func measureOp(backend ml.Backend, op string, gridPoint []int64, computeDtype string, cfg BenchmarkConfig) LatencyPoint {
	runner, ok := LookupRegistry(op)
	if !ok {
		slog.Warn("unknown op in registry", "op", op)
		return LatencyPoint{Shape: gridPoint}
	}

	dt, ok := parseDType(computeDtype)
	if !ok {
		slog.Warn("unsupported dtype", "dtype", computeDtype)
		return LatencyPoint{Shape: gridPoint}
	}

	tensorShapes := expandShapes(op, gridPoint)

	ctx := backend.NewContext()
	defer ctx.Close()

	// Create input tensors
	inputs := make([]ml.Tensor, runner.NumInputs)
	for i := 0; i < runner.NumInputs; i++ {
		if i < len(tensorShapes) {
			shape := tensorShapes[i]
			intShape := make([]int, len(shape))
			for j, s := range shape {
				intShape[j] = int(s)
			}
			inputs[i] = ctx.Input().Zeros(dt, intShape...)
		}
	}

	// Build computation graph
	out := runner.Run(ctx, inputs)
	if out == nil {
		slog.Warn("op runner returned nil", "op", op)
		return LatencyPoint{Shape: gridPoint}
	}
	ctx.Forward(out)

	// Warmup
	for i := 0; i < cfg.WarmupReps; i++ {
		ctx.Compute(out)
	}

	// Measure
	latencies := make([]float64, cfg.MeasureReps)
	for i := 0; i < cfg.MeasureReps; i++ {
		start := time.Now()
		ctx.Compute(out)
		latencies[i] = float64(time.Since(start).Microseconds())
	}

	median := trimmedMedian(latencies, cfg.TrimPercent)

	// Compute stddev of trimmed set
	sort.Float64s(latencies)
	trimCount := int(math.Round(float64(len(latencies)) * cfg.TrimPercent))
	if trimCount*2 >= len(latencies) {
		trimCount = 0
	}
	trimmed := latencies[trimCount : len(latencies)-trimCount]
	mean := 0.0
	for _, l := range trimmed {
		mean += l
	}
	mean /= float64(len(trimmed))
	variance := 0.0
	for _, l := range trimmed {
		d := l - mean
		variance += d * d
	}
	stddev := math.Sqrt(variance / float64(len(trimmed)))

	return LatencyPoint{
		Shape:     gridPoint,
		LatencyUs: median,
		StddevUs:  stddev,
		Reps:      cfg.MeasureReps,
	}
}

// RunBenchmark executes the full v2 calibration pipeline:
// 1. Hardware characterization (peak TOPS, BW)
// 2. For each op × dtype: adaptive sampling → OperatorCurves
// Returns a complete Profile ready for estimation.
func RunBenchmark(backend ml.Backend, ops []string, dtypes []string, cfg BenchmarkConfig) (*Profile, error) {
	// Step 1: Hardware characterization
	hwResult, err := CharacterizeHardware(backend, cfg)
	if err != nil {
		return nil, fmt.Errorf("hardware characterization: %w", err)
	}
	hwProfile := HWCharResultToHardwareProfile(hwResult, backend)

	profile := &Profile{
		Version:   2,
		Timestamp: time.Now(),
		Hardware:  hwProfile,
	}

	slog.Info("starting operator calibration", "ops", len(ops), "dtypes", len(dtypes))

	// Step 2: Benchmark each op × dtype
	for _, op := range ops {
		opDtypes := dtypes
		// FLASH_ATTN_EXT only uses f16
		if op == "FLASH_ATTN_EXT" {
			opDtypes = []string{"f16"}
		}
		// 1D element-wise ops only use f32
		runner, ok := LookupRegistry(op)
		if !ok {
			slog.Warn("skipping unknown op", "op", op)
			continue
		}
		if len(runner.Dimensions) == 1 && runner.Dimensions[0] == "N" && op != "MUL_MAT" {
			opDtypes = []string{"f32"}
		}

		for _, dtype := range opDtypes {
			weightDtype := ""
			if op == "MUL_MAT" {
				weightDtype = dtype
			}

			grids := buildSamplingGrids(op, dtype, weightDtype)

			for _, grid := range grids {
				slog.Info("benchmarking", "op", op, "dtype", dtype, "fixed", grid.FixedDims)

				var curve OperatorCurve
				curve.Op = op
				curve.ComputeDtype = dtype
				curve.WeightDtype = weightDtype
				curve.Dimensions = sweepDimensions(op)
				curve.FixedDims = grid.FixedDims

				// Determine backend name from first device
				devices := backend.BackendDevices()
				if len(devices) > 0 {
					curve.Backend = devices[0].Library
				}

				switch op {
				case "FLASH_ATTN_EXT":
					curve.Points = benchmarkFlashAttn(backend, dtype, grid.FixedDims, cfg)
				case "MUL_MAT":
					curve.Points = benchmarkMulMat(backend, dtype, grid.FixedDims, cfg)
				default:
					curve.Points = benchmarkElementwise(backend, op, dtype, cfg)
				}

				if len(curve.Points) > 0 {
					profile.Operators = append(profile.Operators, curve)
				}
			}
		}
	}

	return profile, nil
}

// sweepDimensions returns the sweep (non-fixed) dimensions for an op.
func sweepDimensions(op string) []string {
	switch op {
	case "MUL_MAT":
		return []string{"N"}
	case "FLASH_ATTN_EXT":
		return []string{"seq_q", "seq_kv"}
	default:
		return []string{"N"}
	}
}

// benchmarkElementwise uses AdaptiveSample1D for 1D ops.
func benchmarkElementwise(backend ml.Backend, op, dtype string, cfg BenchmarkConfig) []LatencyPoint {
	measure := func(shape []int64) LatencyPoint {
		return measureOp(backend, op, shape, dtype, cfg)
	}
	return AdaptiveSample1D(measure, 1024, 64*1024*1024, 8, cfg)
}

// benchmarkMulMat uses AdaptiveSample1D over N with fixed (M, K).
func benchmarkMulMat(backend ml.Backend, dtype string, fixedDims map[string]int64, cfg BenchmarkConfig) []LatencyPoint {
	M := fixedDims["M"]
	K := fixedDims["K"]
	measure := func(shape []int64) LatencyPoint {
		N := shape[0]
		return measureOp(backend, "MUL_MAT", []int64{M, K, N}, dtype, cfg)
	}
	// N range: 1 to 4096 (batch sizes in inference)
	return AdaptiveSample1D(measure, 1, 4096, 8, cfg)
}

// benchmarkFlashAttn samples two regimes: decode and prefill.
// IMPORTANT: AdaptiveSample1D works internally with 1D shapes (Shape[0] is the sweep dimension).
// The measure callbacks must NOT override pt.Shape to 2D during sampling, because
// AdaptiveSample1D uses Shape[0] for sorting and interpolation error analysis.
// We keep shapes 1D during sampling, then convert to 2D after sampling completes.
func benchmarkFlashAttn(backend ml.Backend, dtype string, fixedDims map[string]int64, cfg BenchmarkConfig) []LatencyPoint {
	var points []LatencyPoint

	// Decode: seq_q=1, sweep seq_kv (1D: shape[0] = seq_kv)
	decodeMeasure := func(shape []int64) LatencyPoint {
		seqKV := shape[0]
		pt := measureOp(backend, "FLASH_ATTN_EXT", []int64{1, seqKV}, dtype, cfg)
		// Keep shape 1D for AdaptiveSample1D's internal sorting/interpolation
		pt.Shape = []int64{seqKV}
		return pt
	}
	decodePts := AdaptiveSample1D(decodeMeasure, 64, 16384, 8, cfg)
	// Convert to 2D after sampling: [seq_kv] → [1, seq_kv]
	for i := range decodePts {
		seqKV := decodePts[i].Shape[0]
		decodePts[i].Shape = []int64{1, seqKV}
	}
	points = append(points, decodePts...)

	// Prefill: seq_q=seq_kv, sweep both (1D: shape[0] = seq_len)
	prefillMeasure := func(shape []int64) LatencyPoint {
		seqLen := shape[0]
		pt := measureOp(backend, "FLASH_ATTN_EXT", []int64{seqLen, seqLen}, dtype, cfg)
		// Keep shape 1D for AdaptiveSample1D
		pt.Shape = []int64{seqLen}
		return pt
	}
	prefillPts := AdaptiveSample1D(prefillMeasure, 64, 16384, 8, cfg)
	// Convert to 2D after sampling: [seq_len] → [seq_len, seq_len]
	for i := range prefillPts {
		seqLen := prefillPts[i].Shape[0]
		prefillPts[i].Shape = []int64{seqLen, seqLen}
	}
	points = append(points, prefillPts...)

	return points
}
