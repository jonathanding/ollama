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
// 2. MUL_MAT: one reference curve + extract efficiency constants (roofline extrapolation)
// 3. Other ops: adaptive sampling → OperatorCurves
// Returns a complete Profile ready for estimation.
func RunBenchmark(backend ml.Backend, ops []string, dtypes []string, cfg BenchmarkConfig) (*Profile, error) {
	benchStart := time.Now()

	// Step 1: Hardware characterization
	hwStart := time.Now()
	hwResult, err := CharacterizeHardware(backend, cfg)
	if err != nil {
		return nil, fmt.Errorf("hardware characterization: %w", err)
	}
	hwProfile := HWCharResultToHardwareProfile(hwResult, backend)
	slog.Info("hardware characterization complete", "duration", time.Since(hwStart).Round(time.Second))

	profile := &Profile{
		Version:   2,
		Timestamp: time.Now(),
		Hardware:  hwProfile,
	}

	// Count grids: MUL_MAT = 1 reference curve, others = normal counting
	totalGrids := countGrids(ops, dtypes)
	slog.Info("starting operator calibration", "ops", len(ops), "dtypes", len(dtypes), "total_grids", totalGrids)
	calibrationStart := time.Now()
	gridIdx := 0

	// Step 2: Benchmark each op
	for _, op := range ops {
		if op == "MUL_MAT" {
			// MUL_MAT: measure ONE reference curve, extract efficiency constants
			gridIdx++
			elapsed := time.Since(calibrationStart).Round(time.Second)
			slog.Info("benchmarking MUL_MAT reference curve",
				"progress", fmt.Sprintf("[%d/%d]", gridIdx, totalGrids),
				"M", 4096, "K", 4096, "dtype", "f32", "elapsed", elapsed)

			gridStart := time.Now()
			refPoints := benchmarkMulMat(backend, "f32", map[string]int64{"M": 4096, "K": 4096}, cfg)

			if len(refPoints) == 0 {
				slog.Warn("MUL_MAT reference curve produced no points")
				continue
			}

			// Store reference curve in profile
			refCurve := OperatorCurve{
				Op:           "MUL_MAT",
				ComputeDtype: "f32",
				WeightDtype:  "f32",
				Dimensions:   []string{"N"},
				FixedDims:    map[string]int64{"M": 4096, "K": 4096},
				Points:       refPoints,
			}
			devices := backend.BackendDevices()
			if len(devices) > 0 {
				refCurve.Backend = devices[0].Library
			}
			profile.Operators = append(profile.Operators, refCurve)

			// Extract efficiency constants from reference curve
			eff := extractEfficiencyConstants(refPoints, 4096, 4096, hwResult.PeakTOPS["f32"], hwResult.PeakBW)
			if profile.Hardware.EfficiencyConstants == nil {
				profile.Hardware.EfficiencyConstants = make(map[string]OpEfficiency)
			}
			profile.Hardware.EfficiencyConstants["MUL_MAT"] = eff

			gridDuration := time.Since(gridStart).Round(time.Second)
			slog.Info("completed MUL_MAT reference",
				"progress", fmt.Sprintf("[%d/%d]", gridIdx, totalGrids),
				"points", len(refPoints),
				"eff_compute", fmt.Sprintf("%.2f", eff.ComputeEff),
				"eff_bw", fmt.Sprintf("%.2f", eff.BWEff),
				"overhead_us", fmt.Sprintf("%.0f", eff.OverheadUs),
				"grid_duration", gridDuration)
			continue
		}

		// Non-MUL_MAT ops: adaptive sampling as before
		opDtypes := dtypes
		if op == "FLASH_ATTN_EXT" {
			opDtypes = []string{"f16"}
		}
		runner, ok := LookupRegistry(op)
		if !ok {
			slog.Warn("skipping unknown op", "op", op)
			continue
		}
		if len(runner.Dimensions) == 1 && runner.Dimensions[0] == "N" {
			opDtypes = []string{"f32"}
		}

		for _, dtype := range opDtypes {
			grids := buildSamplingGrids(op, dtype, "")

			for _, grid := range grids {
				gridIdx++
				elapsed := time.Since(calibrationStart).Round(time.Second)
				slog.Info("benchmarking", "progress", fmt.Sprintf("[%d/%d]", gridIdx, totalGrids),
					"op", op, "dtype", dtype, "fixed", grid.FixedDims, "elapsed", elapsed)

				gridStart := time.Now()

				var curve OperatorCurve
				curve.Op = op
				curve.ComputeDtype = dtype
				curve.Dimensions = sweepDimensions(op)
				curve.FixedDims = grid.FixedDims

				devices := backend.BackendDevices()
				if len(devices) > 0 {
					curve.Backend = devices[0].Library
				}

				switch op {
				case "FLASH_ATTN_EXT":
					curve.Points = benchmarkFlashAttn(backend, dtype, grid.FixedDims, cfg)
				default:
					curve.Points = benchmarkElementwise(backend, op, dtype, cfg)
				}

				gridDuration := time.Since(gridStart).Round(time.Second)
				if len(curve.Points) > 0 {
					minLat, maxLat := curve.Points[0].LatencyUs, curve.Points[0].LatencyUs
					for _, p := range curve.Points[1:] {
						if p.LatencyUs < minLat {
							minLat = p.LatencyUs
						}
						if p.LatencyUs > maxLat {
							maxLat = p.LatencyUs
						}
					}
					slog.Info("completed", "progress", fmt.Sprintf("[%d/%d]", gridIdx, totalGrids),
						"op", op, "dtype", dtype, "fixed", grid.FixedDims,
						"points", len(curve.Points),
						"latency_range_us", fmt.Sprintf("%.0f-%.0f", minLat, maxLat),
						"grid_duration", gridDuration)
					profile.Operators = append(profile.Operators, curve)
				} else {
					slog.Warn("no points collected", "op", op, "dtype", dtype, "fixed", grid.FixedDims)
				}
			}
		}
	}

	totalDuration := time.Since(benchStart).Round(time.Second)
	slog.Info("calibration complete", "grids", len(profile.Operators), "total_duration", totalDuration)

	return profile, nil
}

// countGrids pre-counts the total number of sampling grids to run.
// MUL_MAT counts as 1 (reference curve only), not 6×4=24.
func countGrids(ops []string, dtypes []string) int {
	total := 0
	for _, op := range ops {
		if op == "MUL_MAT" {
			total++ // one reference curve
			continue
		}
		opDtypes := dtypes
		if op == "FLASH_ATTN_EXT" {
			opDtypes = []string{"f16"}
		}
		runner, ok := LookupRegistry(op)
		if !ok {
			continue
		}
		if len(runner.Dimensions) == 1 && runner.Dimensions[0] == "N" {
			opDtypes = []string{"f32"}
		}
		for _, dtype := range opDtypes {
			grids := buildSamplingGrids(op, dtype, "")
			total += len(grids)
		}
	}
	return total
}

// extractEfficiencyConstants computes compute and BW efficiency from a MUL_MAT reference curve.
// The reference curve is measured at (refM, refK) with varying N.
func extractEfficiencyConstants(points []LatencyPoint, refM, refK int64, peakTOPS, peakBW float64) OpEfficiency {
	var computeEffs, bwEffs []float64
	var overheads []float64

	for _, pt := range points {
		N := pt.Shape[0]
		if pt.LatencyUs <= 0 {
			continue
		}

		flops := 2.0 * float64(refM) * float64(refK) * float64(N)
		bytes := float64(refM*refK+refK*N+refM*N) * 4 // f32 = 4 bytes

		computeTime := flops / peakTOPS * 1e6  // ideal compute time in us
		bwTime := bytes / peakBW * 1e6          // ideal BW time in us

		if computeTime > bwTime {
			// Compute-bound point: efficiency = ideal / measured
			computeEffs = append(computeEffs, computeTime/pt.LatencyUs)
		} else {
			// BW-bound point: efficiency = ideal / measured
			bwEffs = append(bwEffs, bwTime/pt.LatencyUs)
			// Also extract overhead: measured - ideal_bw
			overheads = append(overheads, pt.LatencyUs-bwTime)
		}
	}

	eff := OpEfficiency{}
	if len(computeEffs) > 0 {
		sort.Float64s(computeEffs)
		eff.ComputeEff = computeEffs[len(computeEffs)/2] // median
	} else {
		eff.ComputeEff = 0.9 // reasonable default
	}
	if len(bwEffs) > 0 {
		sort.Float64s(bwEffs)
		eff.BWEff = bwEffs[len(bwEffs)/2]
	} else {
		eff.BWEff = 0.45 // reasonable default
	}
	if len(overheads) > 0 {
		sort.Float64s(overheads)
		eff.OverheadUs = math.Max(0, overheads[len(overheads)/2])
	}

	return eff
}

// PredictMulMatLatency computes MUL_MAT latency using the roofline model
// with measured efficiency constants.
func PredictMulMatLatency(hw *HardwareProfile, M, K, N int64, dtype string) float64 {
	eff, ok := hw.EfficiencyConstants["MUL_MAT"]
	if !ok {
		return 0
	}
	peakTOPS, ok := hw.PeakTOPS[dtype]
	if !ok {
		// Fall back to f32 if dtype not measured
		peakTOPS = hw.PeakTOPS["f32"]
	}
	peakBW := hw.PeakBandwidthBytesPerSec

	flops := 2.0 * float64(M) * float64(K) * float64(N)
	elemBytes := float64(elemSizeFromDtype(dtype))
	bytes := float64(M*K+K*N+M*N) * elemBytes

	computeTime := flops / (eff.ComputeEff * peakTOPS) * 1e6 // microseconds
	bwTime := bytes / (eff.BWEff * peakBW) * 1e6              // microseconds

	return math.Max(computeTime, bwTime) + eff.OverheadUs
}

// elemSizeFromDtype returns bytes per element for a dtype string.
func elemSizeFromDtype(dtype string) int {
	switch dtype {
	case "f32":
		return 4
	case "f16":
		return 2
	case "q4_0":
		// q4_0: 32 elements in 18 bytes (16 nibbles + 2 byte scale)
		// Effective: 0.5625 bytes per element, but for memory estimation use block size
		return 1 // approximate: actual is 4.5 bits
	case "q8_0":
		return 1 // 8 bits per element
	default:
		return 4
	}
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
// IMPORTANT: AdaptiveSample1D works with 1D shapes (Shape[0] is the sweep dimension).
// We keep shapes 1D during sampling, matching benchmarkFlashAttn's pattern.
func benchmarkMulMat(backend ml.Backend, dtype string, fixedDims map[string]int64, cfg BenchmarkConfig) []LatencyPoint {
	M := fixedDims["M"]
	K := fixedDims["K"]
	measure := func(shape []int64) LatencyPoint {
		N := shape[0]
		pt := measureOp(backend, "MUL_MAT", []int64{M, K, N}, dtype, cfg)
		// Keep shape 1D for AdaptiveSample1D's internal sorting/interpolation
		pt.Shape = []int64{N}
		return pt
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
