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
// For 1D ops: one grid. For MUL_MAT/MUL_MAT_ADD: one grid per (M, K) pair. For FLASH_ATTN: one grid.
func buildSamplingGrids(op, computeDtype, weightDtype string) []SamplingGridWithFixed {
	switch op {
	case "MUL_MAT", "MUL_MAT_ADD":
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

	ctx := backend.NewContext()
	defer ctx.Close()

	// Create input tensors — use custom CreateInputs if provided, else default
	var inputs []ml.Tensor
	if runner.CreateInputs != nil {
		inputs = runner.CreateInputs(ctx, computeDtype, gridPoint)
	} else {
		tensorShapes := expandShapes(op, gridPoint)
		inputs = make([]ml.Tensor, len(tensorShapes))
		for i, shape := range tensorShapes {
			intShape := make([]int, len(shape))
			for j, s := range shape {
				intShape[j] = int(s)
			}
			inputs[i] = randomTensor(ctx, dt, intShape...)
		}
	}

	// Build computation graph
	out := runner.Run(ctx, inputs)
	if out == nil {
		slog.Warn("op runner returned nil", "op", op)
		return LatencyPoint{Shape: gridPoint}
	}
	ctx.Forward(out)

	// Adaptive warmup: reduce for slow ops to avoid wasting minutes
	warmupStart := time.Now()
	for i := 0; i < cfg.WarmupReps; i++ {
		ctx.Compute(out)
		// After first warmup, if it took >1s, reduce remaining warmups
		if i == 0 {
			elapsed := float64(time.Since(warmupStart).Microseconds())
			if elapsed > 5e6 {
				// >5s per op: 1 warmup is enough
				break
			} else if elapsed > 1e6 {
				// >1s per op: 2 warmups total
				ctx.Compute(out)
				break
			}
		}
	}

	// Measure with convergence-based early stopping
	med, sd, actualReps := convergentMeasure(func() float64 {
		start := time.Now()
		ctx.Compute(out)
		return float64(time.Since(start).Microseconds())
	}, cfg)

	return LatencyPoint{
		Shape:     gridPoint,
		LatencyUs: med,
		StddevUs:  sd,
		Reps:      actualReps,
	}
}

// trimmedStats computes the trimmed median and standard deviation of a sample.
// It sorts the values, removes the outermost trimPercent fraction from each tail,
// then computes median and population stddev of the remaining values.
func trimmedStats(values []float64, trimPercent float64) (median, stddev float64) {
	if len(values) == 0 {
		return 0, 0
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	trimCount := int(math.Round(float64(len(sorted)) * trimPercent))
	if trimCount*2 >= len(sorted) {
		trimCount = 0
	}
	trimmed := sorted[trimCount : len(sorted)-trimCount]
	if len(trimmed) == 0 {
		// Fallback: no trimming possible
		trimmed = sorted
	}

	// Median
	median = trimmed[len(trimmed)/2]

	// Stddev (population)
	if len(trimmed) <= 1 {
		return median, 0
	}
	mean := 0.0
	for _, v := range trimmed {
		mean += v
	}
	mean /= float64(len(trimmed))
	variance := 0.0
	for _, v := range trimmed {
		d := v - mean
		variance += d * d
	}
	stddev = math.Sqrt(variance / float64(len(trimmed)))
	return median, stddev
}

// measureOpGPU benchmarks an operator using GPU timestamps instead of wall-clock.
// This eliminates Vulkan dispatch overhead from measurements.
// Requires backend that implements EnableGPUTimestamps/GetOpTimings.
func measureOpGPU(backend ml.Backend, op string, gridPoint []int64, computeDtype string, cfg BenchmarkConfig) LatencyPoint {
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

	backend.EnableGPUTimestamps(true)
	defer backend.EnableGPUTimestamps(false)

	ctx := backend.NewContext()
	defer ctx.Close()

	// Create input tensors — use custom CreateInputs if provided, else default
	var inputs []ml.Tensor
	if runner.CreateInputs != nil {
		inputs = runner.CreateInputs(ctx, computeDtype, gridPoint)
	} else {
		tensorShapes := expandShapes(op, gridPoint)
		inputs = make([]ml.Tensor, len(tensorShapes))
		for i, shape := range tensorShapes {
			intShape := make([]int, len(shape))
			for j, s := range shape {
				intShape[j] = int(s)
			}
			inputs[i] = randomTensor(ctx, dt, intShape...)
		}
	}

	// Build computation graph
	out := runner.Run(ctx, inputs)
	if out == nil {
		slog.Warn("op runner returned nil", "op", op)
		return LatencyPoint{Shape: gridPoint}
	}
	ctx.Forward(out)

	// Warmup (2 iterations — GPU timestamps make warmup fast)
	for range 2 {
		ctx.Compute(out)
	}

	// Measure using GPU timestamps
	samples := make([]float64, 0, cfg.MeasureReps)
	for i := 0; i < cfg.MeasureReps; i++ {
		ctx.Compute(out)
		timings := backend.GetOpTimings()
		if len(timings) == 0 {
			slog.Warn("no GPU timings returned, falling back to wall-clock", "op", op)
			// Fall back to wall-clock measurement for remaining iterations
			return measureOp(backend, op, gridPoint, computeDtype, cfg)
		}
		// Sum all op timings in the graph
		var gpuUs float64
		for _, t := range timings {
			gpuUs += t.GPUTimeUs
		}
		samples = append(samples, gpuUs)

		// Early stopping with convergence check
		if len(samples) >= cfg.MinReps {
			med, sd := trimmedStats(samples, cfg.TrimPercent)
			if med > 0 && sd/med < cfg.ConvergenceCV {
				return LatencyPoint{
					Shape:     gridPoint,
					LatencyUs: med,
					StddevUs:  sd,
					Reps:      len(samples),
				}
			}
		}
	}

	if len(samples) == 0 {
		return LatencyPoint{Shape: gridPoint}
	}

	med, sd := trimmedStats(samples, cfg.TrimPercent)
	return LatencyPoint{
		Shape:     gridPoint,
		LatencyUs: med,
		StddevUs:  sd,
		Reps:      len(samples),
	}
}

// measureOpForBackend dispatches to GPU timestamp or wall-clock measurement
// based on backend capabilities.
func measureOpForBackend(backend ml.Backend, caps BackendCapabilities, op string, gridPoint []int64, computeDtype string, cfg BenchmarkConfig) LatencyPoint {
	if caps.HasGPUTimestamp {
		return measureOpGPU(backend, op, gridPoint, computeDtype, cfg)
	}
	return measureOp(backend, op, gridPoint, computeDtype, cfg)
}

// benchOrchestrationOverhead measures CPU orchestration overhead for different graph sizes.
// Builds N chained trivial ops (16-element ADD), GPU compute ≈ 0, wall-clock ≈ CPU overhead.
// Used by estimate to add CPU-side cost to GPU op time sum.
func benchOrchestrationOverhead(backend ml.Backend, cfg BenchmarkConfig) []LatencyPoint {
	graphSizes := []int{50, 100, 200, 300, 500}
	var points []LatencyPoint

	for _, n := range graphSizes {
		ctx := backend.NewContext()

		a := randomTensor(ctx, ml.DTypeF32, 16)
		b := randomTensor(ctx, ml.DTypeF32, 16)
		last := a.Add(ctx, b)
		for i := 1; i < n; i++ {
			last = last.Add(ctx, b) // Chain: each op depends on the previous output
		}
		ctx.Forward(last)

		// Warmup
		for range 3 {
			ctx.Compute(last)
		}

		// Measure wall-clock (GPU compute ≈ 0 for 16-element ADD)
		med, sd, reps := convergentMeasure(func() float64 {
			start := time.Now()
			ctx.Compute(last)
			return float64(time.Since(start).Microseconds())
		}, cfg)

		points = append(points, LatencyPoint{
			Shape:     []int64{int64(n)},
			LatencyUs: med,
			StddevUs:  sd,
			Reps:      reps,
		})

		ctx.Close()
		slog.Info("orchestration overhead", "nodes", n, "latency_us", fmt.Sprintf("%.0f", med))
	}

	return points
}

// RunBenchmark executes the full v3 calibration pipeline:
// 1. Hardware characterization (peak TOPS, BW) + backend discovery
// 2. MUL_MAT: one reference curve + extract efficiency constants (roofline extrapolation)
// 3. Other ops: adaptive sampling → OperatorCurves
// 4. Fused ops: benchmark fused kernels when backend supports fusion
// 5. Orchestration overhead: measure CPU dispatch cost for GPU backends
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

	caps := DiscoverBackend(backend)
	slog.Info("backend capabilities", "name", caps.Name,
		"gpu_timestamp", caps.HasGPUTimestamp, "fusion_rules", len(caps.FusionRules),
		"mul_mat_vec", caps.HasMulMatVec)

	profile := &Profile{
		Version:   3,
		Timestamp: time.Now(),
		Hardware:  hwProfile,
		BackendCaps: map[string]BackendCapabilitiesJSON{
			caps.Name: caps.ToJSON(),
		},
	}

	// Count grids: MUL_MAT = 1 reference curve, others = normal counting
	totalGrids := countGrids(ops, dtypes)
	slog.Info("starting operator calibration", "ops", len(ops), "dtypes", len(dtypes), "total_grids", totalGrids)
	calibrationStart := time.Now()
	gridIdx := 0

	// Step 2: Benchmark each op
	for _, op := range ops {
		if op == "MUL_MAT" {
			// MUL_MAT: one reference curve per weight dtype
			for _, wdt := range Phase1Dtypes() {
				gridIdx++
				elapsed := time.Since(calibrationStart).Round(time.Second)
				slog.Info("benchmarking MUL_MAT reference curve",
					"progress", fmt.Sprintf("[%d/%d]", gridIdx, totalGrids),
					"M", 4096, "K", 4096, "weight_dtype", wdt, "elapsed", elapsed)

				gridStart := time.Now()
				refPoints := benchmarkMulMat(backend, caps, wdt, map[string]int64{"M": 4096, "K": 4096}, cfg)

				if len(refPoints) == 0 {
					slog.Warn("MUL_MAT reference curve produced no points", "weight_dtype", wdt)
					continue
				}

				refCurve := OperatorCurve{
					Op:           "MUL_MAT",
					ComputeDtype: wdt,
					WeightDtype:  wdt,
					Dimensions:   []string{"N"},
					FixedDims:    map[string]int64{"M": 4096, "K": 4096},
					Points:       refPoints,
				}
				devices := backend.BackendDevices()
				if len(devices) > 0 {
					refCurve.Backend = devices[0].Library
				}
				profile.Operators = append(profile.Operators, refCurve)

				// Extract per-dtype efficiency constants
				peakTOPS := hwResult.PeakTOPS["f32"]
				if tops, ok := hwResult.PeakTOPS[wdt]; ok {
					peakTOPS = tops
				}
				eff := extractEfficiencyConstants(refPoints, 4096, 4096, peakTOPS, hwResult.PeakBW, wdt)
				if profile.Hardware.EfficiencyConstants == nil {
					profile.Hardware.EfficiencyConstants = make(map[string]OpEfficiency)
				}
				effKey := "MUL_MAT_" + wdt
				profile.Hardware.EfficiencyConstants[effKey] = eff

				gridDuration := time.Since(gridStart).Round(time.Second)
				slog.Info("completed MUL_MAT reference",
					"progress", fmt.Sprintf("[%d/%d]", gridIdx, totalGrids),
					"weight_dtype", wdt, "points", len(refPoints),
					"eff_compute", fmt.Sprintf("%.2f", eff.ComputeEff),
					"eff_bw", fmt.Sprintf("%.2f", eff.BWEff),
					"grid_duration", gridDuration)
			}
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
					curve.Points = benchmarkFlashAttn(backend, caps, dtype, grid.FixedDims, cfg)
				default:
					curve.Points = benchmarkElementwise(backend, caps, op, dtype, cfg)
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

	// Step 3: Benchmark fused ops (if backend supports fusion)
	if len(caps.FusionRules) > 0 {
		fusedOps := []string{"RMS_NORM_MUL", "RMS_NORM_MUL_ROPE", "MUL_MAT_ADD"}
		for _, fop := range fusedOps {
			if _, ok := LookupRegistry(fop); !ok {
				continue
			}
			slog.Info("benchmarking fused op", "op", fop)

			switch fop {
			case "MUL_MAT_ADD":
				for _, wdt := range Phase1Dtypes() {
					points := benchmarkMulMat(backend, caps, wdt,
						map[string]int64{"M": 4096, "K": 4096}, cfg)
					var vecPoints []LatencyPoint
					for _, p := range points {
						if len(p.Shape) > 0 && p.Shape[0] <= 8 {
							vecPoints = append(vecPoints, p)
						}
					}
					if len(vecPoints) > 0 {
						profile.Operators = append(profile.Operators, OperatorCurve{
							Op:           fop,
							Backend:      caps.Name,
							ComputeDtype: "f32",
							WeightDtype:  wdt,
							Dimensions:   []string{"N"},
							FixedDims:    map[string]int64{"M": 4096, "K": 4096},
							Points:       vecPoints,
						})
					}
				}
			default:
				measure := func(shape []int64) LatencyPoint {
					return measureOpForBackend(backend, caps, fop, shape, "f32", cfg)
				}
				points := AdaptiveSample1D(measure, 1024, 8*1024*1024, 8, cfg)
				if len(points) > 0 {
					profile.Operators = append(profile.Operators, OperatorCurve{
						Op:           fop,
						Backend:      caps.Name,
						ComputeDtype: "f32",
						Dimensions:   []string{"N"},
						Points:       points,
					})
				}
			}
		}
	}

	// Step 4: Benchmark orchestration overhead (GPU backends only)
	if caps.HasGPUTimestamp {
		slog.Info("benchmarking orchestration overhead")
		ohPoints := benchOrchestrationOverhead(backend, cfg)
		if len(ohPoints) > 0 {
			profile.Operators = append(profile.Operators, OperatorCurve{
				Op:           "ORCHESTRATION_OVERHEAD",
				Backend:      caps.Name,
				ComputeDtype: "f32",
				Dimensions:   []string{"num_nodes"},
				Points:       ohPoints,
			})
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
		if op == "MUL_MAT" || op == "MUL_MAT_ADD" {
			total += len(Phase1Dtypes()) // one reference curve per weight dtype
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
func extractEfficiencyConstants(points []LatencyPoint, refM, refK int64, peakTOPS, peakBW float64, weightDtype string) OpEfficiency {
	var computeEffs, bwEffs []float64
	var overheads []float64

	wElemBytes := elemBytesFromDtype(weightDtype)

	for _, pt := range points {
		N := pt.Shape[0]
		if pt.LatencyUs <= 0 {
			continue
		}

		flops := 2.0 * float64(refM) * float64(refK) * float64(N)
		// Weight: M*K at weightDtype, activation: K*N at f32, output: M*N at f32
		bytes := float64(refM*refK)*wElemBytes + float64(refK*N+refM*N)*4

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
	return predictMulMatLatencyKeyed(hw, M, K, N, dtype, "MUL_MAT_"+dtype)
}

// PredictMulMatVecLatency computes MUL_MAT_VEC latency using the roofline model
// with VEC-specific efficiency constants. Returns 0 if no VEC constants are available.
func PredictMulMatVecLatency(hw *HardwareProfile, M, K, N int64, dtype string) float64 {
	return predictMulMatLatencyKeyed(hw, M, K, N, dtype, "MUL_MAT_VEC_"+dtype)
}

// predictMulMatLatencyKeyed is the shared roofline computation with a caller-specified
// efficiency constant key. The dtype parameter is used for TOPS and elem-bytes lookup.
func predictMulMatLatencyKeyed(hw *HardwareProfile, M, K, N int64, dtype, effKey string) float64 {
	eff, ok := hw.EfficiencyConstants[effKey]
	if !ok {
		eff, ok = hw.EfficiencyConstants["MUL_MAT"]
		if !ok {
			return 0
		}
	}
	peakTOPS, ok := hw.PeakTOPS[dtype]
	if !ok {
		// Fall back to f32 if dtype not measured
		peakTOPS = hw.PeakTOPS["f32"]
	}
	peakBW := hw.PeakBandwidthBytesPerSec

	flops := 2.0 * float64(M) * float64(K) * float64(N)
	wElemBytes := elemBytesFromDtype(dtype)
	// Weight: M*K at dtype, activation: K*N at f32, output: M*N at f32
	bytes := float64(M*K)*wElemBytes + float64(K*N+M*N)*4

	computeTime := flops / (eff.ComputeEff * peakTOPS) * 1e6 // microseconds
	bwTime := bytes / (eff.BWEff * peakBW) * 1e6              // microseconds

	return math.Max(computeTime, bwTime) + eff.OverheadUs
}

// elemBytesFromDtype returns bytes per element for a dtype string.
// Quantized types return fractional values based on block structure.
func elemBytesFromDtype(dtype string) float64 {
	switch dtype {
	case "f32":
		return 4.0
	case "f16":
		return 2.0
	case "q4_0":
		return 18.0 / 32.0 // 18 bytes per 32-element block = 0.5625
	case "q8_0":
		return 34.0 / 32.0 // 34 bytes per 32-element block = 1.0625
	default:
		return 4.0
	}
}

// sweepDimensions returns the sweep (non-fixed) dimensions for an op.
func sweepDimensions(op string) []string {
	switch op {
	case "MUL_MAT", "MUL_MAT_ADD":
		return []string{"N"}
	case "FLASH_ATTN_EXT":
		return []string{"seq_q", "seq_kv"}
	default:
		return []string{"N"}
	}
}

// benchmarkElementwise uses AdaptiveSample1D for 1D ops.
func benchmarkElementwise(backend ml.Backend, caps BackendCapabilities, op, dtype string, cfg BenchmarkConfig) []LatencyPoint {
	measure := func(shape []int64) LatencyPoint {
		return measureOpForBackend(backend, caps, op, shape, dtype, cfg)
	}
	return AdaptiveSample1D(measure, 1024, 8*1024*1024, 8, cfg)
}

// benchmarkMulMat uses AdaptiveSample1D over N with fixed (M, K).
// IMPORTANT: AdaptiveSample1D works with 1D shapes (Shape[0] is the sweep dimension).
// We keep shapes 1D during sampling, matching benchmarkFlashAttn's pattern.
func benchmarkMulMat(backend ml.Backend, caps BackendCapabilities, weightDtype string, fixedDims map[string]int64, cfg BenchmarkConfig) []LatencyPoint {
	M := fixedDims["M"]
	K := fixedDims["K"]
	measure := func(shape []int64) LatencyPoint {
		N := shape[0]
		pt := measureOpForBackend(backend, caps, "MUL_MAT", []int64{M, K, N}, weightDtype, cfg)
		pt.Shape = []int64{N} // 1D for AdaptiveSample1D
		return pt
	}
	return AdaptiveSample1D(measure, 1, 4096, 8, cfg)
}

// benchmarkFlashAttn samples two regimes: decode and prefill.
// IMPORTANT: AdaptiveSample1D works internally with 1D shapes (Shape[0] is the sweep dimension).
// The measure callbacks must NOT override pt.Shape to 2D during sampling, because
// AdaptiveSample1D uses Shape[0] for sorting and interpolation error analysis.
// We keep shapes 1D during sampling, then convert to 2D after sampling completes.
func benchmarkFlashAttn(backend ml.Backend, caps BackendCapabilities, dtype string, fixedDims map[string]int64, cfg BenchmarkConfig) []LatencyPoint {
	var points []LatencyPoint

	// Decode: seq_q=1, sweep seq_kv (1D: shape[0] = seq_kv)
	decodeMeasure := func(shape []int64) LatencyPoint {
		seqKV := shape[0]
		pt := measureOpForBackend(backend, caps, "FLASH_ATTN_EXT", []int64{1, seqKV}, dtype, cfg)
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
		pt := measureOpForBackend(backend, caps, "FLASH_ATTN_EXT", []int64{seqLen, seqLen}, dtype, cfg)
		// Keep shape 1D for AdaptiveSample1D
		pt.Shape = []int64{seqLen}
		return pt
	}
	prefillPts := AdaptiveSample1D(prefillMeasure, 64, 2048, 8, cfg)
	// Convert to 2D after sampling: [seq_len] → [seq_len, seq_len]
	for i := range prefillPts {
		seqLen := prefillPts[i].Shape[0]
		prefillPts[i].Shape = []int64{seqLen, seqLen}
	}
	points = append(points, prefillPts...)

	return points
}
