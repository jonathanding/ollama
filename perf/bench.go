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

// measureOp benchmarks an operator at one shape point using wall-clock timing.
// backendIdx specifies which backend to execute on (-1 = use scheduler).
func measureOp(backend ml.Backend, op string, gridPoint []int64, computeDtype string, cfg BenchmarkConfig, backendIdx int) LatencyPoint {
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

	cache := make(BytesCache)

	ctx := backend.NewContext()
	defer ctx.Close()

	// Create input tensors — use custom CreateInputs if provided, else default
	var inputs []ml.Tensor
	if runner.CreateInputs != nil {
		inputs = runner.CreateInputs(ctx, backend, computeDtype, gridPoint)
	} else {
		tensorShapes := expandShapes(op, gridPoint)
		inputs = make([]ml.Tensor, len(tensorShapes))
		for i, shape := range tensorShapes {
			intShape := make([]int, len(shape))
			for j, s := range shape {
				intShape[j] = int(s)
			}
			inputs[i] = randomLeafTensor(ctx, backend, cache, dt, intShape...)
		}
	}

	// Build computation graph
	out := runner.Run(ctx, inputs)
	if out == nil {
		slog.Warn("op runner returned nil", "op", op)
		return LatencyPoint{Shape: gridPoint}
	}
	ctx.Forward(out)

	// Choose compute function based on backendIdx
	compute := func() { ctx.Compute(out) }
	if backendIdx >= 0 {
		compute = func() { ctx.ComputeOnBackend(backendIdx, out) }
	}

	// Adaptive warmup: reduce for slow ops to avoid wasting minutes
	warmupStart := time.Now()
	for i := 0; i < cfg.WarmupReps; i++ {
		compute()
		// After first warmup, if it took >1s, reduce remaining warmups
		if i == 0 {
			elapsed := float64(time.Since(warmupStart).Microseconds())
			if elapsed > 5e6 {
				// >5s per op: 1 warmup is enough
				break
			} else if elapsed > 1e6 {
				// >1s per op: 2 warmups total
				compute()
				break
			}
		}
	}

	// Measure with convergence-based early stopping
	med, sd, actualReps := convergentMeasure(func() float64 {
		start := time.Now()
		compute()
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

	cache := make(BytesCache)

	backend.EnableGPUTimestamps(true)
	defer backend.EnableGPUTimestamps(false)

	ctx := backend.NewContext()
	defer ctx.Close()

	// Create input tensors — use custom CreateInputs if provided, else default
	var inputs []ml.Tensor
	if runner.CreateInputs != nil {
		inputs = runner.CreateInputs(ctx, backend, computeDtype, gridPoint)
	} else {
		tensorShapes := expandShapes(op, gridPoint)
		inputs = make([]ml.Tensor, len(tensorShapes))
		for i, shape := range tensorShapes {
			intShape := make([]int, len(shape))
			for j, s := range shape {
				intShape[j] = int(s)
			}
			inputs[i] = randomLeafTensor(ctx, backend, cache, dt, intShape...)
		}
	}

	// Build computation graph
	out := runner.Run(ctx, inputs)
	if out == nil {
		slog.Warn("op runner returned nil", "op", op)
		return LatencyPoint{Shape: gridPoint}
	}
	ctx.Forward(out)

	// Warmup — use direct backend execution
	for range 2 {
		ctx.ComputeOnBackend(0, out)
	}

	// Measure using GPU timestamps
	samples := make([]float64, 0, cfg.MeasureReps)
	for i := 0; i < cfg.MeasureReps; i++ {
		ctx.ComputeOnBackend(0, out)
		timings := backend.GetOpTimings()
		if len(timings) == 0 {
			slog.Warn("no GPU timings returned, falling back to wall-clock",
				"op", op, "shape", gridPoint)
			return measureOp(backend, op, gridPoint, computeDtype, cfg, 0)
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
	return measureOp(backend, op, gridPoint, computeDtype, cfg, 0)
}

// benchOrchestrationOverhead measures CPU orchestration overhead for different graph sizes.
// Builds N chained trivial ops (16-element ADD), GPU compute ≈ 0, wall-clock ≈ CPU overhead.
// Used by estimate to add CPU-side cost to GPU op time sum.
// IMPORTANT: This intentionally uses ctx.Compute (scheduler path) because we're measuring
// the scheduler's dispatch overhead itself, not GPU kernel time.
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

// RunBenchmark executes the full v3 calibration pipeline using a work plan pattern:
// buildBenchmarkPlan produces a flat list of BenchmarkStep entries, then this function
// iterates the list uniformly with a single loop — no scattered conditionals.
// Returns a complete Profile ready for estimation.
func RunBenchmark(backend ml.Backend, ops []string, dtypes []string, cfg BenchmarkConfig) (*Profile, error) {
	benchStart := time.Now()

	caps := DiscoverBackend(backend)
	slog.Info("backend capabilities", "name", caps.Name,
		"gpu_timestamp", caps.HasGPUTimestamp, "fusion_rules", len(caps.FusionRules),
		"mul_mat_vec", caps.HasMulMatVec)

	plan := buildBenchmarkPlan(ops, dtypes, caps, cfg)
	slog.Info("benchmark plan", "steps", len(plan))

	profile := &Profile{
		Version:   3,
		Timestamp: time.Now(),
		BackendCaps: map[string]BackendCapabilitiesJSON{
			caps.Name: caps.ToJSON(),
		},
	}

	for i, step := range plan {
		elapsed := time.Since(benchStart).Round(time.Second)
		progress := fmt.Sprintf("[%d/%d]", i+1, len(plan))

		switch step.Type {
		case StepHWChar:
			slog.Info("hardware characterization", "progress", progress, "elapsed", elapsed)
			hwStart := time.Now()
			hwResult, err := CharacterizeHardware(backend, cfg)
			if err != nil {
				return nil, fmt.Errorf("hardware characterization: %w", err)
			}
			profile.Hardware = HWCharResultToHardwareProfile(hwResult, backend)
			slog.Info("hardware characterization complete",
				"progress", progress, "duration", time.Since(hwStart).Round(time.Second))

		case StepMulMatRef:
			slog.Info("benchmarking MUL_MAT", "progress", progress,
				"weight_dtype", step.WeightDtype, "M", step.FixedDims["M"], "K", step.FixedDims["K"],
				"elapsed", elapsed)
			gridStart := time.Now()

			refPoints := benchmarkMulMat(backend, caps, step.WeightDtype, step.FixedDims, cfg)
			if len(refPoints) == 0 {
				slog.Warn("MUL_MAT reference curve produced no points", "weight_dtype", step.WeightDtype)
				continue
			}

			refCurve := OperatorCurve{
				Op: "MUL_MAT", ComputeDtype: step.WeightDtype, WeightDtype: step.WeightDtype,
				Dimensions: []string{"N"}, FixedDims: step.FixedDims, Points: refPoints,
			}
			if devices := backend.BackendDevices(); len(devices) > 0 {
				refCurve.Backend = devices[0].Library
			}
			profile.Operators = append(profile.Operators, refCurve)

			slog.Info("completed MUL_MAT reference", "progress", progress,
				"weight_dtype", step.WeightDtype, "points", len(refPoints),
				"duration", time.Since(gridStart).Round(time.Second))

		case StepOperator:
			slog.Info("benchmarking", "progress", progress,
				"op", step.Op, "dtype", step.Dtype, "fixed", step.FixedDims, "elapsed", elapsed)
			gridStart := time.Now()

			var curve OperatorCurve
			curve.Op = step.Op
			curve.ComputeDtype = step.Dtype
			curve.Dimensions = sweepDimensions(step.Op)
			curve.FixedDims = step.FixedDims
			if devices := backend.BackendDevices(); len(devices) > 0 {
				curve.Backend = devices[0].Library
			}

			switch step.Op {
			case "FLASH_ATTN_EXT":
				curve.Points = benchmarkFlashAttn(backend, caps, step.Dtype, step.FixedDims, cfg)
			default:
				curve.Points = benchmarkElementwise(backend, caps, step.Op, step.Dtype, cfg)
			}

			if len(curve.Points) > 0 {
				slog.Info("completed", "progress", progress,
					"op", step.Op, "dtype", step.Dtype, "points", len(curve.Points),
					"duration", time.Since(gridStart).Round(time.Second))
				profile.Operators = append(profile.Operators, curve)
			} else {
				slog.Warn("no points collected", "op", step.Op, "dtype", step.Dtype)
			}

		case StepFusedOp:
			slog.Info("benchmarking fused op", "progress", progress,
				"op", step.Op, "weight_dtype", step.WeightDtype, "elapsed", elapsed)

			switch step.Op {
			case "MUL_MAT_ADD":
				points := benchmarkMulMat(backend, caps, step.WeightDtype, step.FixedDims, cfg)
				var vecPoints []LatencyPoint
				for _, p := range points {
					if len(p.Shape) > 0 && p.Shape[0] <= 8 {
						vecPoints = append(vecPoints, p)
					}
				}
				if len(vecPoints) > 0 {
					profile.Operators = append(profile.Operators, OperatorCurve{
						Op: step.Op, Backend: caps.Name, ComputeDtype: "f32",
						WeightDtype: step.WeightDtype, Dimensions: []string{"N"},
						FixedDims: step.FixedDims, Points: vecPoints,
					})
				}
			default:
				measure := func(shape []int64) LatencyPoint {
					return measureOpForBackend(backend, caps, step.Op, shape, "f32", cfg)
				}
				points := AdaptiveSample1D(measure, 1024, 8*1024*1024, 8, cfg)
				if len(points) > 0 {
					profile.Operators = append(profile.Operators, OperatorCurve{
						Op: step.Op, Backend: caps.Name, ComputeDtype: "f32",
						Dimensions: []string{"N"}, Points: points,
					})
				}
			}

		case StepOverhead:
			slog.Info("benchmarking orchestration overhead", "progress", progress, "elapsed", elapsed)
			ohPoints := benchOrchestrationOverhead(backend, cfg)
			if len(ohPoints) > 0 {
				profile.Operators = append(profile.Operators, OperatorCurve{
					Op: "ORCHESTRATION_OVERHEAD", Backend: caps.Name,
					ComputeDtype: "f32", Dimensions: []string{"num_nodes"},
					Points: ohPoints,
				})
			}
		}
	}

	totalDuration := time.Since(benchStart).Round(time.Second)
	slog.Info("calibration complete", "operators", len(profile.Operators), "total_duration", totalDuration)

	return profile, nil
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

// PredictMulMatDirect predicts MUL_MAT latency by directly interpolating from
// measured reference curves, bypassing the roofline model entirely.
// Uses inverse distance weighting in (M,K) space between reference curves.
// This is more accurate than roofline for VEC shaders (N≤8) where the roofline
// model fundamentally cannot capture dequantization, memory latency, and warp
// scheduling effects. Returns 0 if no reference curves exist for this dtype.
func PredictMulMatDirect(profile *Profile, M, K, N int64, weightDtype string) float64 {
	var curves []OperatorCurve
	for _, c := range profile.Operators {
		if c.Op == "MUL_MAT" && c.WeightDtype == weightDtype && c.FixedDims != nil {
			curves = append(curves, c)
		}
	}
	if len(curves) == 0 {
		return 0
	}
	return InterpolateMulMat(curves, M, K, N)
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

	// Don't add per-op OverheadUs here. The overhead from the reference curve fit
	// is a benchmark artifact (per-invocation context/sync cost), not a real per-op
	// cost in inference. Real dispatch overhead is async and pipelined — accounted
	// for once via ORCHESTRATION_OVERHEAD in estimatePhaseV3.
	return math.Max(computeTime, bwTime)
}

// elemBytesFromDtype returns bytes per element for a dtype string.
// Quantized types return fractional values based on block structure.
func elemBytesFromDtype(dtype string) float64 {
	switch dtype {
	case "f32":
		return 4.0
	case "f16":
		return 2.0
	case "q4_0", "q4_K":
		return 18.0 / 32.0 // q4_0: 18B/32elem; q4_K: 144B/256elem; both = 0.5625 B/elem
	case "q8_0":
		return 34.0 / 32.0 // 34 bytes per 32-element block = 1.0625
	case "q6_K":
		return 210.0 / 256.0 // 210 bytes per 256-element super-block = 0.8203
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

// strategicNcross is the fixed N value for the BW→compute transition zone.
// Empirically chosen: large enough to be past kernel launch overhead,
// small enough to still show BW effects. The roofline formula
// (peak_tops * elemBytes / (2 * peak_bw)) gives values too small (~3)
// because it ignores kernel overhead and partial overlap.
const strategicNcross = 32

// benchmarkMulMat measures MUL_MAT latency at 3 strategic N values per (M,K) grid point.
// N=1 (decode/BW-bound), N=32 (transition zone), N=512 (prefill/compute-bound).
// Each "curve" has exactly 3 points — sufficient for piecewise linear interpolation in log-N.
func benchmarkMulMat(backend ml.Backend, caps BackendCapabilities, weightDtype string, fixedDims map[string]int64, cfg BenchmarkConfig) []LatencyPoint {
	M := fixedDims["M"]
	K := fixedDims["K"]

	strategicNs := []int64{1, strategicNcross, 512}

	var points []LatencyPoint
	for _, N := range strategicNs {
		pt := measureOpForBackend(backend, caps, "MUL_MAT", []int64{M, K, N}, weightDtype, cfg)
		pt.Shape = []int64{N} // 1D for InterpolateMulMat compatibility
		points = append(points, pt)
	}

	return points
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
