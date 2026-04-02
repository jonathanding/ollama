package perf

import (
	"fmt"
	"log/slog"
	"math"
	"time"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
)

// OpSpec defines an operator to benchmark with its dtype combinations.
type OpSpec struct {
	Op     string
	Dtypes []string
}

// PredefinedOps returns the Layer 1 predefined operator set.
func PredefinedOps() []OpSpec {
	return []OpSpec{
		{Op: "MUL_MAT", Dtypes: []string{"f16", "f32", "bf16", "q4_0", "q4_K", "q5_K", "q6_K", "q8_0"}},
		{Op: "FLASH_ATTN_EXT", Dtypes: []string{"f16", "bf16"}},
		{Op: "RMS_NORM", Dtypes: []string{"f32"}},
		{Op: "SOFTMAX", Dtypes: []string{"f32"}},
		{Op: "SILU", Dtypes: []string{"f32"}},
		{Op: "GELU", Dtypes: []string{"f32"}},
		{Op: "ADD", Dtypes: []string{"f32"}},
		{Op: "MUL", Dtypes: []string{"f32"}},
		{Op: "ROPE", Dtypes: []string{"f32", "f16"}},
		{Op: "GET_ROWS", Dtypes: []string{"f16", "q4_0", "q8_0"}},
		{Op: "CONT", Dtypes: []string{"f32", "f16"}},
	}
}

// SelectBenchmarkShapes returns 5 shape configurations for benchmarking an op.
func SelectBenchmarkShapes(op string, balancePoint float64, computeDtype, weightDtype string) [][][]int64 {
	K := int64(4096)

	if op == "MUL_MAT" || op == "MUL_MAT_ID" {
		Ns := []int64{1, 32, 256, 1024, 4096}
		shapes := make([][][]int64, len(Ns))
		for i, N := range Ns {
			shapes[i] = [][]int64{{K, K}, {K, N}}
		}
		return shapes
	}

	if op == "FLASH_ATTN_EXT" {
		seqLens := []int64{1, 64, 256, 512, 2048}
		shapes := make([][][]int64, len(seqLens))
		for i, S := range seqLens {
			shapes[i] = [][]int64{{1, 32, S, 128}, {1, 32, S, 128}}
		}
		return shapes
	}

	// Fixed-intensity ops: vary tensor size
	sizes := []int64{1024, 65536, 1048576, 16777216, 67108864}
	shapes := make([][][]int64, len(sizes))
	for i, size := range sizes {
		shapes[i] = [][]int64{{size}}
	}
	return shapes
}

// ShouldAdaptiveExtend returns true if η values have high variance (CV > 10%).
func ShouldAdaptiveExtend(etas []float64) bool {
	if len(etas) < 3 {
		return false
	}
	mean := 0.0
	for _, e := range etas {
		mean += e
	}
	mean /= float64(len(etas))
	if mean == 0 {
		return false
	}

	variance := 0.0
	for _, e := range etas {
		d := e - mean
		variance += d * d
	}
	variance /= float64(len(etas))
	cv := math.Sqrt(variance) / mean
	return cv > 0.10
}

// BenchmarkConfig controls benchmark behavior.
type BenchmarkConfig struct {
	Backends    []string
	WarmupReps  int
	MeasureReps int
	MaxAdaptive int
}

// DefaultBenchmarkConfig returns sensible defaults.
func DefaultBenchmarkConfig() BenchmarkConfig {
	return BenchmarkConfig{
		WarmupReps:  3,
		MeasureReps: 50,
		MaxAdaptive: 5,
	}
}

// buildHWProfile extracts backend profiles from hardware benchmarks.
func buildHWProfile(hbs []HardwareBenchmark) []BackendProfile {
	byBackend := make(map[string]*BackendProfile)
	for _, hb := range hbs {
		bp, ok := byBackend[hb.Backend]
		if !ok {
			bp = &BackendProfile{
				Name:          hb.Backend,
				PeakFLOPS:     make(map[string]float64),
				BalancePoints: make(map[string]float64),
			}
			byBackend[hb.Backend] = bp
		}
		switch hb.Test {
		case "peak_flops":
			bp.PeakFLOPS[hb.Dtype] = hb.Value
		case "peak_bandwidth":
			bp.PeakBandwidth = hb.Value
		}
	}
	var result []BackendProfile
	for _, bp := range byBackend {
		for dtype, flops := range bp.PeakFLOPS {
			if bp.PeakBandwidth > 0 {
				bp.BalancePoints[dtype] = flops / bp.PeakBandwidth
			}
		}
		result = append(result, *bp)
	}
	return result
}

// computePointEtas computes per-point η values from benchmark results.
func computePointEtas(points []BenchmarkPoint, bp BackendProfile, dtype string) []float64 {
	peakFLOPS := bp.PeakFLOPS[dtype]
	if peakFLOPS == 0 {
		peakFLOPS = bp.PeakFLOPS["f32"]
	}
	peakBW := bp.PeakBandwidth
	etas := make([]float64, 0, len(points))
	for _, pt := range points {
		tComp := pt.FLOPs / peakFLOPS
		tMem := pt.BytesMoved / peakBW
		tPred := math.Max(tComp, tMem)
		tMeas := pt.LatencyUs * 1e-6
		if tMeas > 0 && tPred > 0 {
			eta := tPred / tMeas
			if eta > 0 && eta <= 2.0 {
				etas = append(etas, eta)
			}
		}
	}
	return etas
}

// interpolateShapes picks a midpoint shape for adaptive extension.
func interpolateShapes(shapes [][][]int64, etas []float64) [][]int64 {
	if len(shapes) < 2 {
		return shapes[0]
	}
	return shapes[len(shapes)/2]
}

// parseDType converts string dtype to ml.DType.
func parseDType(s string) ml.DType {
	switch s {
	case "f16":
		return ml.DTypeF16
	case "f32":
		return ml.DTypeF32
	default:
		return ml.DTypeF32
	}
}

// dtypeToString converts ml.DType to string.
func dtypeToString(dt ml.DType) string {
	switch dt {
	case ml.DTypeF16:
		return "f16"
	case ml.DTypeF32:
		return "f32"
	default:
		return "unknown"
	}
}

func findBackendProfile(profiles []BackendProfile, name string) *BackendProfile {
	for i := range profiles {
		if profiles[i].Name == name {
			return &profiles[i]
		}
	}
	return nil
}

// --- Runtime benchmark functions (require CGo GGML backend) ---

// RunFullBenchmark executes the complete Layer 1 benchmark.
func RunFullBenchmark(backend ml.Backend, cfg BenchmarkConfig) (*RawData, error) {
	raw := &RawData{
		Version:   1,
		Timestamp: time.Now(),
	}

	devices := backend.BackendDevices()
	if len(devices) == 0 {
		return nil, fmt.Errorf("no backend devices found")
	}

	for _, dev := range devices {
		raw.Hardware.Backends = append(raw.Hardware.Backends, RawBackendInfo{
			Name:   dev.Library,
			Device: dev.Name,
		})
	}

	slog.Info("starting hardware characterization", "devices", len(devices))

	dtypes := []ml.DType{ml.DTypeF16, ml.DTypeF32}
	for _, dev := range devices {
		for _, dtype := range dtypes {
			flops, err := benchPeakFLOPS(backend, dtype, cfg)
			if err != nil {
				slog.Warn("peak FLOPS benchmark failed", "device", dev.Name, "dtype", dtype, "error", err)
				continue
			}
			raw.HardwareBenchmarks = append(raw.HardwareBenchmarks, HardwareBenchmark{
				Backend: dev.Library, Dtype: dtypeToString(dtype), Test: "peak_flops", Value: flops, Unit: "FLOPS",
			})
		}
		bw, err := benchPeakBandwidth(backend, cfg)
		if err != nil {
			slog.Warn("peak bandwidth benchmark failed", "device", dev.Name, "error", err)
			continue
		}
		raw.HardwareBenchmarks = append(raw.HardwareBenchmarks, HardwareBenchmark{
			Backend: dev.Library, Test: "peak_bandwidth", Value: bw, Unit: "bytes/sec",
		})
	}

	slog.Info("starting operator calibration (Layer 1)")

	hwProfile := buildHWProfile(raw.HardwareBenchmarks)
	for _, opSpec := range PredefinedOps() {
		for _, dtypeStr := range opSpec.Dtypes {
			for _, bp := range hwProfile {
				balancePoint := bp.BalancePoints["f32"]
				if v, ok := bp.BalancePoints[dtypeStr]; ok {
					balancePoint = v
				}
				shapes := SelectBenchmarkShapes(opSpec.Op, balancePoint, dtypeStr, dtypeStr)
				ob := OperatorBenchmark{
					Op: opSpec.Op, Backend: bp.Name, ComputeDtype: dtypeStr,
				}
				if opSpec.Op == "MUL_MAT" || opSpec.Op == "MUL_MAT_ID" {
					ob.WeightDtype = dtypeStr
				}
				for _, shape := range shapes {
					pt, err := benchSingleOp(backend, opSpec.Op, shape, dtypeStr, cfg)
					if err != nil {
						slog.Warn("op benchmark failed", "op", opSpec.Op, "error", err)
						continue
					}
					ob.Points = append(ob.Points, pt)
				}
				if len(ob.Points) >= 3 {
					etas := computePointEtas(ob.Points, bp, dtypeStr)
					if ShouldAdaptiveExtend(etas) {
						for extra := 0; extra < cfg.MaxAdaptive && len(ob.Points) < 10; extra++ {
							midShape := interpolateShapes(shapes, etas)
							pt, err := benchSingleOp(backend, opSpec.Op, midShape, dtypeStr, cfg)
							if err != nil {
								break
							}
							ob.Points = append(ob.Points, pt)
						}
					}
				}
				raw.OperatorBenchmarks = append(raw.OperatorBenchmarks, ob)
			}
		}
	}

	return raw, nil
}

// benchPeakFLOPS measures peak FLOPS via large MUL_MAT.
func benchPeakFLOPS(backend ml.Backend, dtype ml.DType, cfg BenchmarkConfig) (float64, error) {
	const M, K, N = 4096, 4096, 4096
	ctx := backend.NewContext()
	defer ctx.Close()

	a := ctx.Zeros(dtype, M, K)
	b := ctx.Zeros(dtype, K, N)
	out := a.Mulmat(ctx, b)
	ctx.Forward(out)

	for i := 0; i < cfg.WarmupReps; i++ {
		ctx.Compute(out)
	}
	start := time.Now()
	for i := 0; i < cfg.MeasureReps; i++ {
		ctx.Compute(out)
	}
	elapsed := time.Since(start)

	latencySec := elapsed.Seconds() / float64(cfg.MeasureReps)
	flops := 2.0 * M * K * N
	return flops / latencySec, nil
}

// benchPeakBandwidth measures peak memory bandwidth via large CONT (copy).
func benchPeakBandwidth(backend ml.Backend, cfg BenchmarkConfig) (float64, error) {
	const size = 64 * 1024 * 1024
	ctx := backend.NewContext()
	defer ctx.Close()

	src := ctx.Zeros(ml.DTypeF32, size)
	dst := src.Contiguous(ctx)
	ctx.Forward(dst)

	for i := 0; i < cfg.WarmupReps; i++ {
		ctx.Compute(dst)
	}
	start := time.Now()
	for i := 0; i < cfg.MeasureReps; i++ {
		ctx.Compute(dst)
	}
	elapsed := time.Since(start)

	latencySec := elapsed.Seconds() / float64(cfg.MeasureReps)
	bytesTotal := 2.0 * size * 4
	return bytesTotal / latencySec, nil
}

// benchSingleOp benchmarks one op at one shape.
func benchSingleOp(backend ml.Backend, op string, shapes [][]int64, dtype string, cfg BenchmarkConfig) (BenchmarkPoint, error) {
	ctx := backend.NewContext()
	defer ctx.Close()

	dt := parseDType(dtype)

	var out ml.Tensor
	switch op {
	case "MUL_MAT":
		a := ctx.Zeros(dt, int(shapes[0][0]), int(shapes[0][1]))
		b := ctx.Zeros(dt, int(shapes[1][0]), int(shapes[1][1]))
		out = a.Mulmat(ctx, b)
	case "ADD":
		a := ctx.Zeros(dt, int(shapes[0][0]))
		b := ctx.Zeros(dt, int(shapes[0][0]))
		out = a.Add(ctx, b)
	case "SILU":
		a := ctx.Zeros(dt, int(shapes[0][0]))
		out = a.SILU(ctx)
	case "GELU":
		a := ctx.Zeros(dt, int(shapes[0][0]))
		out = a.GELU(ctx)
	case "RMS_NORM":
		a := ctx.Zeros(dt, int(shapes[0][0]))
		w := ctx.Zeros(dt, int(shapes[0][0]))
		out = a.RMSNorm(ctx, w, 1e-5)
	case "SOFTMAX":
		a := ctx.Zeros(dt, int(shapes[0][0]))
		out = a.Softmax(ctx)
	case "CONT":
		a := ctx.Zeros(dt, int(shapes[0][0]))
		out = a.Contiguous(ctx)
	case "MUL":
		a := ctx.Zeros(dt, int(shapes[0][0]))
		b := ctx.Zeros(dt, int(shapes[0][0]))
		out = a.Mul(ctx, b)
	default:
		return BenchmarkPoint{}, fmt.Errorf("benchmark not implemented for op %s", op)
	}

	ctx.Forward(out)

	for i := 0; i < cfg.WarmupReps; i++ {
		ctx.Compute(out)
	}

	var latencies []float64
	for i := 0; i < cfg.MeasureReps; i++ {
		start := time.Now()
		ctx.Compute(out)
		latencies = append(latencies, float64(time.Since(start).Microseconds()))
	}

	mean := 0.0
	for _, l := range latencies {
		mean += l
	}
	mean /= float64(len(latencies))

	stddev := 0.0
	for _, l := range latencies {
		d := l - mean
		stddev += d * d
	}
	stddev = math.Sqrt(stddev / float64(len(latencies)))

	flops := ComputeFLOPs(op, shapes)
	bytes := ComputeBytes(op, shapes, dtype, dtype)
	intensity := 0.0
	if bytes > 0 {
		intensity = flops / bytes
	}

	return BenchmarkPoint{
		InputShapes: shapes,
		OutputShape: shapes[0],
		FLOPs:       flops,
		BytesMoved:  bytes,
		Intensity:   intensity,
		LatencyUs:   mean,
		Reps:        cfg.MeasureReps,
		StddevUs:    stddev,
	}, nil
}

// RunUpdateBenchmark executes Layer 2 graph-driven discovery.
func RunUpdateBenchmark(backend ml.Backend, existingProfile *Profile,
	modelPaths []string, cfg BenchmarkConfig) (*RawData, error) {

	raw := &RawData{
		Version:   1,
		Timestamp: time.Now(),
	}

	needed := make(map[OpKey]bool)
	for _, modelPath := range modelPaths {
		nodes, err := buildModelGraphNodes(modelPath)
		if err != nil {
			slog.Warn("failed to build graph for model", "path", modelPath, "error", err)
			continue
		}
		for _, node := range nodes {
			if IsZeroCostOp(node.Op) {
				continue
			}
			needed[OpKey{node.Op, node.Backend, node.ComputeDtype, node.WeightDtype}] = true
		}
	}

	for _, op := range existingProfile.Operators {
		delete(needed, OpKey{op.Op, op.Backend, op.ComputeDtype, op.WeightDtype})
	}

	if len(needed) == 0 {
		slog.Info("all operators already calibrated")
		return raw, nil
	}

	slog.Info("operator calibration (Layer 2)", "missing_ops", len(needed))

	for key := range needed {
		bp2, _ := LookupBackend(existingProfile, key.Backend)
		if bp2 == nil {
			continue
		}
		balancePoint := bp2.BalancePoints[key.ComputeDtype]
		shapes := SelectBenchmarkShapes(key.Op, balancePoint, key.ComputeDtype, key.WeightDtype)
		ob := OperatorBenchmark{
			Op: key.Op, Backend: key.Backend,
			ComputeDtype: key.ComputeDtype, WeightDtype: key.WeightDtype,
		}
		for _, shape := range shapes {
			pt, err := benchSingleOp(backend, key.Op, shape, key.ComputeDtype, cfg)
			if err != nil {
				continue
			}
			ob.Points = append(ob.Points, pt)
		}
		if len(ob.Points) > 0 {
			raw.OperatorBenchmarks = append(raw.OperatorBenchmarks, ob)
		}
	}

	return raw, nil
}

// buildModelGraphNodes loads a model, builds prefill+decode graphs, returns all graph nodes.
func buildModelGraphNodes(modelPath string) ([]ml.GraphNode, error) {
	m, err := model.New(modelPath, ml.BackendParams{})
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	defer m.Backend().Close()

	// TODO: implement graph building once model.Forward() batch API is validated
	// For now, return empty — this will be filled when estimate.go is implemented
	_ = m
	return nil, fmt.Errorf("buildModelGraphNodes not yet implemented")
}
