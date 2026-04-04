package perf

import (
	"fmt"
	"log/slog"
	"math"
	"sort"
	"time"

	"github.com/ollama/ollama/ml"
)

// CharacterizeHardware measures peak TOPS and bandwidth for all backend devices.
// Returns an HWCharResult and populates the given HardwareProfile.
func CharacterizeHardware(backend ml.Backend, cfg BenchmarkConfig) (*HWCharResult, error) {
	devices := backend.BackendDevices()

	result := &HWCharResult{
		PeakTOPS:     make(map[string]float64),
		BalancePoint: make(map[string]float64),
	}

	slog.Info("hardware characterization", "devices", len(devices))

	// Measure peak TOPS for f16 and f32
	for _, dtypeStr := range []string{"f16", "f32"} {
		dt, ok := parseDType(dtypeStr)
		if !ok {
			continue
		}
		slog.Info("measuring peak TOPS", "dtype", dtypeStr)
		tops, err := benchPeakTOPS(backend, dt, cfg)
		if err != nil {
			slog.Warn("peak TOPS failed", "dtype", dtypeStr, "error", err)
			continue
		}
		result.PeakTOPS[dtypeStr] = tops
		slog.Info("peak TOPS", "dtype", dtypeStr, "TOPS", tops)
	}

	// Measure peak bandwidth
	slog.Info("measuring peak bandwidth")
	bw, err := benchPeakBandwidth(backend, cfg)
	if err != nil {
		return nil, fmt.Errorf("peak bandwidth failed: %w", err)
	}
	result.PeakBW = bw
	slog.Info("peak bandwidth", "bytes_per_sec", bw)

	// Compute balance points
	for dtype, tops := range result.PeakTOPS {
		if result.PeakBW > 0 {
			result.BalancePoint[dtype] = tops / result.PeakBW
		}
	}

	return result, nil
}

// benchPeakTOPS measures peak TOPS via large MUL_MAT (M=K=N=4096).
// TOPS = FLOPs / latency, where FLOPs = 2 * M * K * N.
func benchPeakTOPS(backend ml.Backend, dtype ml.DType, cfg BenchmarkConfig) (float64, error) {
	const M, K, N = 4096, 4096, 4096
	ctx := backend.NewContext()
	defer ctx.Close()

	a := ctx.Input().Zeros(dtype, M, K)
	b := ctx.Input().Zeros(dtype, K, N)
	out := a.Mulmat(ctx, b)
	ctx.Forward(out)

	// Warmup
	for i := 0; i < cfg.WarmupReps; i++ {
		ctx.Compute(out)
	}

	// Measure with trimming
	latencies := make([]float64, cfg.MeasureReps)
	for i := 0; i < cfg.MeasureReps; i++ {
		start := time.Now()
		ctx.Compute(out)
		latencies[i] = time.Since(start).Seconds()
	}

	median := trimmedMedian(latencies, cfg.TrimPercent)
	flops := 2.0 * M * K * N
	return flops / median, nil
}

// benchPeakBandwidth measures peak memory bandwidth via large CONT (copy).
// Size: 64M elements * 4 bytes = 256MB. Bytes = 2 * 256MB (read + write).
func benchPeakBandwidth(backend ml.Backend, cfg BenchmarkConfig) (float64, error) {
	const size = 64 * 1024 * 1024 // 64M elements
	ctx := backend.NewContext()
	defer ctx.Close()

	src := ctx.Input().Zeros(ml.DTypeF32, size)
	dst := src.Contiguous(ctx)
	ctx.Forward(dst)

	// Warmup
	for i := 0; i < cfg.WarmupReps; i++ {
		ctx.Compute(dst)
	}

	// Measure with trimming
	latencies := make([]float64, cfg.MeasureReps)
	for i := 0; i < cfg.MeasureReps; i++ {
		start := time.Now()
		ctx.Compute(dst)
		latencies[i] = time.Since(start).Seconds()
	}

	median := trimmedMedian(latencies, cfg.TrimPercent)
	bytesTotal := 2.0 * size * 4 // read + write, 4 bytes per f32
	return bytesTotal / median, nil
}

// convergentMeasure runs repeated measurements with convergence-based early stopping.
// The compute callback should perform one timed operation and return latency in microseconds.
// Returns (trimmedMedian, trimmedStddev, actualReps).
//
// Algorithm:
//  1. Collect MinReps samples
//  2. After MinReps, set tiered maxReps based on median latency
//  3. After each additional sample, check CV on trimmed data
//  4. Stop early if CV < ConvergenceCV, or when maxReps reached
func convergentMeasure(compute func() float64, cfg BenchmarkConfig) (median, stddev float64, reps int) {
	maxReps := cfg.MeasureReps
	latencies := make([]float64, 0, maxReps)

	for i := 0; i < maxReps; i++ {
		lat := compute()
		latencies = append(latencies, lat)

		// After MinReps samples, apply tiered ceiling and check convergence
		if i+1 >= cfg.MinReps {
			// On first eligibility check, set tiered max reps
			if i+1 == cfg.MinReps {
				med := trimmedMedian(latencies, 0)
				switch {
				case med > 5e6:
					maxReps = cfg.MinReps // cap at MinReps for very slow ops
				case med > 1e6:
					if 10 < maxReps {
						maxReps = 10
					}
				case med > 1e5:
					if 20 < maxReps {
						maxReps = 20
					}
				}
			}

			// Check convergence: CV on trimmed data
			sorted := make([]float64, len(latencies))
			copy(sorted, latencies)
			sort.Float64s(sorted)
			trimCount := int(math.Round(float64(len(sorted)) * cfg.TrimPercent))
			if trimCount*2 >= len(sorted) {
				trimCount = 0
			}
			trimmed := sorted[trimCount : len(sorted)-trimCount]

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
			sd := math.Sqrt(variance / float64(len(trimmed)))

			if mean > 0 && sd/mean < cfg.ConvergenceCV {
				return trimmedMedian(latencies, cfg.TrimPercent), sd, i + 1
			}
		}
	}

	// Did not converge — return trimmed stats anyway
	sorted := make([]float64, len(latencies))
	copy(sorted, latencies)
	sort.Float64s(sorted)
	trimCount := int(math.Round(float64(len(sorted)) * cfg.TrimPercent))
	if trimCount*2 >= len(sorted) {
		trimCount = 0
	}
	trimmed := sorted[trimCount : len(sorted)-trimCount]
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
	sd := math.Sqrt(variance / float64(len(trimmed)))

	return trimmedMedian(latencies, cfg.TrimPercent), sd, len(latencies)
}

// trimmedMedian sorts the values, trims outliers, and returns the median.
func trimmedMedian(values []float64, trimPercent float64) float64 {
	if len(values) == 0 {
		return 0
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
		return sorted[len(sorted)/2]
	}
	return trimmed[len(trimmed)/2]
}

// HWCharResultToHardwareProfile converts benchmark results into a HardwareProfile.
func HWCharResultToHardwareProfile(result *HWCharResult, backend ml.Backend) HardwareProfile {
	hp := HardwareProfile{
		PeakTOPS:                 result.PeakTOPS,
		PeakBandwidthBytesPerSec: result.PeakBW,
		BalancePoints:            result.BalancePoint,
	}

	for _, dev := range backend.BackendDevices() {
		hp.Backends = append(hp.Backends, BackendInfo{
			Name:   dev.Library,
			Device: dev.Name,
		})
	}

	return hp
}
