package perf

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
)

// EstimateConfig controls estimation behavior.
type EstimateConfig struct {
	InputLength  int
	OutputLength int
	MaxBatchSize int
	ProfilePath  string
	JSON         bool
	Detail       bool
}

// DefaultEstimateConfig returns sensible defaults.
func DefaultEstimateConfig() EstimateConfig {
	return EstimateConfig{
		InputLength:  512,
		OutputLength: 128,
		MaxBatchSize: 512,
	}
}

// OpStats tracks aggregate stats for an op key across graph nodes.
type OpStats struct {
	Key       OpKey
	Count     int
	TotalSec  float64
	MemCount  int
	CompCount int
}

// EstimateGraphLatency estimates total latency for a set of graph nodes.
func EstimateGraphLatency(p *Profile, nodes []ml.GraphNode) (float64, map[OpKey]*OpStats, []string) {
	totalLatency := 0.0
	stats := make(map[OpKey]*OpStats)
	var warnings []string
	seenWarnings := make(map[string]bool)

	var prevBackend string
	for _, node := range nodes {
		if IsZeroCostOp(node.Op) {
			continue
		}

		// Cross-backend transfer cost
		if prevBackend != "" && node.Backend != prevBackend {
			transferBytes := float64(0)
			if len(node.InputShapes) > 0 {
				transferBytes = product(node.InputShapes[0]) * elemSize(node.ComputeDtype)
			}
			transferTime := EstimateTransferCost(p, prevBackend, node.Backend, transferBytes)
			totalLatency += transferTime
		}
		prevBackend = node.Backend

		flops := ComputeFLOPs(node.Op, node.InputShapes)
		bytes := ComputeBytes(node.Op, node.InputShapes, node.ComputeDtype, node.WeightDtype)

		key := OpKey{
			Op:           node.Op,
			Backend:      node.Backend,
			ComputeDtype: node.ComputeDtype,
			WeightDtype:  node.WeightDtype,
		}

		if flops == 0 && bytes == 0 {
			continue
		}

		var cost OpCost
		var err error

		if !CanComputeFLOPs(node.Op) && bytes > 0 {
			bp, bpErr := LookupBackend(p, node.Backend)
			if bpErr != nil {
				continue
			}
			cost = OpCost{
				BytesMoved:   bytes,
				TMemory:      bytes / bp.PeakBandwidth,
				TActual:      bytes / bp.PeakBandwidth,
				Bound:        "memory",
				Eta:          1.0,
				Uncalibrated: true,
			}
			warnKey := fmt.Sprintf("unknown op: %s(%s,%s)", node.Op, node.Backend, node.ComputeDtype)
			if !seenWarnings[warnKey] {
				warnings = append(warnings, warnKey)
				seenWarnings[warnKey] = true
			}
		} else {
			cost, err = EstimateOpCost(p, key, flops, bytes)
			if err != nil {
				continue
			}
			if cost.Uncalibrated {
				warnKey := fmt.Sprintf("uncalibrated: %s(%s,%s,%s)", key.Op, key.Backend, key.ComputeDtype, key.WeightDtype)
				if !seenWarnings[warnKey] {
					warnings = append(warnings, warnKey)
					seenWarnings[warnKey] = true
				}
			}
		}

		totalLatency += cost.TActual

		if _, ok := stats[key]; !ok {
			stats[key] = &OpStats{Key: key}
		}
		s := stats[key]
		s.Count++
		s.TotalSec += cost.TActual
		if cost.Bound == "memory" {
			s.MemCount++
		} else {
			s.CompCount++
		}
	}

	return totalLatency, stats, warnings
}

// ComputePhaseEstimation wraps EstimateGraphLatency for a phase.
func ComputePhaseEstimation(p *Profile, nodes []ml.GraphNode, tokenCount, batchSize int) *PhaseEstimation {
	latencySec, stats, _ := EstimateGraphLatency(p, nodes)

	tokPerSec := 0.0
	if latencySec > 0 {
		tokPerSec = float64(tokenCount) / latencySec
	}

	memTotal, compTotal := 0.0, 0.0
	for _, s := range stats {
		memFrac := float64(s.MemCount) / float64(max(s.Count, 1))
		compFrac := float64(s.CompCount) / float64(max(s.Count, 1))
		memTotal += s.TotalSec * memFrac
		compTotal += s.TotalSec * compFrac
	}
	bottleneck := "memory"
	if compTotal > memTotal {
		bottleneck = "compute"
	}

	topOps := buildTopOps(stats, latencySec)
	nBatches := int(math.Ceil(float64(tokenCount) / float64(batchSize)))

	return &PhaseEstimation{
		TotalLatencyMs: latencySec * 1000,
		TokensPerSec:   tokPerSec,
		TTFTMs:         latencySec * 1000,
		NumBatches:     nBatches,
		Bottleneck:     bottleneck,
		TopOps:         topOps,
	}
}

func buildTopOps(stats map[OpKey]*OpStats, totalSec float64) []OpBreakdown {
	var ops []OpBreakdown
	for _, s := range stats {
		pct := 0.0
		if totalSec > 0 {
			pct = (s.TotalSec / totalSec) * 100
		}
		bound := fmt.Sprintf("%dx mem + %dx compute", s.MemCount, s.CompCount)
		if s.MemCount == 0 {
			bound = fmt.Sprintf("%dx compute", s.CompCount)
		} else if s.CompCount == 0 {
			bound = fmt.Sprintf("%dx memory", s.MemCount)
		}

		ops = append(ops, OpBreakdown{
			Op:             s.Key.Op,
			Backend:        s.Key.Backend,
			ComputeDtype:   s.Key.ComputeDtype,
			WeightDtype:    s.Key.WeightDtype,
			Count:          s.Count,
			TotalMs:        s.TotalSec * 1000,
			Percentage:     pct,
			BoundBreakdown: bound,
		})
	}

	sort.Slice(ops, func(i, j int) bool {
		return ops[i].TotalMs > ops[j].TotalMs
	})

	if len(ops) > 10 {
		ops = ops[:10]
	}
	return ops
}

// BuildSummary generates the human-readable summary line.
func BuildSummary(r *EstimateResult) {
	var parts []string
	parts = append(parts, fmt.Sprintf("%s | input=%d | output=%d", r.Model, r.InputLength, r.OutputLength))

	if r.Prefill.TokensPerSec > 0 {
		parts = append(parts, fmt.Sprintf("Prefill: ~%.0f tok/s (batch=%d, %d batches, TTFT ≈ %.0fms)",
			r.Prefill.TokensPerSec, r.MaxBatchSize, r.Prefill.NumBatches, r.Prefill.TTFTMs))
	}
	if r.Decode.TokensPerSec > 0 {
		parts = append(parts, fmt.Sprintf("Decode: ~%.0f tok/s (%s-bound)",
			r.Decode.TokensPerSec, r.Decode.Bottleneck))
	}

	r.Summary = strings.Join(parts, "\n  ")
}

// RunEstimate is the main entry point for the estimate command.
// Requires CGo runtime — loads model via model.New() to build computation graphs.
func RunEstimate(modelRef string, cfg EstimateConfig) (*EstimateResult, error) {
	profilePath := cfg.ProfilePath
	if profilePath == "" {
		profilePath = ProfilePath()
	}
	profile, err := LoadProfile(profilePath)
	if err != nil {
		return nil, fmt.Errorf("load profile: %w (have you run 'ollama daop-bench'?)", err)
	}

	result := &EstimateResult{
		Model:        modelRef,
		InputLength:  cfg.InputLength,
		OutputLength: cfg.OutputLength,
		MaxBatchSize: cfg.MaxBatchSize,
	}

	for _, bp := range profile.Hardware.Backends {
		primaryDtype := "f16"
		peakFLOPS := bp.PeakFLOPS[primaryDtype]
		balancePoint := bp.BalancePoints[primaryDtype]
		if peakFLOPS == 0 {
			primaryDtype = "f32"
			peakFLOPS = bp.PeakFLOPS[primaryDtype]
			balancePoint = bp.BalancePoints[primaryDtype]
		}
		result.Backends = append(result.Backends, BackendInfo{
			Name:          bp.Name,
			Device:        bp.Device,
			PeakFLOPS:     peakFLOPS,
			PeakBandwidth: bp.PeakBandwidth,
			BalancePoint:  balancePoint,
		})
	}

	ggufPath, err := ResolveModelPath(modelRef)
	if err != nil {
		return nil, err
	}
	m, err := model.New(ggufPath, ml.BackendParams{})
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	defer m.Backend().Close()

	// TODO: build prefill + decode graphs once batch API is validated
	// For now, return partial result
	BuildSummary(result)
	return result, nil
}

func deduplicateStrings(ss []string) []string {
	seen := make(map[string]bool)
	var result []string
	for _, s := range ss {
		if !seen[s] {
			seen[s] = true
			result = append(result, s)
		}
	}
	return result
}
