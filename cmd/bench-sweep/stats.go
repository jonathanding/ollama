package main

import (
	"math"
	"sort"
)

// MetricStats holds summary statistics for a single metric across epochs.
type MetricStats struct {
	Mean   float64 `json:"mean"`
	Median float64 `json:"median"`
	P99    float64 `json:"p99"`
	Stddev float64 `json:"stddev"`
	CVPct  float64 `json:"cv_pct"`
}

// computeStats returns summary statistics for values.
// With N < 100 values, P99 equals the maximum value.
// Returns zero MetricStats for an empty slice.
func computeStats(values []float64) MetricStats {
	n := len(values)
	if n == 0 {
		return MetricStats{}
	}

	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(n)

	variance := 0.0
	for _, v := range values {
		d := v - mean
		variance += d * d
	}
	if n > 1 {
		variance /= float64(n - 1)
	}
	stddev := math.Sqrt(variance)

	sorted := make([]float64, n)
	copy(sorted, values)
	sort.Float64s(sorted)

	var median float64
	if n%2 == 0 {
		median = (sorted[n/2-1] + sorted[n/2]) / 2
	} else {
		median = sorted[n/2]
	}

	idx := int(math.Ceil(float64(n)*0.99)) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= n {
		idx = n - 1
	}
	p99 := sorted[idx]

	cvPct := 0.0
	if mean > 0 {
		cvPct = stddev / mean * 100
	}

	return MetricStats{
		Mean:   mean,
		Median: median,
		P99:    p99,
		Stddev: stddev,
		CVPct:  cvPct,
	}
}
