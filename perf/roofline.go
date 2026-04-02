package perf

import (
	"fmt"
	"math"
)

// LookupBackend finds a backend profile by name in the given profile.
func LookupBackend(p *Profile, backendName string) (*BackendProfile, error) {
	for i := range p.Hardware.Backends {
		if p.Hardware.Backends[i].Name == backendName {
			return &p.Hardware.Backends[i], nil
		}
	}
	return nil, fmt.Errorf("backend %q not found in profile", backendName)
}

// LookupEta finds the calibration coefficient (eta) for a specific operator configuration.
// Returns (eta, true) if found, (0, false) otherwise.
func LookupEta(p *Profile, key OpKey) (float64, bool) {
	for _, op := range p.Operators {
		if op.Op == key.Op && op.Backend == key.Backend &&
			op.ComputeDtype == key.ComputeDtype && op.WeightDtype == key.WeightDtype {
			return op.Eta, true
		}
	}
	return 0, false
}

// LookupInterconnectBW finds the bandwidth between two locations.
// Returns (bandwidth, true) if found, (0, false) otherwise.
func LookupInterconnectBW(p *Profile, from, to string) (float64, bool) {
	for _, ic := range p.Interconnects {
		if ic.From == from && ic.To == to {
			return ic.Bandwidth, true
		}
	}
	return 0, false
}

// EstimateOpCost computes the estimated latency for an operation using the Roofline model.
// It takes into account:
// - Peak compute throughput (FLOPS) and memory bandwidth
// - Operational intensity (FLOPs/byte)
// - Calibration coefficient (eta) if available
//
// Returns an OpCost struct with detailed breakdown of the estimate.
func EstimateOpCost(p *Profile, key OpKey, flops, bytesMoved float64) (OpCost, error) {
	bp, err := LookupBackend(p, key.Backend)
	if err != nil {
		return OpCost{}, err
	}

	// Look up peak FLOPS for the compute dtype
	peakFLOPS, ok := bp.PeakFLOPS[key.ComputeDtype]
	if !ok {
		// Fall back to f32 if specific dtype not found
		peakFLOPS, ok = bp.PeakFLOPS["f32"]
		if !ok {
			return OpCost{}, fmt.Errorf("no peak FLOPS for dtype %q on backend %q", key.ComputeDtype, key.Backend)
		}
	}
	peakBW := bp.PeakBandwidth

	// Calculate operational intensity
	var intensity float64
	if bytesMoved > 0 {
		intensity = flops / bytesMoved
	} else {
		intensity = math.Inf(1)
	}

	// Roofline model: time is limited by either compute or memory
	balancePoint := peakFLOPS / peakBW

	tCompute := flops / peakFLOPS
	tMemory := bytesMoved / peakBW
	tPredicted := math.Max(tCompute, tMemory)

	// Determine which resource is the bottleneck
	bound := "memory"
	if intensity > balancePoint {
		bound = "compute"
	}

	// Look up calibration coefficient
	eta, found := LookupEta(p, key)
	uncalibrated := !found
	if !found {
		// Default to 1.0 for uncalibrated operations
		eta = 1.0
	}

	// Apply calibration: actual time = predicted time / eta
	tActual := tPredicted / eta

	return OpCost{
		FLOPs:        flops,
		BytesMoved:   bytesMoved,
		Intensity:    intensity,
		TCompute:     tCompute,
		TMemory:      tMemory,
		TActual:      tActual,
		Bound:        bound,
		Eta:          eta,
		Uncalibrated: uncalibrated,
	}, nil
}

// EstimateTransferCost estimates the time to transfer data between two locations.
// If from == to or bytes == 0, returns 0.
// If no interconnect bandwidth is found, returns 0 (assumed to be internal/instant).
func EstimateTransferCost(p *Profile, from, to string, bytes float64) float64 {
	if from == to || bytes == 0 {
		return 0
	}
	bw, found := LookupInterconnectBW(p, from, to)
	if !found || bw == 0 {
		return 0
	}
	return bytes / bw
}
