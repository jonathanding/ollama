package perf

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"time"
)

func BenchDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, ".ollama", "bench")
}

func ProfilePath() string {
	return filepath.Join(BenchDir(), "profile.json")
}

func LoadProfile(path string) (*Profile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading profile: %w", err)
	}
	var p Profile
	if err := json.Unmarshal(data, &p); err != nil {
		return nil, fmt.Errorf("parsing profile: %w", err)
	}
	return &p, nil
}

func WriteProfile(path string, p *Profile) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func LoadRawData(path string) (*RawData, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading raw data: %w", err)
	}
	var r RawData
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, fmt.Errorf("parsing raw data: %w", err)
	}
	return &r, nil
}

func WriteRawData(path string, r *RawData) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func RawDataPath() string {
	ts := time.Now().Format("20060102-150405")
	return filepath.Join(BenchDir(), fmt.Sprintf("raw-%s.json", ts))
}

func ComputeEtaFromPoints(points []BenchmarkPoint, peakFLOPS, peakBW float64) (float64, float64) {
	if len(points) == 0 {
		return 1.0, 0
	}

	etas := make([]float64, 0, len(points))
	for _, pt := range points {
		tCompute := pt.FLOPs / peakFLOPS
		tMemory := pt.BytesMoved / peakBW
		tPredicted := math.Max(tCompute, tMemory)
		tMeasured := pt.LatencyUs * 1e-6

		if tMeasured <= 0 || tPredicted <= 0 {
			continue
		}
		eta := tPredicted / tMeasured
		if eta > 0 && eta <= 2.0 {
			etas = append(etas, eta)
		}
	}

	if len(etas) == 0 {
		return 1.0, 0
	}

	sort.Float64s(etas)
	median := etas[len(etas)/2]

	mean := 0.0
	for _, e := range etas {
		mean += e
	}
	mean /= float64(len(etas))

	variance := 0.0
	for _, e := range etas {
		d := e - mean
		variance += d * d
	}
	variance /= float64(len(etas))

	return median, variance
}

func ProcessRawToProfile(rawFiles []string) (*Profile, error) {
	p := &Profile{
		Version:       1,
		GeneratedFrom: rawFiles,
		GeneratedAt:   time.Now(),
	}

	for _, path := range rawFiles {
		raw, err := LoadRawData(path)
		if err != nil {
			return nil, err
		}

		if len(raw.Hardware.Backends) > 0 {
			p.Hardware.Backends = make([]BackendProfile, 0, len(raw.Hardware.Backends))
			for _, rb := range raw.Hardware.Backends {
				p.Hardware.Backends = append(p.Hardware.Backends, BackendProfile{
					Name:          rb.Name,
					Device:        rb.Device,
					PeakFLOPS:     make(map[string]float64),
					BalancePoints: make(map[string]float64),
				})
			}
		}

		for _, hb := range raw.HardwareBenchmarks {
			bp := findOrCreateBackend(&p.Hardware, hb.Backend)
			switch hb.Test {
			case "peak_flops":
				bp.PeakFLOPS[hb.Dtype] = hb.Value
			case "peak_bandwidth":
				bp.PeakBandwidth = hb.Value
			}
		}

		for i := range p.Hardware.Backends {
			bp := &p.Hardware.Backends[i]
			for dtype, flops := range bp.PeakFLOPS {
				if bp.PeakBandwidth > 0 {
					bp.BalancePoints[dtype] = flops / bp.PeakBandwidth
				}
			}
		}

		for _, ob := range raw.OperatorBenchmarks {
			bp := findBackend(&p.Hardware, ob.Backend)
			if bp == nil {
				continue
			}
			peakFLOPS := bp.PeakFLOPS[ob.ComputeDtype]
			if peakFLOPS == 0 {
				peakFLOPS = bp.PeakFLOPS["f32"]
			}
			eta, variance := ComputeEtaFromPoints(ob.Points, peakFLOPS, bp.PeakBandwidth)
			p.Operators = append(p.Operators, OperatorProfile{
				Op:           ob.Op,
				Backend:      ob.Backend,
				ComputeDtype: ob.ComputeDtype,
				WeightDtype:  ob.WeightDtype,
				Eta:          eta,
				EtaVariance:  variance,
				NumPoints:    len(ob.Points),
			})
		}

		for _, ic := range raw.InterconnectBenchmarks {
			p.Interconnects = append(p.Interconnects, InterconnectInfo{
				From:      ic.From,
				To:        ic.To,
				Bandwidth: ic.Bandwidth,
			})
		}
	}

	return p, nil
}

func MergeProfile(existing *Profile, update *Profile) *Profile {
	merged := *existing
	merged.GeneratedFrom = append(merged.GeneratedFrom, update.GeneratedFrom...)
	merged.GeneratedAt = time.Now()

	existingKeys := make(map[OpKey]bool)
	for _, op := range existing.Operators {
		existingKeys[OpKey{op.Op, op.Backend, op.ComputeDtype, op.WeightDtype}] = true
	}
	for _, op := range update.Operators {
		key := OpKey{op.Op, op.Backend, op.ComputeDtype, op.WeightDtype}
		if !existingKeys[key] {
			merged.Operators = append(merged.Operators, op)
		}
	}

	for _, ic := range update.Interconnects {
		found := false
		for _, eic := range merged.Interconnects {
			if eic.From == ic.From && eic.To == ic.To {
				found = true
				break
			}
		}
		if !found {
			merged.Interconnects = append(merged.Interconnects, ic)
		}
	}

	return &merged
}

func findBackend(hw *HardwareProfile, name string) *BackendProfile {
	for i := range hw.Backends {
		if hw.Backends[i].Name == name {
			return &hw.Backends[i]
		}
	}
	return nil
}

func findOrCreateBackend(hw *HardwareProfile, name string) *BackendProfile {
	if bp := findBackend(hw, name); bp != nil {
		return bp
	}
	hw.Backends = append(hw.Backends, BackendProfile{
		Name:          name,
		PeakFLOPS:     make(map[string]float64),
		BalancePoints: make(map[string]float64),
	})
	return &hw.Backends[len(hw.Backends)-1]
}
