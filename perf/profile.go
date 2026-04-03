package perf

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// fixedDimsKey returns a canonical string representation of FixedDims
// for use as part of a map key. Keys are sorted for determinism.
func fixedDimsKey(fd map[string]int64) string {
	if len(fd) == 0 {
		return ""
	}
	keys := make([]string, 0, len(fd))
	for k := range fd {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	for i, k := range keys {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%s=%d", k, fd[k])
	}
	return b.String()
}

// BenchDir returns the directory for benchmark data.
func BenchDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, ".ollama", "bench")
}

// ProfilePath returns the default profile file path.
func ProfilePath() string {
	return filepath.Join(BenchDir(), "profile.json")
}

// LoadProfile reads a v2 profile from disk.
func LoadProfile(path string) (*Profile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading profile: %w", err)
	}
	var p Profile
	if err := json.Unmarshal(data, &p); err != nil {
		return nil, fmt.Errorf("parsing profile: %w", err)
	}
	if p.Version != 2 {
		return nil, fmt.Errorf("unsupported profile version %d (expected 2)", p.Version)
	}
	return &p, nil
}

// WriteProfile writes a v2 profile to disk.
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

// MergeProfile merges new operator curves into an existing profile.
// Curves with matching (Op, Backend, ComputeDtype, WeightDtype, FixedDims) are skipped.
// New curves are appended. Hardware profile is taken from the existing profile.
func MergeProfile(existing, update *Profile) *Profile {
	merged := &Profile{
		Version:   existing.Version,
		Timestamp: update.Timestamp,
		Hardware:  existing.Hardware,
	}

	// Copy existing curves
	merged.Operators = make([]OperatorCurve, len(existing.Operators))
	copy(merged.Operators, existing.Operators)

	// Build lookup of existing curves using a generic key that works for all ops.
	type curveKey struct {
		op, backend, cdt, wdt, fixedDims string
	}
	makeCurveKey := func(c OperatorCurve) curveKey {
		return curveKey{c.Op, c.Backend, c.ComputeDtype, c.WeightDtype, fixedDimsKey(c.FixedDims)}
	}
	seen := make(map[curveKey]bool)
	for _, c := range existing.Operators {
		seen[makeCurveKey(c)] = true
	}

	// Add new curves that don't conflict
	for _, c := range update.Operators {
		k := makeCurveKey(c)
		if !seen[k] {
			merged.Operators = append(merged.Operators, c)
			seen[k] = true
		}
	}

	return merged
}

// LookupBackendInfo finds a backend by name in the profile.
func LookupBackendInfo(p *Profile, backendName string) (*BackendInfo, error) {
	for i := range p.Hardware.Backends {
		if p.Hardware.Backends[i].Name == backendName {
			return &p.Hardware.Backends[i], nil
		}
	}
	return nil, fmt.Errorf("backend %q not found in profile", backendName)
}

// writeFile is a helper for tests.
func writeFile(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}
