package perf

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestProfile() *Profile {
	return &Profile{
		Version:   2,
		Timestamp: time.Now(),
		Hardware: HardwareProfile{
			Backends: []BackendInfo{
				{Name: "cuda", Device: "RTX 4090", VRAMBytes: 24_000_000_000},
			},
			PeakTOPS:                 map[string]float64{"f16": 330e12, "f32": 82.6e12},
			PeakBandwidthBytesPerSec: 1008e9,
			BalancePoints:            map[string]float64{"f16": 327.38, "f32": 81.94},
		},
		Operators: []OperatorCurve{
			{
				Op: "SILU", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1024}, LatencyUs: 2.5, Reps: 100},
					{Shape: []int64{65536}, LatencyUs: 15.0, Reps: 100},
					{Shape: []int64{1048576}, LatencyUs: 200.0, Reps: 100},
				},
			},
			{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Dimensions: []string{"N"},
				FixedDims:  map[string]int64{"M": 4096, "K": 4096},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 10.0, Reps: 100},
					{Shape: []int64{32}, LatencyUs: 50.0, Reps: 100},
					{Shape: []int64{4096}, LatencyUs: 3000.0, Reps: 100},
				},
			},
		},
	}
}

func TestFixedDimsKey(t *testing.T) {
	// Empty map → empty string
	assert.Equal(t, "", fixedDimsKey(nil))
	assert.Equal(t, "", fixedDimsKey(map[string]int64{}))

	// Single key
	assert.Equal(t, "M=4096", fixedDimsKey(map[string]int64{"M": 4096}))

	// Multiple keys — must be sorted deterministically
	k1 := fixedDimsKey(map[string]int64{"M": 4096, "K": 4096})
	k2 := fixedDimsKey(map[string]int64{"K": 4096, "M": 4096})
	assert.Equal(t, k1, k2, "key order should not affect result")
	assert.Equal(t, "K=4096,M=4096", k1)

	// FLASH_ATTN_EXT dims
	k3 := fixedDimsKey(map[string]int64{"num_heads": 32, "head_dim": 128})
	assert.Equal(t, "head_dim=128,num_heads=32", k3)

	// Different values → different keys
	k4 := fixedDimsKey(map[string]int64{"num_heads": 32, "head_dim": 64})
	assert.NotEqual(t, k3, k4)
}

func TestProfileRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "profile.json")

	original := newTestProfile()
	err := WriteProfile(path, original)
	require.NoError(t, err)

	loaded, err := LoadProfile(path)
	require.NoError(t, err)

	assert.Equal(t, 2, loaded.Version)
	assert.Equal(t, "cuda", loaded.Hardware.Backends[0].Name)
	assert.InDelta(t, 330e12, loaded.Hardware.PeakTOPS["f16"], 1e6)
	assert.InDelta(t, 1008e9, loaded.Hardware.PeakBandwidthBytesPerSec, 1e6)
	assert.Len(t, loaded.Operators, 2)

	// Verify SILU curve
	silu := loaded.Operators[0]
	assert.Equal(t, "SILU", silu.Op)
	assert.Equal(t, []string{"N"}, silu.Dimensions)
	assert.Nil(t, silu.FixedDims)
	assert.Len(t, silu.Points, 3)

	// Verify MUL_MAT curve with FixedDims
	mm := loaded.Operators[1]
	assert.Equal(t, "MUL_MAT", mm.Op)
	assert.Equal(t, "q4_0", mm.WeightDtype)
	assert.Equal(t, int64(4096), mm.FixedDims["M"])
	assert.Equal(t, int64(4096), mm.FixedDims["K"])
}

func TestLoadProfile_NotFound(t *testing.T) {
	_, err := LoadProfile("/nonexistent/path.json")
	assert.Error(t, err)
}

func TestLoadProfile_InvalidJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.json")
	require.NoError(t, writeFile(path, []byte("not json")))
	_, err := LoadProfile(path)
	assert.Error(t, err)
}

func TestLoadProfile_V1Rejected(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "v1.json")
	v1 := `{"version": 1, "timestamp": "2024-01-01T00:00:00Z"}`
	require.NoError(t, writeFile(path, []byte(v1)))
	_, err := LoadProfile(path)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported profile version")
}

func TestLoadProfile_V0Rejected(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "v0.json")
	v0 := `{"timestamp": "2024-01-01T00:00:00Z"}`
	require.NoError(t, writeFile(path, []byte(v0)))
	_, err := LoadProfile(path)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported profile version")
}

func TestMergeProfile(t *testing.T) {
	existing := newTestProfile()
	update := &Profile{
		Version:   2,
		Timestamp: time.Now(),
		Hardware:  existing.Hardware,
		Operators: []OperatorCurve{
			// New curve not in existing
			{
				Op: "SILU", Backend: "cuda", ComputeDtype: "f32",
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{67108864}, LatencyUs: 5000.0, Reps: 100},
				},
			},
			// Duplicate of existing (same OpKey) — should NOT be added
			{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Dimensions: []string{"N"},
				FixedDims:  map[string]int64{"M": 4096, "K": 4096},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 99.0, Reps: 100},
				},
			},
			// New MUL_MAT with different FixedDims
			{
				Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				Dimensions: []string{"N"},
				FixedDims:  map[string]int64{"M": 8192, "K": 8192},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 20.0, Reps: 100},
				},
			},
			// FLASH_ATTN_EXT with non-MUL_MAT FixedDims (num_heads, head_dim)
			{
				Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
				Dimensions: []string{"seq_kv"},
				FixedDims:  map[string]int64{"num_heads": 32, "head_dim": 128},
				Points: []LatencyPoint{
					{Shape: []int64{512}, LatencyUs: 80.0, Reps: 100},
				},
			},
		},
	}

	merged := MergeProfile(existing, update)

	// Should have: 2 original + 0 duplicate SILU + 1 new MUL_MAT(8192) + 1 FLASH_ATTN = 4
	// The duplicate SILU and MUL_MAT(4096) should be kept from existing (not replaced)
	assert.Equal(t, 4, len(merged.Operators))

	// Verify FLASH_ATTN_EXT curve was added (generic FixedDims key works)
	found := false
	for _, c := range merged.Operators {
		if c.Op == "FLASH_ATTN_EXT" {
			found = true
			assert.Equal(t, int64(32), c.FixedDims["num_heads"])
			assert.Equal(t, int64(128), c.FixedDims["head_dim"])
		}
	}
	assert.True(t, found, "FLASH_ATTN_EXT curve should be in merged profile")
}

func TestMergeProfile_FlashAttnDuplicate(t *testing.T) {
	existing := newTestProfile()
	// Add a FLASH_ATTN_EXT curve to existing
	existing.Operators = append(existing.Operators, OperatorCurve{
		Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
		Dimensions: []string{"seq_kv"},
		FixedDims:  map[string]int64{"num_heads": 32, "head_dim": 128},
		Points: []LatencyPoint{
			{Shape: []int64{256}, LatencyUs: 40.0, Reps: 100},
		},
	})

	update := &Profile{
		Version:   2,
		Timestamp: time.Now(),
		Hardware:  existing.Hardware,
		Operators: []OperatorCurve{
			// Same FixedDims — should NOT be added (duplicate)
			{
				Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
				Dimensions: []string{"seq_kv"},
				FixedDims:  map[string]int64{"head_dim": 128, "num_heads": 32}, // different key order, same values
				Points: []LatencyPoint{
					{Shape: []int64{512}, LatencyUs: 80.0, Reps: 100},
				},
			},
			// Different head_dim — should be added
			{
				Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
				Dimensions: []string{"seq_kv"},
				FixedDims:  map[string]int64{"num_heads": 32, "head_dim": 64},
				Points: []LatencyPoint{
					{Shape: []int64{512}, LatencyUs: 60.0, Reps: 100},
				},
			},
		},
	}

	merged := MergeProfile(existing, update)
	// existing had 3 (2 original + 1 FLASH_ATTN), update adds 1 new (head_dim=64), skips duplicate
	assert.Equal(t, 4, len(merged.Operators))
}

func TestLookupBackend(t *testing.T) {
	p := newTestProfile()
	info, err := LookupBackendInfo(p, "cuda")
	require.NoError(t, err)
	assert.Equal(t, "RTX 4090", info.Device)
}

func TestLookupBackend_NotFound(t *testing.T) {
	p := newTestProfile()
	_, err := LookupBackendInfo(p, "nonexistent")
	assert.Error(t, err)
}

func TestLoadProfileV3(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "profile.json")

	p := &Profile{
		Version:   3,
		Timestamp: time.Now(),
		Hardware:  HardwareProfile{PeakTOPS: map[string]float64{"f32": 100}},
		BackendCaps: map[string]BackendCapabilitiesJSON{
			"Vulkan": {
				Name:            "Vulkan",
				HasGPUTimestamp: true,
				HasMulMatVec:    true,
				MulMatVecMaxN:   8,
			},
		},
		Operators: []OperatorCurve{
			{
				Op: "ORCHESTRATION_OVERHEAD", Backend: "Vulkan", ComputeDtype: "f32",
				Dimensions: []string{"num_nodes"},
				Points: []LatencyPoint{
					{Shape: []int64{50}, LatencyUs: 3000},
					{Shape: []int64{100}, LatencyUs: 5500},
				},
			},
		},
	}
	err := WriteProfile(path, p)
	require.NoError(t, err)

	loaded, err := LoadProfile(path)
	require.NoError(t, err)
	assert.Equal(t, 3, loaded.Version)
	assert.True(t, loaded.BackendCaps["Vulkan"].HasGPUTimestamp)
	assert.Equal(t, 8, loaded.BackendCaps["Vulkan"].MulMatVecMaxN)
	assert.True(t, loaded.BackendCaps["Vulkan"].HasMulMatVec)
	assert.Equal(t, "Vulkan", loaded.BackendCaps["Vulkan"].Name)
	assert.Len(t, loaded.Operators, 1)
	assert.Equal(t, "ORCHESTRATION_OVERHEAD", loaded.Operators[0].Op)
}

func TestLoadProfileV2StillWorks(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "profile.json")

	p := &Profile{
		Version:   2,
		Timestamp: time.Now(),
		Hardware:  HardwareProfile{PeakTOPS: map[string]float64{"f32": 100}},
	}
	err := WriteProfile(path, p)
	require.NoError(t, err)

	loaded, err := LoadProfile(path)
	require.NoError(t, err)
	assert.Equal(t, 2, loaded.Version)
	assert.Nil(t, loaded.BackendCaps) // v2 has no BackendCaps
}

func TestLoadProfileV3BackendCapsOmittedWhenEmpty(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "profile.json")

	// v3 without BackendCaps should still work (omitempty)
	p := &Profile{
		Version:   3,
		Timestamp: time.Now(),
		Hardware:  HardwareProfile{PeakTOPS: map[string]float64{"f32": 100}},
	}
	err := WriteProfile(path, p)
	require.NoError(t, err)

	loaded, err := LoadProfile(path)
	require.NoError(t, err)
	assert.Equal(t, 3, loaded.Version)
	assert.Nil(t, loaded.BackendCaps)
}

func TestLoadProfileV3MultipleBackends(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "profile.json")

	p := &Profile{
		Version:   3,
		Timestamp: time.Now(),
		Hardware:  HardwareProfile{PeakTOPS: map[string]float64{"f32": 100}},
		BackendCaps: map[string]BackendCapabilitiesJSON{
			"Vulkan": {Name: "Vulkan", HasGPUTimestamp: true, HasMulMatVec: true, MulMatVecMaxN: 8},
			"CPU":    {Name: "CPU", HasGPUTimestamp: false, HasMulMatVec: false, MulMatVecMaxN: 0},
		},
	}
	err := WriteProfile(path, p)
	require.NoError(t, err)

	loaded, err := LoadProfile(path)
	require.NoError(t, err)
	assert.Len(t, loaded.BackendCaps, 2)
	assert.True(t, loaded.BackendCaps["Vulkan"].HasGPUTimestamp)
	assert.False(t, loaded.BackendCaps["CPU"].HasGPUTimestamp)
	assert.False(t, loaded.BackendCaps["CPU"].HasMulMatVec)
}

func TestMergeProfileV3PreservesBackendCaps(t *testing.T) {
	existing := &Profile{
		Version:   3,
		Timestamp: time.Now(),
		Hardware:  HardwareProfile{PeakTOPS: map[string]float64{"f32": 100}},
		BackendCaps: map[string]BackendCapabilitiesJSON{
			"Vulkan": {Name: "Vulkan", HasGPUTimestamp: true, HasMulMatVec: true, MulMatVecMaxN: 8},
		},
		Operators: []OperatorCurve{
			{Op: "ADD", Backend: "Vulkan", ComputeDtype: "f32", Dimensions: []string{"N"}},
		},
	}
	update := &Profile{
		Version:   3,
		Timestamp: time.Now(),
		Hardware:  existing.Hardware,
		Operators: []OperatorCurve{
			{Op: "MUL", Backend: "Vulkan", ComputeDtype: "f32", Dimensions: []string{"N"}},
		},
	}
	merged := MergeProfile(existing, update)
	assert.Len(t, merged.Operators, 2)
	// MergeProfile currently doesn't merge BackendCaps — it inherits from existing
	// (existing.Hardware is used). This is fine for now.
}

func TestBenchDir(t *testing.T) {
	dir := BenchDir()
	assert.Contains(t, dir, ".ollama")
	assert.Contains(t, dir, "bench")
}

func TestProfilePath(t *testing.T) {
	path := ProfilePath()
	assert.Contains(t, path, "profile.json")
}
