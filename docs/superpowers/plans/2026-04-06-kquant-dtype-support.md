# K-Quant DType Support (q4_K, q6_K) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native q4_K and q6_K dtype support to the benchmark system so MUL_MAT can be calibrated directly with K-quant types, eliminating the ~2x prefill prediction error caused by the q4_K->q4_0 approximation.

**Architecture:** Thread two new `ml.DType` constants (Q4K, Q6K) through: Go enum -> GGML C type mapping -> perf parseDType -> Phase1Dtypes -> mapWeightDtype identity passthrough. Add explicit fallback chain with warnings in lookupLatencyV3 so data provenance is always visible. No C code changes needed — `GGML_TYPE_Q4_K` (=12) and `GGML_TYPE_Q6_K` (=14) already exist in ggml.h.

**Tech Stack:** Go, CGo (GGML bindings), testify

**Benchmark time impact:** Adding 2 new dtypes adds 2 × 9 (M,K) pairs × 3 N values = 54 extra measurements to `daop-bench`. Estimated +2-3 minutes.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `ml/backend.go:460-468` | Modify | Add `DTypeQ4K`, `DTypeQ6K` to DType enum |
| `ml/backend/ggml/ggml.go:1512-1548` | Modify | Add Go<->C type mapping in `DType()` and `ggmlDType()` |
| `perf/registry.go:354-389` | Modify | Add q4_K/q6_K to `parseDType()`, `dtypeToString()`, `Phase1Dtypes()` |
| `perf/dtype_map.go` | Modify | Make q4_K/q6_K identity; add `dtypeFallback()` |
| `perf/bench.go:590-605` | Modify | Add q4_K/q6_K to `elemBytesFromDtype()` |
| `perf/estimate.go:335-364` | Modify | Add fallback chain with warnings in `lookupLatencyV3` |
| `perf/dtype_map_test.go` | Modify | Update test expectations |
| `perf/registry_test.go` | Modify | Update `TestPhase1Dtypes` and `TestParseDType` |
| `perf/estimate_test.go` | Modify | Add fallback chain tests |

---

### Task 1: Add DType enum constants

**Files:**
- Modify: `ml/backend.go:460-468`

- [ ] **Step 1: Add the two new constants**

In `ml/backend.go`, add `DTypeQ4K` and `DTypeQ6K` to the enum after `DTypeMXFP4`:

```go
const (
	DTypeOther DType = iota
	DTypeF32
	DTypeF16
	DTypeQ80
	DTypeQ40
	DTypeI32
	DTypeMXFP4
	DTypeQ4K
	DTypeQ6K
)
```

- [ ] **Step 2: Verify build**

Run: `go build ./ml/...`
Expected: PASS (enum is additive, no existing code breaks)

- [ ] **Step 3: Commit**

```bash
git add ml/backend.go
git commit -m "perf: add DTypeQ4K and DTypeQ6K to ml.DType enum"
```

---

### Task 2: Add Go<->C type mapping in GGML backend

**Files:**
- Modify: `ml/backend/ggml/ggml.go:1512-1548`

- [ ] **Step 1: Add C->Go mapping in `DType()` method**

In `ml/backend/ggml/ggml.go`, function `(t *Tensor) DType()` (line 1512), add two cases before the `default`:

```go
	case C.GGML_TYPE_Q4_K:
		return ml.DTypeQ4K
	case C.GGML_TYPE_Q6_K:
		return ml.DTypeQ6K
```

- [ ] **Step 2: Add Go->C mapping in `ggmlDType()` function**

In same file, function `ggmlDType()` (line 1531), add two cases before the `default`:

```go
	case ml.DTypeQ4K:
		return C.GGML_TYPE_Q4_K
	case ml.DTypeQ6K:
		return C.GGML_TYPE_Q6_K
```

- [ ] **Step 3: Verify build**

Run: `go build ./ml/...`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add ml/backend/ggml/ggml.go
git commit -m "perf: add q4_K/q6_K Go<->C type mapping in GGML backend"
```

---

### Task 3: Update perf dtype helpers and Phase1Dtypes

**Files:**
- Modify: `perf/registry.go:354-400`
- Modify: `perf/registry_test.go`
- Modify: `perf/bench.go:590-605`

- [ ] **Step 1: Write failing tests**

In `perf/registry_test.go`, add new tests:

```go
func TestParseDType_KQuants(t *testing.T) {
	dt, ok := parseDType("q4_K")
	assert.True(t, ok)
	assert.Equal(t, ml.DTypeQ4K, dt)
	assert.Equal(t, "q4_K", dtypeToString(dt))

	dt, ok = parseDType("q6_K")
	assert.True(t, ok)
	assert.Equal(t, ml.DTypeQ6K, dt)
	assert.Equal(t, "q6_K", dtypeToString(dt))
}

func TestPhase1Dtypes_IncludesKQuants(t *testing.T) {
	dtypes := Phase1Dtypes()
	assert.Contains(t, dtypes, "q4_K")
	assert.Contains(t, dtypes, "q6_K")
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./perf/ -run "TestParseDType_KQuants|TestPhase1Dtypes_IncludesKQuants" -v`
Expected: FAIL — `parseDType("q4_K")` returns `(0, false)`

- [ ] **Step 3: Update `parseDType()`**

In `perf/registry.go`, add two cases:

```go
func parseDType(s string) (ml.DType, bool) {
	switch s {
	case "f32":
		return ml.DTypeF32, true
	case "f16":
		return ml.DTypeF16, true
	case "q4_0":
		return ml.DTypeQ40, true
	case "q8_0":
		return ml.DTypeQ80, true
	case "q4_K":
		return ml.DTypeQ4K, true
	case "q6_K":
		return ml.DTypeQ6K, true
	default:
		return 0, false
	}
}
```

- [ ] **Step 4: Update `dtypeToString()`**

```go
func dtypeToString(dt ml.DType) string {
	switch dt {
	case ml.DTypeF32:
		return "f32"
	case ml.DTypeF16:
		return "f16"
	case ml.DTypeQ40:
		return "q4_0"
	case ml.DTypeQ80:
		return "q8_0"
	case ml.DTypeQ4K:
		return "q4_K"
	case ml.DTypeQ6K:
		return "q6_K"
	default:
		return "unknown"
	}
}
```

- [ ] **Step 5: Update `Phase1Dtypes()`**

```go
func Phase1Dtypes() []string {
	return []string{"f32", "f16", "q4_0", "q8_0", "q4_K", "q6_K"}
}
```

- [ ] **Step 6: Update `elemBytesFromDtype()` in `perf/bench.go`**

Add q4_K and q6_K cases. q4_K has same bytes/element as q4_0 (144 bytes / 256 elements = 0.5625). q6_K is 210 bytes / 256 elements = 0.8203.

```go
func elemBytesFromDtype(dtype string) float64 {
	switch dtype {
	case "f32":
		return 4.0
	case "f16":
		return 2.0
	case "q4_0", "q4_K":
		return 18.0 / 32.0 // q4_0: 18B/32elem; q4_K: 144B/256elem = same ratio
	case "q8_0":
		return 34.0 / 32.0 // 34 bytes per 32-element block = 1.0625
	case "q6_K":
		return 210.0 / 256.0 // 210 bytes per 256-element super-block = 0.8203
	default:
		return 4.0
	}
}
```

- [ ] **Step 7: Update comment in `Phase1MulMatFixedDims()`**

In `perf/registry.go:395`, update the measurement count comment:

```go
// Total: 9 (M,K) pairs × 6 dtypes × 3 N values = 162 measurements.
```

- [ ] **Step 8: Update existing `TestPhase1Dtypes`**

The existing test in `perf/registry_test.go:150-156` — update it to include new dtypes:

```go
func TestPhase1Dtypes(t *testing.T) {
	dtypes := Phase1Dtypes()
	assert.Contains(t, dtypes, "f32")
	assert.Contains(t, dtypes, "f16")
	assert.Contains(t, dtypes, "q4_0")
	assert.Contains(t, dtypes, "q8_0")
	assert.Contains(t, dtypes, "q4_K")
	assert.Contains(t, dtypes, "q6_K")
}
```

- [ ] **Step 9: Run all perf tests**

Run: `go test ./perf/ -count=1`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add perf/registry.go perf/registry_test.go perf/bench.go
git commit -m "perf: add q4_K/q6_K to parseDType, dtypeToString, Phase1Dtypes, elemBytesFromDtype"
```

---

### Task 4: Update mapWeightDtype and add dtypeFallback

**Files:**
- Modify: `perf/dtype_map.go`
- Modify: `perf/dtype_map_test.go`

- [ ] **Step 1: Write failing tests**

In `perf/dtype_map_test.go`, update `TestMapWeightDtype_DirectlyMeasured` and add fallback tests:

```go
func TestMapWeightDtype_DirectlyMeasured(t *testing.T) {
	for _, dt := range []string{"f32", "f16", "q4_0", "q8_0", "q4_K", "q6_K"} {
		assert.Equal(t, dt, mapWeightDtype(dt), "measured dtype %s should map to itself", dt)
	}
}

func TestDtypeFallback(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"q4_K", "q4_0"},
		{"q6_K", "q8_0"},
		{"q4_0", ""},  // no fallback for base dtypes
		{"q8_0", ""},
		{"f16", ""},
		{"f32", ""},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			assert.Equal(t, tt.expected, dtypeFallback(tt.input))
		})
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./perf/ -run "TestMapWeightDtype_DirectlyMeasured|TestDtypeFallback" -v`
Expected: FAIL — `mapWeightDtype("q4_K")` returns `"q4_0"` not `"q4_K"`; `dtypeFallback` doesn't exist.

- [ ] **Step 3: Update `mapWeightDtype()` and add `dtypeFallback()`**

In `perf/dtype_map.go`:

```go
package perf

// mapWeightDtype maps model weight dtypes to the nearest benchmark-measured dtype.
// K-quant types that are directly benchmarked pass through as identity.
func mapWeightDtype(wdt string) string {
	switch wdt {
	case "f32", "f16", "q4_0", "q8_0", "q4_K", "q6_K":
		return wdt
	case "q4_1":
		return "q4_0"
	case "q5_K", "q5_0", "q5_1":
		return "q6_K"
	case "q3_K", "q2_K":
		return "q4_0"
	case "q8_K":
		return "q8_0"
	default:
		return "f16"
	}
}

// dtypeFallback returns the approximate fallback dtype for when direct calibration
// curves are not yet available. Returns "" if no fallback exists (base dtype).
// Used by lookupLatencyV3 to provide estimates before re-benchmarking.
func dtypeFallback(wdt string) string {
	switch wdt {
	case "q4_K":
		return "q4_0"
	case "q6_K":
		return "q8_0"
	default:
		return ""
	}
}
```

- [ ] **Step 4: Update `TestMapWeightDtype_KQuants` expectations**

```go
func TestMapWeightDtype_KQuants(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"q4_1", "q4_0"},
		{"q5_K", "q6_K"},
		{"q5_0", "q6_K"},
		{"q5_1", "q6_K"},
		{"q3_K", "q4_0"},
		{"q2_K", "q4_0"},
		{"q8_K", "q8_0"},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			assert.Equal(t, tt.expected, mapWeightDtype(tt.input))
		})
	}
}
```

- [ ] **Step 5: Run all dtype map tests**

Run: `go test ./perf/ -run "TestMapWeightDtype|TestDtypeFallback" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add perf/dtype_map.go perf/dtype_map_test.go
git commit -m "perf: make q4_K/q6_K identity in mapWeightDtype, add dtypeFallback"
```

---

### Task 5: Add fallback chain with warnings in ALL lookup paths

This is the critical correctness task. Every lookup path must have explicit warnings when using approximate data. This covers:
- `lookupLatencyV3` MUL_MAT path: q4_K curves → q4_0 fallback → roofline fallback
- `lookupLatency` default path (1D ops): exact dtype match → fallback dtype → bandwidth roofline → error
- All fallbacks emit `slog.Warn` so the user always knows where data came from.

**Files:**
- Modify: `perf/estimate.go:284-364` (both `lookupLatency` and `lookupLatencyV3`)
- Modify: `perf/estimate_test.go`

- [ ] **Step 1: Write failing tests for MUL_MAT fallback**

In `perf/estimate_test.go`, add tests:

```go
func TestLookupLatencyV3_MulMat_DirectQ4K(t *testing.T) {
	// Profile has q4_K curves — should use them directly, no fallback
	p := makeTestProfileForEstimation()
	p.Operators = append(p.Operators, OperatorCurve{
		Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "q4_K", WeightDtype: "q4_K",
		Dimensions: []string{"N"},
		FixedDims:  map[string]int64{"M": 4096, "K": 4096},
		Points:     []LatencyPoint{{Shape: []int64{1}, LatencyUs: 200.0}, {Shape: []int64{512}, LatencyUs: 5000.0}},
	})
	caps := &BackendCapabilities{Name: "cuda"}
	lat, err := lookupLatencyV3(p, "MUL_MAT", []int64{4096, 4096, 1}, "f16", "q4_K", "cuda", caps)
	require.NoError(t, err)
	assert.InDelta(t, 200.0, lat, 10.0)
}

func TestLookupLatencyV3_MulMat_FallbackQ4KtoQ40(t *testing.T) {
	// Profile has NO q4_K curves but has q4_0 — should fall back with warning
	p := makeTestProfileForEstimation()
	caps := &BackendCapabilities{Name: "cuda"}
	lat, err := lookupLatencyV3(p, "MUL_MAT", []int64{4096, 4096, 1}, "f16", "q4_K", "cuda", caps)
	require.NoError(t, err)
	assert.Greater(t, lat, 0.0, "should get a non-zero fallback estimate")
}

func TestLookupLatencyV3_MulMat_FallbackQ6KtoQ80(t *testing.T) {
	// Profile has NO q6_K curves but has q8_0 — should fall back with warning
	p := makeTestProfileForEstimation()
	caps := &BackendCapabilities{Name: "cuda"}
	lat, err := lookupLatencyV3(p, "MUL_MAT", []int64{4096, 4096, 1}, "f16", "q6_K", "cuda", caps)
	require.NoError(t, err)
	assert.Greater(t, lat, 0.0, "should get a non-zero fallback estimate")
}
```

- [ ] **Step 2: Write failing tests for 1D op fallback**

```go
func TestLookupLatency_1DOp_FallbackQ4KDtype(t *testing.T) {
	// Profile has SILU calibrated for f32 but not q4_K.
	// dtypeFallback(q4_K) = q4_0, but q4_0 is also not calibrated for SILU.
	// Should fall through to bandwidth roofline.
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "SILU", []int64{4096}, "q4_K", "", "cuda")
	require.NoError(t, err)
	assert.Greater(t, lat, 0.0)
	// Bandwidth roofline: 4096 * 0.5625 * 2 / bandwidth * 1e6
	expectedUs := float64(4096) * (18.0 / 32.0) * 2 / p.Hardware.PeakBandwidthBytesPerSec * 1e6
	assert.InDelta(t, expectedUs, lat, 0.001)
}

func TestLookupLatency_1DOp_DtypeFallback(t *testing.T) {
	// Profile has SILU calibrated for q4_0.
	// When asked for SILU q4_K, dtypeFallback returns q4_0 — should use it with warning.
	p := makeTestProfileForEstimation()
	p.Operators = append(p.Operators, OperatorCurve{
		Op: "SILU", Backend: "cuda", ComputeDtype: "q4_0",
		Dimensions: []string{"N"},
		Points:     []LatencyPoint{{Shape: []int64{4096}, LatencyUs: 5.0}},
	})
	lat, err := lookupLatency(p, "SILU", []int64{4096}, "q4_K", "", "cuda")
	require.NoError(t, err)
	assert.InDelta(t, 5.0, lat, 0.1)
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `go test ./perf/ -run "TestLookupLatencyV3_MulMat_DirectQ4K|TestLookupLatencyV3_MulMat_Fallback|TestLookupLatency_1DOp" -v`
Expected: FAIL

- [ ] **Step 4: Update `lookupLatencyV3` MUL_MAT path with fallback chain**

In `perf/estimate.go`, replace the MUL_MAT/MUL_MAT_ADD case in `lookupLatencyV3`:

```go
	case "MUL_MAT", "MUL_MAT_ADD":
		if len(shape) < 3 {
			return 0, fmt.Errorf("MUL_MAT requires 3 shape dims, got %d", len(shape))
		}
		M, K, N := shape[0], shape[1], shape[2]
		mappedWdt := mapWeightDtype(weightDtype)

		// Primary: direct interpolation from reference curves
		lat := PredictMulMatDirect(profile, M, K, N, mappedWdt)
		if lat > 0 {
			return lat, nil
		}

		// Fallback 1: try approximate dtype (e.g. q4_K -> q4_0)
		if fb := dtypeFallback(mappedWdt); fb != "" {
			lat = PredictMulMatDirect(profile, M, K, N, fb)
			if lat > 0 {
				slog.Warn("MUL_MAT using fallback dtype curves",
					"requested", mappedWdt, "fallback", fb,
					"M", M, "K", K, "N", N)
				return lat, nil
			}
		}

		// Fallback 2: roofline model
		lat = PredictMulMatLatency(&profile.Hardware, M, K, N, mappedWdt)
		if lat > 0 {
			slog.Warn("MUL_MAT using roofline model (no calibration curves)",
				"dtype", mappedWdt, "M", M, "K", K, "N", N)
			return lat, nil
		}
		return 0, fmt.Errorf("no MUL_MAT calibration data for dtype %s — run daop-bench first", mappedWdt)
```

- [ ] **Step 5: Update `lookupLatency` default path with dtype fallback**

In `perf/estimate.go`, update the `default` case in `lookupLatency` to try fallback dtypes before returning uncalibrated error:

```go
	default:
		if len(shape) < 1 {
			return 0, fmt.Errorf("op %s requires at least 1 shape dim", op)
		}
		// Primary: exact dtype match
		for _, c := range profile.Operators {
			if c.Op == op && c.ComputeDtype == computeDtype && c.Backend == backend {
				return Interpolate1D(c.Points, shape[0]), nil
			}
		}
		// Fallback: try approximate dtype
		if fb := dtypeFallback(computeDtype); fb != "" {
			for _, c := range profile.Operators {
				if c.Op == op && c.ComputeDtype == fb && c.Backend == backend {
					slog.Warn("op using fallback dtype curves",
						"op", op, "requested", computeDtype, "fallback", fb,
						"backend", backend)
					return Interpolate1D(c.Points, shape[0]), nil
				}
			}
		}
		// Fallback: bandwidth roofline for elementwise ops
		if profile.Hardware.PeakBandwidthBytesPerSec > 0 {
			elemBytes := elemBytesFromDtype(computeDtype)
			dataBytes := float64(shape[0]) * elemBytes * 2 // read + write
			lat := dataBytes / profile.Hardware.PeakBandwidthBytesPerSec * 1e6
			slog.Warn("op using bandwidth roofline (no calibration curves)",
				"op", op, "dtype", computeDtype, "N", shape[0], "backend", backend)
			return lat, nil
		}
		return 0, fmt.Errorf("uncalibrated: %s(%s on %s)", op, computeDtype, backend)
```

Note: `slog` is already imported in `estimate.go`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `go test ./perf/ -run "TestLookupLatencyV3_MulMat|TestLookupLatency_1DOp" -v`
Expected: PASS

- [ ] **Step 7: Run all perf tests**

Run: `go test ./perf/ -count=1`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add perf/estimate.go perf/estimate_test.go
git commit -m "perf: add explicit fallback chain with warnings in all lookup paths"
```

---

### Task 6: Commit FLASH_ATTN_EXT shape fix

This task commits the FLASH_ATTN_EXT shape index fix that was already made earlier in this session. It fixes FLASH_ATTN_EXT prefill prediction from 1000x error to ~1x.

**Files:**
- Already modified: `perf/estimate.go:38-44` — changed `InputShapes[x][2]` to `InputShapes[x][1]`
- Already modified: `perf/estimate_test.go` — updated FLASH_ATTN InputShapes in multiple tests
- Already modified: `perf/integration_test.go` — updated FLASH_ATTN InputShapes

- [ ] **Step 1: Verify all tests pass**

Run: `go test ./perf/ -count=1`
Expected: PASS

- [ ] **Step 2: Commit FLASH_ATTN fix**

```bash
git add perf/estimate.go perf/estimate_test.go perf/integration_test.go
git commit -m "perf: fix FLASH_ATTN_EXT shape extraction — use ne[1] (seqlen) not ne[2] (num_heads)"
```

---

### Task 7: E2E validation

**Files:** None (validation only)

- [ ] **Step 1: Run `daop-estimate` with fixed code**

Run: `go run . daop-estimate qwen3:1.7b 2>&1`

Expected: q4_K MUL_MAT ops should emit "using fallback dtype" warnings (q4_K -> q4_0) since q4_K calibration data doesn't exist yet. FLASH_ATTN_EXT should now show large values for prefill (not 1.2ms).

- [ ] **Step 2: Verify no regressions**

Check that:
- Decode estimate is still reasonable (~64ms range for qwen3:1.7b before re-benchmarking)
- Prefill estimate now includes realistic FLASH_ATTN_EXT values
- Warnings are printed for q4_K/q6_K fallback

- [ ] **Step 3: Re-run `daop-bench` (optional, takes ~10min)**

After re-benchmarking, q4_K/q6_K direct curves will be available. Warnings should disappear. Prefill MUL_MAT accuracy should improve from ~0.55x to ~1.0x.

Run: `go run . daop-bench`

---

## Execution Notes

- Tasks 1-5 form the dtype pipeline — must be done in order.
- Task 6 is independent (FLASH_ATTN fix already done, just needs commit). Can be done first or in parallel.
- Task 7 is validation — run after all code changes.
- **Block size constraint:** q4_K super-blocks are 256 elements. The benchmark grid `{512, 2048, 8192}` is divisible by 256, so no alignment issues. If grid values change in the future, they must remain multiples of 256.
