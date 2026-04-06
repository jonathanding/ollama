# MUL_MAT Interpolation Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace roofline-based MUL_MAT prediction with direct interpolation from a 3×3 (M,K) grid + 3 strategic N values per grid point. Target: <2x error for qwen3:1.7b decode (currently 3.6-6x).

**Architecture:** 9 (M,K) pairs × 4 dtypes × 3 N values = 108 measurements. Two-stage interpolation: (1) piecewise linear in log-N per grid point, (2) IDW blend in log-(M,K) space. Roofline demoted to validator/fallback. `InterpolateMulMat()` already implements the two-stage pattern — each "curve" simply has 3 points instead of 8-10.

**Tech Stack:** Go 1.24, testify

**Design Spec:** `docs/superpowers/specs/2026-04-03-daop-v2-empirical-design.md` — Phase 2 (Sections 2.1–2.7)

---

## File Structure

### Modified Files

| File | Changes |
|------|---------|
| `perf/registry.go` | `Phase1MulMatFixedDims()` → 3×3 grid {512, 2048, 8192}² |
| `perf/bench.go` | `benchmarkMulMat()` → measure at 3 fixed N values (1, 32, 512); remove efficiency extraction from benchmark loop |
| `perf/interpolate.go` | `InterpolateMulMat()` → add scaling fallback for (M,K) outside grid range |
| `perf/estimate.go` | `lookupLatencyV3()` → route ALL MUL_MAT through `PredictMulMatDirect`, remove VEC/MAT split |
| `perf/bench_test.go` | Update `TestBuildSamplingGrids_MulMat`, `TestBenchmarkMulMat_OutputShapeContract` |
| `perf/interpolate_test.go` | Add `TestInterpolateMulMat_Extrapolation` |
| `perf/plan_test.go` | Update `TestBuildBenchmarkPlan_MulMatGeneratesRefCurves` ref count |
| `perf/bench_integration_test.go` | Update `TestPlanFusedOpsExcludedFromMainSteps` ref count |

---

## Task 1: Update Phase1MulMatFixedDims to 3×3 Grid

**Files:**
- Modify: `perf/registry.go:315-326`
- Modify: `perf/bench_test.go:55-66`

The 6 hardcoded (M,K) pairs (including 70B shapes that crash iGPU) are replaced with a systematic 3×3 log-spaced grid.

- [ ] **Step 1: Write test for the new grid**

Update `TestBuildSamplingGrids_MulMat` in `perf/bench_test.go:55-66`:

```go
func TestBuildSamplingGrids_MulMat(t *testing.T) {
	grids := buildSamplingGrids("MUL_MAT", "f16", "q4_0")
	// 3×3 grid = 9 (M,K) pairs
	assert.Equal(t, 9, len(grids), "MUL_MAT should have 9 (M,K) grids from 3×3 grid")
	for _, g := range grids {
		assert.Equal(t, "MUL_MAT", g.Op)
		assert.Equal(t, "q4_0", g.WeightDtype)
		assert.NotNil(t, g.FixedDims)
		assert.Contains(t, g.FixedDims, "M")
		assert.Contains(t, g.FixedDims, "K")
	}

	// Verify specific grid values
	expectedDims := []int64{512, 2048, 8192}
	seen := make(map[[2]int64]bool)
	for _, g := range grids {
		M := g.FixedDims["M"]
		K := g.FixedDims["K"]
		seen[[2]int64{M, K}] = true
		assert.Contains(t, expectedDims, M, "M=%d not in expected grid values", M)
		assert.Contains(t, expectedDims, K, "K=%d not in expected grid values", K)
	}
	assert.Equal(t, 9, len(seen), "should have 9 unique (M,K) pairs")
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./perf/ -run TestBuildSamplingGrids_MulMat -v`
Expected: FAIL — currently returns 6 pairs, not 9

- [ ] **Step 3: Update Phase1MulMatFixedDims**

Replace `perf/registry.go:315-326`:

```go
// Phase1MulMatFixedDims returns the 3×3 log-spaced (M, K) grid for MUL_MAT benchmarks.
// Grid values: {512, 2048, 8192} — factor of 4 between steps.
// Covers any transformer architecture with dimensions in [512, 8192].
// Total: 9 (M,K) pairs × 4 dtypes × 3 N values = 108 measurements.
func Phase1MulMatFixedDims() [][2]int64 {
	gridValues := []int64{512, 2048, 8192}
	var pairs [][2]int64
	for _, m := range gridValues {
		for _, k := range gridValues {
			pairs = append(pairs, [2]int64{m, k})
		}
	}
	return pairs
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./perf/ -run TestBuildSamplingGrids_MulMat -v`
Expected: PASS

- [ ] **Step 5: Update ref count assertions in other tests**

In `perf/plan_test.go:79`, update the expected ref count:

```go
assert.Equal(t, len(Phase1Dtypes())*len(Phase1MulMatFixedDims()), refCount)
```

This line already uses `len(Phase1MulMatFixedDims())` dynamically — no change needed. But verify by running:

Run: `go test ./perf/ -run TestBuildBenchmarkPlan_MulMatGeneratesRefCurves -v`
Expected: PASS (the assertion is already dynamic)

Similarly, `perf/bench_integration_test.go:157`:

```go
assert.Equal(t, len(Phase1Dtypes())*len(Phase1MulMatFixedDims()), refCount, "MUL_MAT ref curves")
```

Run: `go test ./perf/ -run TestPlanFusedOpsExcludedFromMainSteps -v`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `go test ./perf/ -v 2>&1 | tail -30`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add perf/registry.go perf/bench_test.go
git commit -m "perf: replace MUL_MAT fixed dims with 3×3 log-spaced grid"
```

---

## Task 2: Replace benchmarkMulMat Adaptive Sweep with Strategic N Sampling

**Files:**
- Modify: `perf/bench.go:635-648` (benchmarkMulMat)
- Modify: `perf/bench.go:370-398` (RunBenchmark StepMulMatRef case — remove efficiency extraction)
- Modify: `perf/bench_test.go:119-154` (TestBenchmarkMulMat_OutputShapeContract)

Instead of `AdaptiveSample1D` sweeping N from 1 to 4096 (8+ points), measure at exactly 3 strategic N values per (M,K,dtype): N=1 (decode), N=32 (transition zone), N=512 (prefill).

N_cross is fixed at 32 — an empirical value in the BW→compute transition zone. The roofline formula gives values too small (~3 for f32) because it doesn't account for kernel launch overhead and partial compute/BW overlap.

- [ ] **Step 1: Replace benchmarkMulMat with strategic N sampling**

Replace `perf/bench.go:635-648`:

```go
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
```

- [ ] **Step 2: Remove efficiency extraction from RunBenchmark**

In `perf/bench.go`, in the `StepMulMatRef` case (around line 383-398), remove the entire efficiency extraction block. The prediction now uses direct interpolation, not roofline. Replace with just storing the curve:

```go
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
```

- [ ] **Step 3: Update TestBenchmarkMulMat_OutputShapeContract**

Replace `perf/bench_test.go:119-154`:

```go
// TestBenchmarkMulMat_OutputShapeContract verifies that benchmarkMulMat produces
// exactly 3 points with 1D shapes [N] compatible with InterpolateMulMat.
func TestBenchmarkMulMat_OutputShapeContract(t *testing.T) {
	// The new benchmarkMulMat measures at 3 strategic N values: 1, 32, 512
	strategicNs := []int64{1, strategicNcross, 512}

	for _, N := range strategicNs {
		// Simulate what benchmarkMulMat's measurement does:
		pt := LatencyPoint{
			Shape:     []int64{4096, 4096, N}, // what measureOp returns
			LatencyUs: float64(N) * 10,
		}
		pt.Shape = []int64{N} // what benchmarkMulMat overrides

		assert.Len(t, pt.Shape, 1, "MUL_MAT points must be 1D for InterpolateMulMat")
		assert.Equal(t, N, pt.Shape[0], "Shape[0] must be the sweep dimension N")
	}

	// Verify InterpolateMulMat can consume 3-point curves
	curves := []OperatorCurve{{
		Op: "MUL_MAT", FixedDims: map[string]int64{"M": 2048, "K": 2048},
		Points: []LatencyPoint{
			{Shape: []int64{1}, LatencyUs: 100.0},
			{Shape: []int64{32}, LatencyUs: 800.0},
			{Shape: []int64{512}, LatencyUs: 5000.0},
		},
	}}
	result := InterpolateMulMat(curves, 2048, 2048, 128)
	assert.Greater(t, result, 800.0, "N=128 should be > N=32 latency")
	assert.Less(t, result, 5000.0, "N=128 should be < N=512 latency")
	assert.False(t, math.IsNaN(result))
}
```

- [ ] **Step 4: Run tests**

Run: `go test ./perf/ -run "TestBenchmarkMulMat_OutputShapeContract|TestBuildBenchmarkPlan" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add perf/bench.go perf/bench_test.go
git commit -m "perf: replace MUL_MAT adaptive sweep with 3 strategic N points (1, 32, 512)"
```

---

## Task 3: Add Scaling Fallback to InterpolateMulMat for (M,K) Extrapolation

**Files:**
- Modify: `perf/interpolate.go:160-202` (InterpolateMulMat)
- Modify: `perf/interpolate_test.go` (add extrapolation tests)

Current `InterpolateMulMat` uses IDW (inverse distance weighting) which is an interpolation technique — it blends between nearby curves but does NOT extrapolate. For query (M,K) outside the grid (e.g., M=14336 for Llama-8B FFN), IDW degrades to nearest-neighbor without scaling.

Add physics-informed scaling: when query is outside the grid convex hull, scale the nearest curve's latency by `(M_q×K_q)/(M_ref×K_ref)` (BW-bound) or `(M_q×K_q×N_q)/(M_ref×K_ref×N_ref)` (compute-bound), blended by the roofline balance point.

- [ ] **Step 1: Write test for (M,K) extrapolation**

Add to `perf/interpolate_test.go`:

```go
func TestInterpolateMulMat_Extrapolation(t *testing.T) {
	// Grid with max M=8192. Query at M=16384 (2x outside grid).
	curves := []OperatorCurve{
		{
			Op: "MUL_MAT", WeightDtype: "f32",
			FixedDims: map[string]int64{"M": 8192, "K": 2048},
			Points: []LatencyPoint{
				{Shape: []int64{1}, LatencyUs: 1000},
				{Shape: []int64{32}, LatencyUs: 5000},
				{Shape: []int64{512}, LatencyUs: 80000},
			},
		},
		{
			Op: "MUL_MAT", WeightDtype: "f32",
			FixedDims: map[string]int64{"M": 2048, "K": 2048},
			Points: []LatencyPoint{
				{Shape: []int64{1}, LatencyUs: 250},
				{Shape: []int64{32}, LatencyUs: 1250},
				{Shape: []int64{512}, LatencyUs: 20000},
			},
		},
	}

	// At N=1 (BW-bound), latency ∝ M*K (weight size).
	// M=16384, K=2048: nearest is (8192, 2048) with lat=1000μs.
	// Scale factor = (16384*2048)/(8192*2048) = 2.0
	// Expected: ~2000μs
	lat := InterpolateMulMat(curves, 16384, 2048, 1)
	assert.InDelta(t, 2000, lat, 500, "BW-bound extrapolation should scale ~2x for 2x M")

	// At N=512 (compute-bound), latency ∝ M*K*N (FLOPs).
	// Same scale factor for fixed N: (16384*2048)/(8192*2048) = 2.0
	// Expected: ~160000μs
	lat = InterpolateMulMat(curves, 16384, 2048, 512)
	assert.InDelta(t, 160000, lat, 40000, "compute-bound extrapolation should scale ~2x for 2x M")

	// Inside grid range should still use IDW (no scaling)
	latInside := InterpolateMulMat(curves, 4096, 2048, 1)
	assert.Greater(t, latInside, 250.0, "inside grid should be > smallest curve")
	assert.Less(t, latInside, 1000.0, "inside grid should be < largest curve")
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./perf/ -run TestInterpolateMulMat_Extrapolation -v`
Expected: FAIL — current IDW returns ~1000μs (nearest neighbor) instead of ~2000μs for M=16384

- [ ] **Step 3: Add scaling fallback to InterpolateMulMat**

Replace the body of `InterpolateMulMat` in `perf/interpolate.go:165-202`:

```go
func InterpolateMulMat(curves []OperatorCurve, queryM, queryK, queryN int64) float64 {
	if len(curves) == 0 {
		return 0
	}

	type candidate struct {
		curve   *OperatorCurve
		logDist float64
	}

	var candidates []candidate
	for i := range curves {
		curveM := curves[i].FixedDims["M"]
		curveK := curves[i].FixedDims["K"]
		dM := math.Log(float64(queryM)) - math.Log(float64(curveM))
		dK := math.Log(float64(queryK)) - math.Log(float64(curveK))
		dist := math.Sqrt(dM*dM + dK*dK)
		candidates = append(candidates, candidate{&curves[i], dist})
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].logDist < candidates[j].logDist
	})

	// Exact match or single curve
	if candidates[0].logDist == 0 || len(candidates) == 1 {
		lat := Interpolate1DByDim(candidates[0].curve.Points, 0, queryN)
		if candidates[0].logDist == 0 {
			return lat
		}
		// Single curve, may need scaling
		return scaleMulMatLatency(lat, candidates[0].curve, queryM, queryK)
	}

	// Check if query is outside the grid convex hull.
	// Heuristic: if nearest distance > ln(2) (~0.69), the query is more than
	// 2x away from the nearest grid point in some dimension — use scaling.
	const extrapolationThreshold = 0.69 // ln(2)
	if candidates[0].logDist > extrapolationThreshold {
		lat := Interpolate1DByDim(candidates[0].curve.Points, 0, queryN)
		return scaleMulMatLatency(lat, candidates[0].curve, queryM, queryK)
	}

	// Inside grid: IDW blend between two nearest curves
	lat1 := Interpolate1DByDim(candidates[0].curve.Points, 0, queryN)
	lat2 := Interpolate1DByDim(candidates[1].curve.Points, 0, queryN)
	w1 := 1.0 / candidates[0].logDist
	w2 := 1.0 / candidates[1].logDist
	return (lat1*w1 + lat2*w2) / (w1 + w2)
}

// scaleMulMatLatency applies physics-informed scaling when extrapolating
// beyond the measured (M,K) grid. At any N, MUL_MAT latency scales
// proportionally to M*K (weight size dominates both BW and compute terms).
func scaleMulMatLatency(nearestLat float64, nearest *OperatorCurve, queryM, queryK int64) float64 {
	refM := nearest.FixedDims["M"]
	refK := nearest.FixedDims["K"]
	scaleFactor := float64(queryM*queryK) / float64(refM*refK)
	return nearestLat * scaleFactor
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./perf/ -run TestInterpolateMulMat_Extrapolation -v`
Expected: PASS

- [ ] **Step 5: Run existing interpolation tests**

Run: `go test ./perf/ -run "TestInterpolateMulMat" -v`
Expected: All existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add perf/interpolate.go perf/interpolate_test.go
git commit -m "perf: add scaling fallback to InterpolateMulMat for (M,K) extrapolation"
```

---

## Task 4: Route ALL MUL_MAT Through PredictMulMatDirect

**Files:**
- Modify: `perf/estimate.go:333-397` (lookupLatencyV3)
- Modify: `perf/estimate_test.go` (if exists, or create new tests in bench_test.go)

The current `lookupLatencyV3` splits MUL_MAT into VEC (N≤8 → `PredictMulMatDirect`) and MAT (N>8 → roofline). The new design routes ALL N through `PredictMulMatDirect`, with roofline as fallback only for old v2 profiles.

- [ ] **Step 1: Write test for unified MUL_MAT routing**

Add to `perf/bench_test.go` (or appropriate test file):

```go
func TestLookupLatencyV3_MulMatAllNUsesDirect(t *testing.T) {
	// Profile with 3×3 grid reference curves (3 points each)
	profile := &Profile{
		Version: 3,
		Hardware: HardwareProfile{
			PeakTOPS:                 map[string]float64{"f32": 64.3e9},
			PeakBandwidthBytesPerSec: 40.7e9,
			EfficiencyConstants: map[string]OpEfficiency{
				"MUL_MAT_f32": {ComputeEff: 0.93, BWEff: 0.45, OverheadUs: 500},
			},
		},
		Operators: []OperatorCurve{
			{
				Op: "MUL_MAT", WeightDtype: "f32",
				FixedDims:  map[string]int64{"M": 2048, "K": 2048},
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 500},
					{Shape: []int64{8}, LatencyUs: 800},
					{Shape: []int64{512}, LatencyUs: 50000},
				},
			},
			{
				Op: "MUL_MAT", WeightDtype: "f32",
				FixedDims:  map[string]int64{"M": 8192, "K": 2048},
				Dimensions: []string{"N"},
				Points: []LatencyPoint{
					{Shape: []int64{1}, LatencyUs: 2000},
					{Shape: []int64{8}, LatencyUs: 3000},
					{Shape: []int64{512}, LatencyUs: 200000},
				},
			},
		},
	}
	caps := GetBackendCapabilities("Vulkan")

	// N=1 (VEC range) — should use direct interpolation, NOT roofline
	lat1, err := lookupLatencyV3(profile, "MUL_MAT", []int64{2048, 2048, 1}, "f32", "f32", "Vulkan", &caps)
	require.NoError(t, err)
	assert.InDelta(t, 500.0, lat1, 50.0, "N=1 at exact grid point should be ~500μs from curve")

	// N=256 (MAT range) — should ALSO use direct interpolation
	lat256, err := lookupLatencyV3(profile, "MUL_MAT", []int64{2048, 2048, 256}, "f32", "f32", "Vulkan", &caps)
	require.NoError(t, err)
	assert.Greater(t, lat256, 800.0, "N=256 should be > N=8 latency")
	assert.Less(t, lat256, 50000.0, "N=256 should be < N=512 latency")

	// Query at (M,K) between grid points — should IDW blend
	latBlend, err := lookupLatencyV3(profile, "MUL_MAT", []int64{4096, 2048, 1}, "f32", "f32", "Vulkan", &caps)
	require.NoError(t, err)
	assert.Greater(t, latBlend, 500.0, "IDW blend should be > nearest small curve")
	assert.Less(t, latBlend, 2000.0, "IDW blend should be < nearest large curve")
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./perf/ -run TestLookupLatencyV3_MulMatAllNUsesDirect -v`
Expected: FAIL — current code routes N=256 through roofline, not direct

- [ ] **Step 3: Simplify lookupLatencyV3 MUL_MAT case**

Replace the `case "MUL_MAT":` block in `perf/estimate.go:338-365`:

```go
	case "MUL_MAT", "MUL_MAT_ADD":
		if len(shape) < 3 {
			return 0, fmt.Errorf("MUL_MAT requires 3 shape dims, got %d", len(shape))
		}
		M, K, N := shape[0], shape[1], shape[2]
		mappedWdt := mapWeightDtype(weightDtype)

		// Primary: direct interpolation from reference curves (Phase 2 design)
		lat := PredictMulMatDirect(profile, M, K, N, mappedWdt)
		if lat > 0 {
			return lat, nil
		}
		// Fallback: roofline (for backward compatibility with v2 profiles
		// that lack multi-(M,K) reference curves)
		lat = PredictMulMatLatency(&profile.Hardware, M, K, N, mappedWdt)
		if lat > 0 {
			return lat, nil
		}
		return 0, fmt.Errorf("no MUL_MAT calibration data for dtype %s — run daop-bench first", mappedWdt)
```

This removes:
- The VEC/MAT split (N≤MulMatVecMaxN check)
- The `PredictMulMatVecLatency` call
- The separate `MUL_MAT_ADD` case

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./perf/ -run TestLookupLatencyV3_MulMatAllNUsesDirect -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `go test ./perf/ -v 2>&1 | tail -30`
Expected: All tests pass. Some existing tests may need adjustment if they relied on the VEC/MAT split.

- [ ] **Step 6: Commit**

```bash
git add perf/estimate.go perf/bench_test.go
git commit -m "perf: route all MUL_MAT through direct interpolation, roofline as fallback"
```

---

## Task 5: End-to-End Validation

**Files:** No new files — benchmark run + estimate comparison

This task runs the benchmark with the new grid and validates accuracy against actual inference.

- [ ] **Step 1: Run the benchmark**

```bash
go run . daop-bench 2>&1 | tee /tmp/bench-output.txt
```

Verify:
- 108 MUL_MAT measurements complete without TDR crash (max shape 8192×8192 is safe)
- Benchmark takes ~3-4 minutes for MUL_MAT portion
- Profile contains 9 (M,K) × 4 dtype = 36 MUL_MAT curves, each with 3 points

- [ ] **Step 2: Check profile structure**

```bash
go run . daop-viewer --profile ~/.ollama/bench/profile.json 2>&1 | head -50
```

Or inspect JSON directly:
```bash
python3 -c "
import json
with open('$HOME/.ollama/bench/profile.json') as f:
    p = json.load(f)
mulmat = [c for c in p['operators'] if c['op'] == 'MUL_MAT']
print(f'MUL_MAT curves: {len(mulmat)}')
for c in mulmat:
    dims = c.get('fixed_dims', {})
    pts = len(c.get('points', []))
    print(f'  M={dims.get(\"M\")}, K={dims.get(\"K\")}, dtype={c.get(\"weight_dtype\")}, points={pts}')
"
```

Expected: 36 curves (9 × 4), each with exactly 3 points.

- [ ] **Step 3: Run estimate for qwen3:1.7b**

```bash
go run . daop-estimate qwen3:1.7b --json 2>&1 | tee /tmp/estimate-output.txt
```

Compare:
- Actual: ~75 ms/tok (13.3 tok/s) for decode
- Target: estimate within 2x → 37-150 ms/tok (6.7-27 tok/s)
- Previous: 272-450 ms/tok (3.6-6x error)

- [ ] **Step 4: Diagnose if needed**

If error > 2x, check:
1. `--json` output for per-op breakdown: which ops contribute most error?
2. Are the reference curves reasonable? Check that N=1 points match actual decode perf.
3. IDW interpolation: are qwen3:1.7b shapes well-covered by the grid?
   - (512, 2048) — exact hit
   - (2048, 2048) — exact hit
   - (8960, 2048) — interpolates from (8192, 2048), should be close
   - (2048, 8960) — interpolates from (2048, 8192), should be close

- [ ] **Step 5: Commit validated state**

```bash
git add -A
git commit -m "perf: validate MUL_MAT interpolation redesign — <Xx error for qwen3:1.7b"
```

(Replace `<X` with actual measured error ratio.)

---

## Self-Review Notes

1. **InterpolateMulMat already works with 3 points**: The existing implementation (`perf/interpolate.go:165`) uses `Interpolate1DByDim` per curve (piecewise linear in log-log), then IDW blends in (M,K) space. 3 points per curve is fine — just 2 segments instead of 7-9.

2. **No efficiency extraction**: Efficiency constants are no longer extracted during benchmark. The prediction is entirely interpolation-based. Roofline with efficiency constants only serves as a fallback for old v2 profiles. The `extractEfficiencyConstants` function and `OpEfficiency` type remain in the codebase for backward compatibility but are not called from the new benchmark path.

3. **N_cross=32 is empirical**: The roofline formula gives N_cross ≈ 3 for f32 on Intel iGPU, which is too small (N=1 and N=3 would be nearly identical measurements). N=32 sits in the transition zone where both BW and compute effects are visible, providing a meaningful middle sample point.

4. **Scaling fallback uses M×K proportionality**: Both BW-bound (`lat ∝ M×K×elemBytes`) and compute-bound (`lat ∝ M×K×N`) have M×K as a common factor. For a fixed N, scaling by `(M_q×K_q)/(M_ref×K_ref)` is correct in both regimes. This avoids needing to detect which regime the query is in.

5. **Extrapolation threshold ln(2)**: A query more than 2× away from the nearest grid point in any dimension triggers scaling instead of IDW. For the {512, 2048, 8192} grid with 4× steps, this means shapes up to ~11,585 (8192×√2) use IDW, and beyond that use scaling. Llama-8B's 14336 would use scaling.

6. **MUL_MAT_ADD routing**: Both `MUL_MAT` and `MUL_MAT_ADD` now go through the same path. Since fusion replaces separate MUL_MAT+ADD nodes with a single `MUL_MAT_ADD` node before estimation, and the latency of the fused op ≈ MUL_MAT alone (ADD overhead <1μs), this is correct.

7. **Backward compatibility**: Profiles without multi-(M,K) curves (v2 or early v3) fall back to roofline via `PredictMulMatLatency`. No breakage.
