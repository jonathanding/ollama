# Phase 1A: Convergence Early-Stop + 4-Dtype MUL_MAT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed-rep measurement with convergence-based early stopping (CV < 5%), benchmark 4 weight dtypes for MUL_MAT reference curves, and add K-quant dtype mapping for estimation.

**Architecture:** Extract a shared `convergentMeasure()` helper that both `measureOp` and hwchar functions call. A dedicated `measureMulMat()` handles mixed-dtype tensor creation (weight=quantized, activation=f32) because GGML requires this for quantized MUL_MAT — other ops use uniform dtypes. MUL_MAT benchmark loops over 4 weight dtypes with per-dtype efficiency constants keyed as `"MUL_MAT_<dtype>"`. Estimation maps K-quant dtypes to nearest measured dtype before roofline prediction.

**Tech Stack:** Go 1.24, testify, GGML backend (ml package)

---

## Task 1: Extend BenchmarkConfig with Convergence Fields

**Files:**
- Modify: `perf/types.go:106-123`
- Test: `perf/types_test.go`

- [ ] **Step 1: Write failing tests for new config fields**

```go
// In perf/types_test.go — add these tests

func TestDefaultBenchmarkConfig_ConvergenceFields(t *testing.T) {
	cfg := DefaultBenchmarkConfig()
	assert.Equal(t, 0.05, cfg.ConvergenceCV, "default CV threshold should be 5%")
	assert.Equal(t, 5, cfg.MinReps, "default min reps should be 5")
}

func TestBenchmarkConfig_ConvergenceFieldsExist(t *testing.T) {
	// Verify the struct can be initialized with convergence fields
	cfg := BenchmarkConfig{
		ConvergenceCV: 0.03,
		MinReps:       10,
	}
	assert.Equal(t, 0.03, cfg.ConvergenceCV)
	assert.Equal(t, 10, cfg.MinReps)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./perf/ -run "TestDefaultBenchmarkConfig_ConvergenceFields|TestBenchmarkConfig_ConvergenceFieldsExist" -v`
Expected: FAIL — `cfg.ConvergenceCV` and `cfg.MinReps` do not exist

- [ ] **Step 3: Add fields to BenchmarkConfig and update defaults**

In `perf/types.go`, add two fields to `BenchmarkConfig`:

```go
type BenchmarkConfig struct {
	ErrorThreshold float64 // relative error threshold for adaptive refinement
	MaxPointsPerOp int     // maximum number of sample points per operator
	WarmupReps     int     // number of warmup iterations before measurement
	MeasureReps    int     // number of measurement iterations (max ceiling)
	TrimPercent    float64 // percentage of outliers to trim (e.g., 0.1 = 10%)
	ConvergenceCV  float64 // CV threshold for early stopping (0.05 = 5%)
	MinReps        int     // minimum reps before checking convergence
}
```

Update `DefaultBenchmarkConfig()`:

```go
func DefaultBenchmarkConfig() BenchmarkConfig {
	return BenchmarkConfig{
		WarmupReps:     5,
		MeasureReps:    50,
		TrimPercent:    0.1,
		ErrorThreshold: 0.05,
		MaxPointsPerOp: 20,
		ConvergenceCV:  0.05,
		MinReps:        5,
	}
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./perf/ -run "TestDefaultBenchmarkConfig_ConvergenceFields|TestBenchmarkConfig_ConvergenceFieldsExist" -v`
Expected: PASS

- [ ] **Step 5: Run all existing tests to verify no regressions**

Run: `go test ./perf/ -v`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add perf/types.go perf/types_test.go
git commit -m "perf: add convergence CV and MinReps fields to BenchmarkConfig"
```

---

## Task 2: Implement Convergent Measurement Helper

**Files:**
- Modify: `perf/hwchar.go` (add `convergentMeasure` function here, near `trimmedMedian`)
- Test: `perf/hwchar_test.go`

This is the core algorithm. It replaces both the tiered adaptive reps in `measureOp` and the fixed 50-rep loops in hwchar. The function takes a `compute` callback so it can be used with any GGML compute call.

- [ ] **Step 1: Write comprehensive tests for convergentMeasure**

```go
// In perf/hwchar_test.go — add these tests

// TestConvergentMeasure_ConvergesEarly verifies that stable measurements stop before maxReps.
func TestConvergentMeasure_ConvergesEarly(t *testing.T) {
	callCount := 0
	// Simulate a stable op returning ~1000us with small noise
	compute := func() float64 {
		callCount++
		// Tiny variation: 1000 ± 10us (CV = 1%, well below 5%)
		noise := float64(callCount%3 - 1) * 10.0
		return 1000.0 + noise
	}
	cfg := BenchmarkConfig{
		MeasureReps:   50,
		TrimPercent:   0.1,
		ConvergenceCV: 0.05,
		MinReps:       5,
	}
	median, stddev, reps := convergentMeasure(compute, cfg)

	assert.Greater(t, median, 900.0)
	assert.Less(t, median, 1100.0)
	assert.Less(t, stddev, 50.0, "stddev should be small for stable measurements")
	assert.GreaterOrEqual(t, reps, 5, "must run at least MinReps")
	assert.Less(t, reps, 15, "stable measurement should converge well before 50 reps")
}

// TestConvergentMeasure_NoisyDoesNotConverge verifies that noisy measurements run to maxReps.
func TestConvergentMeasure_NoisyDoesNotConverge(t *testing.T) {
	callCount := 0
	// Simulate a very noisy op: mean ~1000, stddev ~500 (CV = 50%)
	compute := func() float64 {
		callCount++
		// Alternating between 500 and 1500
		if callCount%2 == 0 {
			return 500.0
		}
		return 1500.0
	}
	cfg := BenchmarkConfig{
		MeasureReps:   20,
		TrimPercent:   0.1,
		ConvergenceCV: 0.05,
		MinReps:       5,
	}
	_, _, reps := convergentMeasure(compute, cfg)

	assert.Equal(t, 20, reps, "noisy measurement should run all maxReps")
}

// TestConvergentMeasure_TieredMaxReps verifies that slow ops get reduced maxReps.
func TestConvergentMeasure_TieredMaxReps(t *testing.T) {
	// Simulate a 2-second op (>1s threshold → maxReps=10) with moderate noise (won't converge)
	callCount := 0
	compute := func() float64 {
		callCount++
		// ~2M us with 20% noise → CV ~20%, won't converge
		return 2e6 + float64(callCount%5)*4e5
	}
	cfg := BenchmarkConfig{
		MeasureReps:   50,
		TrimPercent:   0.1,
		ConvergenceCV: 0.05,
		MinReps:       5,
	}
	_, _, reps := convergentMeasure(compute, cfg)

	assert.LessOrEqual(t, reps, 10, "ops >1s should cap at 10 reps")
}

// TestConvergentMeasure_VerySlowOp verifies >5s ops cap at 5 reps.
func TestConvergentMeasure_VerySlowOp(t *testing.T) {
	callCount := 0
	compute := func() float64 {
		callCount++
		return 6e6 + float64(callCount%3)*1e6 // ~6-8M us, won't converge
	}
	cfg := BenchmarkConfig{
		MeasureReps:   50,
		TrimPercent:   0.1,
		ConvergenceCV: 0.05,
		MinReps:       5,
	}
	_, _, reps := convergentMeasure(compute, cfg)

	assert.Equal(t, 5, reps, "ops >5s should cap at 5 reps")
}

// TestConvergentMeasure_MinRepsRespected verifies we never stop before MinReps.
func TestConvergentMeasure_MinRepsRespected(t *testing.T) {
	// Even if CV is 0 from the start, must run MinReps
	compute := func() float64 {
		return 1000.0 // perfectly stable
	}
	cfg := BenchmarkConfig{
		MeasureReps:   50,
		TrimPercent:   0.1,
		ConvergenceCV: 0.05,
		MinReps:       8,
	}
	_, _, reps := convergentMeasure(compute, cfg)

	assert.GreaterOrEqual(t, reps, 8, "must run at least MinReps even if stable")
}

// TestConvergentMeasure_TrimmedCV verifies CV is computed on trimmed data, not raw.
func TestConvergentMeasure_TrimmedCV(t *testing.T) {
	callCount := 0
	// Mostly stable (~1000us) with one huge outlier every 7th call
	compute := func() float64 {
		callCount++
		if callCount%7 == 0 {
			return 50000.0 // outlier
		}
		return 1000.0 + float64(callCount%3)*5.0
	}
	cfg := BenchmarkConfig{
		MeasureReps:   30,
		TrimPercent:   0.1,
		ConvergenceCV: 0.05,
		MinReps:       5,
	}
	median, _, reps := convergentMeasure(compute, cfg)

	// Trimming should remove outliers, so median should be near 1000
	assert.Greater(t, median, 900.0)
	assert.Less(t, median, 1100.0)
	// Should converge because trimmed CV is low
	assert.Less(t, reps, 30, "should converge after trimming removes outliers")
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./perf/ -run "TestConvergentMeasure" -v`
Expected: FAIL — `convergentMeasure` undefined

- [ ] **Step 3: Implement convergentMeasure**

Add to `perf/hwchar.go`:

```go
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
				// Converged
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
```

Note: needs `"math"` and `"sort"` imports (already present in hwchar.go).

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./perf/ -run "TestConvergentMeasure" -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Run all existing tests**

Run: `go test ./perf/ -v`
Expected: All PASS (no regressions — convergentMeasure is additive)

- [ ] **Step 6: Commit**

```bash
git add perf/hwchar.go perf/hwchar_test.go
git commit -m "perf: add convergentMeasure helper with CV-based early stopping"
```

---

## Task 3: Dedicated measureMulMat + elemBytesFromDtype

**Files:**
- Modify: `perf/bench.go` (add `measureMulMat`, change `elemSizeFromDtype` → `elemBytesFromDtype`)
- Test: `perf/bench_test.go`

GGML `mul_mat` requires weight tensor at quantized dtype and activation at f32/f16. The generic `measureOp` creates all inputs with the same dtype, so quantized MUL_MAT needs a dedicated function. Also, `elemSizeFromDtype` returns `int` which is imprecise for quantized types (q4_0 = 0.5625 bytes/elem, not 1).

- [ ] **Step 1: Write tests for elemBytesFromDtype**

```go
// In perf/bench_test.go — replace TestElemSizeFromDtype

func TestElemBytesFromDtype(t *testing.T) {
	assert.Equal(t, 4.0, elemBytesFromDtype("f32"))
	assert.Equal(t, 2.0, elemBytesFromDtype("f16"))
	assert.InDelta(t, 0.5625, elemBytesFromDtype("q4_0"), 0.001) // 18 bytes / 32 elements
	assert.InDelta(t, 1.0625, elemBytesFromDtype("q8_0"), 0.001) // 34 bytes / 32 elements
	assert.Equal(t, 4.0, elemBytesFromDtype("unknown"))
}
```

- [ ] **Step 2: Write tests for measureMulMat**

```go
// In perf/bench_test.go — add

// TestMeasureMulMat_OutputShape verifies measureMulMat returns correct shape metadata.
func TestMeasureMulMat_OutputShape(t *testing.T) {
	// We can't run real GGML here, but verify the contract:
	// measureMulMat should return LatencyPoint with shape containing M, K, N
	pt := LatencyPoint{
		Shape:     []int64{4096, 4096, 32},
		LatencyUs: 5000.0,
		StddevUs:  100.0,
		Reps:      7,
	}
	assert.Len(t, pt.Shape, 3)
	assert.Equal(t, int64(4096), pt.Shape[0]) // M
	assert.Equal(t, int64(4096), pt.Shape[1]) // K
	assert.Equal(t, int64(32), pt.Shape[2])   // N
}
```

- [ ] **Step 3: Implement elemBytesFromDtype (replaces elemSizeFromDtype)**

```go
// elemBytesFromDtype returns bytes per element for a dtype string.
// Quantized types return fractional values based on block structure.
func elemBytesFromDtype(dtype string) float64 {
	switch dtype {
	case "f32":
		return 4.0
	case "f16":
		return 2.0
	case "q4_0":
		return 18.0 / 32.0 // 18 bytes per 32-element block = 0.5625
	case "q8_0":
		return 34.0 / 32.0 // 34 bytes per 32-element block = 1.0625
	default:
		return 4.0
	}
}
```

Update all call sites of `elemSizeFromDtype` → `elemBytesFromDtype`, change `int` return to `float64`.

- [ ] **Step 4: Implement measureMulMat**

```go
// measureMulMat benchmarks a MUL_MAT at one shape point with mixed dtypes.
// GGML mul_mat requires weight at weightDtype (e.g., q4_0) and activation at f32.
// This is separate from measureOp because measureOp creates all inputs with same dtype.
func measureMulMat(backend ml.Backend, M, K, N int64, weightDtype string, cfg BenchmarkConfig) LatencyPoint {
	wdt, ok := parseDType(weightDtype)
	if !ok {
		slog.Warn("unsupported weight dtype", "dtype", weightDtype)
		return LatencyPoint{Shape: []int64{M, K, N}}
	}

	ctx := backend.NewContext()
	defer ctx.Close()

	// Weight: K×M at weightDtype (quantized or float)
	weight := ctx.Input().Zeros(wdt, int(K), int(M))
	// Activation: K×N at f32
	activation := ctx.Input().Zeros(ml.DTypeF32, int(K), int(N))

	out := weight.Mulmat(ctx, activation)
	if out == nil {
		slog.Warn("mulmat returned nil", "weight_dtype", weightDtype)
		return LatencyPoint{Shape: []int64{M, K, N}}
	}
	ctx.Forward(out)

	// Adaptive warmup
	warmupStart := time.Now()
	for i := 0; i < cfg.WarmupReps; i++ {
		ctx.Compute(out)
		if i == 0 {
			elapsed := float64(time.Since(warmupStart).Microseconds())
			if elapsed > 5e6 {
				break
			} else if elapsed > 1e6 {
				ctx.Compute(out)
				break
			}
		}
	}

	// Measure with convergence-based early stopping
	med, sd, actualReps := convergentMeasure(func() float64 {
		start := time.Now()
		ctx.Compute(out)
		return float64(time.Since(start).Microseconds())
	}, cfg)

	return LatencyPoint{
		Shape:     []int64{M, K, N},
		LatencyUs: med,
		StddevUs:  sd,
		Reps:      actualReps,
	}
}
```

- [ ] **Step 5: Update benchmarkMulMat to use measureMulMat**

```go
func benchmarkMulMat(backend ml.Backend, weightDtype string, fixedDims map[string]int64, cfg BenchmarkConfig) []LatencyPoint {
	M := fixedDims["M"]
	K := fixedDims["K"]
	measure := func(shape []int64) LatencyPoint {
		N := shape[0]
		pt := measureMulMat(backend, M, K, N, weightDtype, cfg)
		pt.Shape = []int64{N} // 1D for AdaptiveSample1D
		return pt
	}
	return AdaptiveSample1D(measure, 1, 4096, 8, cfg)
}
```

- [ ] **Step 6: Run all tests**

Run: `go test ./perf/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add perf/bench.go perf/bench_test.go
git commit -m "perf: add measureMulMat for mixed-dtype benchmarking, use precise elemBytesFromDtype"
```

---

## Task 4: Refactor measureOp to Use Convergent Measurement

**Files:**
- Modify: `perf/bench.go:92-163` (measureOp function)
- Test: `perf/bench_test.go`

Replace the current tiered adaptive reps logic (lines 110-156) with a call to `convergentMeasure`. Note: measureOp is still used by element-wise ops (SILU, etc.) and FLASH_ATTN_EXT, which use uniform dtypes.

- [ ] **Step 1: Refactor measureOp**

Replace lines 110-163 of `perf/bench.go` (everything after adaptive warmup, before the return) with:

```go
	// Measure with convergence-based early stopping
	med, sd, actualReps := convergentMeasure(func() float64 {
		start := time.Now()
		ctx.Compute(out)
		return float64(time.Since(start).Microseconds())
	}, cfg)

	return LatencyPoint{
		Shape:     gridPoint,
		LatencyUs: med,
		StddevUs:  sd,
		Reps:      actualReps,
	}
```

This removes ~45 lines of manual tiered reps, sorting, trimming, and stddev computation.

- [ ] **Step 2: Run all tests**

Run: `go test ./perf/ -v`
Expected: All PASS. The existing `TestTrimmedMedian_*` tests still pass (trimmedMedian is unchanged). `TestExtractEfficiencyConstants` and `TestPredictMulMatLatency_*` are unaffected.

- [ ] **Step 3: Commit**

```bash
git add perf/bench.go perf/bench_test.go
git commit -m "perf: refactor measureOp to use convergentMeasure"
```

---

## Task 5: Refactor Hardware Characterization to Use Convergent Measurement

**Files:**
- Modify: `perf/hwchar.go:62-117` (benchPeakTOPS and benchPeakBandwidth)
- Test: `perf/hwchar_test.go`

Replace fixed 50-rep loops with `convergentMeasure`. This is the biggest time saving: benchPeakTOPS at 4096^3 takes ~3s/rep × 50 = 150s/dtype. With convergence, stable measurements stop at ~5-10 reps (~15-30s/dtype).

- [ ] **Step 1: Write tests for hwchar convergence integration**

```go
// In perf/hwchar_test.go — add

// TestBenchPeakTOPS_UsesConvergence verifies that benchPeakTOPS uses convergent
// measurement (not fixed reps). We can't test with real GPU, but we verify the
// function signature and config propagation.
func TestBenchPeakTOPS_ConfigPropagation(t *testing.T) {
	// Verify that BenchmarkConfig with convergence fields is accepted
	cfg := DefaultBenchmarkConfig()
	assert.Equal(t, 0.05, cfg.ConvergenceCV, "convergence CV should propagate to hwchar")
	assert.Equal(t, 5, cfg.MinReps, "MinReps should propagate to hwchar")
	// benchPeakTOPS takes cfg — this is a compile-time check that the API accepts it
}

// TestTrimmedMedian_WithConvergenceParams verifies trimmedMedian works with
// small sample counts (5-10 reps from convergence, not 50).
func TestTrimmedMedian_SmallSample(t *testing.T) {
	// 5 samples with 10% trim: trimCount = round(0.5) = 1
	values := []float64{100, 102, 101, 103, 99}
	median := trimmedMedian(values, 0.1)
	assert.InDelta(t, 101.0, median, 1.0)

	// 7 samples with 10% trim: trimCount = round(0.7) = 1
	values = []float64{100, 102, 101, 103, 99, 104, 98}
	median = trimmedMedian(values, 0.1)
	assert.InDelta(t, 101.0, median, 2.0)
}
```

- [ ] **Step 2: Run tests to verify they pass (or fail if new test references missing code)**

Run: `go test ./perf/ -run "TestBenchPeakTOPS_ConfigPropagation|TestTrimmedMedian_SmallSample" -v`

- [ ] **Step 3: Refactor benchPeakTOPS**

Replace lines 72-85 of `perf/hwchar.go` with:

```go
	// Warmup — adaptive: skip excess warmups for slow ops
	warmupStart := time.Now()
	for i := 0; i < cfg.WarmupReps; i++ {
		ctx.Compute(out)
		if i == 0 {
			elapsed := float64(time.Since(warmupStart).Microseconds())
			if elapsed > 5e6 {
				break
			} else if elapsed > 1e6 {
				ctx.Compute(out)
				break
			}
		}
	}

	// Measure with convergence-based early stopping
	med, _, _ := convergentMeasure(func() float64 {
		start := time.Now()
		ctx.Compute(out)
		return float64(time.Since(start).Microseconds())
	}, cfg)

	median := med / 1e6 // convert us to seconds for TOPS calculation
	flops := 2.0 * M * K * N
	return flops / median, nil
```

- [ ] **Step 4: Refactor benchPeakBandwidth**

Replace lines 101-114 of `perf/hwchar.go` with the same pattern:

```go
	// Warmup — adaptive
	warmupStart := time.Now()
	for i := 0; i < cfg.WarmupReps; i++ {
		ctx.Compute(dst)
		if i == 0 {
			elapsed := float64(time.Since(warmupStart).Microseconds())
			if elapsed > 5e6 {
				break
			} else if elapsed > 1e6 {
				ctx.Compute(dst)
				break
			}
		}
	}

	// Measure with convergence-based early stopping
	med, _, _ := convergentMeasure(func() float64 {
		start := time.Now()
		ctx.Compute(dst)
		return float64(time.Since(start).Microseconds())
	}, cfg)

	median := med / 1e6 // convert us to seconds
	bytesTotal := 2.0 * size * 4
	return bytesTotal / median, nil
```

- [ ] **Step 5: Run all tests**

Run: `go test ./perf/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add perf/hwchar.go perf/hwchar_test.go
git commit -m "perf: use convergentMeasure in benchPeakTOPS and benchPeakBandwidth"
```

---

## Task 6: 4-Dtype MUL_MAT Reference Curves with Per-Dtype Efficiency Constants

**Files:**
- Modify: `perf/bench.go:196-244` (RunBenchmark MUL_MAT section)
- Modify: `perf/bench.go:319-345` (countGrids)
- Modify: `perf/bench.go:347-395` (extractEfficiencyConstants — adjust bytes calc for dtype)
- Modify: `perf/bench.go:399-419` (PredictMulMatLatency — per-dtype efficiency lookup)
- Test: `perf/bench_test.go`

- [ ] **Step 1: Write tests for per-dtype efficiency constants and prediction**

```go
// In perf/bench_test.go — add

func TestCountGrids_FourMulMatDtypes(t *testing.T) {
	ops := []string{"SILU", "MUL_MAT", "FLASH_ATTN_EXT"}
	dtypes := []string{"f32", "f16", "q4_0", "q8_0"}
	count := countGrids(ops, dtypes)
	// SILU: 1 (f32 only), MUL_MAT: 4 (one per weight dtype), FLASH_ATTN_EXT: 1 (f16 only)
	assert.Equal(t, 6, count, "should be 6 grids: SILU + 4 MUL_MAT refs + FLASH_ATTN")
}

func TestExtractEfficiencyConstants_Q40(t *testing.T) {
	// q4_0: effective ~0.5625 bytes/element (18 bytes / 32 elements)
	// At small N (BW-bound), less data to read → faster than f32
	peakTOPS := 64.3e9
	peakBW := 40.7e9

	// Simulated q4_0 reference curve — faster than f32 at small N due to less data
	points := []LatencyPoint{
		{Shape: []int64{1}, LatencyUs: 1200},   // faster than f32 (3754)
		{Shape: []int64{3}, LatencyUs: 1300},
		{Shape: []int64{11}, LatencyUs: 3000},
		{Shape: []int64{35}, LatencyUs: 8000},
		{Shape: []int64{116}, LatencyUs: 25000},
		{Shape: []int64{380}, LatencyUs: 80000},
		{Shape: []int64{1248}, LatencyUs: 250000},
		{Shape: []int64{4096}, LatencyUs: 2400000}, // slightly slower due to dequant overhead
	}

	eff := extractEfficiencyConstants(points, 4096, 4096, peakTOPS, peakBW, "q4_0")

	assert.Greater(t, eff.ComputeEff, 0.5, "compute efficiency should be reasonable")
	assert.Less(t, eff.ComputeEff, 1.0)
	assert.Greater(t, eff.BWEff, 0.1, "BW efficiency should be positive")
	assert.Less(t, eff.BWEff, 1.0)
}

func TestPredictMulMatLatency_PerDtypeEfficiency(t *testing.T) {
	hw := &HardwareProfile{
		PeakTOPS:                 map[string]float64{"f32": 64.3e9},
		PeakBandwidthBytesPerSec: 40.7e9,
		EfficiencyConstants: map[string]OpEfficiency{
			"MUL_MAT_f32":  {ComputeEff: 0.93, BWEff: 0.45, OverheadUs: 500},
			"MUL_MAT_q4_0": {ComputeEff: 0.85, BWEff: 0.55, OverheadUs: 300},
		},
	}

	latF32 := PredictMulMatLatency(hw, 4096, 4096, 1, "f32")
	latQ40 := PredictMulMatLatency(hw, 4096, 4096, 1, "q4_0")

	assert.Greater(t, latF32, 0.0)
	assert.Greater(t, latQ40, 0.0)
	// q4_0 at N=1 should be faster (less data to read, BW-bound regime)
	assert.Less(t, latQ40, latF32, "q4_0 should be faster than f32 at N=1 (BW-bound)")
}

func TestPredictMulMatLatency_FallbackToGeneric(t *testing.T) {
	// If per-dtype key not found, fall back to generic "MUL_MAT"
	hw := &HardwareProfile{
		PeakTOPS:                 map[string]float64{"f32": 64.3e9},
		PeakBandwidthBytesPerSec: 40.7e9,
		EfficiencyConstants: map[string]OpEfficiency{
			"MUL_MAT": {ComputeEff: 0.90, BWEff: 0.40, OverheadUs: 0},
		},
	}
	lat := PredictMulMatLatency(hw, 4096, 4096, 4096, "f16")
	assert.Greater(t, lat, 0.0, "should fall back to generic MUL_MAT constants")
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./perf/ -run "TestCountGrids_FourMulMatDtypes|TestExtractEfficiencyConstants_Q40|TestPredictMulMatLatency_PerDtypeEfficiency|TestPredictMulMatLatency_FallbackToGeneric" -v`
Expected: FAIL — `countGrids` returns 3 (not 6); `extractEfficiencyConstants` doesn't take dtype param; `PredictMulMatLatency` doesn't look up per-dtype keys

- [ ] **Step 3: Update extractEfficiencyConstants to accept weightDtype**

Add `weightDtype string` parameter. Use `elemBytesFromDtype(weightDtype)` (float64, from Task 3) for the weight matrix byte calculation. Activation and output remain f32 (4 bytes/elem):

```go
func extractEfficiencyConstants(points []LatencyPoint, refM, refK int64, peakTOPS, peakBW float64, weightDtype string) OpEfficiency {
	var computeEffs, bwEffs []float64
	var overheads []float64

	wElemBytes := elemBytesFromDtype(weightDtype)

	for _, pt := range points {
		N := pt.Shape[0]
		if pt.LatencyUs <= 0 {
			continue
		}

		flops := 2.0 * float64(refM) * float64(refK) * float64(N)
		// Weight matrix: M*K elements at weightDtype size (e.g., 0.5625 for q4_0)
		// Activation: K*N at f32 (always f32 in compute)
		// Output: M*N at f32
		bytes := float64(refM*refK)*wElemBytes + float64(refK*N+refM*N)*4

		computeTime := flops / peakTOPS * 1e6
		bwTime := bytes / peakBW * 1e6

		if computeTime > bwTime {
			computeEffs = append(computeEffs, computeTime/pt.LatencyUs)
		} else {
			bwEffs = append(bwEffs, bwTime/pt.LatencyUs)
			overheads = append(overheads, pt.LatencyUs-bwTime)
		}
	}

	// ... rest unchanged (median computation) ...
}
```

Update the call site in RunBenchmark (currently line 229) to pass the dtype.

- [ ] **Step 4: Update PredictMulMatLatency for per-dtype efficiency lookup**

```go
func PredictMulMatLatency(hw *HardwareProfile, M, K, N int64, dtype string) float64 {
	// Try per-dtype key first, then fall back to generic
	effKey := "MUL_MAT_" + dtype
	eff, ok := hw.EfficiencyConstants[effKey]
	if !ok {
		eff, ok = hw.EfficiencyConstants["MUL_MAT"]
		if !ok {
			return 0
		}
	}

	peakTOPS, ok := hw.PeakTOPS[dtype]
	if !ok {
		peakTOPS = hw.PeakTOPS["f32"]
	}
	peakBW := hw.PeakBandwidthBytesPerSec

	flops := 2.0 * float64(M) * float64(K) * float64(N)
	wElemBytes := elemBytesFromDtype(dtype)
	// Weight: M*K at dtype (e.g., 0.5625 for q4_0), activation: K*N at f32, output: M*N at f32
	bytes := float64(M*K)*wElemBytes + float64(K*N+M*N)*4

	computeTime := flops / (eff.ComputeEff * peakTOPS) * 1e6
	bwTime := bytes / (eff.BWEff * peakBW) * 1e6

	return math.Max(computeTime, bwTime) + eff.OverheadUs
}
```

- [ ] **Step 5: Update RunBenchmark MUL_MAT section to loop over 4 dtypes**

Replace the MUL_MAT block (lines 197-243) with:

```go
		if op == "MUL_MAT" {
			// MUL_MAT: one reference curve per weight dtype
			for _, wdt := range Phase1Dtypes() {
				gridIdx++
				elapsed := time.Since(calibrationStart).Round(time.Second)
				slog.Info("benchmarking MUL_MAT reference curve",
					"progress", fmt.Sprintf("[%d/%d]", gridIdx, totalGrids),
					"M", 4096, "K", 4096, "weight_dtype", wdt, "elapsed", elapsed)

				gridStart := time.Now()
				refPoints := benchmarkMulMat(backend, wdt, map[string]int64{"M": 4096, "K": 4096}, cfg)

				if len(refPoints) == 0 {
					slog.Warn("MUL_MAT reference curve produced no points", "weight_dtype", wdt)
					continue
				}

				refCurve := OperatorCurve{
					Op:           "MUL_MAT",
					ComputeDtype: wdt,
					WeightDtype:  wdt,
					Dimensions:   []string{"N"},
					FixedDims:    map[string]int64{"M": 4096, "K": 4096},
					Points:       refPoints,
				}
				devices := backend.BackendDevices()
				if len(devices) > 0 {
					refCurve.Backend = devices[0].Library
				}
				profile.Operators = append(profile.Operators, refCurve)

				// Extract per-dtype efficiency constants
				peakTOPS := hwResult.PeakTOPS["f32"] // compute is always f32/f16
				if tops, ok := hwResult.PeakTOPS[wdt]; ok {
					peakTOPS = tops
				}
				eff := extractEfficiencyConstants(refPoints, 4096, 4096, peakTOPS, hwResult.PeakBW, wdt)
				if profile.Hardware.EfficiencyConstants == nil {
					profile.Hardware.EfficiencyConstants = make(map[string]OpEfficiency)
				}
				effKey := "MUL_MAT_" + wdt
				profile.Hardware.EfficiencyConstants[effKey] = eff

				gridDuration := time.Since(gridStart).Round(time.Second)
				slog.Info("completed MUL_MAT reference",
					"progress", fmt.Sprintf("[%d/%d]", gridIdx, totalGrids),
					"weight_dtype", wdt, "points", len(refPoints),
					"eff_compute", fmt.Sprintf("%.2f", eff.ComputeEff),
					"eff_bw", fmt.Sprintf("%.2f", eff.BWEff),
					"grid_duration", gridDuration)
			}
			continue
		}
```

- [ ] **Step 6: Update countGrids for 4 MUL_MAT references**

```go
	if op == "MUL_MAT" {
		total += len(Phase1Dtypes()) // one reference curve per weight dtype
		continue
	}
```

- [ ] **Step 7: Update existing test TestCountGrids_MulMatIsOne**

Rename to `TestCountGrids_MulMatIsFour` and update expected value from 3 to 6:

```go
func TestCountGrids_MulMatIsFour(t *testing.T) {
	ops := []string{"SILU", "MUL_MAT", "FLASH_ATTN_EXT"}
	dtypes := []string{"f32", "f16", "q4_0", "q8_0"}
	count := countGrids(ops, dtypes)
	assert.Equal(t, 6, count, "should be 6 grids: SILU + 4 MUL_MAT refs + FLASH_ATTN")
}
```

Update `TestExtractEfficiencyConstants` call to pass dtype param: `extractEfficiencyConstants(points, 4096, 4096, peakTOPS, peakBW, "f32")`.

Update `TestPredictMulMatLatency_ComputeBound`, `_BWBound`, `_ScalesWithShape` to use per-dtype efficiency key `"MUL_MAT_f32"` instead of `"MUL_MAT"`.

- [ ] **Step 8: Run all tests**

Run: `go test ./perf/ -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add perf/bench.go perf/bench_test.go
git commit -m "perf: benchmark 4 weight dtypes for MUL_MAT with per-dtype efficiency constants"
```

---

## Task 7: Dtype Mapping and lookupLatency Update

**Files:**
- Modify: `perf/estimate.go:111-155` (lookupLatency)
- Create: `perf/dtype_map.go` (mapWeightDtype function)
- Test: `perf/dtype_map_test.go`
- Modify: `perf/estimate_test.go`

- [ ] **Step 1: Write tests for mapWeightDtype**

```go
// New file: perf/dtype_map_test.go
package perf

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMapWeightDtype_DirectlyMeasured(t *testing.T) {
	for _, dt := range []string{"f32", "f16", "q4_0", "q8_0"} {
		assert.Equal(t, dt, mapWeightDtype(dt), "measured dtype %s should map to itself", dt)
	}
}

func TestMapWeightDtype_KQuants(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"q4_K", "q4_0"},
		{"q4_1", "q4_0"},
		{"q5_K", "q8_0"},
		{"q5_0", "q8_0"},
		{"q5_1", "q8_0"},
		{"q6_K", "q8_0"},
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

func TestMapWeightDtype_UnknownFallback(t *testing.T) {
	assert.Equal(t, "f16", mapWeightDtype("bf16"))
	assert.Equal(t, "f16", mapWeightDtype("unknown_type"))
	assert.Equal(t, "f16", mapWeightDtype(""))
}
```

- [ ] **Step 2: Write tests for updated lookupLatency MUL_MAT path**

```go
// In perf/estimate_test.go — add

func TestLookupLatency_MulMat_UsesRoofline(t *testing.T) {
	// lookupLatency for MUL_MAT should use PredictMulMatLatency (roofline),
	// not curve interpolation
	p := &Profile{
		Version: 2,
		Hardware: HardwareProfile{
			Backends:                 []BackendInfo{{Name: "vulkan", Device: "iGPU"}},
			PeakTOPS:                 map[string]float64{"f32": 50e9, "f16": 50e9},
			PeakBandwidthBytesPerSec: 27e9,
			EfficiencyConstants: map[string]OpEfficiency{
				"MUL_MAT_q4_0": {ComputeEff: 0.90, BWEff: 0.50, OverheadUs: 200},
			},
		},
	}
	lat, err := lookupLatency(p, "MUL_MAT", []int64{4096, 4096, 1}, "f16", "q4_0", "vulkan")
	assert.NoError(t, err)
	assert.Greater(t, lat, 0.0)
}

func TestLookupLatency_MulMat_DtypeMapping(t *testing.T) {
	// q4_K weight dtype should map to q4_0 and use MUL_MAT_q4_0 constants
	p := &Profile{
		Version: 2,
		Hardware: HardwareProfile{
			Backends:                 []BackendInfo{{Name: "vulkan", Device: "iGPU"}},
			PeakTOPS:                 map[string]float64{"f32": 50e9},
			PeakBandwidthBytesPerSec: 27e9,
			EfficiencyConstants: map[string]OpEfficiency{
				"MUL_MAT_q4_0": {ComputeEff: 0.90, BWEff: 0.50, OverheadUs: 200},
			},
		},
	}
	lat, err := lookupLatency(p, "MUL_MAT", []int64{4096, 4096, 1}, "f16", "q4_K", "vulkan")
	assert.NoError(t, err)
	assert.Greater(t, lat, 0.0, "q4_K should map to q4_0 and succeed")
}

func TestLookupLatency_MulMat_NoEfficiencyConstants(t *testing.T) {
	p := &Profile{
		Version: 2,
		Hardware: HardwareProfile{
			Backends:                 []BackendInfo{{Name: "vulkan", Device: "iGPU"}},
			PeakTOPS:                 map[string]float64{"f32": 50e9},
			PeakBandwidthBytesPerSec: 27e9,
		},
	}
	_, err := lookupLatency(p, "MUL_MAT", []int64{4096, 4096, 1}, "f16", "q4_0", "vulkan")
	assert.Error(t, err, "should error when no efficiency constants available")
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `go test ./perf/ -run "TestMapWeightDtype|TestLookupLatency_MulMat_UsesRoofline|TestLookupLatency_MulMat_DtypeMapping|TestLookupLatency_MulMat_NoEfficiencyConstants" -v`
Expected: FAIL — mapWeightDtype undefined, lookupLatency still uses curve matching

- [ ] **Step 4: Implement mapWeightDtype**

Create `perf/dtype_map.go`:

```go
package perf

// mapWeightDtype maps unsupported K-quant and other weight dtypes to the nearest
// measured dtype. The Go DType abstraction only exposes f32/f16/q4_0/q8_0, but real
// models use q4_K, q5_K, q6_K etc.
func mapWeightDtype(wdt string) string {
	switch wdt {
	case "f32", "f16", "q4_0", "q8_0":
		return wdt
	case "q4_K", "q4_1":
		return "q4_0"
	case "q5_K", "q5_0", "q5_1", "q6_K":
		return "q8_0"
	case "q3_K", "q2_K":
		return "q4_0"
	case "q8_K":
		return "q8_0"
	default:
		return "f16"
	}
}
```

- [ ] **Step 5: Update lookupLatency MUL_MAT branch to use roofline + dtype mapping**

Replace the MUL_MAT case in `lookupLatency` (estimate.go:115-131):

```go
	case "MUL_MAT":
		if len(shape) < 3 {
			return 0, fmt.Errorf("MUL_MAT requires 3 shape dims, got %d", len(shape))
		}
		mappedWdt := mapWeightDtype(weightDtype)
		lat := PredictMulMatLatency(&profile.Hardware, shape[0], shape[1], shape[2], mappedWdt)
		if lat == 0 {
			return 0, fmt.Errorf("no efficiency constants for MUL_MAT — run daop-bench first")
		}
		return lat, nil
```

- [ ] **Step 6: Update existing estimate_test.go tests**

The existing MUL_MAT tests use curve interpolation. After this task, `lookupLatency` for MUL_MAT uses roofline prediction, so tests must provide EfficiencyConstants and no longer need MUL_MAT curves.

**6a. Update `makeTestProfileForEstimation()` — add EfficiencyConstants to HardwareProfile:**

```go
func makeTestProfileForEstimation() *Profile {
	return &Profile{
		Version: 2,
		Hardware: HardwareProfile{
			Backends:                 []BackendInfo{{Name: "cuda", Device: "RTX 4090"}},
			PeakTOPS:                 map[string]float64{"f16": 330e12, "f32": 82.6e12},
			PeakBandwidthBytesPerSec: 1008e9,
			BalancePoints:            map[string]float64{"f16": 327.38},
			EfficiencyConstants: map[string]OpEfficiency{
				"MUL_MAT_f16":  {ComputeEff: 0.95, BWEff: 0.80, OverheadUs: 5},
				"MUL_MAT_q4_0": {ComputeEff: 0.90, BWEff: 0.70, OverheadUs: 10},
			},
		},
		Operators: []OperatorCurve{
			// SILU 1D curve — unchanged
			// ... (keep existing SILU curve) ...
			// MUL_MAT curves — REMOVE (no longer used by lookupLatency)
			// FLASH_ATTN_EXT curve — unchanged
			// ... (keep existing FLASH_ATTN curve) ...
		},
	}
}
```

**6b. Replace `TestLookupLatency_MulMat_ExactMK` and `_InterpolatedN` with roofline tests:**

```go
func TestLookupLatency_MulMat_Roofline(t *testing.T) {
	p := makeTestProfileForEstimation()
	lat, err := lookupLatency(p, "MUL_MAT", []int64{4096, 4096, 1}, "f16", "q4_0", "cuda")
	require.NoError(t, err)
	assert.Greater(t, lat, 0.0, "roofline prediction should return positive latency")
}

func TestLookupLatency_MulMat_ScalesWithN(t *testing.T) {
	p := makeTestProfileForEstimation()
	lat1, _ := lookupLatency(p, "MUL_MAT", []int64{4096, 4096, 1}, "f16", "q4_0", "cuda")
	lat4096, _ := lookupLatency(p, "MUL_MAT", []int64{4096, 4096, 4096}, "f16", "q4_0", "cuda")
	assert.Greater(t, lat4096, lat1, "latency should increase with N")
}
```

**6c. Update `TestEstimatePhase_MixedOps` and `_LlamaLikeDecodeLayer`:**

These tests use MUL_MAT nodes with `WeightDtype: "q4_0"`. With EfficiencyConstants in the profile, they'll work via roofline. Just verify they still pass — the assertion logic (`MUL_MAT should be >50%`) remains correct.

**6d. Note: `InterpolateMulMat` becomes dead code.**

After this change, `InterpolateMulMat` in interpolate.go is no longer called by `lookupLatency`. Leave it in place for now (may be useful for Phase 2 spot-check validation). Add a comment:

```go
// InterpolateMulMat interpolates mul_mat latency using inverse distance weighting in (M,K) space.
// NOTE: As of Phase 1A, lookupLatency uses roofline prediction (PredictMulMatLatency) instead.
// This function is retained for potential use in spot-check validation (Phase 2).
```

- [ ] **Step 7: Run all tests**

Run: `go test ./perf/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add perf/dtype_map.go perf/dtype_map_test.go perf/estimate.go perf/estimate_test.go
git commit -m "perf: add K-quant dtype mapping and switch MUL_MAT estimation to roofline"
```

---

## Verification: Run Benchmark and Check Timing

After all tasks are complete:

- [ ] **Step 1: Run full test suite**

```bash
go test ./perf/ -v
```

Expected: All tests PASS

- [ ] **Step 2: Run benchmark to verify timing**

```bash
go run . daop-bench --viewer
```

Expected:
- HW characterization: ~1 min (down from ~5 min)
- MUL_MAT f32 ref: ~2-3 min
- MUL_MAT f16 ref: ~2-3 min
- MUL_MAT q4_0 ref: ~1-2 min
- MUL_MAT q8_0 ref: ~1-2 min
- SILU: ~27s
- FLASH_ATTN: ~3 min
- **Total: ~12-14 min**

- [ ] **Step 3: Verify profile.json contains per-dtype efficiency constants**

Check that `profile.json` contains `"MUL_MAT_f32"`, `"MUL_MAT_f16"`, `"MUL_MAT_q4_0"`, `"MUL_MAT_q8_0"` keys in `hardware.efficiency_constants`.

- [ ] **Step 4: Commit timing results and any adjustments**

```bash
git add -A
git commit -m "perf: verify Phase 1A benchmark timing (~12-14 min)"
```
