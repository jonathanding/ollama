# FLASH_ATTN GQA Full Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark FLASH_ATTN_EXT with all (numQHeads, numKVHeads) combinations where KV≤Q, so GQA models get accurate estimates instead of using wrong MHA curves.

**Architecture:** Extend the existing single-`num_heads` grid to a 2D `(num_q_heads, num_kv_heads)` grid. Each (Q,KV) pair becomes one OperatorCurve with both values in FixedDims. Interpolation uses 2D distance in `(log(Q), log(KV))` space. Backward-compatible: old profiles with only `num_heads` are treated as MHA (KV=Q).

**Tech Stack:** Go, testify (assert/require)

---

### Task 1: Registry — GQA configs and CreateInputs

**Files:**
- Modify: `perf/registry.go:154-181` (CreateInputs), `perf/registry.go:333-346` (expandShapes), `perf/registry.go:427-431` (Phase1FlashAttnHeads)
- Test: `perf/registry_test.go`

- [ ] **Step 1: Write failing tests**

Add to `perf/registry_test.go`:

```go
func TestPhase1FlashAttnGQAConfigs(t *testing.T) {
	configs := Phase1FlashAttnGQAConfigs()
	// 4 Q values × KV ≤ Q: (4,4), (8,4), (8,8), (16,4), (16,8), (16,16), (32,4), (32,8), (32,16), (32,32)
	assert.Len(t, configs, 10)
	// Verify all have KV ≤ Q
	for _, c := range configs {
		assert.LessOrEqual(t, c[1], c[0], "num_kv_heads (%d) must be <= num_q_heads (%d)", c[1], c[0])
	}
	// Verify specific entries
	assert.Contains(t, configs, [2]int64{16, 4}, "should include GQA config Q=16,KV=4")
	assert.Contains(t, configs, [2]int64{32, 8}, "should include GQA config Q=32,KV=8")
	assert.Contains(t, configs, [2]int64{8, 8}, "should include MHA config Q=8,KV=8")
}

func TestExpandShapes_FlashAttn_GQA(t *testing.T) {
	// 4-element gridPoint: [seqQ, seqKV, numQHeads, numKVHeads]
	shapes := expandShapes("FLASH_ATTN_EXT", []int64{1, 2048, 16, 4})
	require.Len(t, shapes, 3)
	assert.Equal(t, []int64{128, 16, 1, 1}, shapes[0], "Q with num_q_heads=16")
	assert.Equal(t, []int64{128, 4, 2048, 1}, shapes[1], "K with num_kv_heads=4")
	assert.Equal(t, []int64{128, 4, 2048, 1}, shapes[2], "V with num_kv_heads=4")
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./perf/ -run "TestPhase1FlashAttnGQAConfigs|TestExpandShapes_FlashAttn_GQA" -v`
Expected: FAIL — `Phase1FlashAttnGQAConfigs` undefined, `expandShapes` 4-arg not handled

- [ ] **Step 3: Implement Phase1FlashAttnGQAConfigs**

In `perf/registry.go`, replace `Phase1FlashAttnHeads`:

```go
// Phase1FlashAttnGQAConfigs returns (numQHeads, numKVHeads) pairs for FLASH_ATTN_EXT benchmarks.
// Covers all combinations where KV ≤ Q from {4, 8, 16, 32}, including both MHA and GQA configs.
func Phase1FlashAttnGQAConfigs() [][2]int64 {
	heads := []int64{4, 8, 16, 32}
	var configs [][2]int64
	for _, q := range heads {
		for _, kv := range heads {
			if kv <= q {
				configs = append(configs, [2]int64{q, kv})
			}
		}
	}
	return configs
}
```

Keep `Phase1FlashAttnHeads` as a deprecated wrapper (used by old tests) or update all callers. Prefer updating all callers to avoid dead code.

- [ ] **Step 4: Update expandShapes for 4-element gridPoint**

In `perf/registry.go`, update the `FLASH_ATTN_EXT` case in `expandShapes`:

```go
case "FLASH_ATTN_EXT":
	// gridPoint = [seq_q, seq_kv] or [seq_q, seq_kv, num_heads] or [seq_q, seq_kv, num_q_heads, num_kv_heads]
	seqQ, seqKV := gridPoint[0], gridPoint[1]
	numQHeads := int64(32)
	numKVHeads := int64(32)
	if len(gridPoint) >= 3 {
		numQHeads = gridPoint[2]
		numKVHeads = numQHeads // default: MHA
	}
	if len(gridPoint) >= 4 {
		numKVHeads = gridPoint[3]
	}
	return [][]int64{
		{128, numQHeads, seqQ, 1},   // Q
		{128, numKVHeads, seqKV, 1}, // K
		{128, numKVHeads, seqKV, 1}, // V
	}
```

- [ ] **Step 5: Update CreateInputs for 4-element gridPoint**

In `perf/registry.go`, update the `FLASH_ATTN_EXT` entry in `Registry`:

```go
CreateInputs: func(ctx ml.Context, backend ml.Backend, dtypeStr string, gridPoint []int64) []ml.Tensor {
	seqQ, seqKV := gridPoint[0], gridPoint[1]
	numQHeads := 32
	numKVHeads := 32
	if len(gridPoint) >= 3 {
		numQHeads = int(gridPoint[2])
		numKVHeads = numQHeads
	}
	if len(gridPoint) >= 4 {
		numKVHeads = int(gridPoint[3])
	}
	q := randomTensor(ctx, ml.DTypeF32, 128, numQHeads, int(seqQ), 1)
	kBytes := materializeTensor(backend, ml.DTypeF16, 128, numKVHeads, int(seqKV), 1)
	vBytes := materializeTensor(backend, ml.DTypeF16, 128, numKVHeads, int(seqKV), 1)
	k := ctx.Input().FromBytes(ml.DTypeF16, kBytes, 128, numKVHeads, int(seqKV), 1)
	v := ctx.Input().FromBytes(ml.DTypeF16, vBytes, 128, numKVHeads, int(seqKV), 1)
	return []ml.Tensor{q, k, v}
},
```

- [ ] **Step 6: Update existing tests**

Update `TestPhase1FlashAttnHeads` → rename or replace with `TestPhase1FlashAttnGQAConfigs` (already written in Step 1).

Update `TestExpandShapes_FlashAttn_NumHeads` to verify backward compat (3-element gridPoint still works with K/V using same heads as Q).

- [ ] **Step 7: Run all registry tests**

Run: `go test ./perf/ -run "TestPhase1FlashAttn|TestExpandShapes_FlashAttn" -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add perf/registry.go perf/registry_test.go
git commit -m "perf: add GQA configs to FLASH_ATTN_EXT registry and expandShapes"
```

---

### Task 2: Benchmark — Generate GQA grids and pass both head counts

**Files:**
- Modify: `perf/bench.go:23-44` (buildSamplingGrids), `perf/bench.go:665-706` (benchmarkFlashAttn)
- Test: `perf/bench_test.go`

- [ ] **Step 1: Write failing tests**

Update existing test and add new one in `perf/bench_test.go`:

```go
func TestBuildSamplingGrids_FlashAttn(t *testing.T) {
	grids := buildSamplingGrids("FLASH_ATTN_EXT", "f16", "")
	configs := Phase1FlashAttnGQAConfigs()
	require.Len(t, grids, len(configs), "should have one grid per GQA config")

	for i, g := range grids {
		assert.Equal(t, "FLASH_ATTN_EXT", g.Op)
		assert.Equal(t, "f16", g.Dtype)
		assert.Equal(t, configs[i][0], g.FixedDims["num_heads"],
			"grid %d should have num_heads=%d", i, configs[i][0])
		assert.Equal(t, configs[i][1], g.FixedDims["num_kv_heads"],
			"grid %d should have num_kv_heads=%d", i, configs[i][1])
		assert.Equal(t, int64(128), g.FixedDims["head_dim"])
	}
}

func TestBenchmarkFlashAttn_GQA_GridPoint(t *testing.T) {
	// Verify benchmarkFlashAttn reads both num_heads and num_kv_heads from fixedDims
	// and passes 4-element gridPoint to measureOp
	fixedDims := map[string]int64{"num_heads": 16, "num_kv_heads": 4, "head_dim": 128}

	// We can't call benchmarkFlashAttn directly without a backend,
	// but we can verify the fixedDims parsing logic.
	numQHeads := int64(32)
	numKVHeads := int64(32)
	if h, ok := fixedDims["num_heads"]; ok {
		numQHeads = h
	}
	if h, ok := fixedDims["num_kv_heads"]; ok {
		numKVHeads = h
	} else {
		numKVHeads = numQHeads
	}
	assert.Equal(t, int64(16), numQHeads)
	assert.Equal(t, int64(4), numKVHeads)
}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `go test ./perf/ -run "TestBuildSamplingGrids_FlashAttn|TestBenchmarkFlashAttn_GQA" -v`
Expected: `TestBuildSamplingGrids_FlashAttn` FAIL (count mismatch: 4 vs 10)

- [ ] **Step 3: Update buildSamplingGrids**

In `perf/bench.go`:

```go
case "FLASH_ATTN_EXT":
	configs := Phase1FlashAttnGQAConfigs()
	grids := make([]SamplingGridWithFixed, len(configs))
	for i, cfg := range configs {
		grids[i] = SamplingGridWithFixed{
			Op: op, Dtype: computeDtype,
			FixedDims: map[string]int64{
				"num_heads":    cfg[0],
				"num_kv_heads": cfg[1],
				"head_dim":     128,
			},
		}
	}
	return grids
```

- [ ] **Step 4: Update benchmarkFlashAttn**

In `perf/bench.go`, update `benchmarkFlashAttn` to read and pass both head counts:

```go
func benchmarkFlashAttn(backend ml.Backend, caps BackendCapabilities, dtype string, fixedDims map[string]int64, cfg BenchmarkConfig) []LatencyPoint {
	var points []LatencyPoint

	numQHeads := int64(32)
	numKVHeads := int64(32)
	if h, ok := fixedDims["num_heads"]; ok {
		numQHeads = h
	}
	if h, ok := fixedDims["num_kv_heads"]; ok {
		numKVHeads = h
	} else {
		numKVHeads = numQHeads
	}

	// Decode: seq_q=1, sweep seq_kv
	decodeMeasure := func(shape []int64) LatencyPoint {
		seqKV := shape[0]
		pt := measureOpForBackend(backend, caps, "FLASH_ATTN_EXT", []int64{1, seqKV, numQHeads, numKVHeads}, dtype, cfg)
		pt.Shape = []int64{seqKV}
		return pt
	}
	decodePts := AdaptiveSample1D(decodeMeasure, 64, 16384, 8, cfg)
	for i := range decodePts {
		seqKV := decodePts[i].Shape[0]
		decodePts[i].Shape = []int64{1, seqKV}
	}
	points = append(points, decodePts...)

	// Prefill: seq_q=seq_kv, sweep both
	prefillMeasure := func(shape []int64) LatencyPoint {
		seqLen := shape[0]
		pt := measureOpForBackend(backend, caps, "FLASH_ATTN_EXT", []int64{seqLen, seqLen, numQHeads, numKVHeads}, dtype, cfg)
		pt.Shape = []int64{seqLen}
		return pt
	}
	prefillPts := AdaptiveSample1D(prefillMeasure, 64, 2048, 8, cfg)
	for i := range prefillPts {
		seqLen := prefillPts[i].Shape[0]
		prefillPts[i].Shape = []int64{seqLen, seqLen}
	}
	points = append(points, prefillPts...)

	return points
}
```

- [ ] **Step 5: Run tests**

Run: `go test ./perf/ -run "TestBuildSamplingGrids_FlashAttn|TestBenchmarkFlashAttn_GQA|TestFlashAttnShapeConversion" -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add perf/bench.go perf/bench_test.go
git commit -m "perf: benchmark FLASH_ATTN_EXT with GQA (Q,KV) grid"
```

---

### Task 3: Interpolation — 2D (numQHeads, numKVHeads) matching

**Files:**
- Modify: `perf/interpolate.go:225-302` (InterpolateFlashAttnMultiHead)
- Test: `perf/interpolate_test.go`

- [ ] **Step 1: Write failing tests**

Add to `perf/interpolate_test.go`:

```go
func TestInterpolateFlashAttnMultiHead_GQA_ExactMatch(t *testing.T) {
	curves := []OperatorCurve{
		{
			Op: "flash_attn", Dimensions: []string{"SeqQ", "SeqKV"},
			FixedDims: map[string]int64{"num_heads": 16, "num_kv_heads": 16},
			Points:    makeDecodePrefillPoints(2.0),
		},
		{
			Op: "flash_attn", Dimensions: []string{"SeqQ", "SeqKV"},
			FixedDims: map[string]int64{"num_heads": 16, "num_kv_heads": 4},
			Points:    makeDecodePrefillPoints(1.2),
		},
	}

	// Exact match on (Q=16, KV=4)
	result := InterpolateFlashAttnMultiHead(curves, 1, 256, 16, 4)
	expected := InterpolateFlashAttn(&curves[1], 1, 256)
	assert.InDelta(t, expected, result, 1e-9)
}

func TestInterpolateFlashAttnMultiHead_GQA_Interpolation(t *testing.T) {
	curves := []OperatorCurve{
		{
			Op: "flash_attn", Dimensions: []string{"SeqQ", "SeqKV"},
			FixedDims: map[string]int64{"num_heads": 16, "num_kv_heads": 4},
			Points:    makeDecodePrefillPoints(1.0),
		},
		{
			Op: "flash_attn", Dimensions: []string{"SeqQ", "SeqKV"},
			FixedDims: map[string]int64{"num_heads": 16, "num_kv_heads": 16},
			Points:    makeDecodePrefillPoints(3.0),
		},
	}

	// Query (Q=16, KV=8) is between the two curves in KV dimension
	result := InterpolateFlashAttnMultiHead(curves, 1, 256, 16, 8)
	lat_kv4 := InterpolateFlashAttn(&curves[0], 1, 256)
	lat_kv16 := InterpolateFlashAttn(&curves[1], 1, 256)
	assert.Greater(t, result, lat_kv4, "should be > KV=4 result")
	assert.Less(t, result, lat_kv16, "should be < KV=16 result")
}

func TestInterpolateFlashAttnMultiHead_BackwardCompat_NoKVHeads(t *testing.T) {
	// Old profile format: only num_heads, no num_kv_heads → treat as MHA
	curves := []OperatorCurve{
		{
			Op: "flash_attn", Dimensions: []string{"SeqQ", "SeqKV"},
			FixedDims: map[string]int64{"num_heads": 8},
			Points:    makeDecodePrefillPoints(1.0),
		},
		{
			Op: "flash_attn", Dimensions: []string{"SeqQ", "SeqKV"},
			FixedDims: map[string]int64{"num_heads": 16},
			Points:    makeDecodePrefillPoints(2.0),
		},
	}

	// Query MHA (Q=16, KV=16) should match num_heads=16 exactly
	result := InterpolateFlashAttnMultiHead(curves, 1, 256, 16, 16)
	expected := InterpolateFlashAttn(&curves[1], 1, 256)
	assert.InDelta(t, expected, result, 1e-9)
}

func TestInterpolateFlashAttnMultiHead_GQA_PrefillRegime(t *testing.T) {
	curves := []OperatorCurve{
		{
			Op: "flash_attn", Dimensions: []string{"SeqQ", "SeqKV"},
			FixedDims: map[string]int64{"num_heads": 16, "num_kv_heads": 4},
			Points:    makeDecodePrefillPoints(1.0),
		},
		{
			Op: "flash_attn", Dimensions: []string{"SeqQ", "SeqKV"},
			FixedDims: map[string]int64{"num_heads": 16, "num_kv_heads": 16},
			Points:    makeDecodePrefillPoints(3.0),
		},
	}

	// Prefill query (Q=16, KV=4)
	result := InterpolateFlashAttnMultiHead(curves, 256, 256, 16, 4)
	expected := InterpolateFlashAttn(&curves[0], 256, 256)
	assert.InDelta(t, expected, result, 1e-9)
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./perf/ -run "TestInterpolateFlashAttnMultiHead_GQA|TestInterpolateFlashAttnMultiHead_BackwardCompat" -v`
Expected: FAIL — function signature mismatch (4 args vs 5)

- [ ] **Step 3: Update InterpolateFlashAttnMultiHead signature and implementation**

In `perf/interpolate.go`, change the function signature to accept both head counts and use 2D distance:

```go
// InterpolateFlashAttnMultiHead interpolates flash_attn latency across multiple
// (numQHeads, numKVHeads) curves using inverse distance weighting in 2D log space.
// Each curve has FixedDims["num_heads"] (Q heads) and optionally FixedDims["num_kv_heads"].
// If num_kv_heads is absent, the curve is treated as MHA (KV = Q).
func InterpolateFlashAttnMultiHead(curves []OperatorCurve, querySeqQ, querySeqKV, queryNumQHeads, queryNumKVHeads int64) float64 {
	if len(curves) == 0 {
		return 0
	}
	if len(curves) == 1 {
		return InterpolateFlashAttn(&curves[0], querySeqQ, querySeqKV)
	}

	type candidate struct {
		curve *OperatorCurve
		dist  float64
	}

	logQ := math.Log(float64(queryNumQHeads))
	logKV := math.Log(float64(queryNumKVHeads))
	var candidates []candidate
	for i := range curves {
		nh := curves[i].FixedDims["num_heads"]
		if nh <= 0 {
			continue
		}
		nkv := curves[i].FixedDims["num_kv_heads"]
		if nkv <= 0 {
			nkv = nh // backward compat: MHA
		}
		dq := logQ - math.Log(float64(nh))
		dkv := logKV - math.Log(float64(nkv))
		dist := math.Sqrt(dq*dq + dkv*dkv)
		candidates = append(candidates, candidate{&curves[i], dist})
	}

	if len(candidates) == 0 {
		return 0
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].dist < candidates[j].dist
	})

	// Exact match
	if candidates[0].dist == 0 {
		return InterpolateFlashAttn(candidates[0].curve, querySeqQ, querySeqKV)
	}

	// Single candidate
	if len(candidates) == 1 {
		return InterpolateFlashAttn(candidates[0].curve, querySeqQ, querySeqKV)
	}

	lat1 := InterpolateFlashAttn(candidates[0].curve, querySeqQ, querySeqKV)
	lat2 := InterpolateFlashAttn(candidates[1].curve, querySeqQ, querySeqKV)

	if lat1 <= 0 || lat2 <= 0 {
		if lat1 > 0 {
			return lat1
		}
		return lat2
	}

	// Check if query is outside the grid — extrapolate using power-law from nearest two
	nh1 := float64(candidates[0].curve.FixedDims["num_heads"])
	nkv1 := float64(candidates[0].curve.FixedDims["num_kv_heads"])
	if nkv1 <= 0 {
		nkv1 = nh1
	}
	nh2 := float64(candidates[1].curve.FixedDims["num_heads"])
	nkv2 := float64(candidates[1].curve.FixedDims["num_kv_heads"])
	if nkv2 <= 0 {
		nkv2 = nh2
	}
	// Use 1D distance along the line connecting the two nearest curves for extrapolation check
	totalDist := candidates[0].dist + candidates[1].dist
	gridSpan := math.Sqrt(math.Pow(math.Log(nh2)-math.Log(nh1), 2) + math.Pow(math.Log(nkv2)-math.Log(nkv1), 2))
	if gridSpan > 0 && totalDist > gridSpan*1.1 {
		// Query is likely outside the grid — power-law extrapolation
		logLat1 := math.Log(lat1)
		logLat2 := math.Log(lat2)
		if gridSpan > 0 {
			slope := (logLat2 - logLat1) / gridSpan
			return math.Exp(logLat1 + slope*candidates[0].dist)
		}
	}

	// IDW blend between two nearest curves
	w1 := 1.0 / candidates[0].dist
	w2 := 1.0 / candidates[1].dist
	return (lat1*w1 + lat2*w2) / (w1 + w2)
}
```

Note: Power-law extrapolation is retained for queries outside the grid (e.g., Q=64 heads). With 10 curves covering {4,8,16,32}², most real models get exact matches; extrapolation handles edge cases.

- [ ] **Step 4: Update ALL callers of InterpolateFlashAttnMultiHead (required for compilation)**

**CRITICAL**: The signature change from 4 to 5 args breaks all callers. ALL must be updated in this task to keep the code compilable.

Caller 1: `perf/estimate.go:266` — add backward-compat KV heads extraction:

```go
case "FLASH_ATTN_EXT":
	if len(shape) < 3 {
		return 0, fmt.Errorf("FLASH_ATTN_EXT requires at least 3 shape dims (seqQ, seqKV, numQHeads[, numKVHeads]), got %d", len(shape))
	}
	numQHeads := shape[2]
	numKVHeads := numQHeads
	if len(shape) >= 4 {
		numKVHeads = shape[3]
	}
	var curves []OperatorCurve
	for _, c := range profile.Operators {
		if c.Op == op && c.Backend == backend {
			curves = append(curves, c)
		}
	}
	if len(curves) == 0 {
		return 0, fmt.Errorf("uncalibrated: %s(%s on %s)", op, computeDtype, backend)
	}
	return InterpolateFlashAttnMultiHead(curves, shape[0], shape[1], numQHeads, numKVHeads), nil
```

Caller 2: Existing tests in `perf/interpolate_test.go` — add 5th argument `numKVHeads = numQHeads` (MHA):

```go
// In TestInterpolateFlashAttnMultiHead_ExactMatch:
result := InterpolateFlashAttnMultiHead(curves, 1, 256, 16, 16)  // was: ..., 16)
expected := InterpolateFlashAttn(&curves[1], 1, 256)

// In TestInterpolateFlashAttnMultiHead_Interpolation:
result := InterpolateFlashAttnMultiHead(curves, 1, 256, 16, 16)  // was: ..., 16)

// In TestInterpolateFlashAttnMultiHead_SingleCurve:
result := InterpolateFlashAttnMultiHead(curves, 1, 256, 16, 16)  // was: ..., 16)

// In TestInterpolateFlashAttnMultiHead_Extrapolation:
result := InterpolateFlashAttnMultiHead(curves, 1, 256, 32, 32)  // was: ..., 32)

// In TestInterpolateFlashAttnMultiHead_Empty:
result := InterpolateFlashAttnMultiHead(nil, 1, 256, 16, 16)     // was: ..., 16)
```

- [ ] **Step 5: Run all perf tests (compilation + correctness)**

Run: `go test ./perf/ -v -count=1`
Expected: ALL PASS (both interpolation changes and estimate.go caller update compile and work)

- [ ] **Step 6: Commit**

```bash
git add perf/interpolate.go perf/interpolate_test.go perf/estimate.go
git commit -m "perf: 2D GQA interpolation for FLASH_ATTN multi-head lookup"
```

---

### Task 4: Estimate — Extract both Q and KV heads from graph nodes + update tests

**Files:**
- Modify: `perf/estimate.go:39-48` (nodeToQueryShape), `perf/estimate.go:250-266` (lookupLatency)
- Test: `perf/estimate_test.go`

- [ ] **Step 1: Write failing tests**

Update existing and add new tests in `perf/estimate_test.go`:

```go
func TestNodeToQueryShape_FlashAttn_GQA(t *testing.T) {
	node := ml.GraphNode{
		Op:           "FLASH_ATTN_EXT",
		Backend:      "cuda",
		ComputeDtype: "f16",
		InputShapes: [][]int64{
			{128, 130, 32, 1}, // Q: num_q_heads=32
			{128, 256, 8, 1},  // K: num_kv_heads=8
		},
	}
	op, shape, _, _ := nodeToQueryShape(node)
	assert.Equal(t, "FLASH_ATTN_EXT", op)
	require.Len(t, shape, 4)
	assert.Equal(t, int64(130), shape[0], "seqQ")
	assert.Equal(t, int64(256), shape[1], "seqKV")
	assert.Equal(t, int64(32), shape[2], "numQHeads from Q tensor")
	assert.Equal(t, int64(8), shape[3], "numKVHeads from K tensor")
}
```

Note: The existing `TestNodeToQueryShape_FlashAttn_GQA` at line 1095 already exists. It currently asserts `Len(t, shape, 3)`. This test must be updated to assert `Len(t, shape, 4)` and check both head counts.

Similarly update `TestNodeToQueryShape_FlashAttn_NumHeads` (line 1167) which also asserts 3-element shape.

Update `TestNodeToQueryShape_FlashAttn` (line 74) and `TestNodeToQueryShape_FlashAttn_Prefill` (line 92): these use MHA (Q and K have same num_heads). Update assertions to expect 4-element shape with `shape[3] == shape[2]`.

Update `TestLookupLatency_FlashAttn_MultiHead` (line 1183): the `lookupLatency` call passes a 3-element shape `[]int64{256, 256, 16}`. Update to 4-element `[]int64{256, 256, 16, 16}`.

Add test for GQA lookupLatency:

```go
func TestLookupLatency_FlashAttn_GQA(t *testing.T) {
	makePoints := func(scale float64) []LatencyPoint {
		return []LatencyPoint{
			{Shape: []int64{1, 64}, LatencyUs: 1.0 * scale},
			{Shape: []int64{1, 256}, LatencyUs: 2.0 * scale},
			{Shape: []int64{1, 1024}, LatencyUs: 4.0 * scale},
			{Shape: []int64{64, 64}, LatencyUs: 3.0 * scale},
			{Shape: []int64{256, 256}, LatencyUs: 20.0 * scale},
			{Shape: []int64{1024, 1024}, LatencyUs: 200.0 * scale},
		}
	}
	profile := &Profile{
		Operators: []OperatorCurve{
			{
				Op: "FLASH_ATTN_EXT", Backend: "Vulkan",
				FixedDims: map[string]int64{"num_heads": 16, "num_kv_heads": 16},
				Points:    makePoints(200.0),
			},
			{
				Op: "FLASH_ATTN_EXT", Backend: "Vulkan",
				FixedDims: map[string]int64{"num_heads": 16, "num_kv_heads": 4},
				Points:    makePoints(80.0),
			},
		},
	}
	// GQA query (Q=16, KV=4) should match the second curve exactly
	lat, err := lookupLatency(profile, "FLASH_ATTN_EXT", []int64{256, 256, 16, 4}, "f16", "", "Vulkan")
	require.NoError(t, err)
	expected := InterpolateFlashAttn(&profile.Operators[1], 256, 256)
	assert.InDelta(t, expected, lat, 1e-9)

	// MHA query (Q=16, KV=16) should match the first curve
	lat2, err := lookupLatency(profile, "FLASH_ATTN_EXT", []int64{256, 256, 16, 16}, "f16", "", "Vulkan")
	require.NoError(t, err)
	expected2 := InterpolateFlashAttn(&profile.Operators[0], 256, 256)
	assert.InDelta(t, expected2, lat2, 1e-9)

	// GQA < MHA for same Q heads
	assert.Less(t, lat, lat2, "GQA (KV=4) should be less than MHA (KV=16)")
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./perf/ -run "TestNodeToQueryShape_FlashAttn|TestLookupLatency_FlashAttn" -v`
Expected: FAIL — shape length assertions fail (3 vs 4), lookupLatency call mismatches

- [ ] **Step 3: Update nodeToQueryShape**

In `perf/estimate.go`:

```go
case "FLASH_ATTN_EXT":
	if len(node.InputShapes) >= 2 && len(node.InputShapes[0]) >= 3 && len(node.InputShapes[1]) >= 3 {
		// GGML layout: Q ne=[head_dim, seqQ, num_q_heads, batch]
		//              K ne=[head_dim, seqKV, num_kv_heads, batch]
		seqQ := node.InputShapes[0][1]
		seqKV := node.InputShapes[1][1]
		numQHeads := node.InputShapes[0][2]
		numKVHeads := node.InputShapes[1][2]
		shape = []int64{seqQ, seqKV, numQHeads, numKVHeads}
		return
	}
	shape = []int64{totalElements(node.Shape)}
	return
```

Guard changed from `>= 2` to `>= 3` for K tensor to safely read `numKVHeads` from `InputShapes[1][2]`.

Note: `lookupLatency` was already updated in Task 3 Step 4 (required for compilation). No changes needed here.

- [ ] **Step 4: Update all affected tests**

Update every test that constructs FLASH_ATTN_EXT GraphNodes or calls lookupLatency with FLASH_ATTN shapes. Key tests to update:

1. `TestNodeToQueryShape_FlashAttn` (line 74): assert `Len(t, shape, 4)`, add `assert.Equal(t, int64(32), shape[3], "numKVHeads")`
2. `TestNodeToQueryShape_FlashAttn_Prefill` (line 92): same pattern
3. `TestNodeToQueryShape_FlashAttn_GQA` (line 1095): change `Len(t, shape, 3)` to `4`, change `shape[2]` assertion from Q heads (32) to Q heads (32) + add KV heads assertion (8)
4. `TestNodeToQueryShape_FlashAttn_NumHeads` (line 1167): change `Len(t, shape, 3)` to `4`, add KV heads assertion
5. `TestLookupLatency_FlashAttn_Decode` (line 304): update shape to 4 elements
6. `TestLookupLatency_FlashAttn_Prefill` (line 311): update shape to 4 elements
7. `TestLookupLatency_FlashAttn_InsufficientShape` (line 448): keep 2-element shape test (should still fail)
8. `TestLookupLatency_FlashAttn_MultiHead` (line 1183): update shape to 4 elements
9. `TestEstimatePhase_FlashAttnScalesWithSeqLen` (line 945): MHA nodes, shape[3]=shape[2] automatically via nodeToQueryShape
10. `TestEstimatePhaseV3_FlashAttnScalesWithSeqLen` (line 1113): same
11. `TestEstimatePhase_EdgeCase_InputLengthOne` (line 1148): MHA nodes
12. `TestEstimatePhase_LlamaDecodeFlashAttnPercentageIncreasesWithKVLen` (line 1016): check if uses lookupLatency directly or through estimatePhase

For tests that go through `estimatePhase` → `nodeToQueryShape` → `lookupLatency`: the GraphNodes have MHA shapes (Q and K have same num_heads), so `nodeToQueryShape` will automatically produce 4-element shapes with `numKVHeads == numQHeads`. These tests may need profile curves that have the new `num_kv_heads` FixedDims, OR the backward-compat code in `InterpolateFlashAttnMultiHead` handles old curves (no `num_kv_heads` → treat as MHA). Verify this works.

- [ ] **Step 5: Run all estimate tests (unit + integration)**

Run: `go test ./perf/ -run "TestNodeToQueryShape|TestLookupLatency" -v`
Expected: ALL PASS

Then run integration tests that go through the full pipeline (estimatePhase → nodeToQueryShape → lookupLatency → InterpolateFlashAttnMultiHead):

Run: `go test ./perf/ -run "TestEstimatePhase" -v`
Expected: ALL PASS — these use MHA GraphNodes so backward compat should handle them. If any fail, check that the test profile's FLASH_ATTN curves work with the new 5-arg InterpolateFlashAttnMultiHead (old curves without num_kv_heads should be treated as MHA).

- [ ] **Step 6: Run ALL perf tests**

Run: `go test ./perf/ -v -count=1`
Expected: ALL PASS (no regressions)

- [ ] **Step 7: Commit**

```bash
git add perf/estimate.go perf/estimate_test.go
git commit -m "perf: extract GQA head counts in estimate and route to 2D interpolation"
```
