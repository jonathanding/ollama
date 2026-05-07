# FLASH_ATTN num_heads Parameterization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `num_heads` as a benchmark grid dimension for FLASH_ATTN_EXT so that estimation accuracy improves from ~3x underestimate to within 1.5x for models with varying head counts.

**Architecture:** Follow the established MUL_MAT multi-curve IDW pattern. Benchmark at 4 num_heads values, store multiple OperatorCurves per num_heads, interpolate across num_heads in log-space.

**Tech Stack:** Go, existing perf package infrastructure

---

### Task 1: Parameterize FLASH_ATTN benchmark with num_heads

**Files:**
- Modify: `perf/registry.go:154-177` (FLASH_ATTN_EXT CreateInputs and expandShapes)
- Modify: `perf/bench.go:23-49` (buildSamplingGrids)
- Modify: `perf/bench.go:655-696` (benchmarkFlashAttn)
- Test: `perf/bench_test.go`, `perf/registry_test.go`

- [ ] **Step 1: Write failing test — buildSamplingGrids returns multiple grids**

In `perf/bench_test.go`, update or add test:

```go
func TestBuildSamplingGrids_FlashAttn_MultiHead(t *testing.T) {
	grids := buildSamplingGrids("FLASH_ATTN_EXT", "f16", "")
	assert.Equal(t, 4, len(grids), "should produce one grid per num_heads value")

	expectedHeads := []int64{4, 8, 16, 32}
	for i, grid := range grids {
		assert.Equal(t, "FLASH_ATTN_EXT", grid.Op)
		assert.Equal(t, expectedHeads[i], grid.FixedDims["num_heads"])
		assert.Equal(t, int64(128), grid.FixedDims["head_dim"])
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./perf/ -run TestBuildSamplingGrids_FlashAttn_MultiHead -v`
Expected: FAIL (currently returns 1 grid, not 4)

- [ ] **Step 3: Add Phase1FlashAttnHeads and update buildSamplingGrids**

In `perf/bench.go`, add:

```go
// Phase1FlashAttnHeads returns the num_heads grid values for FLASH_ATTN benchmarks.
// Covers common head counts: GQA KV heads (4, 8) through standard models (16, 32).
func Phase1FlashAttnHeads() []int64 {
	return []int64{4, 8, 16, 32}
}
```

In `buildSamplingGrids`, change the FLASH_ATTN_EXT case:

```go
case "FLASH_ATTN_EXT":
	heads := Phase1FlashAttnHeads()
	grids := make([]SamplingGridWithFixed, len(heads))
	for i, h := range heads {
		grids[i] = SamplingGridWithFixed{
			Op: op, Dtype: computeDtype,
			FixedDims: map[string]int64{"num_heads": h, "head_dim": 128},
		}
	}
	return grids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./perf/ -run TestBuildSamplingGrids_FlashAttn_MultiHead -v`
Expected: PASS

- [ ] **Step 5: Update registry CreateInputs to use gridPoint[2] for num_heads**

In `perf/registry.go`, change `FLASH_ATTN_EXT.CreateInputs`:

```go
CreateInputs: func(ctx ml.Context, backend ml.Backend, dtypeStr string, gridPoint []int64) []ml.Tensor {
	seqQ, seqKV := gridPoint[0], gridPoint[1]
	numHeads := int64(32) // default for backward compatibility
	if len(gridPoint) >= 3 {
		numHeads = gridPoint[2]
	}
	q := randomTensor(ctx, ml.DTypeF32, 128, int(numHeads), int(seqQ), 1)
	kBytes := materializeTensor(backend, ml.DTypeF16, 128, int(numHeads), int(seqKV), 1)
	vBytes := materializeTensor(backend, ml.DTypeF16, 128, int(numHeads), int(seqKV), 1)
	k := ctx.Input().FromBytes(ml.DTypeF16, kBytes, 128, int(numHeads), int(seqKV), 1)
	v := ctx.Input().FromBytes(ml.DTypeF16, vBytes, 128, int(numHeads), int(seqKV), 1)
	return []ml.Tensor{q, k, v}
},
```

- [ ] **Step 6: Update benchmarkFlashAttn to pass num_heads through gridPoint**

In `perf/bench.go`, change `benchmarkFlashAttn` to read `num_heads` from `fixedDims` and pass as 3rd element of gridPoint:

```go
func benchmarkFlashAttn(backend ml.Backend, caps BackendCapabilities, dtype string, fixedDims map[string]int64, cfg BenchmarkConfig) []LatencyPoint {
	numHeads := fixedDims["num_heads"]
	var points []LatencyPoint

	// Decode: seq_q=1, sweep seq_kv
	decodeMeasure := func(shape []int64) LatencyPoint {
		seqKV := shape[0]
		pt := measureOpForBackend(backend, caps, "FLASH_ATTN_EXT", []int64{1, seqKV, numHeads}, dtype, cfg)
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
		pt := measureOpForBackend(backend, caps, "FLASH_ATTN_EXT", []int64{seqLen, seqLen, numHeads}, dtype, cfg)
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

- [ ] **Step 7: Update expandShapes FLASH_ATTN case for consistency**

In `perf/registry.go`, update `expandShapes` to handle 3-element gridPoint:

```go
case "FLASH_ATTN_EXT":
	seqQ, seqKV := gridPoint[0], gridPoint[1]
	numHeads := int64(32)
	if len(gridPoint) >= 3 {
		numHeads = gridPoint[2]
	}
	return [][]int64{
		{128, numHeads, seqQ, 1},
		{128, numHeads, seqKV, 1},
		{128, numHeads, seqKV, 1},
	}
```

- [ ] **Step 8: Write test for registry CreateInputs with variable num_heads**

In `perf/registry_test.go`, add:

```go
func TestExpandShapes_FlashAttn_NumHeads(t *testing.T) {
	shapes := expandShapes("FLASH_ATTN_EXT", []int64{128, 256, 16})
	assert.Equal(t, 3, len(shapes))
	assert.Equal(t, []int64{128, 16, 128, 1}, shapes[0]) // Q: head_dim=128, num_heads=16
	assert.Equal(t, []int64{128, 16, 256, 1}, shapes[1]) // K
	assert.Equal(t, []int64{128, 16, 256, 1}, shapes[2]) // V
}

func TestExpandShapes_FlashAttn_DefaultHeads(t *testing.T) {
	// 2-element gridPoint should default to 32 heads (backward compat)
	shapes := expandShapes("FLASH_ATTN_EXT", []int64{128, 256})
	assert.Equal(t, []int64{128, 32, 128, 1}, shapes[0])
}
```

- [ ] **Step 9: Update existing tests that expect single grid**

In `perf/bench_test.go`, update `TestBuildSamplingGrids_FlashAttn` to expect 4 grids or rename/remove it (the new multi-head test replaces it).

- [ ] **Step 10: Run all tests**

Run: `go test ./perf/ -run "FlashAttn|Expand" -v`
Expected: All PASS

- [ ] **Step 11: Commit**

```bash
git add perf/registry.go perf/bench.go perf/bench_test.go perf/registry_test.go
git commit -m "perf: parameterize FLASH_ATTN benchmark with num_heads grid"
```

---

### Task 2: Multi-head interpolation function

**Files:**
- Modify: `perf/interpolate.go` (add InterpolateFlashAttnMultiHead)
- Test: `perf/interpolate_test.go`

- [ ] **Step 1: Write failing tests**

In `perf/interpolate_test.go`, add:

```go
func TestInterpolateFlashAttnMultiHead_ExactMatch(t *testing.T) {
	curves := []OperatorCurve{
		{FixedDims: map[string]int64{"num_heads": 16}, Points: makeDecodePrefillPoints(100.0)},
		{FixedDims: map[string]int64{"num_heads": 32}, Points: makeDecodePrefillPoints(200.0)},
	}
	result := InterpolateFlashAttnMultiHead(curves, 256, 256, 16)
	// Should use the 16-head curve exactly
	expected := InterpolateFlashAttn(&curves[0], 256, 256)
	assert.InDelta(t, expected, result, 0.01)
}

func TestInterpolateFlashAttnMultiHead_Interpolation(t *testing.T) {
	curves := []OperatorCurve{
		{FixedDims: map[string]int64{"num_heads": 8}, Points: makeDecodePrefillPoints(50.0)},
		{FixedDims: map[string]int64{"num_heads": 32}, Points: makeDecodePrefillPoints(200.0)},
	}
	lat8 := InterpolateFlashAttn(&curves[0], 256, 256)
	lat32 := InterpolateFlashAttn(&curves[1], 256, 256)
	result := InterpolateFlashAttnMultiHead(curves, 256, 256, 16)
	// 16 is geometric mean of 8 and 32 → IDW should blend
	assert.Greater(t, result, lat8)
	assert.Less(t, result, lat32)
}

func TestInterpolateFlashAttnMultiHead_SingleCurve(t *testing.T) {
	curves := []OperatorCurve{
		{FixedDims: map[string]int64{"num_heads": 32}, Points: makeDecodePrefillPoints(200.0)},
	}
	result := InterpolateFlashAttnMultiHead(curves, 256, 256, 16)
	expected := InterpolateFlashAttn(&curves[0], 256, 256)
	// Single curve: should return that curve's result (backward compat)
	assert.InDelta(t, expected, result, 0.01)
}

func TestInterpolateFlashAttnMultiHead_Extrapolation(t *testing.T) {
	curves := []OperatorCurve{
		{FixedDims: map[string]int64{"num_heads": 8}, Points: makeDecodePrefillPoints(50.0)},
		{FixedDims: map[string]int64{"num_heads": 16}, Points: makeDecodePrefillPoints(100.0)},
	}
	// Query num_heads=32 is outside the grid — should extrapolate upward
	result := InterpolateFlashAttnMultiHead(curves, 256, 256, 32)
	lat16 := InterpolateFlashAttn(&curves[1], 256, 256)
	assert.Greater(t, result, lat16, "extrapolation beyond grid should be higher")
}

func TestInterpolateFlashAttnMultiHead_Empty(t *testing.T) {
	result := InterpolateFlashAttnMultiHead(nil, 256, 256, 16)
	assert.Equal(t, 0.0, result)
}
```

The helper `makeDecodePrefillPoints(scale)` creates a curve with decode and prefill points scaled by `scale`:

```go
func makeDecodePrefillPoints(scale float64) []LatencyPoint {
	return []LatencyPoint{
		// Decode: seqQ=1
		{Shape: []int64{1, 64}, LatencyUs: 1.0 * scale},
		{Shape: []int64{1, 256}, LatencyUs: 2.0 * scale},
		{Shape: []int64{1, 1024}, LatencyUs: 4.0 * scale},
		// Prefill: seqQ=seqKV
		{Shape: []int64{64, 64}, LatencyUs: 3.0 * scale},
		{Shape: []int64{256, 256}, LatencyUs: 20.0 * scale},
		{Shape: []int64{1024, 1024}, LatencyUs: 200.0 * scale},
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./perf/ -run "InterpolateFlashAttnMultiHead" -v`
Expected: FAIL (function does not exist)

- [ ] **Step 3: Implement InterpolateFlashAttnMultiHead**

In `perf/interpolate.go`, add:

```go
// InterpolateFlashAttnMultiHead interpolates flash_attn latency across multiple
// num_heads curves using inverse distance weighting in log-num_heads space.
// Each curve has a FixedDims["num_heads"] value and contains decode/prefill points.
// Falls back to single-curve InterpolateFlashAttn when only one curve is available.
func InterpolateFlashAttnMultiHead(curves []OperatorCurve, querySeqQ, querySeqKV, queryNumHeads int64) float64 {
	if len(curves) == 0 {
		return 0
	}
	if len(curves) == 1 {
		return InterpolateFlashAttn(&curves[0], querySeqQ, querySeqKV)
	}

	type candidate struct {
		curve   *OperatorCurve
		logDist float64
	}

	logQ := math.Log(float64(queryNumHeads))
	var candidates []candidate
	for i := range curves {
		nh := curves[i].FixedDims["num_heads"]
		if nh <= 0 {
			continue
		}
		dist := math.Abs(logQ - math.Log(float64(nh)))
		candidates = append(candidates, candidate{&curves[i], dist})
	}

	if len(candidates) == 0 {
		return 0
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].logDist < candidates[j].logDist
	})

	// Exact match
	if candidates[0].logDist == 0 {
		return InterpolateFlashAttn(candidates[0].curve, querySeqQ, querySeqKV)
	}

	// Single candidate after filtering
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

	// Check if query is outside the grid — extrapolate using power-law
	nh1 := float64(candidates[0].curve.FixedDims["num_heads"])
	nh2 := float64(candidates[1].curve.FixedDims["num_heads"])
	qnh := float64(queryNumHeads)

	// If query is outside [min, max] of nearest two curves, extrapolate
	minNH := math.Min(nh1, nh2)
	maxNH := math.Max(nh1, nh2)
	if qnh < minNH || qnh > maxNH {
		// Power-law extrapolation from nearest two points
		logNH1 := math.Log(nh1)
		logNH2 := math.Log(nh2)
		logLat1 := math.Log(lat1)
		logLat2 := math.Log(lat2)
		slope := (logLat2 - logLat1) / (logNH2 - logNH1)
		return math.Exp(logLat1 + slope*(logQ-logNH1))
	}

	// IDW blend between two nearest curves
	w1 := 1.0 / candidates[0].logDist
	w2 := 1.0 / candidates[1].logDist
	return (lat1*w1 + lat2*w2) / (w1 + w2)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./perf/ -run "InterpolateFlashAttnMultiHead" -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add perf/interpolate.go perf/interpolate_test.go
git commit -m "perf: add InterpolateFlashAttnMultiHead for multi-head-count interpolation"
```

---

### Task 3: Wire estimation to use num_heads from graph and multi-head interpolation

**Files:**
- Modify: `perf/estimate.go:18-54` (nodeToQueryShape FLASH_ATTN case)
- Modify: `perf/estimate.go:232-244` (lookupLatency FLASH_ATTN case)
- Test: `perf/estimate_test.go`

- [ ] **Step 1: Write failing test — nodeToQueryShape extracts num_heads**

In `perf/estimate_test.go`, add:

```go
func TestNodeToQueryShape_FlashAttn_NumHeads(t *testing.T) {
	node := ml.GraphNode{
		Op: "FLASH_ATTN_EXT",
		InputShapes: [][4]int64{
			{128, 256, 16, 1}, // Q: head_dim=128, seqQ=256, num_heads=16
			{128, 512, 4, 1},  // K: head_dim=128, seqKV=512, num_kv_heads=4
		},
	}
	op, shape, _, _ := nodeToQueryShape(node)
	assert.Equal(t, "FLASH_ATTN_EXT", op)
	assert.Equal(t, int64(256), shape[0], "seqQ")
	assert.Equal(t, int64(512), shape[1], "seqKV")
	assert.Equal(t, int64(16), shape[2], "numHeads from Q tensor")
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./perf/ -run TestNodeToQueryShape_FlashAttn_NumHeads -v`
Expected: FAIL (shape only has 2 elements currently)

- [ ] **Step 3: Update nodeToQueryShape to extract num_heads**

In `perf/estimate.go`, change the FLASH_ATTN_EXT case:

```go
case "FLASH_ATTN_EXT":
	if len(node.InputShapes) >= 2 && len(node.InputShapes[0]) >= 3 && len(node.InputShapes[1]) >= 2 {
		seqQ := node.InputShapes[0][1]
		seqKV := node.InputShapes[1][1]
		numHeads := node.InputShapes[0][2]
		shape = []int64{seqQ, seqKV, numHeads}
		return
	}
	shape = []int64{totalElements(node.Shape)}
	return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./perf/ -run TestNodeToQueryShape_FlashAttn_NumHeads -v`
Expected: PASS

- [ ] **Step 5: Write test — lookupLatency uses multi-head interpolation**

In `perf/estimate_test.go`, add:

```go
func TestLookupLatency_FlashAttn_MultiHead(t *testing.T) {
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
				FixedDims: map[string]int64{"num_heads": 8},
				Points: makePoints(50.0),
			},
			{
				Op: "FLASH_ATTN_EXT", Backend: "Vulkan",
				FixedDims: map[string]int64{"num_heads": 32},
				Points: makePoints(200.0),
			},
		},
	}
	// Query with 16 heads — should interpolate between 8-head and 32-head curves
	lat, err := lookupLatency(profile, "FLASH_ATTN_EXT", []int64{256, 256, 16}, "f16", "", "Vulkan")
	assert.NoError(t, err)

	lat8, _ := lookupLatency(profile, "FLASH_ATTN_EXT", []int64{256, 256, 8}, "f16", "", "Vulkan")
	lat32, _ := lookupLatency(profile, "FLASH_ATTN_EXT", []int64{256, 256, 32}, "f16", "", "Vulkan")
	assert.Greater(t, lat, lat8)
	assert.Less(t, lat, lat32)
}
```

- [ ] **Step 6: Update lookupLatency FLASH_ATTN case**

In `perf/estimate.go`, change the FLASH_ATTN_EXT case:

```go
case "FLASH_ATTN_EXT":
	if len(shape) < 3 {
		return 0, fmt.Errorf("FLASH_ATTN_EXT requires 3 shape dims (seqQ, seqKV, numHeads), got %d", len(shape))
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
	return InterpolateFlashAttnMultiHead(curves, shape[0], shape[1], shape[2]), nil
```

- [ ] **Step 7: Update existing tests for 3-element shape**

Update tests that construct FLASH_ATTN shapes to include num_heads as the 3rd element. Tests that test FLASH_ATTN with 2-element shapes need updating:

- `TestLookupLatency_FlashAttn_Decode`: shape `[1, 256]` → `[1, 256, 32]`
- `TestLookupLatency_FlashAttn_Prefill`: shape `[128, 128]` → `[128, 128, 32]`
- `TestLookupLatency_FlashAttn_InsufficientShape`: update expected error message
- `TestEstimatePhase_FlashAttnScalesWithSeqLen`: update node InputShapes to include num_heads dim
- `TestEstimatePhase_FlashAttnDecodeScalesWithKVLen`: same
- `TestEstimatePhase_LlamaDecodeFlashAttnPercentageIncreasesWithKVLen`: same
- `TestEstimatePhaseV3_FlashAttnScalesWithSeqLen`: same
- `TestEstimatePhase_EdgeCase_InputLengthOne`: same (if it uses FLASH_ATTN)

In all `makeFlashAttnNode` helpers, ensure InputShapes[0] has a 3rd element for num_heads.

- [ ] **Step 8: Run all perf tests**

Run: `go test ./perf/ -v -count=1`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add perf/estimate.go perf/estimate_test.go
git commit -m "perf: wire estimation to use num_heads from graph for FLASH_ATTN"
```

---

### Task 4: Update TODO.md with deferred items

**Files:**
- Modify: `docs/TODO.md`

- [ ] **Step 1: Add FLASH_ATTN deferred items to TODO.md**

Add under a new section or existing section:

```markdown
**Phase 1I: FLASH_ATTN Accuracy**
- [x] FLASH_ATTN benchmark: num_heads 网格化 — 从固定 32 头改为 {4, 8, 16, 32} 多头测量 + 插值 (来源: 2026-04-06, 完成: 2026-04-06)
- [ ] FLASH_ATTN benchmark: GQA-specific 测量 — Q/K/V 不同头数的 benchmark (来源: 2026-04-06)
- [ ] Sliding window seqKV 修正: `capacity = max(inputLength, sliding_window)` (来源: 2026-04-06)
- [ ] GLU, SET_ROWS 算子校准 (qwen3 模型使用，影响 <1%) (来源: 2026-04-06)
```

- [ ] **Step 2: Commit**

```bash
git add docs/TODO.md
git commit -m "docs: update TODO with FLASH_ATTN deferred items"
```
