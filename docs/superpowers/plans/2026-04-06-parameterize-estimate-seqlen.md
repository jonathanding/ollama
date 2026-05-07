# 参数化 Estimate 序列长度 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `daop-estimate` 使用实际的 input length 来 capture graph，替代硬编码的 512/2048，修复 FLASH_ATTN_EXT 和 MUL_MAT prefill 的严重高估问题（FLASH_ATTN prefill 0.09x、decode 0.41x；MUL_MAT prefill 0.50x）。

**Architecture:** KV cache 的 `capacity` 参数直接控制 K/V tensor 的 seqKV 维度（通过 `reserve=true` 时 `curCellRange.max = len(c.cells) - 1`）。将 `cache.Init(backend, dtype, 1, 2048, 512)` 改为 `cache.Init(backend, dtype, 1, inputLength, inputLength)`，prefill 用 `captureGraph(inputLength)`，decode 用 `captureGraph(1)` 但 seqKV=inputLength（由 cache capacity 决定）。CLI 已有未接线的 `--input-length` flag，只需贯穿到 `EstimateModel`。

**注意：`captureGraph(inputLength)` 影响所有 op 的 shape，不仅仅是 FLASH_ATTN。** Prefill graph 中 MUL_MAT 的 activation N 维度也从固定 512 变为 inputLength，这同样修复了 MUL_MAT prefill 的高估（0.50x ratio，因为 512/130≈3.9 被计入 latency）。

**Tech Stack:** Go, testify

**背景：当前问题**

`EstimateModel` 中：
- `cache.Init(m.Backend(), ml.DTypeF16, 1, 2048, 512)` — capacity=2048 决定了 decode 时 seqKV=2048
- `captureGraph(512)` — prefill 固定 seqQ=seqKV=512
- 实际推理中，130 token prompt 的 FLASH_ATTN 是 seqQ=seqKV=130，但 estimate 用了 512×512，高估 ~11x

**KV cache shape 机制：**
- `cache.Init(backend, dtype, maxSequences=1, capacity, maxBatch)` → `cacheSize = maxSequences * capacity`
- `StartForward(ctx, batch, reserve=true)` → `curCellRange.max = len(c.cells) - 1 = capacity - 1`
- `cache.Get()` → `cachedSize = c.curMask.Dim(0)` = 基于 curCellRange 的长度 = capacity
- K tensor shape: `(head_dim, capacity, num_kv_heads, 1)` → `ne[1] = capacity = seqKV`

**因此：改 capacity 就能精确控制 seqKV。**

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `perf/estimate.go:536-657` | Modify | `EstimateModel` + `RunEstimate` 加 inputLength 参数 |
| `perf/estimate.go:216-283` | Delete | 移除死代码 `buildModelGraphNodes` |
| `perf/cmd.go:65-80,120-125` | Modify | `RunEstimateCLI` + `EstimateCLIOptions` 加 InputLength |
| `cmd/cmd.go:2107-2113` | Modify | `daopEstimateHandler` 接线 `--input-length` flag |
| `perf/estimate_test.go` | Modify | 更新测试 |

---

### Task 1: 接线 CLI `--input-length` flag 到 EstimateModel

这个任务将已有的 `--input-length` CLI flag 贯穿到 graph capture 逻辑。

**Files:**
- Modify: `perf/cmd.go:65-80,120-125`
- Modify: `perf/estimate.go:536-657`
- Modify: `cmd/cmd.go:2107-2113`

- [ ] **Step 1: 更新 `EstimateCLIOptions` 添加 InputLength**

在 `perf/cmd.go:121-125`:

```go
// EstimateCLIOptions controls `ollama daop-estimate`.
type EstimateCLIOptions struct {
	Profile     string // --profile: profile path
	JSON        bool   // --json: output as JSON
	Verbose     bool   // --verbose: show per-op breakdown
	InputLength int    // --input-length: input prompt length in tokens (default 512)
}
```

- [ ] **Step 2: 更新 `RunEstimateCLI` 传递 InputLength**

在 `perf/cmd.go:66-80`，将 `opts.InputLength` 传给 `RunEstimate`:

```go
func RunEstimateCLI(modelRef string, opts EstimateCLIOptions) error {
	inputLength := opts.InputLength
	if inputLength <= 0 {
		inputLength = 512
	}
	result, err := RunEstimate(modelRef, opts.Profile, inputLength)
	if err != nil {
		return err
	}

	if opts.JSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(result)
	}

	PrintEstimateResult(os.Stdout, result, opts.Verbose)
	return nil
}
```

- [ ] **Step 3: 更新 `RunEstimate` 接受 inputLength 参数**

在 `perf/estimate.go:642-657`:

```go
func RunEstimate(modelRef string, profilePath string, inputLength int) (*EstimateResult, error) {
	if profilePath == "" {
		profilePath = ProfilePath()
	}
	profile, err := LoadProfile(profilePath)
	if err != nil {
		return nil, fmt.Errorf("load profile: %w (have you run 'ollama daop-bench'?)", err)
	}

	ggufPath, err := ResolveModelPath(modelRef)
	if err != nil {
		return nil, err
	}

	return EstimateModel(profile, ggufPath, inputLength)
}
```

- [ ] **Step 4: 更新 `EstimateModel` 接受 inputLength 参数并参数化 graph capture**

在 `perf/estimate.go:536`，修改函数签名和两处硬编码：

```go
func EstimateModel(profile *Profile, modelPath string, inputLength int) (*EstimateResult, error) {
```

将 `cache.Init` 行（当前在第 562 行）从：
```go
cache.Init(m.Backend(), ml.DTypeF16, 1, 2048, 512)
```
改为：
```go
cache.Init(m.Backend(), ml.DTypeF16, 1, inputLength, inputLength)
```

将 prefill captureGraph 调用（当前在第 601 行）从：
```go
prefillNodes, err := captureGraph(512)
```
改为：
```go
prefillNodes, err := captureGraph(inputLength)
```

decode captureGraph 调用保持不变：
```go
decodeNodes, err := captureGraph(1)
```

这样：
- Prefill: Q 的 seqQ = inputLength, K 的 seqKV = inputLength（cache capacity）
- Decode: Q 的 seqQ = 1, K 的 seqKV = inputLength（cache capacity）

- [ ] **Step 5: 更新 `daopEstimateHandler` 接线 flag**

在 `cmd/cmd.go:2107-2113`:

```go
func daopEstimateHandler(cmd *cobra.Command, args []string) error {
	jsonOutput, _ := cmd.Flags().GetBool("json")
	verbose, _ := cmd.Flags().GetBool("detail")
	inputLength, _ := cmd.Flags().GetInt("input-length")
	return perf.RunEstimateCLI(args[0], perf.EstimateCLIOptions{
		JSON:        jsonOutput,
		Verbose:     verbose,
		InputLength: inputLength,
	})
}
```

- [ ] **Step 6: 验证编译**

Run: `go build ./perf/ && go build ./cmd/`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add perf/cmd.go perf/estimate.go cmd/cmd.go
git commit -m "$(cat <<'EOF'
perf: parameterize estimate graph capture with actual input length

Replace hardcoded captureGraph(512) and cache.Init capacity=2048 with
user-provided inputLength. Wire existing --input-length CLI flag through
EstimateCLIOptions -> RunEstimate -> EstimateModel.

Fixes FLASH_ATTN_EXT estimation: prefill was 0.09x (seqQ/seqKV=512 vs
actual 130), decode was 0.41x (seqKV=2048 vs actual ~256).
EOF
)"
```

---

### Task 2: 添加有意义的测试

这个 task 的核心改动（cache.Init capacity + captureGraph batchSize）发生在 `EstimateModel` 内部，需要加载真实模型才能测。但我们可以在 `estimatePhase` / `estimatePhaseV3` 层面验证：**不同的 InputShapes 确实产生不同的 latency 估计**。这是 inputLength 参数化的直接效果——改 inputLength 就改了 graph 中 FLASH_ATTN 的 seqQ/seqKV，以及 MUL_MAT 的 activation N 维度。

**测试覆盖矩阵：**
| 测试 | 验证什么 | v2/v3 | 阶段 |
|---|---|---|---|
| FlashAttnScalesWithSeqLen | prefill FLASH_ATTN 随 seqQ×seqKV 缩放 | v2 | prefill |
| FlashAttnDecodeScalesWithKVLen | decode FLASH_ATTN 随 seqKV 缩放 | v2 | decode |
| LlamaDecodeFlashAttnPercentageIncreasesWithKVLen | 完整 layer 中 FLASH_ATTN 占比随 KV 增长 | v2 | decode |
| PrefillMulMatScalesWithInputLength | prefill MUL_MAT 随 N(=inputLength) 缩放 | v2 | prefill |
| FlashAttn_GQA | GQA 模型 ne[1] 索引正确性 | shape | — |
| FlashAttnScalesWithSeqLen_V3 | v3 路径的 FLASH_ATTN 缩放 | v3 | prefill |
| EdgeCase_InputLengthOne | inputLength=1 不 panic | v2 | decode |

**Files:**
- Modify: `perf/estimate_test.go`
- Modify: `perf/cmd_test.go`

- [ ] **Step 1: 检查现有测试是否直接调用 EstimateModel 或 RunEstimate**

Run: `grep -n "EstimateModel\|RunEstimate" perf/estimate_test.go perf/cmd_test.go perf/integration_test.go`

如果有直接调用这两个函数的测试，需要补上第三个参数 `inputLength`。

- [ ] **Step 2: 添加 FLASH_ATTN prefill 估计随 seqlen 缩放的测试**

在 `perf/estimate_test.go` 添加。这验证了核心行为：inputLength=130 的 graph 应该比 inputLength=512 的 graph 产生更小的 FLASH_ATTN 估计。

```go
func TestEstimatePhase_FlashAttnScalesWithSeqLen(t *testing.T) {
	// Core test: FLASH_ATTN_EXT latency should scale with seqQ×seqKV.
	// This is the direct observable effect of parameterizing captureGraph(inputLength):
	// smaller inputLength → smaller seqQ/seqKV in graph → lower FLASH_ATTN estimate.
	p := makeTestProfileForEstimation()

	makeFlashAttnNode := func(seqQ, seqKV int64) ml.GraphNode {
		return ml.GraphNode{
			Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
			InputShapes: [][]int64{
				{128, seqQ, 32, 1},  // Q: [head_dim, seqQ, num_heads, batch]
				{128, seqKV, 32, 1}, // K: [head_dim, seqKV, num_kv_heads, batch]
			},
		}
	}

	// Simulate prefill graphs at different input lengths
	nodes130 := []ml.GraphNode{makeFlashAttnNode(130, 130)}
	nodes512 := []ml.GraphNode{makeFlashAttnNode(512, 512)}

	var w1, w2 []string
	result130 := estimatePhase(p, nodes130, &w1)
	result512 := estimatePhase(p, nodes512, &w2)

	require.NotEmpty(t, result130.TopOps)
	require.NotEmpty(t, result512.TopOps)

	lat130 := result130.TopOps[0].TotalUs
	lat512 := result512.TopOps[0].TotalUs

	assert.Greater(t, lat512, lat130,
		"FLASH_ATTN at seqlen=512 (%.1fus) should be greater than seqlen=130 (%.1fus)",
		lat512, lat130)
	// FLASH_ATTN is roughly O(seqQ × seqKV). Log-space interpolation gives:
	// seqlen=130 → ~20.37us, seqlen=512 → 100.0us, ratio ≈ 4.91x
	ratio := lat512 / lat130
	assert.Greater(t, ratio, 4.0,
		"latency ratio should reflect quadratic scaling, got %.1fx", ratio)
}
```

- [ ] **Step 3: 添加 decode FLASH_ATTN 随 seqKV 缩放的测试**

这验证 decode 场景：seqQ=1 固定，seqKV 变化（由 cache capacity 决定）。

```go
func TestEstimatePhase_FlashAttnDecodeScalesWithKVLen(t *testing.T) {
	// Decode: seqQ=1, seqKV varies with inputLength (= cache capacity).
	// Smaller inputLength → smaller seqKV → lower decode FLASH_ATTN estimate.
	p := makeTestProfileForEstimation()

	makeDecodeFlashAttn := func(seqKV int64) ml.GraphNode {
		return ml.GraphNode{
			Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
			InputShapes: [][]int64{
				{128, 1, 32, 1},     // Q: seqQ=1
				{128, seqKV, 32, 1}, // K: seqKV varies
			},
		}
	}

	var w1, w2 []string
	result130 := estimatePhase(p, []ml.GraphNode{makeDecodeFlashAttn(130)}, &w1)
	result2048 := estimatePhase(p, []ml.GraphNode{makeDecodeFlashAttn(2048)}, &w2)

	lat130 := result130.TopOps[0].TotalUs
	lat2048 := result2048.TopOps[0].TotalUs

	assert.Greater(t, lat2048, lat130,
		"decode FLASH_ATTN at seqKV=2048 (%.1fus) should be greater than seqKV=130 (%.1fus)",
		lat2048, lat130)
	// Decode FLASH_ATTN is roughly O(seqKV), so ratio ≈ 2048/130 ≈ 15.8x
	ratio := lat2048 / lat130
	assert.Greater(t, ratio, 3.0,
		"decode latency ratio should reflect linear KV scaling, got %.1fx", ratio)
}
```

- [ ] **Step 4: 添加完整 Llama layer decode 测试 — FLASH_ATTN 占比随 KV 增长**

这是最接近真实场景的测试：一个完整的 transformer layer graph。

**注意（Issue 2 修复）：** MUL_MAT 在任何 KV 长度下都是绝对量最大的（6 个 MUL_MAT ~147.5us vs 1 个 FLASH_ATTN 5~55us）。因此不能断言"MUL_MAT 在短 KV 时 dominate 而长 KV 时不 dominate"——MUL_MAT 始终 dominate。正确的断言是 **FLASH_ATTN 占总延迟的百分比随 KV 增长而显著增加**。

```go
func TestEstimatePhase_LlamaDecodeFlashAttnPercentageIncreasesWithKVLen(t *testing.T) {
	// In a full Llama decode layer, MUL_MAT always dominates by count (6 MUL_MATs vs 1 FLASH_ATTN).
	// But FLASH_ATTN's percentage of total latency should increase with longer KV cache.
	// This validates that inputLength parameterization meaningfully changes the estimate balance.
	p := makeTestProfileForEstimation()

	makeLlamaDecodeLayer := func(seqKV int64) []ml.GraphNode {
		return []ml.GraphNode{
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
			{Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
				InputShapes: [][]int64{{128, 1, 32, 1}, {128, seqKV, 32, 1}}},
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				InputShapes: [][]int64{{14336, 4096}, {4096, 1}}},
			{Op: "SILU", Backend: "cuda", Shape: [4]int64{14336, 1, 1, 1}, ComputeDtype: "f32"},
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				InputShapes: [][]int64{{4096, 14336}, {14336, 1}}},
		}
	}

	var w1, w2 []string
	resultShort := estimatePhase(p, makeLlamaDecodeLayer(128), &w1)
	resultLong := estimatePhase(p, makeLlamaDecodeLayer(2048), &w2)

	// Find FLASH_ATTN percentage in each result
	flashPctShort := 0.0
	flashPctLong := 0.0
	totalShortUs := resultShort.TotalLatencyMs * 1000
	totalLongUs := resultLong.TotalLatencyMs * 1000
	for _, op := range resultShort.TopOps {
		if op.Op == "FLASH_ATTN_EXT" {
			flashPctShort = op.TotalUs / totalShortUs * 100
		}
	}
	for _, op := range resultLong.TopOps {
		if op.Op == "FLASH_ATTN_EXT" {
			flashPctLong = op.TotalUs / totalLongUs * 100
		}
	}

	assert.Greater(t, flashPctLong, flashPctShort,
		"FLASH_ATTN percentage should increase with longer KV: short=%.1f%%, long=%.1f%%",
		flashPctShort, flashPctLong)

	// With seqKV=2048, FLASH_ATTN should be a meaningful fraction (>15%) of total
	assert.Greater(t, flashPctLong, 15.0,
		"with seqKV=2048, FLASH_ATTN should be >15%% of total, got %.1f%%", flashPctLong)

	// Total latency should increase with longer KV
	assert.Greater(t, resultLong.TotalLatencyMs, resultShort.TotalLatencyMs,
		"longer KV cache should increase total decode latency")
}
```

- [ ] **Step 5: 添加 prefill MUL_MAT 随 inputLength 缩放的测试（Issue 3）**

`captureGraph(inputLength)` 不仅影响 FLASH_ATTN，也改变 MUL_MAT 的 activation N 维度。Prefill 时 N=inputLength（不是固定 512）。这就是 MUL_MAT prefill 0.50x ratio 的原因。

```go
func TestEstimatePhase_PrefillMulMatScalesWithInputLength(t *testing.T) {
	// captureGraph(inputLength) changes MUL_MAT activation's N dimension.
	// Prefill N=130 should be cheaper than N=512 (less compute work).
	// This is why MUL_MAT prefill was 0.50x with hardcoded 512 vs actual 130.
	p := makeTestProfileForEstimation()

	makePrefillMulMatNodes := func(seqLen int64) []ml.GraphNode {
		return []ml.GraphNode{
			// Q/K/V projections: weight={4096,4096}, activation={4096,seqLen}
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				InputShapes: [][]int64{{4096, 4096}, {4096, seqLen}}},
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				InputShapes: [][]int64{{4096, 4096}, {4096, seqLen}}},
			// FFN up: weight={14336,4096}, activation={4096,seqLen}
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				InputShapes: [][]int64{{14336, 4096}, {4096, seqLen}}},
			// FFN down: weight={4096,14336}, activation={14336,seqLen}
			{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
				InputShapes: [][]int64{{4096, 14336}, {14336, seqLen}}},
		}
	}

	var w1, w2 []string
	result130 := estimatePhase(p, makePrefillMulMatNodes(130), &w1)
	result512 := estimatePhase(p, makePrefillMulMatNodes(512), &w2)

	assert.Greater(t, result512.TotalLatencyMs, result130.TotalLatencyMs,
		"prefill MUL_MAT at N=512 (%.3fms) should be greater than N=130 (%.3fms)",
		result512.TotalLatencyMs, result130.TotalLatencyMs)

	// Roofline: for large matrices, latency scales linearly with N (bandwidth-bound).
	// Ratio should be close to 512/130 ≈ 3.94x
	ratio := result512.TotalLatencyMs / result130.TotalLatencyMs
	assert.Greater(t, ratio, 2.0,
		"prefill MUL_MAT latency ratio should reflect N scaling, got %.1fx", ratio)
}
```

- [ ] **Step 6: 添加 nodeToQueryShape GQA 配置测试**

验证 GQA（Grouped Query Attention）时 ne[2] 不同不影响 seqQ/seqKV 提取——确保 ne[1] 索引正确。

```go
func TestNodeToQueryShape_FlashAttn_GQA(t *testing.T) {
	// GQA: Q has 32 heads, K/V have 8 heads. ne[2] differs but ne[1] (seqlen) is the same.
	// This verifies the ne[1] fix is correct for GQA models (like Qwen3, Llama3).
	node := ml.GraphNode{
		Op:           "FLASH_ATTN_EXT",
		Backend:      "cuda",
		ComputeDtype: "f16",
		InputShapes: [][]int64{
			{128, 130, 32, 1}, // Q: head_dim=128, seqQ=130, num_heads=32
			{128, 256, 8, 1},  // K: head_dim=128, seqKV=256, num_kv_heads=8 (GQA)
		},
	}
	op, shape, _, _ := nodeToQueryShape(node)
	assert.Equal(t, "FLASH_ATTN_EXT", op)
	require.Len(t, shape, 2)
	assert.Equal(t, int64(130), shape[0], "seqQ should come from Q ne[1], not ne[2]")
	assert.Equal(t, int64(256), shape[1], "seqKV should come from K ne[1], not ne[2]")
}
```

- [ ] **Step 7: 添加 v3 路径 FLASH_ATTN 缩放测试（Issue 4）**

实际代码在 `profile.Version >= 3` 时走 `estimatePhaseV3`。需要至少一个 v3 路径测试验证 FLASH_ATTN 缩放在 v3 下同样成立。

```go
func TestEstimatePhaseV3_FlashAttnScalesWithSeqLen(t *testing.T) {
	// Same as v2 test but through estimatePhaseV3 path.
	// Verifies that v3 (with fusion + orchestration overhead) still correctly
	// reflects FLASH_ATTN scaling with different input lengths.
	p := makeTestProfileForEstimation()
	p.Version = 3

	makeFlashAttnNode := func(seqQ, seqKV int64) ml.GraphNode {
		return ml.GraphNode{
			Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
			InputShapes: [][]int64{
				{128, seqQ, 32, 1},
				{128, seqKV, 32, 1},
			},
		}
	}

	caps := &BackendCapabilities{Name: "cuda", HasGPUTimestamp: false}
	var w1, w2 []string
	result130 := estimatePhaseV3(p, []ml.GraphNode{makeFlashAttnNode(130, 130)}, caps, &w1)
	result512 := estimatePhaseV3(p, []ml.GraphNode{makeFlashAttnNode(512, 512)}, caps, &w2)

	require.NotEmpty(t, result130.TopOps)
	require.NotEmpty(t, result512.TopOps)

	lat130 := result130.TopOps[0].TotalUs
	lat512 := result512.TopOps[0].TotalUs

	assert.Greater(t, lat512, lat130,
		"v3: FLASH_ATTN at seqlen=512 (%.1fus) should be greater than seqlen=130 (%.1fus)",
		lat512, lat130)
	ratio := lat512 / lat130
	assert.Greater(t, ratio, 4.0,
		"v3: latency ratio should reflect quadratic scaling, got %.1fx", ratio)
}
```

- [ ] **Step 8: 添加 edge case inputLength=1 测试（Issue 5）**

验证极端情况 inputLength=1（单 token decode）不会 panic。这也是实际的 decode 场景（seqQ=1, seqKV=1 = 第一个 decode token 紧接 1-token prompt）。

```go
func TestEstimatePhase_EdgeCase_InputLengthOne(t *testing.T) {
	// inputLength=1 is an edge case: both prefill and decode have minimal seqlen.
	// Must not panic and should produce valid (non-zero) estimates.
	p := makeTestProfileForEstimation()

	// Prefill with seqQ=seqKV=1
	prefillNodes := []ml.GraphNode{
		{Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
			InputShapes: [][]int64{{128, 1, 32, 1}, {128, 1, 32, 1}}},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
	}

	// Decode with seqQ=1, seqKV=1
	decodeNodes := []ml.GraphNode{
		{Op: "FLASH_ATTN_EXT", Backend: "cuda", ComputeDtype: "f16",
			InputShapes: [][]int64{{128, 1, 32, 1}, {128, 1, 32, 1}}},
		{Op: "MUL_MAT", Backend: "cuda", ComputeDtype: "f16", WeightDtype: "q4_0",
			InputShapes: [][]int64{{4096, 4096}, {4096, 1}}},
	}

	var w1, w2 []string
	resultPrefill := estimatePhase(p, prefillNodes, &w1)
	resultDecode := estimatePhase(p, decodeNodes, &w2)

	// Should produce valid non-zero results without panicking
	assert.Greater(t, resultPrefill.TotalLatencyMs, 0.0, "prefill with inputLength=1 should have non-zero latency")
	assert.Greater(t, resultDecode.TotalLatencyMs, 0.0, "decode with inputLength=1 should have non-zero latency")
	assert.NotEmpty(t, resultPrefill.TopOps, "prefill should have op breakdown")
	assert.NotEmpty(t, resultDecode.TopOps, "decode should have op breakdown")
}
```

- [ ] **Step 9: 更新 `perf/cmd_test.go` 的 EstimateCLIOptions 测试**

在 `perf/cmd_test.go:19-24`，更新 `TestEstimateCLIOptions_Defaults`：

```go
func TestEstimateCLIOptions_Defaults(t *testing.T) {
	opts := EstimateCLIOptions{}
	assert.Empty(t, opts.Profile)
	assert.False(t, opts.JSON)
	assert.False(t, opts.Verbose)
	assert.Equal(t, 0, opts.InputLength, "zero means RunEstimateCLI applies default 512")
}
```

- [ ] **Step 10: 运行测试**

Run: `go test ./perf/ -count=1 -v -run "FlashAttnScale|FlashAttnDecode|FlashAttnPercentage|PrefillMulMat|FlashAttn_GQA|InputLengthOne|EstimateCLIOptions"`
Expected: PASS

Run: `go test ./perf/ -count=1`
Expected: PASS (全量回归)

- [ ] **Step 11: Commit**

```bash
git add perf/estimate_test.go perf/cmd_test.go
git commit -m "$(cat <<'EOF'
perf: add tests for estimation scaling with sequence length

Tests verify core parameterization behavior:
- Prefill FLASH_ATTN scales quadratically with seqQ×seqKV (v2 + v3)
- Decode FLASH_ATTN scales linearly with seqKV (cache length)
- Full Llama layer: FLASH_ATTN percentage increases with longer KV
- Prefill MUL_MAT scales with activation N dimension (inputLength)
- GQA shape extraction: ne[1] is seqlen regardless of num_heads
- Edge case: inputLength=1 produces valid estimates without panic
EOF
)"
```

---

### Task 3: 清理死代码 `buildModelGraphNodes`

`buildModelGraphNodes`（`perf/estimate.go:216-283`）没有任何调用者，是 `EstimateModel` 引入后的遗留代码。

**Files:**
- Modify: `perf/estimate.go:216-283`

- [ ] **Step 1: 确认无调用者**

Run: `grep -rn "buildModelGraphNodes" perf/`
Expected: 只有定义处（estimate.go:219），无调用。

- [ ] **Step 2: 删除 `buildModelGraphNodes` 函数**

删除 `perf/estimate.go` 中从 `// buildModelGraphNodes loads a model...`（第 216 行）到函数结尾 `return prefill, decode, nil` + 右括号（第 283 行）的整段代码。

- [ ] **Step 3: 验证编译和测试**

Run: `go build ./perf/ && go test ./perf/ -count=1`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add perf/estimate.go
git commit -m "perf: remove dead code buildModelGraphNodes"
```

---

### Task 4: E2E 验证

**Files:** 无（验证 only）

- [ ] **Step 1: 用默认参数运行 estimate**

Run: `go run . daop-estimate qwen3:1.7b 2>&1`

Expected: prefill 用 inputLength=512 capture graph，结果应该和之前相同（回归测试）。

- [ ] **Step 2: 用 `--input-length 130` 运行 estimate**

Run: `go run . daop-estimate qwen3:1.7b --input-length 130 2>&1`

Expected:
- Prefill FLASH_ATTN_EXT 的估计值应该大幅下降（从 ~2.5s 降到 ~200-300ms 量级）
- Decode FLASH_ATTN_EXT 的估计值也应该下降（seqKV 从 2048 降到 130）
- 总 prefill 延迟应该从 ~4000ms 降到更接近实际 ~856ms 的值

- [ ] **Step 3: 与实际 Vulkan timing 对比**

实际 prefill（130 tokens）Vulkan timing = 856,094 us。
实际 decode（per token）Vulkan timing = 81,560 us。

对比新 estimate 结果，计算 ratio，确认精度改善。

---

## Execution Notes

- Task 1 是核心改动，Task 2-3 可并行。Task 4 依赖 Task 1。
- **不改 `reserve` 逻辑**：继续用 `reserve=true`。通过 `cache.Init` 的 capacity 参数控制 KV cache 大小，这是最小侵入性的方案。
- **Decode seqKV = inputLength**：这代表 decode 第一个 token 时的状态。实际生成过程中 seqKV 会增长，但第一个 token 是最常被关注的（也是 TTFT 之后的首个 decode latency）。如果以后需要 average decode latency，可以用 `inputLength + outputLength/2` 作为 capacity。
- **`--output-length` flag 暂不接线**：当前只解决 input length 问题。output length 影响 decode 阶段平均 seqKV，是未来改进项。
