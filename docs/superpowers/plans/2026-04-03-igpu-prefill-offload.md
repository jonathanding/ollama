# iGPU Prefill Offload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Intel Arc iGPU (Arrow Lake, UMA) to accelerate prefill for models that overflow the dGPU VRAM, using a two-phase approach gated on bench-sweep validation.

**Architecture:** Phase 1 injects the iGPU Vulkan backend into the GGML scheduler (bypassing the empty-tensor guard) behind `OLLAMA_IGPU_OFFLOAD=1`, relying on the built-in `op->ne[1] >= 32` gate to route only prefill matmuls to iGPU while decode stays on CPU. Phase 2 (gated on Phase 1 bench results) adds explicit cross-library layer assignment in `createLayout()`, so overflow layers are given to iGPU rather than staying on CPU.

**Tech Stack:** Go, CGo (ggml C bindings), `ml/backend/ggml/ggml.go`, `llm/server.go`, `envconfig/config.go`, `bench-sweep` (cmd/bench-sweep on feat/bench-sweep-standalone branch)

---

## File Map

| File | Change |
|------|--------|
| `envconfig/config.go` | Add `IGPUOffload = Bool("OLLAMA_IGPU_OFFLOAD")` |
| `ml/backend/ggml/ggml.go` | Phase 1: bypass empty-tensor check for iGPU when IGPUOffload enabled |
| `llm/server.go` | Phase 2: add `assignOverflowToIGPU()`, call from `createLayout()` |
| `llm/server_test.go` | Phase 2: add `TestAssignOverflowToIGPU` |
| `docs/internals/05-cross-library-gpu-mixing.md` | Correct §6.7 UMA zero-copy claim |

---

## Task 1: Add `OLLAMA_IGPU_OFFLOAD` env var

**Files:**
- Modify: `envconfig/config.go` (near line 226, after `SchedSpread`)

- [ ] **Step 1: Add the env var**

In `envconfig/config.go`, add one line after the `SchedSpread` declaration (around line 226):

```go
// SchedSpread allows scheduling models across all GPUs.
SchedSpread = Bool("OLLAMA_SCHED_SPREAD")
// IGPUOffload enables iGPU Vulkan acceleration for prefill (compute-bound) operations
// when a model partially overflows GPU VRAM onto CPU. See docs/superpowers/specs/2026-04-03-igpu-offload-design.md.
IGPUOffload = Bool("OLLAMA_IGPU_OFFLOAD")
```

- [ ] **Step 2: Add it to the info table**

Find the `Info()` map near line 324, and add an entry alongside `SchedSpread`:

```go
"OLLAMA_IGPU_OFFLOAD":   {"OLLAMA_IGPU_OFFLOAD", IGPUOffload(), "Offload prefill to iGPU Vulkan when VRAM overflows to CPU"},
```

- [ ] **Step 3: Verify it compiles**

```bash
go build ./envconfig/...
```

Expected: no output (clean build).

- [ ] **Step 4: Commit**

```bash
git add envconfig/config.go
git commit -m "envconfig: add OLLAMA_IGPU_OFFLOAD env var for iGPU prefill offload"
```

---

## Task 2: Phase 1 — Inject iGPU into GGML scheduler

**Background:** `ml/backend/ggml/ggml.go` loops over GPU devices at ~line 359 and builds `schedBackends`/`schedBufts` for the GGML scheduler. The guard at ~line 364 skips any GPU backend whose `ctxs[bt]` is nil or empty — which is always the case for the iGPU when no layers are assigned to it. Phase 1 bypasses this guard for iGPU devices when `OLLAMA_IGPU_OFFLOAD=1`, injecting iGPU into the scheduler so that `ggml_backend_vk_device_offload_op()` can route prefill matmuls to it via `op_offload`. Decode matmuls (batch=1) are excluded by the Vulkan backend's built-in `op->ne[1] >= 32` check.

**Files:**
- Modify: `ml/backend/ggml/ggml.go` (~lines 364–368)

- [ ] **Step 1: Read the current code**

Read `ml/backend/ggml/ggml.go` around line 356–380 to confirm the exact lines before editing.

The current guard looks like:

```go
// Always include CPU as a fallback but otherwise, just use the devices where we assigned layers
if !slices.Contains(cpuDeviceBufferType.bts, bt) {
    if c, ok := ctxs[bt]; !ok || C.ggml_get_first_tensor(c) == nil {
        continue
    }
}
```

- [ ] **Step 2: Apply the Phase 1 patch**

Replace the guard block with:

```go
// Always include CPU as a fallback but otherwise, just use the devices where we assigned layers.
// Exception: when OLLAMA_IGPU_OFFLOAD is set, inject iGPU Vulkan into the scheduler even if it
// has no assigned layers — this allows op_offload to route prefill matmuls (ne[1]>=32) to iGPU.
if !slices.Contains(cpuDeviceBufferType.bts, bt) {
    isIGPU := envconfig.IGPUOffload() && C.ggml_backend_dev_type(d) == C.GGML_BACKEND_DEVICE_TYPE_IGPU
    if !isIGPU {
        if c, ok := ctxs[bt]; !ok || C.ggml_get_first_tensor(c) == nil {
            continue
        }
    }
}
```

- [ ] **Step 3: Verify it compiles**

```bash
go build ./ml/...
```

Expected: no output.

- [ ] **Step 4: Run existing ml tests**

```bash
go test ./ml/...
```

Expected: all tests pass (PASS).

- [ ] **Step 5: Commit**

```bash
git add ml/backend/ggml/ggml.go
git commit -m "ml/backend/ggml: inject iGPU Vulkan into scheduler when OLLAMA_IGPU_OFFLOAD=1 (Phase 1)"
```

---

## Task 3: Run full test suite to verify Phase 1 regression-free

**Files:** None modified.

- [ ] **Step 1: Run llm tests**

```bash
go test ./llm/...
```

Expected: PASS.

- [ ] **Step 2: Run server tests**

```bash
go test ./server/...
```

Expected: PASS.

- [ ] **Step 3: Run all tests**

```bash
go test ./...
```

Expected: PASS. If any test fails, investigate before proceeding.

---

## Task 4: Phase 1 bench-sweep validation (gate for Phase 2)

This task validates whether Phase 1 actually accelerates prefill and doesn't regress decode.
**Phase 2 must not be implemented until these criteria are met:**

| Metric | Pass criterion |
|--------|----------------|
| `prefill_tps` improvement vs baseline | ≥ 20% |
| `gen_tps` regression vs baseline | ≤ 5% |
| `prefill_tps` CV% | < 10% |

- [ ] **Step 1: Check out bench-sweep branch and build**

```bash
git fetch origin feat/bench-sweep-standalone
git worktree add ../bench-sweep-build feat/bench-sweep-standalone
cd ../bench-sweep-build
go build -o ../ollama-bench-sweep.exe ./cmd/bench-sweep/
cd -
```

Expected: `ollama-bench-sweep.exe` created.

Alternatively, if already built: confirm `cmd/bench-sweep/` exists on current branch.

- [ ] **Step 2: Start ollama in baseline mode (no IGPU_OFFLOAD)**

```bash
# In a separate terminal, without OLLAMA_IGPU_OFFLOAD:
go run . serve
```

Wait for model to load: `ollama run qwen3-coder-next:latest` (just to load, then Ctrl-C the chat).

- [ ] **Step 3: Run baseline bench**

```bash
./ollama-bench-sweep.exe run \
  -model qwen3-coder-next \
  -name baseline \
  -sizes 512,1024,2048,4096 \
  -epochs 8 \
  -warmup 4
```

Expected output: JSON results saved to `~/.ollama/bench/baseline-*.json` with `prefill_tps`, `ttft_ms`, `gen_tps` per size.

- [ ] **Step 4: Restart ollama with Phase 1 enabled**

Stop the current `ollama serve`. Start with:

```bash
# In a separate terminal:
OLLAMA_IGPU_OFFLOAD=1 go run . serve
```

Load the model again: `ollama run qwen3-coder-next:latest`.

During prefill: open Windows Task Manager → Performance → GPU (Intel) → watch for Compute utilization spike above baseline. This visually confirms iGPU is being used.

- [ ] **Step 5: Run Phase 1 bench**

```bash
./ollama-bench-sweep.exe run \
  -model qwen3-coder-next \
  -name igpu-phase1 \
  -sizes 512,1024,2048,4096 \
  -epochs 8 \
  -warmup 4
```

Expected: results saved to `~/.ollama/bench/igpu-phase1-*.json`.

- [ ] **Step 6: Diff results**

```bash
./ollama-bench-sweep.exe diff baseline igpu-phase1
```

Expected output: table comparing `prefill_tps`, `gen_tps`, `ttft_ms`, CV% for each prompt size.

**Evaluate against pass criteria:**
- `prefill_tps` improvement ≥ 20% at sizes 512–4096: **proceed to Task 5**
- `gen_tps` regression ≤ 5%: **proceed to Task 5**
- If criteria NOT met: stop, investigate. Do not implement Phase 2. File findings in `docs/TODO.md`.

---

## Task 5: Phase 2 — Explicit cross-library iGPU layer assignment

**Background:** Phase 1 is a scheduler-level hint; iGPU still has no layers assigned in `createLayout()`, so it only participates via op_offload routing of CPU-overflow matmuls. Phase 2 explicitly assigns overflow layers to the iGPU device, giving it permanent Vulkan device buffer allocations. This makes iGPU first-class in the layer layout, enables better memory accounting, and works regardless of scheduler behavior.

**Prerequisite:** Task 4 bench criteria PASSED.

**Files:**
- Modify: `llm/server.go` — add `assignOverflowToIGPU()`, call from `createLayout()`

- [ ] **Step 1: Read `createLayout` and `buildLayout` signatures**

Read `llm/server.go` lines 920–1000 to confirm the current `createLayout` / `buildLayout` / `verifyLayout` call chain and `systemGPUs` parameter availability.

Key: `createLayout(systemInfo ml.SystemInfo, systemGPUs []ml.DeviceInfo, memory *ml.BackendMemory, requireFull bool, backoff float32)` has both `systemGPUs` and `systemInfo` in scope after `buildLayout` returns.

- [ ] **Step 2: Add `assignOverflowToIGPU` function**

After the closing brace of `buildLayout` (around line 999), add:

```go
// assignOverflowToIGPU assigns layers not covered by the ByLibrary winner to any
// integrated (UMA) GPU device. It is called from createLayout when OLLAMA_IGPU_OFFLOAD=1
// and gpuLayers does not cover all model layers.
//
// UMA iGPU devices typically report FreeMemory=0 at layout time because the runner has
// not yet allocated any buffers on them. In that case, systemInfo.FreeMemory is used as
// a proxy for available memory (DDR5 is shared between CPU and iGPU).
func assignOverflowToIGPU(allLayers []uint64, gpus []ml.DeviceInfo, assigned ml.GPULayersList, systemInfo ml.SystemInfo) ml.GPULayersList {
	// Collect integrated GPU devices; use systemInfo.FreeMemory as fallback when FreeMemory==0
	var igpuDevs []ml.DeviceInfo
	for _, g := range gpus {
		if !g.Integrated {
			continue
		}
		igpu := g
		if igpu.FreeMemory == 0 {
			igpu.FreeMemory = systemInfo.FreeMemory
		}
		igpuDevs = append(igpuDevs, igpu)
	}
	if len(igpuDevs) == 0 {
		return nil
	}

	// Build set of already-assigned layer indices
	assignedSet := make(map[int]bool)
	for _, gl := range assigned {
		for _, l := range gl.Layers {
			assignedSet[l] = true
		}
	}

	// Collect overflow (unassigned) layers in order
	var overflowSizes []uint64
	var overflowIndices []int
	for i, sz := range allLayers {
		if !assignedSet[i] {
			overflowSizes = append(overflowSizes, sz)
			overflowIndices = append(overflowIndices, i)
		}
	}
	if len(overflowSizes) == 0 {
		return nil
	}

	// Run layer assignment against the overflow subset
	iGPULayers := assignLayers(overflowSizes, igpuDevs, false, -1, 0)

	// Remap relative indices (into overflowSizes) to absolute layer indices
	for i := range iGPULayers {
		for j, rel := range iGPULayers[i].Layers {
			iGPULayers[i].Layers[j] = overflowIndices[rel]
		}
	}
	return iGPULayers
}
```

- [ ] **Step 3: Wire `assignOverflowToIGPU` into `createLayout`**

In `createLayout` (around line 933), change:

```go
gpuLayers, layers := s.buildLayout(systemGPUs, memory, requireFull, backoff)
err := s.verifyLayout(systemInfo, systemGPUs, memory, requireFull, gpuLayers, layers)
```

to:

```go
gpuLayers, layers := s.buildLayout(systemGPUs, memory, requireFull, backoff)
if envconfig.IGPUOffload() && gpuLayers.Sum() < len(layers) {
    igpuOverflow := assignOverflowToIGPU(layers, systemGPUs, gpuLayers, systemInfo)
    gpuLayers = append(gpuLayers, igpuOverflow...)
}
err := s.verifyLayout(systemInfo, systemGPUs, memory, requireFull, gpuLayers, layers)
```

- [ ] **Step 4: Verify it compiles**

```bash
go build ./llm/...
```

Expected: no output.

---

## Task 6: Phase 2 test — `TestAssignOverflowToIGPU`

**Files:**
- Modify: `llm/server_test.go`

- [ ] **Step 1: Write the failing test**

Add this new test function to `llm/server_test.go` (after the closing brace of `TestLLMServerFitGPU`):

```go
// TestAssignOverflowToIGPU verifies that when OLLAMA_IGPU_OFFLOAD=1 and CUDA wins
// the ByLibrary competition but cannot fit all layers, overflow layers are assigned
// to the integrated GPU using systemInfo.FreeMemory as the UMA memory proxy.
func TestAssignOverflowToIGPU(t *testing.T) {
	t.Setenv("OLLAMA_IGPU_OFFLOAD", "1")

	const minMemory = 457 * format.MebiByte
	const layerSize = 100 * format.MebiByte
	const numLayers = 6
	const cudaFit = 3 // CUDA can fit exactly 3 layers

	gpus := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{Library: "cuda", ID: "gpu0"}, FreeMemory: uint64(cudaFit*layerSize + minMemory)},
		{DeviceID: ml.DeviceID{Library: "vulkan", ID: "igpu0"}, Integrated: true, FreeMemory: 0},
	}

	var systemInfo ml.SystemInfo
	systemInfo.TotalMemory = 128 * format.GibiByte
	systemInfo.FreeMemory = format.GibiByte  // large enough for iGPU UMA fallback (3 layers = 300MB)
	systemInfo.FreeSwap = 512 * format.MebiByte

	s := &ollamaServer{
		llmServer: llmServer{
			totalLayers: uint64(numLayers),
			options: api.Options{
				Runner: api.Runner{NumGPU: -1},
			},
		},
	}

	s.mem = &ml.BackendMemory{
		CPU: ml.DeviceMemory{
			Weights: make([]uint64, numLayers),
			Cache:   make([]uint64, numLayers),
		},
		GPUs: make([]ml.DeviceMemory, len(gpus)),
	}
	for i := range numLayers {
		s.mem.CPU.Weights[i] = layerSize
	}
	for i := range s.mem.GPUs {
		s.mem.GPUs[i].DeviceID = gpus[i].DeviceID
		s.mem.GPUs[i].Weights = make([]uint64, numLayers)
		s.mem.GPUs[i].Cache = make([]uint64, numLayers)
	}

	gpuLayers, err := s.createLayout(systemInfo, gpus, s.mem, false, 0)
	if err != nil {
		t.Fatalf("createLayout returned unexpected error: %v", err)
	}

	// All layers should be assigned (CUDA: 3 + iGPU: 3)
	if got := gpuLayers.Sum(); got != numLayers {
		t.Errorf("expected all %d layers assigned, got %d; layout: %v", numLayers, got, gpuLayers)
	}

	var cudaCount, igpuCount int
	for _, gl := range gpuLayers {
		switch gl.DeviceID.Library {
		case "cuda":
			cudaCount += len(gl.Layers)
		case "vulkan":
			igpuCount += len(gl.Layers)
		}
	}
	if cudaCount != cudaFit {
		t.Errorf("CUDA layer count: want %d, got %d", cudaFit, cudaCount)
	}
	if igpuCount != numLayers-cudaFit {
		t.Errorf("iGPU layer count: want %d, got %d", numLayers-cudaFit, igpuCount)
	}
}

// TestAssignOverflowToIGPUDisabled verifies that without OLLAMA_IGPU_OFFLOAD, the
// cross-library overflow assignment does NOT happen (baseline behavior preserved).
func TestAssignOverflowToIGPUDisabled(t *testing.T) {
	// OLLAMA_IGPU_OFFLOAD is NOT set

	const minMemory = 457 * format.MebiByte
	const layerSize = 100 * format.MebiByte
	const numLayers = 6
	const cudaFit = 3

	gpus := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{Library: "cuda", ID: "gpu0"}, FreeMemory: uint64(cudaFit*layerSize + minMemory)},
		{DeviceID: ml.DeviceID{Library: "vulkan", ID: "igpu0"}, Integrated: true, FreeMemory: 0},
	}

	var systemInfo ml.SystemInfo
	systemInfo.TotalMemory = 128 * format.GibiByte
	systemInfo.FreeMemory = format.GibiByte
	systemInfo.FreeSwap = 512 * format.MebiByte

	s := &ollamaServer{
		llmServer: llmServer{
			totalLayers: uint64(numLayers),
			options:     api.Options{Runner: api.Runner{NumGPU: -1}},
		},
	}

	s.mem = &ml.BackendMemory{
		CPU: ml.DeviceMemory{
			Weights: make([]uint64, numLayers),
			Cache:   make([]uint64, numLayers),
		},
		GPUs: make([]ml.DeviceMemory, len(gpus)),
	}
	for i := range numLayers {
		s.mem.CPU.Weights[i] = layerSize
	}
	for i := range s.mem.GPUs {
		s.mem.GPUs[i].DeviceID = gpus[i].DeviceID
		s.mem.GPUs[i].Weights = make([]uint64, numLayers)
		s.mem.GPUs[i].Cache = make([]uint64, numLayers)
	}

	gpuLayers, err := s.createLayout(systemInfo, gpus, s.mem, false, 0)
	if err != nil {
		t.Fatalf("createLayout returned unexpected error: %v", err)
	}

	// Without IGPU_OFFLOAD: only CUDA layers (no cross-library overflow assignment)
	if got := gpuLayers.Sum(); got != cudaFit {
		t.Errorf("expected only CUDA layers (%d) without IGPU_OFFLOAD, got %d; layout: %v", cudaFit, got, gpuLayers)
	}
	for _, gl := range gpuLayers {
		if gl.DeviceID.Library == "vulkan" {
			t.Errorf("expected no Vulkan layers without IGPU_OFFLOAD, but found: %v", gl)
		}
	}
}
```

- [ ] **Step 2: Run the new tests to verify they fail (before Phase 2 code is added)**

If you have not yet applied the Phase 2 changes from Task 5, run:

```bash
go test ./llm/... -run TestAssignOverflowToIGPU -v
```

Expected: `TestAssignOverflowToIGPU` FAIL (total layers assigned = 3, not 6). `TestAssignOverflowToIGPUDisabled` PASS.

- [ ] **Step 3: Run tests after Phase 2 code is applied**

After completing Task 5:

```bash
go test ./llm/... -run TestAssignOverflowToIGPU -v
```

Expected: both tests PASS.

- [ ] **Step 4: Run the full llm test suite**

```bash
go test ./llm/...
```

Expected: all tests PASS (including existing `TestLLMServerFitGPU` table cases).

- [ ] **Step 5: Commit Phase 2**

```bash
git add llm/server.go llm/server_test.go
git commit -m "llm: add iGPU cross-library overflow layer assignment for OLLAMA_IGPU_OFFLOAD (Phase 2)"
```

---

## Task 7: Update §6.7 in cross-library GPU mixing docs

**Files:**
- Modify: `docs/internals/05-cross-library-gpu-mixing.md`

- [ ] **Step 1: Find §6.7**

Read `docs/internals/05-cross-library-gpu-mixing.md` and locate the section describing UMA zero-copy (§6.7 or similar heading containing "UMA" or "零拷贝").

- [ ] **Step 2: Replace the inaccurate claim**

Replace any statement claiming "UMA zero-copy has solved the data sharing problem" with the corrected description:

```markdown
### 6.7 UMA 数据共享机制（已校正）

CPU 上的权重通过 `ggml_backend_cpu_buffer_type()` → `malloc()` 分配，**不在**
Vulkan `pinned_memory` 表中。`ggml_backend_vk_host_buffer_type()` 当前硬编码为
`vk_instance.devices[0]`（上游已知问题，有注释 `"Should be changed to return
device-specific host buffer type"`），在多 GPU 场景下不提供 per-device 零拷贝。

实际执行路径：当 op_offload 路由 matmul 至 iGPU 时，`compute_splits` 通过
`ggml_vk_buffer_write_2d()` 将权重从 CPU buffer 复制到 Vulkan device buffer。
在 UMA（`eHostVisible`）系统上，这是一次 DDR5 内部 `memcpy()`，消耗约 3×
带宽（copy 读 + copy 写 + shader 读）。

**性能影响**（prefill B=512，per N×K 元素）：

| 路径 | 时间 (s/N×K) |
|------|-------------|
| iGPU（含 memcpy 开销） | 3×0.5/90GB/s + 512×2/4T ≈ 2.73e-10 |
| CPU（直接计算）         | 1×0.5/90GB/s + 512×2/1.9T ≈ 5.45e-10 |

iGPU 仍约 2× 快；memcpy 开销约占收益的 6.5%，可忽略。
```

- [ ] **Step 3: Verify doc looks correct**

Read back the modified section to confirm no broken formatting.

- [ ] **Step 4: Commit**

```bash
git add docs/internals/05-cross-library-gpu-mixing.md
git commit -m "docs: correct §6.7 UMA zero-copy claim in cross-library GPU mixing doc"
```

---

## Task 8: Phase 2 bench-sweep validation

- [ ] **Step 1: Restart ollama with Phase 2 code and IGPU_OFFLOAD=1**

```bash
OLLAMA_IGPU_OFFLOAD=1 go run . serve
```

Load the model: `ollama run qwen3-coder-next:latest`.

Check that `ollama ps` shows the iGPU as a co-processor (the layer split should now show layers on both CUDA and Vulkan backends).

- [ ] **Step 2: Run Phase 2 bench**

```bash
./ollama-bench-sweep.exe run \
  -model qwen3-coder-next \
  -name igpu-phase2 \
  -sizes 512,1024,2048,4096 \
  -epochs 8 \
  -warmup 4
```

- [ ] **Step 3: Diff against baseline and Phase 1**

```bash
./ollama-bench-sweep.exe diff baseline igpu-phase2
./ollama-bench-sweep.exe diff igpu-phase1 igpu-phase2
```

**Pass criteria:**

| Metric | Criterion |
|--------|-----------|
| `prefill_tps` vs baseline | ≥ 20% improvement |
| `prefill_tps` vs phase1 | ≥ 0% (no regression) |
| `gen_tps` vs baseline | ≤ 10% regression |

If `gen_tps` regression exceeds 10%, investigate Vulkan dispatch overhead for decode layers (see design doc §5.3). Consider limiting iGPU layer count via a separate env var.

- [ ] **Step 4: Record results in TODO.md**

Add a summary to `docs/TODO.md` with the bench-sweep diff output and pass/fail verdict.

---

## Self-Review Checklist

- [x] All spec requirements covered: Phase 1 (Task 2), env var (Task 1), bench gate (Task 4), Phase 2 (Task 5+6), doc correction (Task 7), Phase 2 validation (Task 8)
- [x] No placeholders — all code blocks are complete and directly usable
- [x] Type consistency: `ml.GPULayersList`, `ml.DeviceInfo`, `ml.SystemInfo` used consistently across Tasks 5 and 6; `assignLayers` signature matches `llm/server.go` actual signature
- [x] Test imports: `llm/server_test.go` already imports `api`, `format`, `ml` — no new imports needed for the new test functions
- [x] `format.GibiByte` used for systemInfo.FreeMemory (1 GB > 3×100 MB + 457 MB minimum = 757 MB ✓)
- [x] Phase ordering: Task 4 bench gate is explicit, Phase 2 (Task 5) is conditionally gated
