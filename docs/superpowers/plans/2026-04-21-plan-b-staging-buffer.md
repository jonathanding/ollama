# Plan B Staging Buffer (Option D) Implementation Plan

**Date:** 2026-04-21
**Branch:** `debug/moe-prefetch-plan-b`
**Parent commit:** `965da8e0` (Option A correctness baseline — prefetch on main stream)

**Goal:** Recover copy/compute overlap for Plan B prefetch while keeping the correctness guarantee established by Option A.

**Architecture:** Allocate two dedicated VRAM staging buffers (double-buffered) that are disjoint from any ggml-alloc `input_cpy` region. The independent prefetch stream writes pinned CPU weights to `staging[slot]`; the main compute stream later `cudaStreamWaitEvent`s the per-slot H2D event and D2D-copies `staging[slot] -> input_cpy`. No cross-stream aliasing, no race.

**Tech Stack:** CUDA 13.2, ggml-cuda backend, ggml-backend scheduler.

---

## Why Option D (vs other options)

| Option | Correctness | Overlap | VRAM cost | Complexity |
|---|---|---|---|---|
| A: prefetch on main stream | ✅ | ❌ none | 0 | trivial |
| B: independent stream + event-wait before H2D | ✅ | ❌ none (main stream idle) | 0 | low |
| D: **staging buffer + D2D** | **✅** | **✅ full** | **~840 MiB** | **medium** |
| E: break input_cpy aliasing | ✅ | ✅ | 60+ GiB per run | infeasible |

Only D reclaims the overlap without unacceptable VRAM cost.

---

## Timing Model (per MoE layer, 3 splits A/B/C)

```
main stream:     [compute A]                    [D2D s[1]->cpy_B][compute B]           [D2D s[0]->cpy_C][compute C]
prefetch stream: [H2D pinned_B -> s[1]]─E[1]    [H2D pinned_C -> s[0]]─E[0]
                  └──overlaps compute A────┘     └──overlaps compute B─┘
```

- `s[slot]` = staging VRAM (never aliased by ggml-alloc)
- `E[slot]` = `cudaEvent_t` recorded on prefetch stream after H2D completes
- D2D (~900 GB/s) adds ~0.5 ms per split; negligible vs the H2D time it hides

---

## Design Decisions (confirmed with user)

1. **staging location:** CUDA backend context (`ggml_backend_cuda_context::moe_staging`), exposed via proc address (consistent with existing Plan B stream/event helpers)
2. **size:** one-shot allocation = `max(ggml_nbytes(input))` over all CPU-MoE splits, discovered during `compute_splits` init; grows but never shrinks
3. **buffering:** **double buffer** (slot = counter & 1), avoids H2D-write/D2D-read race on the same staging buffer
4. **D2D location:** inside the `input_copy` block where `moe_prefetch_hit` is detected (D1 — replaces the bare `continue`)
5. **scope:** this plan covers **full-copy + staging** (E1). Selective + staging is a follow-up after validation
6. **cleanup:** proactive in `ggml_backend_sched_free` + safety net in `ggml_backend_cuda_context::~ggml_backend_cuda_context`

---

## File Structure

**Modify:**
- `ml/backend/ggml/ggml/src/ggml-cuda/common.cuh` — add `moe_staging` struct to `ggml_backend_cuda_context`
- `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu` — implement 4 staging helpers + register proc addresses + safety-net cleanup in dtor
- `ml/backend/ggml/ggml/src/ggml-backend.cpp` — add typedefs, rewrite prefetch init block (compute max_size + call staging_init), rewrite fire/hit blocks (H2D→staging / D2D→input_cpy with double buffering), cleanup staging in sched_free

**No new files. No tests in this branch (manual verification via user's `run_single_1k_test.py` + bench-sweep).**

---

## Task Decomposition

### D-Task 1: Add `moe_staging` struct to `ggml_backend_cuda_context` ✅ DONE

**File:** `ml/backend/ggml/ggml/src/ggml-cuda/common.cuh`

Add nested struct holding two `void*` device buffers, two `cudaEvent_t`, and a capacity counter. Zero-initialized so `destroy` is safe pre-init.

### D-Task 2: Implement 4 CUDA staging helpers + register proc addresses ✅ DONE

**File:** `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu`

- `ggml_backend_cuda_moe_staging_init(backend, max_size)` — `cudaMalloc` × 2 + `cudaEventCreateWithFlags(cudaEventDisableTiming)` × 2. Idempotent: short-circuit if already sized adequately.
- `ggml_backend_cuda_moe_staging_destroy(backend)` — `cudaFree` × 2 + `cudaEventDestroy` × 2.
- `ggml_backend_cuda_moe_staging_h2d(backend, prefetch_stream, slot, input)` — `cudaMemcpyAsync(H2D, prefetch_stream)` + `cudaEventRecord(h2d_done[slot], prefetch_stream)`.
- `ggml_backend_cuda_moe_staging_d2d(backend, slot, input_cpy)` — `cudaStreamWaitEvent(main_stream, h2d_done[slot])` + `cudaMemcpyAsync(D2D, main_stream)`.

Register all 4 in `ggml_backend_cuda_reg_get_proc_address`. Also add safety-net cleanup in `~ggml_backend_cuda_context`.

### D-Task 3: Sched-side init — compute max_size + call `staging_init` ✅ DONE

**File:** `ml/backend/ggml/ggml/src/ggml-backend.cpp` — prefetch init block in `ggml_backend_sched_compute_splits`

- Add typedefs for the 4 staging helper signatures
- Resolve all 4 proc addresses via `moe_get_proc`
- Walk `sched->splits` to find max `ggml_nbytes(input)` for CPU-MoE weight inputs; remember the CUDA backend handle
- Call `fn_staging_init(backend, max_size)` once
- On success, set `prefetch_enabled = true` and log `"enabled (staging 2xN MiB)"`
- On any failure, fall back to non-prefetch (Plan A) path with a warning

Note: `fn_staging_init` is idempotent, so being called again on subsequent `compute_splits` invocations is cheap.

### D-Task 4: Rewrite fire/hit blocks for staging path (IN PROGRESS)

**File:** `ml/backend/ggml/ggml/src/ggml-backend.cpp`

**fire block** (after compute N submitted):
```
slot = prefetch_counter & 1
fn_staging_h2d(moe_cuda_backend, prefetch_stream, slot, inp)
prefetch_pending  = true
prefetch_split_id = next_id
prefetch_slot     = slot
prefetch_counter++
```

**hit detection** (top of split loop):
```
bool moe_prefetch_hit = false
int  moe_prefetch_slot = -1
if (pending && split_id == prefetch_split_id) {
    moe_prefetch_hit = true
    moe_prefetch_slot = prefetch_slot
    prefetch_pending = false
}
```

**hit consumption** (inside `input_copy` block, before the existing `continue`):
```
if (moe_prefetch_hit && fn_staging_d2d) {
    fn_staging_d2d(split_backend, moe_prefetch_slot, input_cpy)
    continue    // skip the regular expert copy
}
```

### D-Task 5: sched_free cleanup

**File:** `ml/backend/ggml/ggml/src/ggml-backend.cpp` — `ggml_backend_sched_free`

Call `fn_staging_destroy(moe_cuda_backend)` before freeing sched's internal buffers. Requires stashing the backend handle somewhere on `sched` — either:
- (a) walk sched->backends to find the first GPU backend (same logic as `moe_get_proc`), OR
- (b) store the backend pointer on sched when staging_init succeeds.

Prefer (a) for simplicity — no new sched fields.

### D-Task 6: Build + correctness test

- Run `scripts/build_cuda_incremental.ps1`
- Manually copy `build/cuda_v13/lib/ollama/ggml-base.dll` into `dist/windows-amd64/lib/ollama/`
- User runs `run_single_1k_test.py` with `OLLAMA_MOE_PREFETCH=1 OLLAMA_MOE_PINNED=1`
- Expect: Fibonacci output matches Plan A (no `!!!!!`)

### D-Task 7: Performance benchmark

- User runs `bench-sweep.exe` 1024 tokens, 6 epochs
- Expected prefill_mean: somewhere in [Plan A 1380 ms, Option A 2004 ms]
- If overlap works, prefill should be **< Option A 2004 ms**; magnitude depends on how much H2D fits inside the previous compute window.
- Not expected to beat Plan A 1380 ms in this full-copy variant — that requires selective + staging (Variant A upgrade, separate branch).

### D-Task 8: Decide on selective upgrade

- If D-Task 7 shows clear overlap gain vs Option A → cherry-pick selective logic from Variant A into this branch (or go back to `feat/moe-split-cpu` and port staging there)
- If no gain → diagnose why (stream priorities, PCIe queue depth, etc.) before adding selective complexity

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `cudaMalloc` 840 MiB fails on low-VRAM systems | fallback to non-prefetch path with warning; user can `unset OLLAMA_MOE_PREFETCH` |
| max_size changes across `compute_splits` calls (different graphs) | staging_init grows but never shrinks; short-circuit when capacity sufficient |
| Double-buffer slot collision if >1 prefetch pending | `prefetch_pending` flag enforces "at most 1 pending"; slot rotation safe |
| Proc address lookup returns NULL (old DLL) | init block falls back to non-prefetch path |
| staging leak on sched destruction | proactive destroy in sched_free + safety net in ~cuda_context |

---

## Performance Expectation (rough)

Full-copy H2D per layer: 288 + 288 + 420 = 996 MiB
Compute per split: ~3.5 ms (from prior measurements, Layer 2-31)
H2D per split at ~25 GB/s: ~12-17 ms

Overlap can hide at most ~3.5 ms per compute-behind-H2D pair. Per layer: ~7 ms hidden × 30 layers ≈ **~200 ms saved** vs Option A 2004 ms → projected prefill **~1800 ms**.

Still worse than Plan A 1380 ms because full-copy bulk (~30 GiB total H2D) dominates. Selective + staging is required to undercut Plan A. This plan intentionally validates the overlap mechanism first before combining with selective.

---

## Success Criteria

- **Correctness:** `run_single_1k_test.py` output matches Plan A byte-for-byte
- **Overlap activated:** prefill_mean < Option A 2004 ms (any improvement)
- **No VRAM leak:** repeated model load/unload doesn't grow `vram_used_bytes`
- **Fallback clean:** if staging_init fails, falls back to non-prefetch path without crash

---

## Out of Scope

- Selective copy (Variant A merge)
- Layer-level prefetch (prefetch L+1's weights during L's attention)
- Persistent staging across sched instances
- Multi-GPU staging
