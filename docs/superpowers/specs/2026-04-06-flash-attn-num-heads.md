# FLASH_ATTN num_heads Parameterization

## Problem

FLASH_ATTN_EXT estimation has a systematic ~3x underestimate because the benchmark uses fixed `num_heads=32` while real models use different head counts (e.g., qwen3:1.7b has `num_heads=16, num_kv_heads=4`). The `InterpolateFlashAttn` function only interpolates over `(seqQ, seqKV)` and completely ignores `num_heads`, making it unable to account for the performance differences caused by different head configurations.

### Root Cause Analysis (from Session 32 investigation)

1. **num_heads not parameterized**: Benchmark hardcodes 32 heads. Model actual: 16 Q heads, 4 KV heads (GQA 4:1). Different num_heads = different Vulkan kernel dispatch, parallelism, memory access patterns.
2. **Benchmark vs actual**: At similar seqlen, benchmark underestimates ~3x for qwen3:1.7b due to head count mismatch.
3. **Profile data**: FLASH_ATTN_EXT currently stores one curve with `FixedDims: {num_heads: 32, head_dim: 128}`.

## Solution

Add `num_heads` as a benchmark grid dimension for FLASH_ATTN_EXT, following the established MUL_MAT multi-curve IDW pattern.

### Benchmark Changes

- Measure FLASH_ATTN at `num_heads in {4, 8, 16, 32}` (covers GQA KV heads through large models)
- Each num_heads value produces one `OperatorCurve` with `FixedDims: {num_heads: X, head_dim: 128}`
- Each curve contains the existing decode + prefill sweep points
- `benchmarkFlashAttn` already has `fixedDims` param — pass `num_heads` through `gridPoint[2]` to `CreateInputs`
- `CreateInputs` in registry: read `gridPoint[2]` as `num_heads` instead of hardcoding 32
- Total measurement increase: ~4x (4 num_heads values x existing ~16 points = ~64 measurements)

### How gridPoint carries num_heads

Current `benchmarkFlashAttn` calls `measureOpForBackend` with `gridPoint = [seqQ, seqKV]`.
Change to `gridPoint = [seqQ, seqKV, numHeads]`.
Registry `CreateInputs` reads `gridPoint[2]` instead of hardcoded 32.
No signature changes needed — `gridPoint []int64` already supports variable length.
The stored `LatencyPoint.Shape` remains `[seqQ, seqKV]` (num_heads is in `FixedDims`, not shape).

### Interpolation Changes

New function `InterpolateFlashAttnMultiHead(curves []OperatorCurve, seqQ, seqKV, numHeads int64) float64`:
1. For each curve, extract `num_heads` from `curve.FixedDims["num_heads"]`
2. Compute log-distance: `|log(queryHeads) - log(curveHeads)|`
3. Call existing `InterpolateFlashAttn(curve, seqQ, seqKV)` on the two nearest curves
4. IDW blend the two results using inverse log-distance weights
5. If only one curve exists (old profile), use it directly (backward compatible)
6. For exact match, return that curve's result directly
7. For extrapolation (query outside grid), use power-law scaling from nearest curve

### Estimation Changes

- `nodeToQueryShape` FLASH_ATTN: extract `num_heads` from Q tensor `InputShapes[0][2]`, return shape `[seqQ, seqKV, numHeads]`
- `lookupLatency` FLASH_ATTN case: shape now has 3 elements `[seqQ, seqKV, numHeads]`. Collect all FLASH_ATTN curves matching op+backend, call `InterpolateFlashAttnMultiHead`
- `lookupLatencyV3`: no change needed — its default case already delegates FLASH_ATTN to `lookupLatency`

### GQA Handling

- Benchmark uses non-GQA (Q/K/V same heads) for simplicity
- Estimation uses Q's `num_heads` (not `num_kv_heads`) for interpolation lookup
- GQA data reuse effects are a second-order correction — defer to future work

### Profile Backward Compatibility

- Old profiles with a single FLASH_ATTN curve (num_heads=32) continue to work
- `InterpolateFlashAttnMultiHead` with one curve = `InterpolateFlashAttn` with that curve
- New profiles store 4 curves (one per num_heads value), each with `FixedDims: {num_heads: X, head_dim: 128}`

## Files Changed

| File | Change |
|------|--------|
| `perf/registry.go` | `FLASH_ATTN_EXT.CreateInputs`: use `gridPoint[2]` as num_heads (was hardcoded 32) |
| `perf/bench.go` | `buildSamplingGrids` FLASH_ATTN: return 4 grids (one per num_heads) |
| `perf/bench.go` | `benchmarkFlashAttn`: pass `fixedDims["num_heads"]` via gridPoint[2] |
| `perf/bench.go` | New `Phase1FlashAttnHeads() []int64` returning `{4, 8, 16, 32}` |
| `perf/interpolate.go` | New `InterpolateFlashAttnMultiHead(curves, seqQ, seqKV, numHeads)` |
| `perf/estimate.go` | `nodeToQueryShape` FLASH_ATTN: shape becomes `[seqQ, seqKV, numHeads]` |
| `perf/estimate.go` | `lookupLatency` FLASH_ATTN: collect curves, call multi-head interpolation |

## Non-Goals

- GQA-specific benchmark (different num_heads for Q vs K/V)
- head_dim parameterization (128 is universal in modern models)
- Sliding window seqKV correction (separate, lower-priority issue)
