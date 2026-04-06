# DAOP v2: Empirical Performance Estimation — Engineering Spec

> High-level design rationale, design evolution (v1 eta → full measurement → hybrid approach),
> log-space theory, and architectural decisions are documented in
> [`docs/daop/design.md`](../../daop/design.md). This spec focuses on engineering details for implementation.

## 1. Scope & Phases

### Phase 1: Three Representative Operators + Validation

Build the complete pipeline end-to-end with three operators that cover 1D, 2D, and 3D performance characteristics:

| Operator | Perf Dimensions | Why |
|----------|----------------|-----|
| SILU | 1D: `f(N)` | Simplest element-wise op; validates full pipeline |
| MUL_MAT | 3D: `f(M, K, N)` | Most important op; validates multi-dim interpolation + quantized dtypes |
| FLASH_ATTN_EXT | 2D: `f(seq_q, seq_kv)` | Attention core; validates 2D interpolation + special interface |

Phase 1 deliverables:
1. Operator registry + benchmark runner for these 3 ops
2. Hardware characterization (peak TOPS, peak BW)
3. Adaptive sampling with log-space interpolation
4. Profile storage (JSON)
5. `buildModelGraphNodes()` implementation
6. Estimation pipeline (graph → latency lookup → sum)
7. HTML viewer for benchmark data visualization
8. Comprehensive tests (TDD, requires GGML build)

### Phase 1B: Extended Operator Coverage + Random Initialization

Extend the benchmark and estimation pipeline to cover all common LLM operators, eliminating "uncalibrated" warnings for standard architectures (Llama, Gemma, Qwen, Mistral, etc.).

**Changes from Phase 1A:**

1. **Random tensor initialization**: Replace `ctx.Input().Zeros()` with random data in `[-1, 1]` for more realistic benchmarking. GPU kernel timing is generally data-independent, but random init avoids edge cases with special values (all-zeros quantization patterns, NaN propagation, etc.).

2. **Extended `OpRunnerML`**: Add optional `CreateInputs` field for ops that need non-standard tensor creation (mixed dtypes, special shapes). Remove `NumInputs` — inferred from `CreateInputs` return or `expandShapes`.

   ```go
   type OpRunnerML struct {
       Dimensions   []string
       CreateInputs func(ctx ml.Context, dt ml.DType, gridPoint []int64) []ml.Tensor  // optional
       Run          func(ctx ml.Context, inputs []ml.Tensor) ml.Tensor
   }
   ```

3. **MUL_MAT tensor creation unified**: `measureMulMat` is removed; MUL_MAT uses `measureOp` with a custom `CreateInputs` that creates weight at quantized dtype and activation at f32. Post-measurement processing (roofline efficiency extraction) remains MUL_MAT-specific in `RunBenchmark`.

4. **New operators** (all 1D `f(N)`, benchmarked at f32 only):

   | GGML Op Name | Category | Inputs | Registry `Run` |
   |---|---|---|---|
   | `ADD` | 2-input element-wise | 2 same-shape | `in[0].Add(ctx, in[1])` |
   | `MUL` | 2-input element-wise | 2 same-shape | `in[0].Mul(ctx, in[1])` |
   | `RMS_NORM` | 1-input norm | 1 | `in[0].RMSNorm(ctx, nil, 1e-5)` (nil weight → only ggml_rms_norm node) |
   | `SOFT_MAX` | 1-input | 1 | `in[0].Softmax(ctx)` |
   | `GELU` | 1-input activation | 1 | `in[0].GELU(ctx)` |
   | `RELU` | 1-input activation | 1 | `in[0].RELU(ctx)` |
   | `CONT` | 1-input memory copy | 1 | `in[0].Contiguous(ctx)` |
   | `NORM` | 1-input norm | 1 | `in[0].LayerNorm(ctx, nil, nil, 1e-5)` |
   | `ROPE` | special (type assert) | custom | Needs positions(i32) + 4D input; uses type assertion for `RoPE` method |
   | `GET_ROWS` | special (index tensor) | custom | Needs i32 index tensor; embedding lookup pattern |

5. **`expandShapes` updated**: 2-input ops (ADD, MUL) return `[][]int64{gridPoint, gridPoint}`.

6. **`IsZeroCostOp` updated**: Add `TRANSPOSE` (metadata-only, like VIEW/RESHAPE/PERMUTE).

7. **Default ops list**: `cmd.go` expands to include all registered ops.

**Benchmark time impact**: ~10 new 1D ops × ~10-15s each = **~2 min additional**. Total from ~6 min → ~8 min.

**ROPE handling**: `RoPE` is not on the `ml.Tensor` interface — it's a backend-specific method accessed via type assertion. The registry uses `CreateInputs` for the 4D input + i32 positions tensor, and `Run` does a type assertion to call `RoPE`.

**GET_ROWS handling**: Removed from benchmark. GET_ROWS is a pure memory copy (embedding lookup) called once per forward pass, with negligible latency (~10μs). Benchmarking it requires creating output tensors of N × hidden_dim which becomes impractical at large N (67M × 4096 = 1TB). Instead, `lookupLatency` returns a fixed 10μs constant to avoid "uncalibrated" warnings.

**NORM (LayerNorm) excluded**: Investigation of 21+ model architectures found 0 model files call `.LayerNorm()` directly. Removed from the operator list.

### Phase 1B Addendum: Sampling Range Analysis

#### Model Architecture Reference

| Model | hidden | intermediate | heads | kv_heads | head_dim | context |
|-------|--------|-------------|-------|----------|----------|---------|
| Llama-3 8B | 4096 | 14336 | 32 | 8 | 128 | 8K |
| Llama-3 70B | 8192 | 28672 | 64 | 8 | 128 | 8K |
| Qwen2.5-32B | 5120 | 27648 | 40 | 8 | 128 | 128K |
| Qwen3-30B-A3B (MoE) | 2048 | 6144* | 32 | 4 | 64 | 128K |

\* MoE: 128 experts, 8 active per token, per-expert intermediate=768.

#### Maximum Element Counts (N) per Operator

The largest tensors occur during **prefill** (batch=512). The table below shows max N for each operator across model sizes.

| Operator | Max tensor source | 8B | 32B | 70B |
|----------|------------------|-----|-----|-----|
| SILU/GELU/RELU | intermediate × batch | 7.3M | 14.2M | 14.7M |
| ADD/MUL | intermediate × batch | 7.3M | 14.2M | 14.7M |
| RMS_NORM | hidden × batch | 2.1M | 2.6M | 4.2M |
| SOFT_MAX | heads × context_len | 262K | 5.1M | 524K |
| ROPE | head_dim × heads × batch | 2.1M | 2.6M | 4.2M |
| CONT | hidden × batch | 2.1M | 2.6M | 4.2M |
| MUL_MAT (N axis) | batch/seq | ≤512 | ≤512 | ≤512 |
| FLASH_ATTN seq_kv | context length | 8K | 128K | 8K |
| FLASH_ATTN seq_q | prefill batch | ≤2048 | ≤2048 | ≤2048 |

#### Chosen Sampling Ranges

| Operator | shapeMin | shapeMax | Rationale |
|----------|----------|----------|-----------|
| 1D elementwise (all) | 1,024 | **8M** | Covers 8B in-range. 32B/70B (~15M) covered by log-log extrapolation — curves are smooth (measured max_error <5%), extrapolation ≈1 octave in log space |
| MUL_MAT (N axis) | 1 | **4,096** | N = batch/sequence dim. Prefill typically ≤512, 4096 provides ample margin |
| FLASH_ATTN decode (seq_kv) | 64 | **16,384** | 128K context covered by extrapolation. Decode latency is O(seq_kv), curve is smooth (max_error 3.1%, zero refinement needed) |
| FLASH_ATTN prefill (seq_len) | 64 | **2,048** | Prefill is O(seq²), measuring beyond 2048 is prohibitively slow (~2s/point on iGPU). Larger values extrapolated |
| GET_ROWS | — | — | Not benchmarked. Fixed 10μs constant (negligible: 1 call/forward, pure memory copy) |

**Key design principle**: The adaptive sampler + log-log interpolation framework means shapeMax does not need to cover the actual maximum tensor size. It only needs to capture enough of the curve shape for reliable extrapolation. All operators showed smooth power-law-like behavior (max_error <5% at convergence), validating this approach.

#### Outlier Handling

Each sampling point goes through a two-level measurement pipeline:

1. **Per-point**: `convergentMeasure()` collects multiple reps, trims outliers (top/bottom `TrimPercent`), checks coefficient-of-variation (CV) convergence, and returns the trimmed median. This eliminates GPU scheduling jitter and cold-start effects.

2. **Adaptive level**: `AdaptiveSample1D` uses existing measured points to find intervals with poor interpolation fit (`worstInterval`), measures one midpoint per round, and stops when interpolation error < `ErrorThreshold` (default 5%). This ensures the curve is well-represented without over-sampling smooth regions.

### Phase 2: MUL_MAT Accuracy Redesign

**Problem**: Phase 1's roofline model has 3.6x prediction error for small models (qwen3:1.7b decode: 272ms predicted vs 75ms actual). Root causes:

1. **Single calibration point**: Only (M,K)=(4096,4096) was benchmarked. Extrapolating to qwen3:1.7b's smaller shapes (e.g., 2048×2048) fails because the relationship between (M,K) size and latency is non-linear at small N.
2. **Roofline structurally wrong for VEC**: At N=1 with q4_0, effective BW=4.68 GB/s (9.6% of peak), effective compute=16.6 GFLOPS (1.06% of peak). The op is "latency-bound" — dominated by dequantization pipeline, memory latency, and warp scheduling. The roofline `max(compute, bw)` model cannot represent this regime.
3. **Roofline self-consistent at calibration point**: At (4096,4096,N=1,q4_0), roofline predicts 2023μs vs measured 2021μs (<0.1% error). The error is purely from extrapolation, not from the model at its calibration shape.

**Design**: Replace roofline as the primary MUL_MAT prediction engine with direct interpolation from multi-(M,K) reference curves. Roofline is retained as a validator and extrapolation fallback.

#### 2.1 Architecture Overview

| Role | Method | When Used |
|------|--------|-----------|
| **Primary prediction** | IDW interpolation from reference curves | Query (M,K) within range of measured reference points |
| **Extrapolation fallback** | Nearest reference × scaling factor `(M_q×K_q)/(M_ref×K_ref)` | Query (M,K) far from all reference points |
| **Anomaly monitoring** | Roofline cross-check + multi-dimensional consistency checks | During benchmark (online) and after benchmark (report) |
| **Confidence scoring** | IDW distance from query to nearest reference | During estimation, warn if low confidence |

#### 2.2 (M,K) Grid + Strategic N Sampling

##### (M,K) Grid: Systematic Log-Spaced Coverage

Instead of hardcoding specific model shapes, use a 3×3 log-spaced grid in (M,K) space:

**Grid values**: {512, 2048, 8192} — log-spaced (factor of 4 between steps)

```
K\M      512        2048       8192
512    (512,512)  (2048,512)  (8192,512)
2048   (512,2048) (2048,2048) (8192,2048)
8192   (512,8192) (2048,8192) (8192,8192)
```

**9 (M,K) pairs** — provides uniform coverage of the (log M, log K) plane. Any query shape within [512, 8192]² falls inside the convex hull, enabling accurate IDW interpolation.

Coverage of qwen3:1.7b actual shapes (hidden=2048, intermediate=8960, GQA kv_heads=4):
- (512, 2048) — exact hit (K/V projection)
- (2048, 2048) — exact hit (Q/O projection)
- (8960, 2048) — near (8192, 2048), IDW interpolates
- (2048, 8960) — near (2048, 8192), IDW interpolates

**Design principle**: The grid is model-agnostic. It covers any transformer architecture with dimensions in [512, 8192]. Larger shapes (>8192) require extrapolation via the scaling fallback (Section 2.4).

##### N Sampling: Roofline-Guided Strategic Points

Instead of full adaptive curves over N, measure at **3 strategic N values** per (M,K,dtype):

| N | Purpose |
|---|---------|
| 1 | Decode (BW-bound / VEC regime) |
| N_cross | Roofline crossover point — BW↔compute transition |
| 512 | Prefill (compute-bound) |

**N_cross** is computed from hardware peaks — where compute time equals BW time:

```
N_cross = peak_tops[dtype] × elemBytes(dtype) / (2 × peak_bw)
```

This depends only on dtype and hardware, not on (M,K):

| dtype | N_cross (Intel iGPU) |
|-------|---------------------|
| q4_0 | ~9 |
| q8_0 | ~6 |
| f16 | ~12 |
| f32 | ~19 |

**Rationale**: Roofline provides the *structure* (where to measure), measurements provide the *accuracy*. Three points define two segments in log-log space:
- N=1 to N_cross: BW-bound segment (relatively flat — weight bytes dominate)
- N_cross to N=512: compute-bound segment (linear in log-log — latency ∝ N)

Piecewise linear interpolation in log-log space between these 3 points covers any query N.

**Benchmark step count**: 9 (M,K) × 4 dtypes × 3 N = **108 measurements**

**Benchmark time**: ~3–4 minutes on Intel iGPU (each measurement ~2s with convergence early stop). This is ~4x faster than full adaptive curves.

#### 2.3 Prediction: Two-Stage Interpolation

For any query (M_q, K_q, N_q, dtype):

1. **Filter** reference points by matching dtype
2. **Stage 1 — N interpolation**: For each (M,K) grid point, interpolate among its 3 measured N values (N=1, N_cross, N=512) in log-log space to get latency at N_q
3. **Stage 2 — (M,K) interpolation**: IDW blend in log (M,K) space across grid points to get final latency

This replaces the Phase 1 split between VEC (N≤8) and MAT (N>8) paths. The two-stage interpolation works uniformly for all N.

```go
// lookupLatencyV3 MUL_MAT case (simplified)
case "MUL_MAT", "MUL_MAT_ADD":
    lat := PredictMulMatDirect(profile, M, K, N, mappedWdt)
    if lat > 0 {
        return lat, nil
    }
    // Fallback: roofline (for backward compatibility with v2 profiles)
    return PredictMulMatLatency(&profile.Hardware, M, K, N, mappedWdt), nil
```

Note: `InterpolateMulMat()` already implements this two-stage pattern (1D interpolation over N per curve, then IDW blend). Each "curve" simply has 3 points instead of 8–10 from adaptive sampling.

#### 2.4 Extrapolation: Scaling Fallback

When the query (M,K) is far from all reference points (IDW distance > threshold), direct interpolation degrades to nearest-neighbor, losing accuracy. In this case, apply physics-informed scaling:

```
lat_predicted = lat_nearest × scale_factor(M_q, K_q, N_q, M_ref, K_ref)
```

Where:
- **BW-bound (small N)**: `scale_factor ≈ (M_q × K_q) / (M_ref × K_ref)` — latency proportional to weight size
- **Compute-bound (large N)**: `scale_factor ≈ (M_q × K_q × N_q) / (M_ref × K_ref × N_ref)` — latency proportional to FLOPs
- **Transition**: blend between the two using the roofline balance point

This is essentially roofline with the nearest reference point as the "anchor" instead of global efficiency constants. It preserves the physics structure without requiring accurate global calibration.

#### 2.5 Anomaly Monitoring

##### 2.5.1 Online Checks (During Benchmark)

Run after each reference curve is measured. Immediate warnings in benchmark log.

**Single-curve checks:**

| Check | Method | Trigger |
|-------|--------|---------|
| **Monotonicity** | Within a single (M,K) curve, latency must increase with N | Adjacent points where `lat[i+1] < lat[i] × 0.9` (>10% decrease) |
| **Physical lower bound** | No measurement should be faster than theoretical minimum | `measured < min(weight_bytes/peak_bw, 2MKN/peak_tops)` |
| **Measurement stability** | Per-point CV from convergent measurement | `CV > 0.20` (20%) after convergence attempts |
| **Adaptive convergence** | Adaptive sampling should converge | `max_error > ErrorThreshold` after all refinement rounds exhausted |

**Cross-curve checks (run after all curves of same dtype):**

| Check | Method | Trigger |
|-------|--------|---------|
| **Scaling consistency** | At large N (compute-bound), latency should scale ∝ M×K across shapes | `lat(M2,K2)/lat(M1,K1)` deviates from `(M2×K2)/(M1×K1)` by >2x |
| **Efficiency consistency** | Extract compute_eff from each curve at large N; should be similar | `max(eff) / min(eff) > 2.0` across same-dtype curves |
| **Roofline cross-check** | Compare measured N=1 latency against roofline prediction using efficiency from large-N points | `measured / roofline_predicted > 5x` (expected ~1-3x due to VEC overhead) |

##### 2.5.2 Offline Report (After Benchmark)

Generated as summary after benchmark completes. Saved to profile or printed to console.

**Cross-dtype checks:**

| Check | Method | Trigger |
|-------|--------|---------|
| **Dtype ordering** | At same (M,K) and large N: q4_0 should be fastest (least data), f32 slowest | Ordering violated |
| **Peak TOPS consistency** | Implied TOPS from large-N measurements should not exceed hardware peak | `implied_tops > 1.1 × peak_tops` |

**Temporal checks (when re-running benchmark):**

| Check | Method | Trigger |
|-------|--------|---------|
| **Reproducibility** | Compare new profile against previous profile | Same (M,K,N,dtype) point changed by >30% |
| **Hardware degradation** | Peak TOPS/BW from hw characterization vs previous | Changed by >20% |

##### 2.5.3 Estimation-Time Checks

Run during `daop-estimate` to flag unreliable predictions.

| Check | Method | Output |
|-------|--------|--------|
| **IDW distance** | Compute min log-distance from query (M,K) to nearest reference curve | Warning if distance > `ln(2)` (query shape >2x away from nearest reference) |
| **N range coverage** | Query N within measured range of reference curves? | Warning if extrapolating beyond measured N range |
| **Confidence score** | `confidence = 1 / (1 + min_distance)` normalized to [0, 1] | Print in estimate output; low confidence (<0.5) gets explicit warning |

#### 2.6 Efficiency Constants: Retained for Validation Only

Phase 1's `extractEfficiencyConstants` and `EfficiencyConstants` in `HardwareProfile` are retained but their role changes:

| Phase 1 | Phase 2 |
|---------|---------|
| Primary prediction input | Validation-only: cross-check measured data |
| Extracted from every (M,K) curve | Extracted from (4096,4096) curve only (canonical reference) |
| Stored in profile, required for estimation | Stored in profile, optional (estimation works without them) |

#### 2.7 Code Changes Summary

| File | Change |
|------|--------|
| `perf/estimate.go` (lookupLatencyV3) | MUL_MAT: call PredictMulMatDirect for all N, not just VEC range. Remove VEC/MAT split. |
| `perf/bench.go` (RunBenchmark) | Extract efficiency only from (4096,4096) curve. Skip for other (M,K). |
| `perf/bench.go` (benchmarkMulMat) | Dynamic N_max cap based on (M,K) size. |
| `perf/registry.go` (Phase1MulMatFixedDims) | Replace 70B shapes with small-model shapes (2048, 5504). |
| `perf/monitor.go` (new) | Anomaly monitoring: online checks, offline report, confidence scoring. |
| `perf/bench.go` (PredictMulMatDirect) | Already implemented in Phase 1; no change needed. |

### Phase 3: Extended Coverage (out of scope)

- Extend to specialized architecture operators (SSM_CONV, SSM_SCAN for Mamba; TRI, SOLVE_TRI for DeltaNet; Conv2D/Conv3D for vision models).
- **Cross-backend transfer cost**: Measure interconnect bandwidth (PCIe, NVLink), add transfer latency when consecutive graph nodes are on different backends. Phase 1 assumes single backend.
- **Incremental calibration (`--update`)**: Load existing profile, diff against model graph to find uncalibrated (op, dtype) combos, benchmark only the missing ones, merge into profile. v1's `RunUpdateBenchmark` pattern is preserved but reimplemented on the new data model.
- **`HardwareProfile.InterconnectBWBytesPerSec`**: Populated in Phase 3 when cross-backend support is added. Phase 1 leaves it as 0.

## 2. Changes from v1

### 2.1 Files to Remove or Gut

These v1 components are replaced by the empirical model:

| File | What to remove | Why |
|------|---------------|-----|
| `perf/ops.go` | `ComputeFLOPs()`, `ComputeBytes()` | Replaced by direct latency measurement |
| `perf/ops.go` | `CanComputeFLOPs()` | No longer needed |
| `perf/roofline.go` | `EstimateOpCost()`, `LookupEta()` | Replaced by curve lookup |
| `perf/bench.go` | `benchSingleOp()`, `SelectBenchmarkShapes()`, `computePointEtas()`, `ShouldAdaptiveExtend()` | Replaced by new benchmark + adaptive sampling |
| `perf/bench.go` | `benchPeakFLOPS()`, `benchPeakBandwidth()` | Kept but moved/refactored — still needed for initial grid |
| `perf/profile.go` | `OperatorProfile.Eta` field | Replaced by latency points |
| `perf/estimate.go` | Roofline-based estimation | Replaced by curve lookup |

### 2.2 Files to Keep (with modifications)

| File | Changes |
|------|---------|
| `perf/types.go` | Rewrite: replace v1 data structures (`OperatorProfile` with `Eta`/`EtaVariance`/`NumPoints`, `BenchmarkPoint` with `FLOPs`/`BytesMoved`/`Intensity`, `OpCost`) with v2 structures (`OperatorCurve` with `FixedDims`/`Points`, `LatencyPoint`, `OpRunner`, `SamplingGrid`, `HWCharResult`). Keep: `OpKey`, `EstimateResult` (extended), `Profile` (restructured). |
| `perf/ops.go` | Keep `IsZeroCostOp()`, `elemSize()`, `product()`. Remove `ComputeFLOPs()`, `ComputeBytes()`, `CanComputeFLOPs()`. |
| `perf/bench.go` | Rewrite: new `benchmarkOp()` using registry, adaptive sampling loop. Remove `benchSingleOp()`, `SelectBenchmarkShapes()`, `computePointEtas()`, `ShouldAdaptiveExtend()`. Keep `benchPeakFLOPS()`, `benchPeakBandwidth()` (move to hwchar.go). |
| `perf/profile.go` | Rewrite: new profile format (latency curves instead of eta). Keep `LoadProfile()`, `WriteProfile()`, `MergeProfile()`. Remove `ComputeEtaFromPoints()`, `ProcessRawToProfile()`. |
| `perf/estimate.go` | Rewrite: log-space interpolation lookup. Keep per-op breakdown, top-ops ranking, bottleneck classification from v1's `ComputePhaseEstimation()`. Keep `RunEstimate()` entry point and `resolve.go` integration. |
| `perf/roofline.go` | Remove entirely. `LookupBackend()` moves to profile.go. |
| `perf/viewer.go` | Keep CLI viewer (`PrintProfile`, `PrintEstimateResult`, `printTopOps`). |
| `perf/resolve.go` | Keep as-is (model path resolution: model name → GGUF path). |
| `perf/cmd.go` | Update CLI commands. |
| `ml/backend.go` | No changes needed (GraphNode, Context already sufficient). |

### 2.3 New Files

| File | Purpose |
|------|---------|
| `perf/registry.go` | Operator registry (OpRunner map + benchmarkOp function) |
| `perf/interpolate.go` | Log-space piecewise linear interpolation (Interpolate1D, Interpolate1DByDim, InterpolateMulMat, InterpolateFlashAttn) |
| `perf/adaptive.go` | Adaptive sampling algorithm |
| `perf/hwchar.go` | Hardware characterization (peak TOPS, BW, balance point) |
| `perf/viewer_html.go` | HTML viewer generation |
| `perf/viewer.html` | HTML template (embedded via `//go:embed`) |

## 3. Data Structures

### 3.1 Profile (v2)

```go
type Profile struct {
    Version   int              `json:"version"`   // 2
    Timestamp time.Time        `json:"timestamp"`
    Hardware  HardwareProfile  `json:"hardware"`
    Operators []OperatorCurve  `json:"operators"`
}

type HardwareProfile struct {
    Backends                []BackendInfo      `json:"backends"`
    PeakTOPS                map[string]float64 `json:"peak_tops"`                  // dtype -> TOPS
    PeakBandwidthBytesPerSec float64           `json:"peak_bandwidth_bytes_sec"`
    InterconnectBWBytesPerSec float64           `json:"interconnect_bandwidth_bytes_sec"`
    BalancePoints           map[string]float64 `json:"balance_points"`             // dtype -> FLOPs/byte
}

type BackendInfo struct {
    Name   string `json:"name"`
    Device string `json:"device"`
    VRAMBytes int64 `json:"vram_bytes"`
}

type OperatorCurve struct {
    Op           string           `json:"op"`
    Backend      string           `json:"backend"`
    ComputeDtype string           `json:"compute_dtype"`
    WeightDtype  string           `json:"weight_dtype,omitempty"`
    Dimensions   []string         `json:"dimensions"`  // sweep dimensions: ["N"] for 1D, ["N"] for MUL_MAT (M,K fixed)
    FixedDims    map[string]int64 `json:"fixed_dims,omitempty"` // e.g., {"M": 4096, "K": 4096} for MUL_MAT
    Points       []LatencyPoint   `json:"points"`
}

// For MUL_MAT, each (M, K, compute_dtype, weight_dtype) combination is a separate
// OperatorCurve with FixedDims={"M": M, "K": K} and Dimensions=["N"].
// This makes each curve a 1D function of N, which simplifies:
//   - Interpolation: 1D lookup per curve, then weight-average across (M,K) pairs
//   - HTML viewer: each curve maps directly to a Plotly trace
//   - Storage: clear what each curve represents
//
// For FLASH_ATTN_EXT, each (num_heads, head_dim) combination is a separate
// OperatorCurve with FixedDims={"num_heads": 32, "head_dim": 128}.
// Dimensions=["seq_q", "seq_kv"] but in practice only two regimes are sampled:
//   - Prefill: seq_q == seq_kv (sweep both together)
//   - Decode: seq_q == 1 (sweep seq_kv)

type LatencyPoint struct {
    Shape     []int64 `json:"shape"`       // values for sweep Dimensions only (not FixedDims)
                                           // e.g., for MUL_MAT with FixedDims={M,K}: Shape=[N]
    LatencyUs float64 `json:"latency_us"`  // median latency in microseconds
    StddevUs  float64 `json:"stddev_us"`   // for confidence reporting
    Reps      int     `json:"reps"`
}
```

### 3.2 Operator Registry

```go
type OpRunner struct {
    // NumInputs is how many input tensors the op requires.
    NumInputs int

    // Dimensions lists ALL performance-relevant shape dimensions for this op.
    // Used to determine the sampling grid structure.
    // Examples: ["N"] for element-wise, ["M", "K", "N"] for MUL_MAT
    //
    // Note: OpRunner.Dimensions is the FULL set of relevant dims.
    // When creating OperatorCurves, some dims become FixedDims and the rest
    // become OperatorCurve.Dimensions (sweep dims).
    // e.g., MUL_MAT OpRunner.Dimensions=["M","K","N"] →
    //   OperatorCurve{FixedDims={"M":4096,"K":4096}, Dimensions=["N"]}
    Dimensions []string

    // Run invokes the operator on the given inputs and returns the output tensor.
    Run func(ctx ml.Context, inputs []ml.Tensor) ml.Tensor
}
```

### 3.3 Sampling Grid

```go
// SamplingGrid defines the points to benchmark for one operator.
type SamplingGrid struct {
    Op         string
    Dtype      string
    WeightDtype string          // for MUL_MAT with quantized weights
    Dimensions []string         // dimension names
    Points     [][]int64        // each entry is one shape to benchmark
}
```

## 4. Operator Registry — Phase 1

```go
// Operator Registry
//
// To add a new operator:
//   1. Add an entry below: "OP_NAME": {NumInputs, Dimensions, RunFunc}
//   2. NumInputs = how many input tensors the op needs
//   3. Dimensions = which shape dimensions affect performance
//   4. inputs[i] are created via ctx.Zeros(dtype, shape...) with shapes
//      from the sampling grid or model graph
//
// Examples:
//   1D unary:  "SILU":     {1, ["N"],          func(...) { return in[0].SILU(ctx) }}
//   3D binary: "MUL_MAT":  {2, ["M","K","N"],  func(...) { return in[0].Mulmat(ctx, in[1]) }}
//   Special:   "FLASH_ATTN_EXT": requires ScaledDotProductAttention interface
//
// The op name must match GGML op names as they appear in GraphNode.Op.

var opRegistry = map[string]OpRunner{
    "SILU": {
        NumInputs:  1,
        Dimensions: []string{"N"},
        Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
            return in[0].SILU(ctx)
        },
    },
    "MUL_MAT": {
        NumInputs:  2,
        Dimensions: []string{"M", "K", "N"},
        Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
            return in[0].Mulmat(ctx, in[1])
        },
    },
    "FLASH_ATTN_EXT": {
        NumInputs:  3, // Q, K, V
        Dimensions: []string{"seq_q", "seq_kv"},
        Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
            // Q=in[0], K=in[1], V=in[2]
            // Requires type assertion to ScaledDotProductAttention
            sdpa, ok := in[0].(ml.ScaledDotProductAttention)
            if !ok {
                return nil
            }
            scale := 1.0 / math.Sqrt(float64(in[0].Dim(0)))
            return sdpa.ScaledDotProductAttention(ctx, in[1], in[2], nil, nil, nil, scale, false)
        },
    },
}
```

### 4.1 FLASH_ATTN_EXT Shape Construction

FLASH_ATTN_EXT requires specially shaped Q, K, V tensors in GGML's column-major
(ne[0..3]) order. Note: this is NOT the common [batch, heads, seq, dim] convention:

```
Q: [head_dim, num_heads, seq_q, 1]     — in[0]  (ne0=128, ne1=32, ne2=seq_q, ne3=1)
K: [head_dim, num_heads, seq_kv, 1]    — in[1]  (ne0=128, ne1=32, ne2=seq_kv, ne3=1)
V: [head_dim, num_heads, seq_kv, 1]    — in[2]  (ne0=128, ne1=32, ne2=seq_kv, ne3=1)
```

The generic `ctx.Zeros(dtype, shape...)` creation works, but the shapes must be constructed correctly. The benchmark function needs per-op shape expansion logic:

```go
// expandShapes converts grid dimensions to full tensor shapes per op.
func expandShapes(op string, gridPoint []int64) [][]int64 {
    switch op {
    case "FLASH_ATTN_EXT":
        // gridPoint = [seq_q, seq_kv], fixed head_dim=128, num_heads=32
        seqQ, seqKV := gridPoint[0], gridPoint[1]
        return [][]int64{
            {128, 32, seqQ, 1},   // Q
            {128, 32, seqKV, 1},  // K
            {128, 32, seqKV, 1},  // V
        }
    case "MUL_MAT":
        // gridPoint = [M, K, N]
        M, K, N := gridPoint[0], gridPoint[1], gridPoint[2]
        return [][]int64{
            {M, K},  // weight
            {K, N},  // activation
        }
    default:
        // 1D ops: gridPoint = [N]
        shapes := make([][]int64, 1)
        shapes[0] = gridPoint
        return shapes
    }
}
```

## 5. Hardware Characterization

### 5.1 What to Measure

```go
type HWCharResult struct {
    PeakTOPS     map[string]float64  // dtype -> TOPS (teraops/sec)
    PeakBW       float64             // bytes/sec
    BalancePoint map[string]float64  // dtype -> FLOPs/byte
}
```

### 5.2 Measurement Method

**Peak TOPS** (per dtype):
- Large square MUL_MAT: M=K=N=4096
- FLOPs = 2 × 4096³ = 137.4 GFLOP
- Measure latency, compute TOPS = FLOPs / latency
- Test dtypes: f16, f32 (and bf16 if supported)

**Peak Bandwidth**:
- Large CONT (copy): 64M elements × 4 bytes = 256MB
- Bytes = 2 × 256MB (read + write) = 512MB
- Measure latency, compute BW = bytes / latency

**Balance Point**: β = PeakTOPS / PeakBW (units: FLOPs/byte)

### 5.3 Usage

These values are used for **both prediction and sanity checking**:
1. **Roofline prediction**: MUL_MAT latency is predicted using `max(FLOPs / (eff_compute × PeakTOPS), bytes / (eff_bw × PeakBW))` with per-op efficiency constants (see Section 5A)
2. **Sanity checks**: Measured latency should never be less than `max(FLOPs/PeakTOPS, bytes/PeakBW)`
3. **Balance point**: Determines the memory-bound → compute-bound transition

> **History**: The original design said "NOT used for prediction" because v1 planned to measure every (M,K,dtype) combination
> individually via adaptive sampling (see Section 6-ARCHIVE). Empirical testing on Intel iGPU showed that 24 MUL_MAT grids
> took ~4 hours, making the approach unscalable. Roofline + efficiency constants achieve <10% error with ~5 minutes of measurement.

## 5A. Roofline Extrapolation for MUL_MAT

> **Design rationale**: The choice to use roofline extrapolation instead of full adaptive sampling was arrived at through empirical testing and iterative reasoning. The complete derivation — including the key insight that this approach is an improved version of v1's eta model — is documented in [Appendix A](#appendix-a-mul_mat-benchmark-strategy--design-rationale).

### 5A.1 Motivation

Measuring every (M, K, dtype) combination via adaptive sampling is prohibitively slow:
- Phase 1: 6 (M,K) pairs × 4 dtypes = 24 grids
- Each grid: ~10 minutes on Intel iGPU (8 initial + ~3 refinement points, each requiring warmup + 100 reps)
- Total: ~4 hours for MUL_MAT alone
- Future phases with more operators make this approach unscalable

Empirical validation (Intel iGPU, Vulkan) showed that roofline predictions match measured latency within 10% for compute-bound regimes (N ≥ ~100), and the deviation at small N follows a predictable pattern.

### 5A.2 Empirical Evidence

Data from Intel iGPU benchmark (peak_TOPS_f32 = 64.3 GFLOPS, peak_BW = 40.7 GB/s):

**MUL_MAT f32, M=K=4096** (reference curve):

| N | Ideal Compute (us) | Ideal BW (us) | Roofline (us) | Measured (us) | Regime | Regime Eff. |
|---|---|---|---|---|---|---|
| 1 | 524 | **1,570** | 1,570 | 3,754 | BW-bound | BW: 0.42 |
| 3 | 1,572 | **1,571** | 1,572 | 3,007 | BW/transition | BW: 0.52 |
| 11 | **5,740** | 1,649 | 5,740 | 8,028 | Transition | Compute: 0.71 |
| 35 | **18,260** | 1,891 | 18,260 | 24,217 | Transition | Compute: 0.75 |
| 116 | **60,527** | 2,555 | 60,527 | 64,610 | Compute | Compute: 0.94 |
| 380 | **198,290** | 4,719 | 198,290 | 219,651 | Compute | Compute: 0.90 |
| 1,248 | **651,263** | 11,829 | 651,263 | 695,931 | Compute | Compute: 0.94 |
| 4,096 | **2,137,466** | 35,266 | 2,137,466 | 2,302,781 | Compute | Compute: 0.93 |

> "Regime Eff." = `dominant_ideal / measured`. BW-bound points compare vs BW ceiling; compute-bound points compare vs compute ceiling. Bold values in Ideal columns show which ceiling dominates.

**MUL_MAT f32, M=14336, K=4096** (different shape, NO full adaptive sampling):

| N | Roofline (us) | Measured (us) | Regime Eff. |
|---|---|---|---|
| 1 | 5,773 | 12,046 | BW: 0.48 |
| 380 | 695,019 | 784,416 | Compute: 0.89 |
| 4,096 | 7,480,000 | 7,509,203 | Compute: **1.00** |

Key observations:
- **Compute-bound (N ≥ ~100)**: efficiency converges to ~0.90-0.93, consistent across (M,K) shapes
- **BW-bound (N ≤ ~3)**: effective BW is ~40-50% of peak CONT bandwidth (expected — matmul kernels have tiled access patterns, not sequential like CONT)
- **The efficiency constants are per-op properties**, not per-shape — they reflect GPU kernel characteristics

### 5A.3 Prediction Model

For MUL_MAT:

```
latency(M, K, N, dtype) = max(
    FLOPs / (eff_compute × peak_TOPS[dtype]),
    bytes / (eff_bw × peak_BW)
) + overhead
```

Where:
- `FLOPs = 2 × M × K × N`
- `bytes = (M×K + K×N + M×N) × elem_size(dtype)`
- `eff_compute` ≈ 0.93 (compute efficiency, measured from reference curve)
- `eff_bw` ≈ 0.45 (BW efficiency for matmul, measured from small-N points)
- `overhead` ≈ small constant per kernel launch (measured once)
- `peak_TOPS[dtype]` and `peak_BW` come from hardware characterization (Section 5)

### 5A.4 Calibration Procedure

Instead of 24 full adaptive sampling grids, the calibration measures:

1. **Hardware characterization** (already in Section 5): peak_TOPS for f16 + f32, peak_BW (~1 min with convergence early stopping)
2. **One reference curve per weight dtype**: Full adaptive sampling at M=K=4096 for each weight dtype
   - f32: M=K=4096, adaptive over N ∈ [1, 4096] (~2-3 min)
   - f16: same (~2-3 min)
   - q4_0: same (~1-2 min, likely faster due to smaller memory footprint)
   - q8_0: same (~1-2 min)
   - Each curve independently extracts eff_compute and eff_bw constants for that weight dtype
3. **Spot checks** (optional): 2 points (small N, large N) for a few other (M,K) to validate scaling
   - Each spot check: ~30 seconds
   - Total: ~2-3 minutes for 4-5 spot checks

**Total calibration time: ~12-14 minutes** (vs ~4 hours with full adaptive sampling)

> **Why per-dtype reference curves**: Different weight dtypes use completely different GPU kernels.
> q4_0 MUL_MAT involves on-the-fly dequantization, which changes both the compute efficiency
> (dequant overhead) and memory bandwidth efficiency (smaller blocks = less data to read). Using
> f32 efficiency constants for q4_0 prediction would be inaccurate. Empirical data is needed to
> confirm the actual magnitude of this difference.
>
> **Dtype mapping for estimation**: The benchmark measures f32, f16, q4_0, q8_0. Real models may
> use K-quant variants (q4_K, q5_K, q6_K) that are not directly benchmarkable via the Go DType
> abstraction (which only exposes f32/f16/q4_0/q8_0). At estimation time, unsupported weight dtypes
> are mapped to the nearest measured dtype:
> - q4_K, q4_1 → q4_0 (same 4.5 bits/element, similar dequant cost)
> - q5_K, q5_0, q5_1 → q8_0 (closer to 8-bit in memory footprint)
> - q6_K → q8_0
> - q3_K, q2_K → q4_0 (conservative: use 4-bit efficiency for smaller quants)
> - q8_K → q8_0
>
> This mapping is approximate. The primary error source is the dequantization kernel difference,
> not the memory footprint (which is similar within each group). Phase 2 may add direct C-level
> benchmarking to bypass the Go DType limitation.
>
> **Mixed-dtype tensor creation for MUL_MAT benchmarking**: GGML's `mul_mat` requires the weight
> tensor to be the quantized dtype (e.g., q4_0) while the activation tensor must be f32 or f16.
> This means the generic `measureOp()` function — which creates all input tensors with the same
> dtype — cannot be used for quantized MUL_MAT benchmarks. Instead, a dedicated `measureMulMat()`
> function creates weight tensors at `weightDtype` and activation tensors at f32. This is
> MUL_MAT-specific: FLASH_ATTN_EXT uses uniform f16 for all inputs (Q/K/V), and element-wise ops
> operate on f32 activations only.

### 5A.5 Efficiency Constant Extraction

From the reference curve:

```go
// eff_compute = median of (roofline_latency / measured_latency) for compute-bound points
// A point is "compute-bound" when FLOPs/peak_TOPS > bytes/peak_BW
computeEffPoints := filterComputeBound(referenceCurve, peakTOPS, peakBW)
effCompute := median(efficiencies(computeEffPoints))

// eff_bw = median of (roofline_bw_latency / measured_latency) for BW-bound points
bwBoundPoints := filterBWBound(referenceCurve, peakTOPS, peakBW)
effBW := median(efficiencies(bwBoundPoints))
```

### 5A.6 What This Changes

| Aspect | Old (Adaptive per grid) | New (Roofline extrapolation) |
|--------|------------------------|------|
| MUL_MAT grids measured | 6 (M,K) × 4 dtype = 24 | 4 reference curves (1 per weight dtype) + spot checks |
| Time | ~4 hours | ~8-10 minutes |
| Accuracy (compute-bound) | Exact measurement | ~10% error |
| Accuracy (BW-bound) | Exact measurement | ~15% error |
| Scalability to new (M,K) | Linear in grid count | Constant (roofline extrapolates) |
| Profile storage | 24 curves | 4 curves + 4 sets of efficiency constants |

### 5A.7 Profile Extension

```go
type HardwareProfile struct {
    // ... existing fields ...
    EfficiencyConstants map[string]OpEfficiency `json:"efficiency_constants,omitempty"`
}

type OpEfficiency struct {
    ComputeEff float64 `json:"compute_eff"` // fraction of peak TOPS achieved
    BWEff      float64 `json:"bw_eff"`      // fraction of peak BW achieved
    OverheadUs float64 `json:"overhead_us"`  // per-kernel-launch overhead
}
```

### 5A.8 Scope and Dtype Strategy

Phase 1 applies roofline extrapolation to **MUL_MAT only** (the bottleneck with 24 grids).
Other ops keep direct measurement:
- Element-wise ops (SILU, etc.): 1 grid each (f32 only), fast (~10-27 seconds per op)
- FLASH_ATTN_EXT: 1 grid (f16 only), moderate (~2-3 minutes)

**Dtype strategy per op type:**

| Op Type | Benchmark Dtypes | Rationale |
|---------|-----------------|-----------|
| Peak TOPS (hw char) | f16, f32 | Measures raw ALU capability; quantization is irrelevant |
| MUL_MAT reference curves | f32, f16, q4_0, q8_0 | Different weight dtypes use different GPU kernels with different efficiency |
| Element-wise (SILU, ADD, ...) | f32 only | Operate on activations, not weights; no quantization involved |
| FLASH_ATTN_EXT | f16 only | Q/K/V tensors are always f16; no quantization involved |

**MUL_MAT estimation with unmeasured weight dtypes**: See dtype mapping table in Section 5A.4.

Future phases can apply the same pattern to any new op that has many shape combinations.

## 6. Adaptive Sampling Algorithm

> **Note**: Adaptive sampling remains the measurement engine for reference curves, SILU, and FLASH_ATTN_EXT.
> For MUL_MAT, only ONE reference curve is measured via adaptive sampling; other (M,K,dtype) combinations
> use roofline extrapolation (Section 5A).

### 6.1 For 1D Operators (SILU)

```go
func adaptiveSample1D(backend ml.Backend, op string, dtype ml.DType,
    hw HWCharResult, cfg BenchmarkConfig) []LatencyPoint {

    // Step 1: Initial log-spaced grid
    // Range: [1K, 64M] for element-wise ops
    logMin, logMax := math.Log(1024), math.Log(64*1024*1024)
    nInitial := 8
    points := make([]LatencyPoint, 0, 20)

    for i := 0; i < nInitial; i++ {
        logN := logMin + float64(i)*(logMax-logMin)/float64(nInitial-1)
        N := int64(math.Round(math.Exp(logN)))
        lat := measureOp(backend, op, []int64{N}, dtype, 0, cfg)
        points = append(points, lat)
    }

    // Step 2: Adaptive refinement
    for len(points) < 20 {  // budget limit
        maxErr, maxIdx := findMaxInterpolationError(points)
        if maxErr < 0.05 {  // 5% threshold
            break
        }
        // Measure midpoint of highest-error interval
        midShape := logMidpoint(points[maxIdx].Shape, points[maxIdx+1].Shape)
        lat := measureOp(backend, op, midShape, dtype, 0, cfg)
        points = insertSorted(points, lat)
    }

    return points
}
```

### 6.2 For MUL_MAT (Reference Curve Only)

Adaptive sampling is used for **one reference (M,K) pair per dtype** to extract efficiency constants.
All other (M,K,dtype) combinations use roofline extrapolation (Section 5A).

```
Reference: (M=4096, K=4096), dtype=f32
    Run 1D adaptive sampling over N ∈ [1, 4096]
    Extract eff_compute, eff_bw from the resulting curve
```

After efficiency constants are extracted, MUL_MAT latency for any (M, K, N, dtype) is predicted analytically.

### 6.2-ARCHIVE: Original Full-Grid MUL_MAT Sampling (Superseded by Section 5A)

> **This was the original design. It is preserved here for historical reference.**
> Empirical testing showed this takes ~4 hours on Intel iGPU. Replaced by roofline extrapolation.

MUL_MAT adaptive sampling was structured differently because (M, K) pairs are discrete (from model architectures) while N is continuous:

```
For each (M, K) pair in {(4096,4096), (14336,4096), (4096,14336), (8192,8192), ...}:
    For each weight_dtype in {f16, q4_0, q8_0, ...}:
        Run 1D adaptive sampling over N ∈ [1, 4096]
        (Same algorithm as 6.1, but with fixed M, K)
```

This produces a collection of 1D curves, one per (M, K, weight_dtype). Estimation interpolates between the nearest (M, K) pairs.

### 6.3 For FLASH_ATTN_EXT

With head_dim=128, num_heads=32 fixed. In practice, transformer inference only produces two regimes:

- **Decode**: seq_q = 1, seq_kv = context_length (single new token attends to full context)
- **Prefill**: seq_q = seq_kv = prompt_length (full self-attention on prompt)

Arbitrary (seq_q, seq_kv) combinations don't occur in real inference. So we sample two 1D curves:

```
Decode curve:  seq_q = 1, sweep seq_kv ∈ [1, 16384] (log-spaced, adaptive)
Prefill curve: seq_q = seq_kv, sweep both ∈ [1, 16384] (log-spaced, adaptive)
```

Both curves are stored in a single OperatorCurve with `FixedDims={"num_heads": 32, "head_dim": 128}` and `Dimensions=["seq_q", "seq_kv"]`. Points have `Shape=[seq_q, seq_kv]`.

### 6.4 Interpolation Error Estimation

For 1D: measure actual latency at the midpoint between two adjacent points, compare with interpolated value in log-log space.

```go
func findMaxInterpolationError(points []LatencyPoint) (float64, int) {
    maxErr := 0.0
    maxIdx := 0
    for i := 0; i < len(points)-1; i++ {
        // Interpolated value at log-midpoint
        logX1 := math.Log(float64(points[i].Shape[0]))
        logX2 := math.Log(float64(points[i+1].Shape[0]))
        logY1 := math.Log(points[i].LatencyUs)
        logY2 := math.Log(points[i+1].LatencyUs)
        logMid := (logX1 + logX2) / 2
        logInterp := logY1 + (logY2-logY1)*(logMid-logX1)/(logX2-logX1)

        // Actual measurement at midpoint (cached or measured)
        midN := int64(math.Round(math.Exp(logMid)))
        actualLogY := math.Log(measureOrLookup(midN))

        relErr := math.Abs(logInterp-actualLogY) / math.Abs(actualLogY)
        if relErr > maxErr {
            maxErr = relErr
            maxIdx = i
        }
    }
    return maxErr, maxIdx
}
```

## 7. Log-Space Interpolation

### 7.1 1D Interpolation

```go
// Interpolate1D performs piecewise linear interpolation in log-log space.
// points must be sorted by Shape[0] ascending.
func Interpolate1D(points []LatencyPoint, queryN int64) float64 {
    logQ := math.Log(float64(queryN))

    // Find bracketing interval
    for i := 0; i < len(points)-1; i++ {
        logX1 := math.Log(float64(points[i].Shape[0]))
        logX2 := math.Log(float64(points[i+1].Shape[0]))
        if logQ >= logX1 && logQ <= logX2 {
            logY1 := math.Log(points[i].LatencyUs)
            logY2 := math.Log(points[i+1].LatencyUs)
            t := (logQ - logX1) / (logX2 - logX1)
            return math.Exp(logY1 + t*(logY2-logY1))
        }
    }

    // Extrapolation: extend slope of nearest segment
    if logQ < math.Log(float64(points[0].Shape[0])) {
        return extrapolateLeft(points, logQ)
    }
    return extrapolateRight(points, logQ)
}
```

### 7.2 MUL_MAT Latency Prediction (Roofline)

For MUL_MAT with query (M, K, N, dtype), latency is predicted analytically using
efficiency constants from the reference curve (Section 5A):

```go
// PredictMulMatLatency computes MUL_MAT latency using the roofline model
// with measured efficiency constants.
func PredictMulMatLatency(hw *HardwareProfile, M, K, N int64, dtype string) float64 {
    eff := hw.EfficiencyConstants["MUL_MAT"]
    peakTOPS := hw.PeakTOPS[dtype]
    peakBW := hw.PeakBandwidthBytesPerSec

    flops := 2.0 * float64(M) * float64(K) * float64(N)
    elemBytes := float64(elemSize(dtype))
    bytes := float64(M*K+K*N+M*N) * elemBytes

    computeTime := flops / (eff.ComputeEff * peakTOPS)   // seconds
    bwTime := bytes / (eff.BWEff * peakBW)                // seconds
    overhead := eff.OverheadUs * 1e-6                      // seconds

    latency := math.Max(computeTime, bwTime) + overhead
    return latency * 1e6  // return microseconds
}
```

No multi-curve interpolation needed — the formula works for any (M, K, N, dtype) directly.

### 7.2-ARCHIVE: Original Multi-Curve Interpolation (Superseded by Section 5A)

> **This was the original design for looking up MUL_MAT latency from multiple measured curves.**
> It required measuring 24 curves and interpolating between the nearest (M,K) pairs.
> Replaced by analytical roofline prediction.

For MUL_MAT with query (M, K, N):

Each MUL_MAT OperatorCurve has `FixedDims={"M": m, "K": k}` and `Dimensions=["N"]`,
making it a 1D curve of N. To look up latency for an arbitrary (M, K, N):

1. Find the two closest (M, K) curves by Euclidean distance in log space
2. For each curve, do 1D interpolation over N
3. Weight-average by inverse log-distance

```go
func InterpolateMulMat(curves []OperatorCurve, queryM, queryK, queryN int64) float64 {
    type candidate struct {
        curve    *OperatorCurve
        logDist  float64
    }

    // Find closest (M, K) curves using FixedDims
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

    if candidates[0].logDist == 0 || len(candidates) == 1 {
        // Exact (M, K) match or only one curve available
        return Interpolate1D(candidates[0].curve.Points, queryN)
    }

    // Inverse-distance weighted average of two nearest (M, K) curves
    lat1 := Interpolate1D(candidates[0].curve.Points, queryN)
    lat2 := Interpolate1D(candidates[1].curve.Points, queryN)
    w1 := 1.0 / candidates[0].logDist
    w2 := 1.0 / candidates[1].logDist
    return (lat1*w1 + lat2*w2) / (w1 + w2)
}
```

### 7.3 FLASH_ATTN_EXT Interpolation

FLASH_ATTN_EXT has `FixedDims={"num_heads": 32, "head_dim": 128}` and two sampled regimes stored as points with `Shape=[seq_q, seq_kv]`:

- **Prefill points**: seq_q == seq_kv (e.g., [128, 128], [512, 512], [2048, 2048])
- **Decode points**: seq_q == 1 (e.g., [1, 128], [1, 512], [1, 2048])

For a query (seq_q, seq_kv):

1. If seq_q == 1 → use decode points, 1D interpolation over seq_kv
2. If seq_q == seq_kv → use prefill points, 1D interpolation over seq_kv
3. Otherwise → interpolate between the two regimes by `t = log(seq_q) / log(seq_kv)`

```go
// Interpolate1DByDim is like Interpolate1D but reads Shape[dimIdx] instead of Shape[0].
// This allows interpolating over any dimension of a multi-dimensional LatencyPoint.
func Interpolate1DByDim(points []LatencyPoint, dimIdx int, queryVal int64) float64 {
    // Same algorithm as Interpolate1D, but uses points[i].Shape[dimIdx]
    // for the x-axis instead of points[i].Shape[0].
    // Points must be sorted by Shape[dimIdx] ascending.
    // (Implementation mirrors Interpolate1D — omitted for brevity.)
}

func InterpolateFlashAttn(curve *OperatorCurve, querySeqQ, querySeqKV int64) float64 {
    // Separate points into prefill (seq_q == seq_kv) and decode (seq_q == 1)
    var prefillPts, decodePts []LatencyPoint
    for _, pt := range curve.Points {
        if pt.Shape[0] == 1 {
            decodePts = append(decodePts, pt)
        } else if pt.Shape[0] == pt.Shape[1] {
            prefillPts = append(prefillPts, pt)
        }
    }

    if querySeqQ == 1 {
        return Interpolate1DByDim(decodePts, 1, querySeqKV) // interpolate over dim index 1 (seq_kv)
    }
    if querySeqQ == querySeqKV {
        return Interpolate1DByDim(prefillPts, 1, querySeqKV)
    }

    // Between regimes: weighted blend
    // t=0 → decode, t=1 → prefill. Use log ratio as interpolation weight.
    // Guard: if seq_kv <= 1, fall back to decode curve (trivial case).
    if querySeqKV <= 1 {
        return Interpolate1DByDim(decodePts, 1, querySeqKV)
    }
    decodeLat := Interpolate1DByDim(decodePts, 1, querySeqKV)
    prefillLat := Interpolate1DByDim(prefillPts, 1, querySeqKV)
    t := math.Log(float64(querySeqQ)) / math.Log(float64(querySeqKV))
    return math.Exp(math.Log(decodeLat)*(1-t) + math.Log(prefillLat)*t)
}
```

## 8. buildModelGraphNodes

### 8.1 Implementation

Uses `AllocMemory: false` to extract graph structure without loading model weights (MB not GB). See [`docs/internals/graph-without-weights.md`](../../internals/graph-without-weights.md) for rationale and code evidence.

#### Backend 分配问题

`GraphNode.Backend` 字段需要正确反映每个算子在哪个 backend（vulkan/cuda/cpu）上执行，以便 estimate 时匹配 profile 中正确的 calibration curve。

**问题**: 原来使用 `ctx.Forward(t).Reserve()` 捕获 graph nodes，但 `ggml_backend_sched_reserve` 内部的调用链是：
```
split_graph (分配 backend + graph_optimize) → gallocr_reserve → gallocr_alloc → reset()
```
`reset()` 会清空 `hash_set` 和 `hv_tensor_backend_ids`，导致 `captureGraphNodes()` 在 `reserve` 返回后调用时，`ggml_backend_sched_get_tensor_backend()` 对所有节点返回 nil → "unknown"。

**不需要 `AllocMemory: true`**: 权重 buffer 在 `ggml_backend_alloc_ctx_tensors_from_buft` 中独立分配（`ml/backend/ggml/ggml.go:400`），不受 `AllocMemory` 参数控制。`split_graph` 通过 `src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS` 判断权重 tensor 属于哪个 backend（`ggml-backend.cpp:862`），只要权重 buffer 存在就能正确分配。

**解决方案**: 在 `ml.Context` 接口上添加 `PlanGraph()` 方法，专为 estimate 路径设计。只调用 `split_graph`（包含 backend 分配 + graph_optimize）+ `captureGraphNodes` + `reset`，不做内存预分配。全部使用 ggml 公开 API，不修改 C 代码：
- `ggml_backend_sched_split_graph` — 拆分图 + 分配 backend + graph_optimize
- `ggml_backend_sched_get_tensor_backend` — 查询节点 backend（在 reset 前有效）
- `ggml_backend_sched_reset` — 清空状态

```go
// ml.Context interface addition
PlanGraph()  // split_graph + captureGraphNodes + reset (no memory allocation)

// ml/backend/ggml implementation
func (c *Context) PlanGraph() {
    if c.batchSize > 0 {
        C.ggml_backend_sched_set_batch_size(c.b.sched, C.int(c.batchSize))
    }
    // split_graph: assigns backends + runs graph_optimize (fusion)
    C.ggml_backend_sched_split_graph(c.b.sched, c.graph)
    // Capture graph nodes while backend assignments are still valid
    c.captureGraphNodes()
    // Clean up scheduler state (no side effects on memory allocation)
    C.ggml_backend_sched_reset(c.b.sched)
}
```

调用方 `buildModelGraphNodes` 改用 `PlanGraph()` 替代 `Reserve()`：
```go
ctx.SetBatchSize(batchSize)
ctx.Forward(t)
ctx.PlanGraph()      // backend-aware graph capture, no memory allocation
return ctx.GraphNodes(), nil
```

`Reserve()` 保持不变，继续用于正常推理路径的内存预分配。

#### 层分配 (Schedule) 与 Backend 分配

`split_graph` 通过检查权重 tensor 的 `buffer` 来判断每个 op 属于哪个 backend。
权重 buffer 的分配取决于 `BackendParams.GPULayers`——哪些层 offload 到 GPU（`ggml.go:203 assignLayer`）。
不传 `GPULayers` → 全部在 CPU → `split_graph` 把所有 op 分到 CPU，这不符合实际推理行为。

**层分配即 schedule**: `GPULayers` 本质上是一个"schedule"——决定哪些层在哪个 device 上执行。
正常推理时，server 根据 GPU 空闲内存和每层权重大小计算 schedule（`llm/server.go:buildLayout`）。

**DAOP 的 schedule 策略**: estimate 路径需要自己构造 `GPULayers` 来模拟不同的 schedule 方案。
这也是 DAOP 的远期目标之一：评估不同 schedule 方案下的预估性能，选择最优方案。

Phase 1C 实现最简单的策略——**全部 offload 到主 GPU**：
```go
// Schedule strategy: full offload to primary GPU
func fullOffloadSchedule(backend ml.Backend, numLayers int) ml.GPULayersList {
    devices := backend.BackendDevices()
    if len(devices) == 0 {
        return nil  // CPU-only, no GPU layers
    }
    layers := make([]int, numLayers+1) // +1 for output layer
    for i := range layers {
        layers[i] = i
    }
    return ml.GPULayersList{{
        DeviceID: devices[0].DeviceID,
        Layers:   layers,
    }}
}
```

这要求 `buildModelGraphNodes` 执行两次 `model.New`：
1. 第一次不传 `GPULayers`（发现模型层数和 GPU 信息）
2. 构造 schedule
3. 第二次传 `GPULayers`（正确的 backend 分配 → `PlanGraph` 拿到正确的 graph nodes）

两次都用 `AllocMemory: false`，不加载权重、不分配计算缓冲区。
第一次只需要获取元信息（层数、设备列表），开销很小。

**未来扩展方向**（记入 TODO）：
- 提取 `llm/server.go:buildLayout` 到公共包，实现"模拟 Ollama 默认分配"策略
- 多 GPU 分配策略（按内存容量分配层）
- 自定义 schedule 评估（用户指定分配方案，比较预估性能）

### 8.2 Return Value Change

Returns prefill and decode graphs **separately** (not merged). Estimation needs both because:
- Prefill: large batch → MUL_MAT compute-bound → latency ∝ seq_len
- Decode: batch=1 → MUL_MAT memory-bound → latency per token is constant

### 8.3 Graph Node to Profile Lookup

```go
// nodeToQueryShape extracts the performance-relevant dimensions from a GraphNode.
func nodeToQueryShape(node ml.GraphNode) (op string, shape []int64, computeDtype, weightDtype string) {
    op = node.Op
    computeDtype = node.ComputeDtype
    weightDtype = node.WeightDtype

    switch op {
    case "MUL_MAT":
        // InputShapes[0] = weight [M, K], InputShapes[1] = activation [K, N]
        if len(node.InputShapes) >= 2 {
            M := node.InputShapes[0][0]
            K := node.InputShapes[0][1]
            N := node.InputShapes[1][1]
            shape = []int64{M, K, N}
        }
    case "FLASH_ATTN_EXT":
        // InputShapes[0] = Q [head_dim, num_heads, seq_q, 1]
        // InputShapes[1] = K [head_dim, num_heads, seq_kv, 1]
        if len(node.InputShapes) >= 2 && len(node.InputShapes[0]) >= 3 {
            seqQ := node.InputShapes[0][2]
            seqKV := node.InputShapes[1][2]
            shape = []int64{seqQ, seqKV}
        }
    default:
        // 1D ops: total elements
        totalElements := int64(1)
        for _, d := range node.Shape {
            if d > 0 { totalElements *= d }
        }
        shape = []int64{totalElements}
    }
    return
}
```

## 9. Estimation Pipeline

### 9.1 Core Function

```go
// EstimateResult preserves per-op breakdown from v1 for analysis and viewer.
type EstimateResult struct {
    Model                  string
    PrefillLatencyUs       float64
    PrefillMs              float64
    DecodeLatencyUsPerToken float64
    DecodeTokensPerSec     float64
    Prefill                PhaseEstimation
    Decode                 PhaseEstimation
    Warnings               []string
}

type PhaseEstimation struct {
    TotalLatencyMs float64
    TokensPerSec   float64
    TopOps         []OpBreakdown  // sorted by TotalMs descending, top 10
}

type OpBreakdown struct {
    Op           string
    Backend      string
    ComputeDtype string
    WeightDtype  string
    Count        int      // how many graph nodes matched this op
    TotalUs      float64  // sum of latencies for all nodes of this op
    Percentage   float64  // fraction of total phase latency
}

func EstimateModel(profile *Profile, modelPath string) (*EstimateResult, error) {
    prefillNodes, decodeNodes, err := buildModelGraphNodes(modelPath)
    if err != nil {
        return nil, err
    }

    result := &EstimateResult{}
    result.Prefill = estimatePhase(profile, prefillNodes, &result.Warnings)
    result.Decode = estimatePhase(profile, decodeNodes, &result.Warnings)

    result.PrefillLatencyUs = result.Prefill.TotalLatencyMs * 1000
    result.PrefillMs = result.Prefill.TotalLatencyMs
    result.DecodeLatencyUsPerToken = result.Decode.TotalLatencyMs * 1000
    result.DecodeTokensPerSec = 1e6 / result.DecodeLatencyUsPerToken

    return result, nil
}

// estimatePhase computes latency for a set of graph nodes with per-op breakdown.
func estimatePhase(profile *Profile, nodes []ml.GraphNode, warnings *[]string) PhaseEstimation {
    opStats := make(map[OpKey]*OpBreakdown)
    var totalUs float64

    for _, node := range nodes {
        if IsZeroCostOp(node.Op) { continue }
        op, shape, cdt, wdt := nodeToQueryShape(node)
        lat, err := lookupLatency(profile, op, shape, cdt, wdt, node.Backend)
        if err != nil {
            *warnings = append(*warnings, err.Error())
            continue
        }
        totalUs += lat

        key := OpKey{op, node.Backend, cdt, wdt}
        if s, ok := opStats[key]; ok {
            s.Count++
            s.TotalUs += lat
        } else {
            opStats[key] = &OpBreakdown{
                Op: op, Backend: node.Backend,
                ComputeDtype: cdt, WeightDtype: wdt,
                Count: 1, TotalUs: lat,
            }
        }
    }

    // Build top-ops list sorted by TotalUs descending
    var topOps []OpBreakdown
    for _, s := range opStats {
        if totalUs > 0 { s.Percentage = s.TotalUs / totalUs }
        topOps = append(topOps, *s)
    }
    sort.Slice(topOps, func(i, j int) bool { return topOps[i].TotalUs > topOps[j].TotalUs })
    if len(topOps) > 10 { topOps = topOps[:10] }

    return PhaseEstimation{
        TotalLatencyMs: totalUs / 1000,
        TokensPerSec:   1e6 / totalUs,
        TopOps:         topOps,
    }
}
```

### 9.2 lookupLatency

```go
// mapWeightDtype maps unsupported K-quant and other weight dtypes to the nearest
// measured dtype. The Go DType abstraction only exposes f32/f16/q4_0/q8_0, but real
// models use q4_K, q5_K, q6_K etc. This mapping enables estimation without
// direct benchmarking of every quant variant.
func mapWeightDtype(wdt string) string {
    switch wdt {
    case "f32", "f16", "q4_0", "q8_0":
        return wdt // directly measured
    case "q4_K", "q4_1":
        return "q4_0" // same ~4.5 bits/element, similar dequant
    case "q5_K", "q5_0", "q5_1", "q6_K":
        return "q8_0" // closer to 8-bit in memory footprint
    case "q3_K", "q2_K":
        return "q4_0" // conservative: use 4-bit efficiency
    case "q8_K":
        return "q8_0"
    default:
        return "f16" // fallback for unknown types
    }
}

func lookupLatency(profile *Profile, op string, shape []int64,
    computeDtype, weightDtype, backend string) (float64, error) {

    switch op {
    case "MUL_MAT":
        // Map weight dtype to nearest measured dtype for efficiency constant lookup
        mappedWdt := mapWeightDtype(weightDtype)

        // Use roofline prediction with per-dtype efficiency constants (Section 5A)
        // Efficiency constants are keyed as "MUL_MAT_<weightDtype>" (e.g., "MUL_MAT_q4_0")
        effKey := "MUL_MAT_" + mappedWdt
        eff, ok := profile.Hardware.EfficiencyConstants[effKey]
        if !ok {
            // Fall back to generic MUL_MAT constants if per-dtype not available
            eff, ok = profile.Hardware.EfficiencyConstants["MUL_MAT"]
            if !ok {
                return 0, fmt.Errorf("no efficiency constants for MUL_MAT — run daop-bench first")
            }
        }
        _ = eff // used by PredictMulMatLatency via profile lookup
        return PredictMulMatLatency(&profile.Hardware, shape[0], shape[1], shape[2], mappedWdt), nil

    case "FLASH_ATTN_EXT":
        // Find matching FLASH_ATTN_EXT curve (direct measurement)
        for i := range profile.Operators {
            c := &profile.Operators[i]
            if c.Op == op && c.ComputeDtype == computeDtype && c.Backend == backend {
                return InterpolateFlashAttn(c, shape[0], shape[1]), nil
            }
        }
        return 0, fmt.Errorf("uncalibrated op: %s (dtype=%s)", op, computeDtype)

    default:
        // 1D ops (direct measurement curves)
        for _, c := range profile.Operators {
            if c.Op == op && c.ComputeDtype == computeDtype && c.Backend == backend {
                return Interpolate1D(c.Points, shape[0]), nil
            }
        }
        return 0, fmt.Errorf("uncalibrated op: %s (dtype=%s)", op, computeDtype)
    }
}
```

## 10. HTML Viewer

### 10.1 Purpose

Interactive visualization of benchmark data in a web browser. Allows visual inspection of:
- Latency curves in log-log space per operator
- Sampling point distribution
- Memory-bound / compute-bound transition (knee point)
- Interpolation accuracy

### 10.2 Implementation

Single self-contained HTML file generated from profile data:

```go
func GenerateHTMLViewer(profile *Profile, outputPath string) error {
    // Embed profile JSON into HTML template
    // Template uses a JS charting library (e.g., Chart.js via CDN, or inline plotly.js)
    // No server needed — just open the HTML file in a browser
}
```

### 10.3 Features

- **All operators on one page**: Each operator rendered as a separate chart card (no dropdown — user preference from empirical testing)
- **Log/linear toggle**: Switch axes between log and linear scale
- **Hover details**: Show exact shape, latency, stddev on hover
- **1D ops**: 2D scatter plot (log N vs log latency) with interpolation line
- **MUL_MAT**: Reference curve(s) per weight dtype (f32, f16, q4_0, q8_0) with efficiency constants
- **FLASH_ATTN**: Two sub-traces per curve — decode (shape[0]=1, varying seq_kv) and prefill (shape[0]=seq_kv, both vary)

### 10.4 Tech Stack

- Single HTML file with embedded JS (no build step, no npm)
- Chart library: Plotly.js loaded from CDN (supports 2D, 3D, heatmaps)
- Profile data embedded as `<script>const PROFILE_DATA = {...}</script>`
- Generated by Go code, opened in browser via `open` / `xdg-open` / `start`

## 11. Benchmark Measurement

### 11.1 Configuration

```go
type BenchmarkConfig struct {
    WarmupReps     int     // Max GPU warmup iterations (default: 5)
    MeasureReps    int     // Max timed iterations, before adaptive reduction (default: 50)
    TrimPercent    float64 // Outlier trim percentage (default: 0.1 = 10%)
    ConvergenceCV  float64 // CV threshold for early stopping (default: 0.05 = 5%)
    MinReps        int     // Minimum reps before convergence check (default: 5)
    ErrorThreshold float64 // Adaptive sampling convergence (default: 0.05 = 5%)
    MaxPointsPerOp int     // Budget limit per (op, dtype) (default: 20)
}
```

### 11.2 Latency Computation — Convergence-Based Adaptive Measurement

> **Design rationale**: Fixed rep counts waste time on slow, stable ops (e.g., 4096³ MUL_MAT at ~3s/rep
> needs only 5-10 reps for stable median) while potentially under-sampling fast, noisy ops. Industry
> practice (Google Benchmark uses CV for warmup detection; criterion.rs uses bootstrap CI) confirms that
> convergence-based stopping is the standard approach. We use CV on trimmed samples — simpler than
> bootstrap, effective for our 5-50 rep range.

**Measurement algorithm:**

1. **Adaptive warmup**: Run up to `WarmupReps` iterations. After the first warmup, check elapsed time:
   - \>5s per iteration → 1 warmup is sufficient (break immediately)
   - \>1s per iteration → 2 warmups total
   - Otherwise → run all `WarmupReps`

2. **Tiered max reps**: After `MinReps` (5) samples, compute median latency and set a ceiling:
   - \>5s → maxReps = 5
   - \>1s → maxReps = 10
   - \>100ms → maxReps = 20
   - Otherwise → maxReps = `MeasureReps` (50)

3. **Convergence early stopping**: After each sample (once ≥ `MinReps`), compute CV on trimmed data:
   - Sort samples, trim top/bottom `TrimPercent`
   - CV = stddev(trimmed) / mean(trimmed)
   - If CV < `ConvergenceCV` (5%) → stop early, measurement has converged
   - Otherwise continue until maxReps

4. **Result**: Take **median** of trimmed set as reported latency; stddev of trimmed set for confidence.

**Why these parameters:**
- **CV threshold 5%** (not 3%): Empirical data from Intel iGPU shows even large stable MUL_MAT (N=4096,
  ~3s/rep) has CV ≈ 9% after 50 trimmed reps. A 3% threshold would never converge. 5% allows
  convergence for most ops while still catching unstable measurements.
- **Trimming before CV computation**: GPU benchmarks have right-skewed distributions (OS interrupts,
  thermal throttling). Without trimming, one spike inflates CV and prevents convergence. The CV
  should match the same trimmed set used for the final median — otherwise "CV says not converged"
  but "trimmed median is already stable".
- **Min 5 reps**: Minimum needed for meaningful trimming (10% of 5 = 0.5, rounds to 0-1 trim).

**Applies uniformly to**: `measureOp()` (operator benchmarking) AND `benchPeakTOPS()`/`benchPeakBandwidth()` (hardware characterization). Previously, hardware characterization ran a fixed 50 reps unconditionally — with 4096³ MUL_MAT at ~3s/rep, this took ~2.5 min/dtype. With convergence early stopping, stable large-matrix measurements converge in ~5-10 reps (~15-30s/dtype).

## 12. CLI Commands

### 12.1 `ollama daop-bench`

```
Usage: ollama daop-bench [flags]

Calibrate operator performance on this hardware.

Flags:
  --output PATH    Profile output path (default: ~/.ollama/daop/profile.json)
  --ops LIST       Comma-separated ops to benchmark (default: all registered)
  --dtypes LIST    Comma-separated dtypes (default: f16,f32,q4_0,q8_0)
  --viewer         Generate HTML viewer after benchmarking
  --verbose        Show per-point results during calibration
```

### 12.2 `ollama daop-estimate`

```
Usage: ollama daop-estimate <model> [flags]

Estimate inference performance for a model.

Arguments:
  model            Model name or path to GGUF file

Flags:
  --profile PATH   Profile to use (default: ~/.ollama/daop/profile.json)
  --json           Output as JSON
  --verbose        Show per-operator breakdown
```

### 12.3 `ollama daop-viewer`

```
Usage: ollama daop-viewer [flags]

Open benchmark data in interactive HTML viewer.

Flags:
  --profile PATH   Profile to visualize (default: ~/.ollama/daop/profile.json)
  --output PATH    Save HTML to file instead of opening browser
```

## 13. Testing Strategy

### 13.1 Test Categories

**Pure Go tests** (no GGML required):
- Interpolation math (Interpolate1D, Interpolate1DByDim, InterpolateMulMat, InterpolateFlashAttn)
- Adaptive sampling logic (with mock measurement function)
- Profile serialization/deserialization
- Shape expansion (expandShapes)
- Node-to-query-shape mapping (nodeToQueryShape)
- HTML viewer generation

**Integration tests** (require GGML build):
- Hardware characterization accuracy
- End-to-end benchmark of 3 ops on real backend
- buildModelGraphNodes with a small test model
- Full estimation pipeline accuracy

### 13.2 TDD Approach

For each component:
1. Write failing test first
2. Implement minimum code to pass
3. Refactor

Test files mirror source files: `registry_test.go`, `interpolate_test.go`, `adaptive_test.go`, `hwchar_test.go`, `estimate_test.go`, `viewer_html_test.go`.

### 13.3 Key Test Cases for Interpolation

```go
// Interpolate1D:
//   Exact match: query at a measured point should return exact value
//   Interior: query between two points should interpolate correctly
//   Boundary: query at first/last point
//   Extrapolation: query beyond measured range
//   Log-space correctness: verify interpolation happens in log not linear
//   Known function: benchmark f(N) = a + b*N, verify interpolation recovers it

// InterpolateMulMat:
//   Exact (M,K) match: should fall through to Interpolate1D
//   Between (M,K) pairs: verify inverse-distance weighting is correct
//   Single curve: should return Interpolate1D result directly

// InterpolateFlashAttn:
//   Decode regime: seq_q=1, verify 1D interpolation over seq_kv
//   Prefill regime: seq_q=seq_kv, verify 1D interpolation over seq_kv
//   Between regimes: verify weighted blend between decode and prefill curves

// Interpolate1DByDim:
//   Same cases as Interpolate1D but with dimIdx > 0
```

### 13.4 Key Test Cases for Adaptive Sampling

```go
// Smooth function: should converge quickly (8-10 points)
// Function with sharp knee: should add points around the knee
// Budget limit: should stop at MaxPointsPerOp even if not converged
// Already converged: initial grid sufficient, no refinement needed
```

---

## Appendix A: MUL_MAT Benchmark Strategy — Design Rationale

This appendix documents the complete reasoning process that led to the hybrid benchmark strategy (roofline extrapolation for MUL_MAT, direct curves for other ops). The conclusion is that the "new" approach is an empirically validated, improved version of v1's eta model — not a regression but a justified convergence.

### A.1 Starting Point: v1 Roofline + eta

v1 DAOP used a single-constant roofline model:

```
latency = max(FLOPs / (eta × peak_TOPS), bytes / peak_BW)
```

Where `eta` is a per-op efficiency constant (0 < eta ≤ 1) that captures the fraction of peak hardware performance achieved by the operator kernel. This is simple and fast but has limitations:
- A single `eta` conflates compute efficiency and memory bandwidth efficiency
- No explicit overhead term for kernel launch latency
- BW-bound regime uses raw `peak_BW` without accounting for memory access pattern differences between ops

### A.2 v2 Original Design: Replace Roofline with Direct Measurement

v2 was designed to **eliminate the roofline model entirely** and replace it with direct latency measurements at representative shape points, connected by log-space interpolation:

```
For each operator:
  For each (M, K) pair from model architectures:
    For each dtype (f16, f32, q4_0, q8_0):
      Adaptively sample latency vs N → one OperatorCurve
```

For MUL_MAT this produces: 6 (M,K) pairs × 4 dtypes = **24 sampling grids**, each containing 8-20 measurement points with 5 warmup + 50 timed repetitions each.

Rationale: direct measurement avoids all modeling assumptions. If the hardware has unusual characteristics (throttling, cache effects, non-linear scaling), the measurements capture them.

### A.3 Empirical Discovery: Full Measurement is Prohibitively Slow

Running `daop-bench` on Intel UHD Graphics 770 (iGPU, Vulkan backend):

| Phase | Duration |
|-------|----------|
| Hardware characterization | ~4 min |
| 1 MUL_MAT grid (M=K=4096, f32, 11 points adaptive) | ~10 min |
| 24 MUL_MAT grids (projected) | **~4 hours** |
| SILU (1 grid, fast) | ~10 sec |
| FLASH_ATTN_EXT (1 grid) | ~2 min |

**~4 hours for MUL_MAT alone**, and Phase 2 adds ~22 more operators. This approach does not scale.

The bottleneck is per-point measurement cost: each point requires warmup (5 reps) + measurement (50 reps), and adaptive refinement adds O(N) midpoint measurements per round via `findMaxInterpolationError`.

### A.4 Hypothesis: Roofline Can Predict Across Shapes

If the roofline model's efficiency is consistent across different (M,K) shapes, we only need to measure ONE reference curve to extract the efficiency constants, then predict all other shapes analytically.

**Test**: Compare roofline prediction against actual measurements for two different (M,K) shapes.

**Data** (Intel iGPU, peak_TOPS_f32 = 64.3 GFLOPS, peak_BW = 40.7 GB/s):

**Reference curve: M=K=4096, f32** (measured via adaptive sampling, converged at 11 points):

| N | FLOPs | Bytes | Arith. Intensity | Ideal Compute (us) | Ideal BW (us) | Measured (us) | Regime | Regime Eff. |
|---|-------|-------|---|---|---|---|---|---|
| 1 | 33.6M | 67.1MB | 0.50 | 524 | **1,570** | 3,754 | BW-bound | BW: **0.42** |
| 3 | 100.7M | 67.2MB | 1.50 | 1,572 | **1,571** | 3,007 | BW/transition | BW: **0.52** |
| 11 | 369.2M | 68.0MB | 5.43 | **5,740** | 1,649 | 8,028 | Transition | Compute: 0.71 |
| 35 | 1,174M | 70.0MB | 16.8 | **18,260** | 1,891 | 24,217 | Transition | Compute: 0.75 |
| 116 | 3,893M | 76.6MB | 50.8 | **60,527** | 2,555 | 64,610 | Compute | Compute: **0.94** |
| 380 | 12,750M | 98.1MB | 130 | **198,290** | 4,719 | 219,651 | Compute | Compute: **0.90** |
| 1,248 | 41,880M | 168.9MB | 248 | **651,263** | 11,829 | 695,931 | Compute | Compute: **0.94** |
| 4,096 | 137,400M | 402MB | 342 | **2,137,466** | 35,266 | 2,302,781 | Compute | Compute: **0.93** |

> **How to read "Regime Eff."**: For each point, we compare measured latency against the **dominant bottleneck ceiling** for that regime. BW-bound points: `Ideal_BW / Measured` (how close to memory bandwidth limit). Compute-bound points: `Ideal_Compute / Measured` (how close to compute limit). These are two different efficiency metrics — GPU matmul kernels achieve ~93% of peak compute but only ~45% of peak BW because matmul uses tiled (non-sequential) memory access.

**Validation curve: M=14336, K=4096, f32** (partial measurement from killed benchmark):

| N | Predicted* (us) | Measured (us) | Prediction Error |
|---|-----------------|---------------|-----------------|
| 1 | 11,618 | 12,046 | −3.6% |
| 380 | 735,150 | 784,416 | −6.3% |
| 4,096 | 7,508,000 | 7,509,203 | −0.02% |

*Predicted using eff_compute=0.93, eff_bw=0.45 extracted from reference curve.

**Key findings**:
1. **Compute-bound regime (N ≥ ~100)**: efficiency converges to 0.90–0.93, **consistent across (M,K) shapes**
2. **BW-bound regime (N ≤ ~3)**: effective BW is ~40–50% of peak CONT bandwidth
3. **Cross-shape prediction error**: <10% for all tested points
4. The efficiency constants are **per-kernel properties** (GPU matmul tiling/dispatch), not per-shape properties

#### Transition Zone Accuracy and the `max()` Overlap Assumption

The low efficiency at N=11 (0.71) and N=35 (0.75) — despite being compute-bound by roofline classification — reveals a limitation of the `max()` model.

The formula `latency = max(compute_time, bw_time)` assumes **perfect overlap** between compute and memory operations. In reality, overlap is partial. Using extracted efficiency constants for N=11:

- Real compute time: 5,740 / 0.93 = 6,172 us
- Real memory time: 1,649 / 0.45 = 3,664 us (59% of compute)
- `max()` predicts: 6,172 us, but measured: 8,028 us — the gap is the non-overlapping memory portion

For N=116+, memory time is <9% of compute, so even with zero overlap the `max()` model is accurate within ~10%. The transition zone (N ≈ 10–50) is where both components are significant and partial overlap causes `max()` to underestimate by up to ~30%.

**Practical impact is minimal**: real transformer inference uses N=1 (decode, firmly BW-bound) or N=prompt_length (prefill, firmly compute-bound). The transition zone is rarely exercised.

### A.5 The Key Insight: This IS v1's eta, Improved

At this point we recognized that the "new" roofline extrapolation model:

```
latency = max(FLOPs / (eff_compute × peak_TOPS), bytes / (eff_bw × peak_BW)) + overhead
```

is structurally identical to v1's eta model:

```
latency = max(FLOPs / (eta × peak_TOPS), bytes / peak_BW)
```

The differences are improvements, not fundamental changes:

| Aspect | v1 eta | v2 efficiency constants |
|--------|--------|------------------------|
| Compute efficiency | Single `eta` constant | Dedicated `eff_compute` |
| BW efficiency | Implicit (uses raw peak_BW) | Dedicated `eff_bw` (captures matmul access patterns) |
| Kernel overhead | Not modeled | Explicit `overhead_us` term |
| Calibration | Measured from 1 large matmul | Measured from full reference curve (8–11 points) |
| BW-bound accuracy | Poor (raw peak_BW ≠ matmul BW) | Better (eff_bw corrects for tiling overhead) |

The v2 version splits eta into two regime-specific constants and adds an overhead term, which explains why BW-bound predictions improve from ~2× error to ~15% error.

### A.6 Why Not Just Keep Full Measurement?

Full measurement is strictly more accurate but fails on practical grounds:

1. **Time**: ~4 hours (MUL_MAT) + future ops = unacceptable for user experience
2. **Scalability**: Phase 2 adds ~22 more operators; each with multiple (M,K,dtype) combos
3. **Diminishing returns**: ±10% error from roofline is acceptable for the use case (relative comparison across models/hardware, bottleneck identification)
4. **Redundancy**: If efficiency constants are consistent across shapes (empirically verified), measuring every shape wastes time measuring the same constant

### A.7 Why Not Use Roofline for ALL Ops?

Roofline works well for MUL_MAT because:
- MUL_MAT kernels are well-optimized and exhibit predictable scaling
- The compute-bound/BW-bound transition is clear
- Efficiency constants are stable across shapes

But it does NOT work for all operators:

**SILU / element-wise ops**: Measured BW efficiency is only ~12% of peak. The discrepancy is too large and variable to capture with a single constant — it reflects memory subsystem effects (strided access, cache behavior) that vary with tensor size in non-linear ways.

**FLASH_ATTN_EXT**: Has two distinct operating modes (decode: seq_q=1, prefill: seq_q=seq_kv) with different computational characteristics. The relationship between FLOPs and latency is not a simple roofline — attention involves softmax, masking, and memory access patterns that don't map cleanly to compute/BW regimes.

### A.8 Final Design: Hybrid Approach

| Operator | Strategy | Rationale | Calibration Time |
|----------|----------|-----------|-----------------|
| MUL_MAT | Roofline + efficiency constants | ±10% accuracy, consistent across shapes | ~3 min (1 reference curve) |
| SILU / element-wise | Direct adaptive sampling | Roofline doesn't fit (12% peak BW) | ~10 sec per op |
| FLASH_ATTN_EXT | Direct adaptive sampling | Dual-mode, complex access patterns | ~2 min |

**Total calibration: ~10 minutes** (vs ~4 hours with full measurement).

This is v1 + v2 combined: the analytical model (improved eta) for where it works, empirical curves for where it doesn't. The empirical data validates both the choice and the error bounds.

### A.9 Open Questions for Future Work

1. **Per-dtype efficiency constants**: Current calibration uses f32 only. Do quantized dtypes (q4_0, q8_0) have different eff_compute/eff_bw? Likely yes — quantized kernels have different arithmetic intensity profiles. Phase 2 should measure one reference curve per dtype.

2. **Cross-GPU validation**: The ±10% error bound was validated on Intel iGPU only. Need to verify on NVIDIA (CUDA), AMD (ROCm), and Apple (Metal/MLX) — these have different memory hierarchies and kernel implementations.

3. **Spot checks**: The design spec mentions optional spot-check measurements (2 points at other M,K values) to validate cross-shape consistency. This is not yet implemented but would add ~2 min and increase confidence.

4. **Adaptive refinement optimization**: The current `findMaxInterpolationError` measures ALL midpoints each round (O(N) measurements). A smarter approach would only measure the highest-error midpoint, reducing refinement cost from ~7 min to ~30 sec.
