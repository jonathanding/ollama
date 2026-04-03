# DAOP v2: Empirical Performance Estimation — Engineering Spec

> High-level design rationale, log-space theory, and architectural decisions are documented in
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

### Phase 2: Full Operator Coverage (out of scope)

- Extend to all remaining operators (~22 more). Each new op = 1 registry line + tests.
- **Cross-backend transfer cost**: Measure interconnect bandwidth (PCIe, NVLink), add transfer latency when consecutive graph nodes are on different backends. Phase 1 assumes single backend.
- **Incremental calibration (`--update`)**: Load existing profile, diff against model graph to find uncalibrated (op, dtype) combos, benchmark only the missing ones, merge into profile. v1's `RunUpdateBenchmark` pattern is preserved but reimplemented on the new data model.
- **`HardwareProfile.InterconnectBWBytesPerSec`**: Populated in Phase 2 when cross-backend support is added. Phase 1 leaves it as 0.

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

These values are NOT used for prediction. They serve two purposes:
1. **Initial grid placement**: Estimate where memory-bound → compute-bound transition occurs
2. **Sanity checks**: Measured latency should never be less than `max(FLOPs/PeakTOPS, bytes/PeakBW)`

## 6. Adaptive Sampling Algorithm

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

### 6.2 For 3D Operators (MUL_MAT)

MUL_MAT adaptive sampling is structured differently because (M, K) pairs are discrete (from model architectures) while N is continuous:

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

### 7.2 Multi-dimensional Lookup

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

Uses `AllocMemory: false` to extract graph structure without loading model weights (MB not GB). See [`docs/daop/design.md` Section 8](../../daop/design.md) for rationale and code evidence.

```go
func buildModelGraphNodes(modelPath string) (prefill, decode []ml.GraphNode, err error) {
    m, err := model.New(modelPath, ml.BackendParams{AllocMemory: false})
    if err != nil {
        return nil, nil, fmt.Errorf("load model: %w", err)
    }
    defer m.Backend().Close()

    // Capture graph for a given batch size.
    // Pattern follows runner/ollamarunner/runner.go:reserveWorstCaseGraph().
    captureGraph := func(batchSize int) ([]ml.GraphNode, error) {
        ctx := m.Backend().NewContext()
        defer ctx.Close()

        // Construct dummy input batch
        batchInputs := make([]int32, batchSize)
        positions := make([]int32, batchSize)
        sequences := make([]int, batchSize)
        for i := 0; i < batchSize; i++ {
            positions[i] = int32(i)
        }
        batch := input.Batch{
            Inputs:    ctx.Input().FromInts(batchInputs, batchSize),
            Outputs:   ctx.Input().Empty(ml.DTypeI32, 1),
            Positions: positions,
            Sequences: sequences,
        }

        // Initialize cache for graph capture (reserve=true)
        if cache := m.Config().Cache; cache != nil {
            if err := cache.StartForward(ctx, batch, true); err != nil {
                return nil, fmt.Errorf("cache start: %w", err)
            }
        }

        // Build computation graph via Forward (no actual computation)
        t, err := m.Forward(ctx, batch)
        if err != nil {
            return nil, fmt.Errorf("forward: %w", err)
        }

        // Capture graph structure
        ctx.SetBatchSize(batchSize)
        ctx.Forward(t).Reserve()

        return ctx.GraphNodes(), nil
    }

    // Prefill graph: batch=512 (representative prompt length)
    prefill, err = captureGraph(512)
    if err != nil {
        return nil, nil, fmt.Errorf("prefill graph: %w", err)
    }

    // Decode graph: batch=1 (single token generation)
    decode, err = captureGraph(1)
    if err != nil {
        return nil, nil, fmt.Errorf("decode graph: %w", err)
    }

    return prefill, decode, nil
}
```

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
func lookupLatency(profile *Profile, op string, shape []int64,
    computeDtype, weightDtype, backend string) (float64, error) {

    switch op {
    case "MUL_MAT":
        // Collect all MUL_MAT curves matching dtype/backend
        var curves []OperatorCurve
        for _, c := range profile.Operators {
            if c.Op == op && c.ComputeDtype == computeDtype &&
               c.WeightDtype == weightDtype && c.Backend == backend {
                curves = append(curves, c)
            }
        }
        if len(curves) == 0 {
            return 0, fmt.Errorf("uncalibrated op: %s (dtype=%s, wdtype=%s)", op, computeDtype, weightDtype)
        }
        return InterpolateMulMat(curves, shape[0], shape[1], shape[2]), nil

    case "FLASH_ATTN_EXT":
        // Find matching FLASH_ATTN_EXT curve
        for i := range profile.Operators {
            c := &profile.Operators[i]
            if c.Op == op && c.ComputeDtype == computeDtype && c.Backend == backend {
                return InterpolateFlashAttn(c, shape[0], shape[1]), nil
            }
        }
        return 0, fmt.Errorf("uncalibrated op: %s (dtype=%s)", op, computeDtype)

    default:
        // 1D ops
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

- **Op selector**: Dropdown to switch between operators
- **Log/linear toggle**: Switch axes between log and linear scale
- **Hover details**: Show exact shape, latency, stddev on hover
- **1D ops**: 2D scatter plot (log N vs log latency) with interpolation line
- **MUL_MAT**: Multiple 2D plots (one per (M,K) pair, each a trace). Each OperatorCurve = one Plotly trace labeled "MUL_MAT [M×K] dtype"
- **FLASH_ATTN**: Two curves overlaid — decode (seq_q=1, varying seq_kv) and prefill (seq_q=seq_kv)

### 10.4 Tech Stack

- Single HTML file with embedded JS (no build step, no npm)
- Chart library: Plotly.js loaded from CDN (supports 2D, 3D, heatmaps)
- Profile data embedded as `<script>const PROFILE_DATA = {...}</script>`
- Generated by Go code, opened in browser via `open` / `xdg-open` / `start`

## 11. Benchmark Measurement

### 11.1 Configuration

```go
type BenchmarkConfig struct {
    WarmupReps    int     // GPU warmup iterations (default: 5)
    MeasureReps   int     // Timed iterations (default: 100)
    TrimPercent   float64 // Outlier trim percentage (default: 0.1 = 10%)
    ErrorThreshold float64 // Adaptive sampling convergence (default: 0.05 = 5%)
    MaxPointsPerOp int    // Budget limit per (op, dtype) (default: 20)
}
```

### 11.2 Latency Computation

Per-iteration timing with outlier trimming:

1. Run `WarmupReps` iterations (discard)
2. Time each of `MeasureReps` iterations individually
3. Sort latencies
4. Trim top and bottom `TrimPercent` (remove GPU clock spikes, OS interrupts)
5. Take **median** of trimmed set as the reported latency
6. Compute stddev of trimmed set for confidence reporting

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
