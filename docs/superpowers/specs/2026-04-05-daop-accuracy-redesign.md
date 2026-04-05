# DAOP Accuracy Redesign: Benchmark + Estimate 修正方案

> GPU timestamp 验证揭示 estimate 18x 偏差的四个根本原因。
> 本 spec 设计修正方案，使 estimate 精度达到 <2x 误差。

## 1. 问题总结

### 1.1 四个根本原因

| # | 问题 | 影响 | 来源 |
|---|------|------|------|
| 1 | **Dispatch overhead** | 1D ops 99.9% 是 overhead，MUL_MAT 也严重 | benchmark 逐 op 提交 Vulkan command buffer |
| 2 | **MUL_MAT vs MUL_MAT_VEC** | N=1 时差 341x（83μs vs 25,978μs） | benchmark 测 MUL_MAT kernel，推理用 MUL_MAT_VEC |
| 3 | **Op fusion** | RMS_NORM+MUL 融合后 12μs vs 独立 3,349μs | estimate 按 unfused graph 逐 op 累加 |
| 4 | **CPU overhead** | GPU 54ms，wall-clock 75ms，差 21ms | estimate 只算 GPU op 时间 |
| 5 | **Benchmark 在 CPU 而非 GPU 上运行** | 所有 GPU timestamp 为空，benchmark 数据不可靠 | tensor 在 CPU buffer，scheduler 不 offload 到 Vulkan（见 Section 2B） |

### 1.2 跨 Backend 调研结论

| | Vulkan | CUDA | CPU |
|--|--------|------|-----|
| Dispatch overhead | 严重（~1ms/op） | 低（CUDA Graphs） | 可忽略 |
| MUL_MAT_VEC | 有，341x 差异 | 有（GEMV vs GEMM） | 差异小 |
| Op fusion | 14 种规则 | 有 | 无 |
| Wall-clock 准确 | 否 | 中等 | 是 |

**策略**：先解决 Vulkan（问题最严重），CPU 现有方案已足够，CUDA 以后再做。

## 2. 总体架构

### 2.1 Benchmark + Estimate 共享基础设施

当前 benchmark（`perf/bench.go`）和 estimate（`perf/estimate.go`）之间缺乏统一的命名和 backend 发现机制。新设计引入 `perf/common.go` 作为共享模块：

```go
// perf/common.go — benchmark 和 estimate 共享的基础设施

// OpVariant 统一 benchmark profile key 和 estimate lookup key 的命名。
// 确保 benchmark 存储的 op 名称与 estimate 查询时使用的名称完全一致。
type OpVariant struct {
    Op          string // GGML op name: "MUL_MAT", "RMS_NORM", etc.
    Variant     string // kernel variant: "", "VEC", "FUSED_RMS_NORM_MUL", etc.
    WeightDtype string // for MUL_MAT: "f16", "q4_0", etc.
    Backend     string // "Vulkan", "CUDA", "CPU", etc.
}

// ProfileKey 返回 profile 中存储/查询用的规范化 key。
// 保证 benchmark 写入和 estimate 读取使用完全相同的 key。
func (v OpVariant) ProfileKey() string {
    key := v.Op
    if v.Variant != "" {
        key += "_" + v.Variant
    }
    return key
}

// BackendCapabilities 描述一个 backend 支持的特性。
// benchmark 和 estimate 共用，避免重复检测。
type BackendCapabilities struct {
    Name            string   // "Vulkan", "CUDA", "CPU"
    HasGPUTimestamp bool     // 是否支持 GPU timestamp 精确计时
    FusionRules     []FusionRule  // 该 backend 支持的 op fusion 规则
    HasMulMatVec    bool     // 是否有 MUL_MAT_VEC 专用 kernel
    MulMatVecMaxN   int      // MUL_MAT_VEC 触发条件（Vulkan: N≤8）
}

// DiscoverBackend 检测当前系统的 primary backend 及其能力。
// benchmark 和 estimate 都调用此函数，确保一致性。
func DiscoverBackend(backend ml.Backend) BackendCapabilities {
    // 根据 backend.BackendDevices()[0].Library 判断类型
    // 返回对应的 capabilities
}
```

### 2.2 命名一致性

当前问题：benchmark 用 `"MUL_MAT"` + `WeightDtype="f16"` 存储，estimate 也用同样的 key 查询。但 MUL_MAT_VEC 和 fused ops 没有对应的 profile entry。

新设计在 profile 中新增 entries：

| Profile Key | 来源 | 用途 |
|------------|------|------|
| `MUL_MAT` (existing) | benchmark MUL_MAT kernel | prefill (N>8) estimate |
| `MUL_MAT_VEC` (new) | benchmark MUL_MAT with N≤8, GPU timestamp | decode (N≤8) estimate |
| `RMS_NORM_MUL` (new) | benchmark fused RMS_NORM+MUL | decode estimate (fusion) |
| `RMS_NORM_MUL_ROPE` (new) | benchmark fused pattern | decode estimate (fusion) |
| `MUL_MAT_ADD` (new) | benchmark fused MUL_MAT+ADD | decode estimate (fusion) |
| `ORCHESTRATION_OVERHEAD` (new) | benchmark N-trivial-op graph | end-to-end overhead |

`OpVariant.ProfileKey()` 确保 benchmark 存储和 estimate 查询使用完全相同的字符串。

## 2B. Direct Backend Execution（第五个根本原因）

> **发现于 Session 17 (2026-04-05)**：E2E 验证中发现 GPU timestamps 始终为空，
> 经调查确认是 benchmark 的所有 op 实际运行在 CPU 而非 GPU 上。

### 2B.1 问题

当前 benchmark 通过 `ggml_backend_sched`（scheduler）执行计算图。Scheduler 根据 tensor 所在 buffer 和 op offload 启发式决定每个 op 运行在哪个 backend。

**致命问题**：benchmark 的所有 tensor 通过 `ctx.Input().FromFloats()` 创建，`Input()` 返回 **CPU buffer type**。Scheduler 看到 tensor 在 CPU 上，对小/中型 op 不做 offload，直接在 CPU 上执行。

**证据**：
- DLL 重编译后，`ggml_backend_is_vk()` 返回 true，`EnableGPUTimestamps(true)` 成功调用
- 但 `GetOpTimings()` 始终返回 `nTimings=0`——Vulkan 的 `graph_compute` 从未被调用
- 所有 op 落入 wall-clock fallback

**影响范围**：

| Benchmark 步骤 | 是否在 GPU 上？ | 原因 |
|---------------|---------------|------|
| Peak TOPS (大 MUL_MAT 4096³) | 不确定 | Scheduler 可能 offload 大 op，但不保证 |
| Peak Bandwidth (大 CONT) | 不确定 | 同上 |
| 1D elementwise (ADD, SILU...) | ❌ CPU | Scheduler 不 offload 小 op |
| MUL_MAT reference curve | 混合 | 大 N 可能在 GPU，小 N 在 CPU——同一条曲线来源不一致 |
| Fused ops | ❌ CPU | 同 elementwise |
| Orchestration overhead | N/A | 本意就是测 wall-clock |

**与理论值的对比**（Intel Iris Xe G7 96EU, Alder Lake）：

| 参数 | 理论值 | 我们测量值 | 差距 |
|------|--------|-----------|------|
| FP32 TFLOPS | ~1.69 TFLOPS | 44 GFLOPS | 2.6%——极低，可能是 CPU |
| FP16 TFLOPS | ~3.38 TFLOPS | 44 GFLOPS | 1.3%——同上 |
| 内存带宽 | 51.2 GB/s (DDR4-3200) | 36.5 GB/s | 71%——合理（iGPU 共享内存） |

注：该 iGPU 无 matrix cores（`matrix cores: none`），MUL_MAT 通过 compute shader 执行。

### 2B.2 解决方案：绕过 Scheduler，直接调用 Backend

**原则**：Benchmark 必须精确控制每个 op 运行在哪个 backend 上，不依赖 scheduler 的启发式。

**方案**：在 CGO 层新增 `ComputeOnBackend` 方法，直接在指定 backend 上分配 buffer 并执行：

```go
// ml/backend/ggml/ggml.go — 新增

// ComputeOnBackend 在指定 backend 上直接执行计算图，绕过 scheduler。
// backendIdx: schedBackends 中的索引（0=Vulkan, 1=CPU 等）
// 步骤：
//   1. 获取目标 backend 的 buffer type
//   2. ggml_backend_alloc_ctx_tensors(ctx, buft) 分配 tensor 到 GPU buffer
//   3. 将输入数据 copy 到 GPU buffer（ggml_backend_tensor_set）
//   4. ggml_backend_graph_compute(backend, graph) 直接执行
func (c *Context) ComputeOnBackend(backendIdx int) {
    // ...
}
```

**影响的所有 benchmark 步骤**：

| 步骤 | 当前实现 | 改造后 |
|------|---------|--------|
| Peak TOPS | `ctx.Compute(out)` → scheduler | `ctx.ComputeOnBackend(0)` → 直接 Vulkan |
| Peak Bandwidth | 同上 | 同上 |
| 1D elementwise | `measureOpGPU` → scheduler | `measureOpGPU` → 直接 Vulkan |
| MUL_MAT reference | 同上 | 同上 |
| Fused ops | 同上 | 同上 |
| Orchestration overhead | wall-clock，保持 scheduler | 不变（测的就是 scheduler 开销） |

### 2B.3 Benchmark 控制流统一

当前 `RunBenchmark` 的 `--ops`/`--skip-hwchar` 实现是分散的条件判断（到处 patch）。
改造为 **work plan 模式**：

```go
func RunBenchmark(backend ml.Backend, ops []string, dtypes []string, cfg BenchmarkConfig) (*Profile, error) {
    // 1. Build work plan: 根据参数确定要执行的步骤列表
    plan := buildBenchmarkPlan(ops, dtypes, cfg)
    // plan = [{Step: "hwchar"}, {Step: "op", Op: "ADD", Dtype: "f32"}, ...]

    // 2. Execute plan: 统一循环执行每个步骤
    for _, step := range plan {
        switch step.Type {
        case StepHWChar:
            // ...
        case StepOperator:
            // ...
        case StepFusedOp:
            // ...
        case StepOverhead:
            // ...
        }
    }
}
```

### 2B.4 与 Estimate 的关系

Estimate 侧计划自己实现 backend 分配（graph nodes → backend），不需要真实执行，所以不受此问题影响。Benchmark 侧需要真实执行，因此必须控制 buffer 分配和 backend 选择。

两侧保持一致的是 `BackendCapabilities`——benchmark 用它决定使用哪个 backend 的 buffer，estimate 用它决定 op 应该查询哪个 backend 的 profile 数据。

## 3. Benchmark 改造

### 3.1 GPU Timestamp C API（核心改动）

**目标**：让 Go 侧能读取 Vulkan op 的 GPU 执行时间，而非 wall-clock。

**C 层新增**（`llama/ggml/src/ggml-vulkan/ggml-vulkan.cpp`）：

```c
// 新增公开 API：获取最近一次 graph_compute 的 per-op GPU 时间
// 类似已有的 vk_perf_logger，但通过结构化 API 返回而非 stderr

struct ggml_vk_op_timing {
    const char * op_name;     // e.g. "MUL_MAT_VEC", "RMS_NORM_MUL"
    int node_idx;
    float gpu_time_us;
};

// 返回最近一次 graph_compute 的 per-op GPU 时间列表
// n_timings: 输出参数，返回条目数
// 调用者不拥有返回的内存（由 backend 管理）
GGML_API struct ggml_vk_op_timing * ggml_vk_get_op_timings(
    ggml_backend_t backend, int * n_timings);

// 启用/禁用 GPU timestamp 收集（默认关闭，有微小性能开销）
GGML_API void ggml_vk_enable_timestamps(ggml_backend_t backend, bool enable);
```

**Go CGO bindings**（`ml/backend/ggml/ggml.go`）：

```go
// EnableGPUTimestamps 启用 GPU timestamp 收集。
func (b *Backend) EnableGPUTimestamps(enable bool) {
    // 找到 Vulkan backend handle，调用 C.ggml_vk_enable_timestamps
}

// GetOpTimings 返回最近一次 Compute 的 per-op GPU 时间。
func (b *Backend) GetOpTimings() []OpTiming {
    // 调用 C.ggml_vk_get_op_timings，转换为 Go slice
}

type OpTiming struct {
    OpName    string
    NodeIdx   int
    GPUTimeUs float64
}
```

**Benchmark 使用方式**（`perf/bench.go`）：

```go
func measureOpGPU(backend ml.Backend, op string, gridPoint []int64, 
    computeDtype string, cfg BenchmarkConfig) LatencyPoint {
    // 1. 构建 graph（和现有 measureOp 相同）
    // 2. EnableGPUTimestamps(true)
    // 3. ctx.Compute(out)  // 执行 + 收集 timestamp
    // 4. timings := backend.GetOpTimings()
    // 5. 从 timings 中提取目标 op 的 GPU 时间
    // 6. 返回 GPU 时间（非 wall-clock）
}
```

**关键**：使用 GPU timestamp 后，dispatch overhead 自动消除。MUL_MAT 在 N=1 时自动测到 MUL_MAT_VEC kernel（因为 Vulkan 自动选择），且 timing 的 op_name 会反映实际执行的 kernel 名称。

### 3.2 MUL_MAT_VEC Benchmark

**不需要单独的 benchmark entry**。GPU timestamp API 返回的 `op_name` 字段已包含实际 kernel 名称：
- N>8 → `op_name = "MUL_MAT"` 
- N≤8 → `op_name = "MUL_MAT_VEC"`

Benchmark 对 MUL_MAT 采样时覆盖 N=1 到 N=4096，GPU timestamp 自动区分两种 kernel。Profile 存储时按 `op_name` 分开：

```json
{
  "operators": [
    {"op": "MUL_MAT", "weight_dtype": "f16", "points": [/* N>8 的点 */]},
    {"op": "MUL_MAT_VEC", "weight_dtype": "f16", "points": [/* N≤8 的点 */]}
  ]
}
```

但这里有一个问题：当前 benchmark 对 MUL_MAT 使用 roofline + efficiency constants（Section 5A of v2 spec），而非直接测量每个 shape。GPU timestamp 方案改变了这个假设——有了准确的 per-op 时间，我们可以重新考虑是否继续用 roofline。

**决策**：
- **MUL_MAT（N>8，prefill）**：保持 roofline extrapolation。Prefill 的 shape 空间太大，仍需 roofline 泛化。但 efficiency constants 从 GPU timestamp 数据提取（更准确）。
- **MUL_MAT_VEC（N≤8，decode）**：**直接测量**。Decode 只有 N=1（偶尔 N=2-8），shape 空间小，直接在 reference curve (M=K=4096) 上测几个点即可。GPU timestamp 消除了 dispatch overhead，测量值可信。

### 3.3 Fused Op Benchmark

需要 benchmark fused kernel 的真实性能。方法：构建包含 fusable pattern 的小图，用 GPU timestamp 读取融合后的 kernel 时间。

```go
// perf/registry.go — 新增 fused op entries

"RMS_NORM_MUL": {
    Dimensions: []string{"N"},
    CreateInputs: func(ctx ml.Context, dtypeStr string, gridPoint []int64) []ml.Tensor {
        // 创建 RMS_NORM 的输入 + MUL 的第二个输入（weight/scale）
        N := gridPoint[0]
        input := randomTensor(ctx, ml.DTypeF32, int(N))
        scale := randomTensor(ctx, ml.DTypeF32, int(N))
        return []ml.Tensor{input, scale}
    },
    Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
        normed := in[0].RMSNorm(ctx, nil, 1e-5)
        return normed.Mul(ctx, in[1])
    },
},
```

当 Vulkan 收到包含 `RMS_NORM` 后跟 `MUL` 的图时，`ggml_vk_build_graph` 会自动融合它们。GPU timestamp 返回的 op_name 为 `"RMS_NORM_MUL"`，时间为融合后的单 kernel 时间。

**前提验证**：融合依赖运行时条件（设备能力、buffer 对齐等）。实现时需要先用 GPU timestamp 确认 benchmark 图确实触发了融合（检查返回的 op_name 是否包含 fused 名称）。如果 2-op 图不触发融合，可能需要构建更大的图来满足融合条件，或者直接用独立 op 的 GPU timestamp 数据（fused 和 unfused 的差异由 estimate 端 fusion 模拟处理）。

需要新增的 fused benchmarks：
1. `RMS_NORM_MUL` — RMS_NORM + MUL（最常见，每层 2 次）
2. `RMS_NORM_MUL_ROPE` — RMS_NORM + MUL + ROPE（attention Q/K）
3. `MUL_MAT_ADD` — MUL_MAT + ADD（有 bias 的线性层）

注意：融合是否发生取决于 backend。CPU 不融合 → 同样的图会返回独立 op 的 timestamp。这由 `BackendCapabilities.FusionRules` 控制 —— benchmark 和 estimate 都参考它。

### 3.4 CPU Orchestration Overhead Benchmark

**目标**：测量 Vulkan graph compute 的 CPU 侧固定开销（command buffer 构建、submit、fence wait 等），作为 end-to-end estimate 的加项。

**方法**：构建包含 N 个极小 op 的合成图，GPU 计算时间 ≈ 0，wall-clock ≈ CPU overhead。

```go
// perf/bench.go

// benchOrchestrationOverhead 测量不同 graph 大小下的 CPU orchestration overhead。
// 构建 N 个 16-element ADD op 的图，GPU 计算 ≈ 0，wall-clock ≈ pure CPU overhead。
func benchOrchestrationOverhead(backend ml.Backend, cfg BenchmarkConfig) []LatencyPoint {
    var points []LatencyPoint
    for _, n := range []int{50, 100, 200, 300, 500} {
        // 构建 n 个链式 trivial op 的图
        // 必须链式连接，否则 Forward(last) 只追踪最后一个 op 的依赖
        ctx := backend.NewContext()
        a := randomTensor(ctx, ml.DTypeF32, 16)
        b := randomTensor(ctx, ml.DTypeF32, 16)
        last := a.Add(ctx, b)
        for i := 1; i < n; i++ {
            last = last.Add(ctx, b) // 链式：每个 op 依赖前一个的输出
        }
        ctx.Forward(last)
        
        // 测量 wall-clock（GPU 计算 ≈ 0）
        lat := convergentMeasure(func() float64 {
            start := time.Now()
            ctx.Compute(last)
            return float64(time.Since(start).Microseconds())
        }, cfg)
        
        points = append(points, LatencyPoint{
            Shape: []int64{int64(n)},
            LatencyUs: lat.LatencyUs,
            StddevUs: lat.StddevUs,
            Reps: lat.Reps,
        })
        ctx.Close()
    }
    return points
}
```

Profile 存储为：
```json
{
  "op": "ORCHESTRATION_OVERHEAD",
  "backend": "Vulkan",
  "compute_dtype": "f32",
  "dimensions": ["num_nodes"],
  "points": [
    {"shape": [50], "latency_us": 3000},
    {"shape": [100], "latency_us": 5500},
    {"shape": [300], "latency_us": 15000}
  ]
}
```

Estimate 使用时：根据图的 node 数量，从 ORCHESTRATION_OVERHEAD 曲线插值得到 CPU overhead，加到 GPU 时间总和上。

## 4. Estimate 改造

### 4.1 Op Fusion 模拟

在 `estimatePhase()` 之前，扫描 graph node 序列，将可融合的 pattern 替换为单个 fused node。

```go
// perf/fusion.go

// FusionRule 定义一个 op fusion pattern。
type FusionRule struct {
    Name     string   // fused op name: "RMS_NORM_MUL"
    Pattern  []string // op sequence: ["RMS_NORM", "MUL"]
    // Match 检查 nodes[start:start+len(Pattern)] 是否满足融合条件。
    // 返回 true 表示可以融合。
    Match    func(nodes []ml.GraphNode, start int) bool
}

// VulkanFusionRules 返回 Vulkan backend 的 fusion 规则。
// 只包含 LLM 推理中常见的 3 条核心规则，覆盖 >95% 的 decode 场景。
// 规则按 Pattern 长度降序排列，确保优先匹配更长的 pattern。
func VulkanFusionRules() []FusionRule {
    return []FusionRule{
        // 3-op pattern 必须在 2-op 之前，否则 RMS_NORM+MUL 会先匹配掉前两个 op
        {
            Name:    "RMS_NORM_MUL_ROPE",
            Pattern: []string{"RMS_NORM", "MUL", "ROPE"},
            Match: func(nodes []ml.GraphNode, i int) bool {
                return nodes[i].Backend == nodes[i+1].Backend &&
                    nodes[i+1].Backend == nodes[i+2].Backend
            },
        },
        {
            Name:    "RMS_NORM_MUL",
            Pattern: []string{"RMS_NORM", "MUL"},
            Match: func(nodes []ml.GraphNode, i int) bool {
                // 约束：同 backend，f32，连续
                return nodes[i].Backend == nodes[i+1].Backend &&
                    nodes[i].ComputeDtype == "f32"
            },
        },
        {
            Name:    "MUL_MAT_ADD",
            Pattern: []string{"MUL_MAT", "ADD"},
            Match: func(nodes []ml.GraphNode, i int) bool {
                // 仅 mat-vec（N≤8）时融合
                if len(nodes[i].InputShapes) < 2 {
                    return false
                }
                N := nodes[i].InputShapes[1][1]
                return N <= 8 && nodes[i].Backend == nodes[i+1].Backend
            },
        },
    }
}

// CUDAFusionRules / CPUFusionRules — 以后添加
func CPUFusionRules() []FusionRule { return nil } // CPU 无 fusion

// ApplyFusion 扫描 graph nodes，将可融合的 pattern 替换为 fused node。
// 返回新的 node 列表（可能比原始列表短）。
//
// 限制：当前仅检查 op 名称序列和 backend 一致性，不检查数据依赖
// （即不验证 MUL 的输入是否来自前面的 RMS_NORM）。Vulkan 原生融合会检查
// src[0] 依赖关系。在 LLM 图中，RMS_NORM 后总是跟消费其输出的 MUL，
// 所以实际误匹配的风险极低。如果未来出现误匹配，需要在 FusionRule.Match
// 中增加 InputShapes/Name 的依赖检查。
func ApplyFusion(nodes []ml.GraphNode, rules []FusionRule) []ml.GraphNode {
    if len(rules) == 0 {
        return nodes
    }
    result := make([]ml.GraphNode, 0, len(nodes))
    i := 0
    for i < len(nodes) {
        fused := false
        // 尝试按规则长度从长到短匹配（优先匹配更长的 pattern）
        for _, rule := range rules { // 已按 Pattern 长度降序排列
            pLen := len(rule.Pattern)
            if i+pLen > len(nodes) {
                continue
            }
            match := true
            for j, opName := range rule.Pattern {
                if nodes[i+j].Op != opName {
                    match = false
                    break
                }
            }
            if match && rule.Match(nodes, i) {
                // 创建 fused node：使用第一个 node 的 shape/dtype，Op 改为 fused name
                fusedNode := nodes[i]
                fusedNode.Op = rule.Name
                result = append(result, fusedNode)
                i += pLen
                fused = true
                break
            }
        }
        if !fused {
            result = append(result, nodes[i])
            i++
        }
    }
    return result
}
```

### 4.2 MUL_MAT_VEC Routing

在 `lookupLatency()` 中，根据 N 维度选择查询 MUL_MAT 还是 MUL_MAT_VEC：

```go
// perf/estimate.go — lookupLatency 修改

case "MUL_MAT":
    M, K, N := shape[0], shape[1], shape[2]
    caps := backendCaps[backend]  // 从共享的 BackendCapabilities 获取
    
    if caps.HasMulMatVec && N <= int64(caps.MulMatVecMaxN) {
        // Decode path: 查询 MUL_MAT_VEC 曲线（GPU timestamp 直接测量值）
        // MUL_MAT_VEC 用 roofline extrapolation（和 MUL_MAT 相同的模型结构，
        // 但使用独立的 efficiency constants，因为 VEC kernel 特性不同）
        lat := PredictMulMatLatency(&profile.Hardware, M, K, N, "MUL_MAT_VEC_"+mappedWdt)
        if lat > 0 {
            return lat, nil
        }
        // fallback: 用 MUL_MAT roofline（不太准但好过没有）
    }
    // Prefill path: 用现有 roofline extrapolation
    return PredictMulMatLatency(&profile.Hardware, M, K, N, mappedWdt), nil
```

### 4.3 CPU Overhead 模型

```go
// perf/estimate.go — estimatePhase 修改

func estimatePhase(profile *Profile, nodes []ml.GraphNode, 
    caps BackendCapabilities, warnings *[]string) PhaseEstimation {
    
    // 1. Apply fusion
    fusedNodes := ApplyFusion(nodes, caps.FusionRules)
    
    // 2. Sum per-op GPU time (existing logic, on fusedNodes)
    var totalGPUUs float64
    for _, node := range fusedNodes {
        // ... lookupLatency with MUL_MAT_VEC routing ...
        totalGPUUs += lat
    }
    
    // 3. Add orchestration overhead
    overheadUs := lookupOrchestrationOverhead(profile, len(nodes), caps.Name)
    totalUs := totalGPUUs + overheadUs
    
    // ...
}

// lookupOrchestrationOverhead 根据 graph node 数量查询 CPU overhead。
func lookupOrchestrationOverhead(profile *Profile, numNodes int, backend string) float64 {
    for _, c := range profile.Operators {
        if c.Op == "ORCHESTRATION_OVERHEAD" && c.Backend == backend {
            return Interpolate1D(c.Points, int64(numNodes))
        }
    }
    return 0 // CPU backend 或无数据时不加 overhead
}
```

### 4.4 End-to-End Estimate 流程

```
Input: model GGUF + profile.json

1. DiscoverBackend() → BackendCapabilities
2. buildModelGraphNodes() → prefill nodes, decode nodes (unfused GGML graph)
3. For each phase (prefill, decode):
   a. ApplyFusion(nodes, caps.FusionRules) → fused nodes
   b. For each fused node:
      - nodeToQueryShape() → op, shape, dtype
      - If MUL_MAT and N≤caps.MulMatVecMaxN → route to MUL_MAT_VEC
      - lookupLatency() → per-op GPU time
   c. Sum per-op GPU time
   d. + lookupOrchestrationOverhead(numNodes)
   e. → PhaseEstimation
4. Output: prefill latency, decode tok/s, top-ops breakdown
```

## 5. Profile Format 扩展

### 5.1 新增字段

```go
type Profile struct {
    Version   int              `json:"version"`   // 3 (bump from 2)
    Timestamp time.Time        `json:"timestamp"`
    Hardware  HardwareProfile  `json:"hardware"`
    Operators []OperatorCurve  `json:"operators"`
    // 新增：backend capabilities snapshot（仅存 flag，不存 FusionRules 函数指针）
    // FusionRules 在加载时根据 backend name 通过 GetBackendCapabilities() 重建
    BackendCaps map[string]BackendCapabilitiesJSON `json:"backend_caps,omitempty"`
}

// BackendCapabilitiesJSON 是 BackendCapabilities 的可序列化子集。
type BackendCapabilitiesJSON struct {
    Name            string `json:"name"`
    HasGPUTimestamp bool   `json:"has_gpu_timestamp"`
    HasMulMatVec    bool   `json:"has_mul_mat_vec"`
    MulMatVecMaxN   int    `json:"mul_mat_vec_max_n"`
    // FusionRules 不序列化——加载时通过 GetBackendCapabilities(Name) 重建
}
```

### 5.2 新增 OperatorCurve entries

| Op | Dimensions | FixedDims | 来源 |
|----|-----------|-----------|------|
| `MUL_MAT_VEC` | `["N"]` | `{"M":4096,"K":4096}` | GPU timestamp, N∈[1,8] |
| `RMS_NORM_MUL` | `["N"]` | — | GPU timestamp, fused kernel |
| `RMS_NORM_MUL_ROPE` | `["N"]` | — | GPU timestamp, fused kernel |
| `MUL_MAT_ADD` | `["M","K","N"]` | `{"M":4096,"K":4096}` | GPU timestamp, fused kernel |
| `ORCHESTRATION_OVERHEAD` | `["num_nodes"]` | — | Wall-clock, trivial-op graph |

## 6. Backend 可移植性

### 6.1 抽象层设计

Benchmark 和 estimate 通过 `BackendCapabilities` 适配不同 backend：

```go
func GetBackendCapabilities(backendName string) BackendCapabilities {
    switch backendName {
    case "Vulkan":
        return BackendCapabilities{
            Name:            "Vulkan",
            HasGPUTimestamp: true,
            FusionRules:     VulkanFusionRules(),
            HasMulMatVec:    true,
            MulMatVecMaxN:   8,
        }
    case "CUDA":
        return BackendCapabilities{
            Name:            "CUDA",
            HasGPUTimestamp: false, // 以后添加
            FusionRules:     nil,   // 以后调研
            HasMulMatVec:    true,
            MulMatVecMaxN:   8,     // 待验证
        }
    case "CPU":
        return BackendCapabilities{
            Name:            "CPU",
            HasGPUTimestamp: false,
            FusionRules:     CPUFusionRules(), // nil
            HasMulMatVec:    false,
            MulMatVecMaxN:   0,
        }
    default:
        return BackendCapabilities{Name: backendName}
    }
}
```

### 6.2 Benchmark 策略按 Backend 分发

```go
func measureOpForBackend(backend ml.Backend, caps BackendCapabilities, 
    op string, gridPoint []int64, dtype string, cfg BenchmarkConfig) LatencyPoint {
    if caps.HasGPUTimestamp {
        return measureOpGPU(backend, op, gridPoint, dtype, cfg)
    }
    return measureOpWallClock(backend, op, gridPoint, dtype, cfg) // 现有逻辑
}
```

CPU backend 继续使用 wall-clock（已验证准确），Vulkan 使用 GPU timestamp。

## 7. 实现计划

### Phase 1: 共享基础设施 + GPU Timestamp C API（核心修正）

1. **common.go**：`OpVariant`, `BackendCapabilities`, `BackendCapabilitiesJSON`, `DiscoverBackend()`, `GetBackendCapabilities()`
   - 这是其他所有改动的基础——benchmark 和 estimate 都依赖 BackendCapabilities
2. **C 层**：在 `ggml-vulkan.cpp` 新增 `ggml_vk_enable_timestamps()` + `ggml_vk_get_op_timings()`
   - 基于已有的 `vk_perf_logger` 基础设施（timestamp query pool 已存在）
   - 把 stderr 输出改为结构化 API 返回
3. **Go CGO**：在 `ml/backend/ggml/ggml.go` 添加 bindings
4. **ml.Backend 接口**：添加 `EnableGPUTimestamps(bool)` + `GetOpTimings() []OpTiming`
5. **bench.go**：新增 `measureOpGPU()`，Vulkan 时使用 GPU timestamp
6. **验证**：对比 GPU timestamp benchmark vs `GGML_VK_PERF_LOGGER=1` stderr 输出

### Phase 2: Op Fusion + Orchestration Overhead + MUL_MAT_VEC

1. **fusion.go**：实现 `FusionRule`, `ApplyFusion()`, `VulkanFusionRules()`
2. **registry.go**：新增 fused op benchmark entries（RMS_NORM_MUL, MUL_MAT_ADD）
3. **bench.go**：新增 `benchOrchestrationOverhead()`
4. **estimate.go**：集成 fusion + MUL_MAT_VEC routing + orchestration overhead
5. **profile.go**：Profile version bump to 3, 新增 `BackendCapabilitiesJSON`

### Phase 3: 端到端验证

1. **端到端验证**：跑 `daop-estimate qwen3:1.7b`，对比实际推理 75ms/tok
2. **目标**：estimate 误差 < 2x
3. **调优**：根据验证结果调整 fusion 规则或 overhead 模型

## 8. 预期效果

| 指标 | 修正前 | 修正后（预期） |
|------|--------|---------------|
| Decode estimate | 1338 ms/tok | ~75-150 ms/tok |
| 误差 | 18x | < 2x |
| Benchmark 时间 | ~8 min | ~10-12 min（多了 fused ops + overhead） |
| 支持的 backend | Vulkan (broken) | Vulkan (fixed), CPU (unchanged) |

## 9. E2E 验证结果 (2026-04-05)

### 9.1 Hardware Characterization (Intel Iris Xe G7 96EU, Alder Lake)

| Dtype | Peak TOPS | vs 理论值 | 备注 |
|-------|-----------|----------|------|
| f16 | 585.6 GFLOPS | ~35% of 1.69 TFLOPS | Vulkan compute shader，无 matrix cores |
| f32 | 466.6 GFLOPS | ~28% of 1.69 TFLOPS | 正常——Vulkan GEMM 效率 |
| q8_0 | 562.2 GFLOPS | ≈ f16 | 未利用 DP4A |
| q4_0 | 1569.1 GFLOPS | 3.4x f32 | 量化解压有特殊优化 |
| BW | 48.8 GB/s | 95% of 51.2 GB/s | DDR4-3200 共享内存 |

### 9.2 Estimate 精度演化

| 阶段 | Decode Estimate | 实际 | 误差 | 改动 |
|------|----------------|------|------|------|
| 初始 (wall-clock, CPU) | 1338 ms/tok | 75 ms | 18x | — |
| + GPU timestamp + Direct Backend | 668 ms/tok | 75 ms | 8.9x | 修复了 CPU vs GPU 问题 |
| + 去除 per-op OverheadUs | **272 ms/tok** | 75 ms | **3.6x** | overhead 是 per-batch，非 per-op |

### 9.3 当前 272ms Decode 分解 (qwen3:1.7b Q4_K_M)

| 类别 | 延迟 | 占比 | 实例数 | 每次 | 瓶颈 |
|------|------|------|--------|------|------|
| MUL_MAT q4_K | 107ms | 39% | 113 | 945μs | 带宽 |
| MUL_MAT q6_K | 72ms | 26% | 1 | 71688μs | 带宽 |
| MUL_MAT_ADD q6_K | 41ms | 15% | 14 | 2901μs | 带宽 |
| MUL_MAT_ADD q4_K | 35ms | 13% | 41 | 852μs | 带宽 |
| MUL_MAT f16 | 10ms | 4% | 28 | 341μs | 带宽 |
| 其他 | 9ms | 3% | — | — | — |

**全部 bandwidth-bound**（decode N=1 时 compute 只占 1-2%）。

### 9.4 剩余误差根因分析

**核心问题：bw_eff 过低**

| Dtype | bw_eff | 含义 |
|-------|--------|------|
| q4_0 | 0.096 | 只用了 peak BW 的 9.6% |
| q8_0 | 0.095 | 9.5% |
| f16 | 0.253 | 25.3% |

这导致 bandwidth-bound 操作的延迟被高估约 **4-10x**。

**根因**：reference curve 只在 M=K=4096 一个固定 shape 上采样 N=1 到 N=8M。
- 大 N 点是 compute-bound，贡献不了 bw_eff 信息
- 小 N 点太少，bw_eff 拟合不准
- 4096x4096 的 GPU tiling/occupancy 不能代表其他 shape (2048x2048, 5504x2048 等)
- N=1 时 Vulkan 走 MUL_MAT_VEC shader，和大 N 的 MUL_MAT 是完全不同的代码路径

**Uncalibrated ops**: GLU 和 SET_ROWS 未在 registry 中，产生大量警告（但对总延迟影响 <3%）。

### 9.5 下一步方向

1. **Benchmark 改进** (Phase 1F)：
   - 直接测量小 N (1,2,4,8) 的 MUL_MAT_VEC 延迟，不依赖 roofline 外推
   - 多 (M,K) shape 覆盖，不只是 4096x4096
   - MUL_MAT_VEC 独立 efficiency constants

2. **Estimate 速度优化** (Phase 1G)：
   - 模型加载产生 ~15k 行 DXGI 日志，耗时数十秒
   - 研究避免真实内存分配的轻量方案

## 10. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| C API 修改需要 patch llama.cpp | 维护成本 | 最小化修改，复用已有 timestamp 基础设施 |
| Fusion 规则不完整 | estimate 仍有误差 | 3 条核心规则覆盖 >95% decode；可增量添加 |
| GPU timestamp 在某些 Vulkan driver 上不工作 | 特定硬件失败 | Fallback 到 wall-clock + 警告 |
| Orchestration overhead 与 graph 结构相关 | 简单线性模型不够准 | 先用线性模型，收集数据后评估是否需要更复杂模型 |

## 11. 实现中的 Bug 修复记录

### 11.1 allocMemory guard bug（预先存在）

`Context.FromFloats/Zeros/FromInts` 有 `if c.b.allocMemory` guard，当 `AllocMemory=false` 时跳过数据写入。但 `newTensor` 总是分配有效 buffer，guard 不应阻止写入。

**修复**：替换为 `tensor_can_set()` C helper（检查 `t->buffer->iface.set_tensor != NULL`），对有效 buffer 写入数据，对 GPU buffer 无 set_tensor 回调时跳过（estimate 路径）。

### 11.2 benchPeakTOPS q8_0 Vulkan 崩溃

两个 tensor 都创建为 q8_0 类型，Vulkan MUL_MAT 要求 activation 为 f32。

**修复**：activation 始终用 f32，weight 用目标 dtype。

### 11.3 FLASH_ATTN_EXT Vulkan 崩溃

默认路径将所有 3 个 tensor 都创建为 f16，Vulkan 需要 Q=f32。

**修复**：添加自定义 CreateInputs，Q=f32, K/V=f16（匹配真实推理）。

### 11.4 per-op OverheadUs 高估

Reference curve 拟合的 OverheadUs (~1827μs for q4_0) 在 estimate 时逐 op 累加，113 个 MUL_MAT × 1827μs = 206ms 纯 overhead。但实际推理中整个 graph 一次 dispatch，GPU kernel 异步流水执行，per-op overhead ≈ 0。

**修复**：roofline 预测中去掉 per-op OverheadUs，dispatch overhead 由 ORCHESTRATION_OVERHEAD 一次性加入。
