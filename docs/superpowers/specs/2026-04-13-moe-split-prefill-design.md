# MoE 感知层内拆分（Phase 1）设计规格

**目标**：将 Qwen3-Coder-Next 80B Q4_K_M 在 Arrow Lake + RTX 3090（24GB VRAM）上的
prefill 1k tokens 延迟从 ~2.0s 降至 ~1.55s（-22%），通过把所有层的 attention/dense
权重常驻 GPU，MoE expert 权重按 VRAM 余量部分留 GPU、其余留 CPU 按需拷贝计算。

**分支**：`feat/moe/split-cpu`（基于 main，与 iGPU 工作无关）

**参考**：`docs/perf/2026-04-10-moe-split-prefill-optimization.md` §4

---

## 1. 背景与收益来源

当前 Ollama 对 Qwen3-Next-80B 的默认分配：

```
~20 层整层 → GPU（dense + MoE 全在 VRAM）
~28 层整层 → CPU（dense + MoE 全在内存，attention 也在 CPU 跑）
```

每层权重构成极度不对称（来自 GGUF metadata）：

| 组件 | 大小 | 占比 |
|------|------|------|
| ffn_gate_exps + ffn_up_exps + ffn_down_exps | ~996 MB | 97% |
| Attention/SSM + shared expert + norms | ~25 MB | 3% |
| **整层合计** | **~1.02 GB** | 100% |

全模型 dense 权重仅 ~1.2 GB，可轻松全部放入 24GB VRAM。

Phase 1 的改变：

```
所有 48 层 dense（attention/SSM）→ GPU（~1.2 GB 总量）
前 ~22 层 MoE experts              → GPU（占用剩余 ~22 GB）
后 ~26 层 MoE experts              → CPU（按需拷贝到 GPU 计算）
```

加速来源：目前 28 层 attention 跑在 CPU（~16ms/层 × 28 = 448ms），改后全部跑 GPU
（~2.5ms/层 × 48 = 120ms）。MoE expert 的计算无论常驻还是临时拷贝，都在 GPU 上执行。

---

## 2. 架构总览

### 改动文件

```
ml/device.go              ← DeviceMemory 新增 MoEWeights 字段
ml/backend.go             ← BackendParams 新增 MoEGPULayers 字段
envconfig/config.go       ← 新增 OLLAMA_MOE_GPU_LAYERS 环境变量
ml/backend/ggml/ggml.go   ← (1) probe 追踪 MoEWeights  (2) 正式分配路由
llm/server.go             ← buildLayout() 两轮分配 + createLayout() 签名变更
```

### 执行流程

```
server.go iteration loop
  └─ ggml.go:New(AllocMemory=false)     ← probe：追踪每层 Weights + MoEWeights
  └─ buildLayout(memory)
       ├─ [Pass 1] 检查：所有 48 层 dense 是否放得进 VRAM（预期 ~1.2GB << 24GB）
       └─ [Pass 2] 贪心：用剩余 VRAM 前向填充 MoE 层，得到 moeGPUCount K
  └─ ggml.go:New(AllocMemory=true, GPULayers=0..47, MoEGPULayers=0..K-1)
       ├─ non-MoE tensor → assignLayer()      → GPU（所有 48 层）
       └─ MoE expert tensor → assignMoELayer() → GPU(0..K-1) / CPU(K..47)
```

Bootstrap 说明：`MoEWeights` 在每次 probe 中无条件追踪（不依赖先验的分配决策），
第一次 probe 结束后 `buildLayout()` 即可获得完整的 dense/MoE size breakdown，
无需额外 probe 轮次。

---

## 3. 数据模型变更

### `ml/device.go` — DeviceMemory

```go
type DeviceMemory struct {
    DeviceID
    Name string

    // Weights: 每层全部权重大小（dense + MoE），语义不变。
    Weights []uint64

    // MoEWeights: 每层 MoE expert tensor 大小（Weights 的子集）。
    // 非 MoE 模型全零。tensor 匹配规则：\.ffn_(up|down|gate)_(ch_)?exps$
    MoEWeights []uint64

    Cache []uint64
    Graph uint64
}
```

`Size()`、`memoryPresent()` 等现有方法无需修改。

### `ml/backend.go` — BackendParams

```go
type BackendParams struct {
    AllocMemory    bool
    NumThreads     int
    GPULayers      GPULayersList  // 语义不变：所有 48 层 dense → GPU
    MoEGPULayers   GPULayersList  // 新增：前 K 层 MoE 也在 GPU（其余 MoE → CPU）
    FlashAttention FlashAttentionType
}
```

`MoEGPULayers` 为空时行为与现有完全相同（零风险回退）。

### `envconfig/config.go` — 用户配置

```go
// OLLAMA_MOE_GPU_LAYERS 控制多少层的 MoE expert 权重常驻 GPU。
//   -1（默认）= 自动从剩余 VRAM 推算
//    0         = 禁用 MoE split（可用于 baseline 对比）
//   >0         = 强制指定层数（等价于 llama.cpp --n-cpu-moe 的互补参数）
var MoeGpuLayers = Int("OLLAMA_MOE_GPU_LAYERS", -1)
```

与 llama.cpp `--n-cpu-moe` 的关系：`n-cpu-moe = total_layers - OLLAMA_MOE_GPU_LAYERS`，
语义互补，视角相反。

### `llm/server.go` — createLayout / buildLayout 签名

```go
// 变更前
func (s *llmServer) createLayout(...) (ml.GPULayersList, error)
func (s *llmServer) buildLayout(...) (ml.GPULayersList, []uint64)

// 变更后
func (s *llmServer) createLayout(...) (ml.GPULayersList, ml.GPULayersList, error)
func (s *llmServer) buildLayout(...) (ml.GPULayersList, ml.GPULayersList, []uint64)
//                                    ^dense gpuLayers  ^moeGPULayers
```

---

## 4. buildLayout() 两轮分配逻辑

```go
func (s *llmServer) buildLayout(systemGPUs, memory, requireFull, backoff) (
    gpuLayers    ml.GPULayersList,
    moeGPULayers ml.GPULayersList,
    layers       []uint64,
) {
    // ── 计算每层 dense / MoE 大小 ─────────────────────────────────────
    denseSize := make([]uint64, totalLayers)
    moeSize   := make([]uint64, totalLayers)
    for i := range totalLayers {
        totalW   := sum(gpu.Weights[i]) + cpu.Weights[i]
        totalMoE := sum(gpu.MoEWeights[i]) + cpu.MoEWeights[i]
        denseSize[i] = totalW - totalMoE
        moeSize[i]   = totalMoE
    }

    // ── MoE split 适用性检查 ──────────────────────────────────────────
    isMoEModel := slices.ContainsFunc(moeSize, func(s uint64) bool { return s > 0 })
    if !isMoEModel {
        slog.Debug("moe split: skipped, no MoE tensors detected")
        return existingAssignLayers(...), nil, layers
    }

    // ── Pass 1：dense-only 是否放得进 VRAM ───────────────────────────
    totalDense := sum(denseSize) + sum(cache) + overhead
    if totalDense > availableVRAM {
        slog.Warn("moe split: dense weights exceed VRAM, falling back",
            "dense_total", format.HumanBytes2(totalDense),
            "vram",        format.HumanBytes2(availableVRAM))
        return existingAssignLayers(...), nil, layers
    }

    slog.Info("moe split: dense weights fit, activating split",
        "dense_total",  format.HumanBytes2(totalDense),
        "vram_total",   format.HumanBytes2(availableVRAM),
        "vram_for_moe", format.HumanBytes2(availableVRAM-totalDense))

    gpuLayers = allLayersOnGPU(systemGPUs)  // Pass 1 结果：所有 48 层 dense → GPU

    // ── Pass 2：MoE 层贪心填充（前向） ───────────────────────────────
    moeGPUCount := envconfig.MoeGpuLayers()
    source := "auto"
    if moeGPUCount >= 0 {
        source = "user-override"
        if moeGPUCount > totalLayers {
            slog.Warn("moe split: OLLAMA_MOE_GPU_LAYERS exceeds total layers, clamped",
                "requested", moeGPUCount, "clamped", totalLayers)
            moeGPUCount = totalLayers
        }
    } else {
        remainingVRAM := availableVRAM - totalDense
        moeGPUCount = 0
        for i := range totalLayers {
            if moeSize[i] > remainingVRAM { break }
            remainingVRAM -= moeSize[i]
            moeGPUCount++
        }
    }

    slog.Info("moe split: layer budget",
        "moe_gpu_layers", moeGPUCount,
        "moe_cpu_layers", totalLayers-moeGPUCount,
        "source",         source)

    for i := range totalLayers {
        loc := "cpu"
        if i < moeGPUCount { loc = "gpu" }
        slog.Debug("moe split: layer layout",
            "layer", i,
            "dense_size", format.HumanBytes2(denseSize[i]),
            "moe_size",   format.HumanBytes2(moeSize[i]),
            "moe_loc",    loc)
    }

    moeGPULayers = buildMoEGPULayersList(systemGPUs, moeGPUCount)
    return gpuLayers, moeGPULayers, layers
}
```

### 回退策略

| 情况 | 处理 |
|------|------|
| 非 MoE 模型 | `Debug` 跳过，原有逻辑 |
| dense 超过 VRAM | `Warn` 回退，原有逻辑 |
| `OLLAMA_MOE_GPU_LAYERS=0` | `moeGPULayers` 为空，所有 MoE 留 CPU |
| `OLLAMA_MOE_GPU_LAYERS > totalLayers` | clamp + `Warn` |

---

## 5. ggml.go Tensor 路由变更

### 5.1 MoE tensor 识别

```go
// 与 llama.cpp partial_moe pattern 保持一致
var moeExpertRE = regexp.MustCompile(`\.ffn_(up|down|gate)_(ch_)?exps$`)

func isMoEExpertTensor(name string) bool {
    return moeExpertRE.MatchString(name)
}
```

### 5.2 Probe 阶段：MoEWeights 追踪

在 `createTensor()` 中，计算完 `size` 后追加：

```go
if layer >= 0 {
    btDeviceMemory[bt].Weights[layer] += uint64(size)  // 原有
    if isMoEExpertTensor(name) {
        btDeviceMemory[bt].MoEWeights[layer] += uint64(size)  // 新增
        slog.Debug("moe split: tracked MoE tensor",
            "name",   name,
            "layer",  layer,
            "size",   format.HumanBytes2(uint64(size)),
            "buffer", C.GoString(C.ggml_backend_buft_name(bt)))
    }
}
```

### 5.3 正式分配阶段：assignMoELayer

在 `assignLayer` 函数定义后新增：

```go
assignMoELayer := func(layer int) deviceBufferType {
    for _, p := range params.MoEGPULayers {
        for _, l := range p.Layers {
            if l == layer {
                for i := range requiredMemory.GPUs {
                    if requiredMemory.GPUs[i].DeviceID == p.DeviceID {
                        return gpuDeviceBufferTypes[i]
                    }
                }
                return cpuDeviceBufferType
            }
        }
    }
    return cpuDeviceBufferType
}
```

在 `switch default` 分支（`ggml.go:336`）：

```go
if layerIndex >= 0 {
    bts := layers[layerIndex].bts
    if isMoEExpertTensor(t.Name) && len(params.MoEGPULayers) > 0 {
        bts = assignMoELayer(layerIndex).bts
    }
    createTensor(tensor{source: t}, bts, layerIndex)
} else {
    createTensor(tensor{source: t}, input.bts, -1)
}
```

启动时（`AllocMemory: true`，`MoEGPULayers` 非空）输出路由摘要：

```go
for i := range layers {
    moeLocation := "cpu"
    if moeGPUSet[i] { moeLocation = "gpu" }
    slog.Info("moe split: tensor routing",
        "layer", i, "dense", "gpu", "moe", moeLocation)
}
```

---

## 6. 验证清单

### Step 1 — 确认 MoE tensor 识别

```bash
OLLAMA_DEBUG=1 ollama serve 2>&1 | grep "moe split: tracked MoE tensor"
```
预期：每层 3 条（`ffn_up_exps`/`ffn_down_exps`/`ffn_gate_exps`），共 144 条。

### Step 2 — 确认层级分配

```bash
OLLAMA_DEBUG=1 ollama serve 2>&1 | grep "moe split: layer layout"
```
预期：所有 48 层 `dense_loc=gpu`，前 ~22 层 `moe_loc=gpu`，后 ~26 层 `moe_loc=cpu`。

### Step 3 — 确认正式 tensor 路由

```bash
OLLAMA_DEBUG=1 ollama serve 2>&1 | grep "moe split: tensor routing"
```
预期：与 Step 2 一致。

### Step 4 — Benchmark 验证

```bash
# Baseline（禁用 MoE split）
OLLAMA_MOE_GPU_LAYERS=0 ollama run qwen3-coder-next /prefill_bench

# Phase 1
OLLAMA_MOE_GPU_LAYERS=-1 ollama run qwen3-coder-next /prefill_bench
```
预期：prefill 1k tokens ~2.0s → ~1.55s（-22%）。

---

## 7. 成功标准

| 指标 | 目标 |
|------|------|
| 48 层 attention/dense 全在 GPU | Step 3 日志确认 |
| ~22 层 MoE 常驻 GPU，~26 层 MoE 在 CPU | Step 2 日志确认 |
| Prefill 1k tokens 延迟 | ≤ 1.6s |
| 非 MoE 模型行为不变 | `OLLAMA_MOE_GPU_LAYERS` 未设置时无 moe split 日志 |
