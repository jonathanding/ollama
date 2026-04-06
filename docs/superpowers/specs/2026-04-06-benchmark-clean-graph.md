# Benchmark Clean Graph: 消除数据准备 ops 对测量的污染

## 问题

当前 `measureOpGPU` 使用 `randomTensor` 创建非 f32 dtype 的输入 tensor 时，`randomTensor` 会在 graph 中插入 `Cast`（GGML_OP_CPY）op 将 f32 数据转换为目标 dtype。这导致 benchmark graph 中除了目标 op 外还包含数据准备 ops，而 `GetOpTimings()` 返回所有 ops 的 GPU timing 后被 `measureOpGPU` 全部求和（bench.go:235-239），导致测量值包含了不应计入的 Cast 开销。

**影响范围：** 所有非 f32 dtype 的 benchmark 都受影响（f16, q4_0, q8_0），包括 MUL_MAT、MUL_MAT_ADD、ROPE 以及任何使用 `randomTensor` 且 dtype != f32 的 op。

**实测证据：** q4_0 MUL_MAT M=2048, K=2048, N=1:
- `GGML_VK_PERF_LOGGER=1` benchmark: MUL_MAT_VEC kernel = ~100μs，Total（含 CPY）= ~560μs
- `GGML_VK_PERF_LOGGER=1` 真实推理: MUL_MAT_VEC = ~117μs（无额外 CPY）
- f32 benchmark: MUL_MAT_VEC = ~275μs，Total = ~275μs（无 Cast，无差异）

这是 qwen3:1.7b decode 预测误差 3.9x（预测 294ms vs 实际 75ms）的主要原因。

## 设计原则

**Benchmark graph 应该只包含被测 op。** 输入数据应在 graph 外部准备好，作为 leaf tensor 传入。这是 benchmark 的标准做法：准备 mock data → 测量真实区间。

## 方案

### 核心机制：两阶段数据准备

1. **Prep 阶段**：在独立 context 中创建 f32 随机数据，Cast 到目标 dtype，执行 graph，通过 `Bytes()` 读回 quantized bytes 到 CPU
2. **Benchmark 阶段**：用 `ctx.Input().FromBytes(dtype, bytes, shape...)` 创建 leaf tensor，graph 中只包含目标 op

CPU→GPU data transfer 不影响 GPU timestamp 测量（timestamp 只记录 kernel 执行时间），且在 warmup 阶段完成。

### Bytes Cache

同一 (dtype, shape) 的 quantized bytes 可跨多个 benchmark 共享。例如 MUL_MAT 的 9 (M,K) 对 × 3 N 值，同一 (M,K,dtype) 的 weight bytes 只需准备一次。

```go
type prepKey struct {
    Dtype string
    Shape string // e.g. "2048,8192"
}

// Shared across benchmarks for same dtype+shape
type BytesCache map[prepKey][]byte
```

## Phases

### Phase 1: PoC — MUL_MAT clean graph 验证

**目标：** 只修改 MUL_MAT 的 benchmark 路径，验证消除 Cast 后测量精度是否显著改善。

**成功标准：** qwen3:1.7b decode 预测误差从 3.9x 降到 <2x。

**修改范围：**

1. **新增 `materializeTensor` 函数**（registry.go）

   ```go
   // materializeTensor creates quantized tensor data by executing Cast in a
   // separate context, then reading back the bytes. The returned bytes can be
   // used with ctx.Input().FromBytes() to create a clean leaf tensor.
   func materializeTensor(backend ml.Backend, dt ml.DType, shape ...int) []byte {
       prepCtx := backend.NewContext()
       defer prepCtx.Close()

       f32Tensor := randomTensor(prepCtx, ml.DTypeF32, shape...)
       castTensor := f32Tensor.Cast(prepCtx, dt)
       prepCtx.Forward(castTensor)
       prepCtx.ComputeOnBackend(0, castTensor)
       return castTensor.Bytes()
   }
   ```

2. **扩展 `CreateInputs` 签名，增加 `backend` 参数**（registry.go）

   ```go
   CreateInputs func(ctx ml.Context, backend ml.Backend, dtypeStr string, gridPoint []int64) []ml.Tensor
   ```

   MUL_MAT 的实现改为：对非 f32 dtype 使用 `materializeTensor` + `FromBytes` 创建 leaf tensor；f32 path 不变。其他 ops 的 CreateInputs 暂时忽略 backend 参数。

3. **修改 `measureOpGPU`**（bench.go）

   将 `backend` 传递给 `CreateInputs`。

4. **验证**

   - 运行 `daop-bench --ops MUL_MAT`
   - 运行 `daop-estimate qwen3:1.7b`
   - 对比新旧结果

**不修改：** 其他 op 的 benchmark 路径、profile 格式、estimation 逻辑。

### Phase 2: 全面推广到所有 ops 和 dtypes

**目标：** 将 clean graph 方案应用到所有受影响的 benchmark。

**前提：** Phase 1 PoC 验证通过（误差 <2x）。

**修改范围：**

1. **通用化 `materializeTensor`**

   添加 `BytesCache` 支持，避免相同 (dtype, shape) 重复准备。在 benchmark session 开始时创建 cache，传递给各 op 的 benchmark 函数。

   ```go
   func materializeTensorCached(backend ml.Backend, cache BytesCache,
       dt ml.DType, shape ...int) []byte {
       key := prepKey{dtypeToString(dt), fmt.Sprint(shape)}
       if b, ok := cache[key]; ok {
           return b
       }
       b := materializeTensor(backend, dt, shape...)
       cache[key] = b
       return b
   }
   ```

2. **修改 `randomTensor` 或新增 `randomLeafTensor`**

   通用的 leaf tensor 创建函数，对 f32 直接 FromFloats，对其他 dtype 使用 cache 中的 bytes + FromBytes。所有 op 的 CreateInputs 和默认路径统一使用此函数。

3. **受影响的 ops 清单**

   | Op | 非 f32 输入 | 说明 |
   |---|---|---|
   | MUL_MAT | weight (q4_0, q8_0, f16) | Phase 1 已修复 |
   | MUL_MAT_ADD | weight (q4_0, q8_0, f16) | 同 MUL_MAT |
   | ROPE | input (f16) | CreateInputs 使用 randomTensor |
   | 1D ops (ADD, MUL, SILU...) | 如果被要求 benchmark 非 f32 | 默认路径使用 randomTensor |
   | Fused ops (RMS_NORM_MUL...) | 通常 f32，不受影响 | 确认无 Cast 即可 |

4. **测试**

   - 所有 op 的 benchmark 输出验证：graph 中无 CPY timing
   - E2E 验证：多个模型的 estimate 精度

### Phase 3: 清理和文档

1. 删除旧的 `randomTensor` 中 Cast 路径（如果不再需要）
2. 更新 benchmark 设计文档
3. 更新 compact-snapshot.md

## 关键 API 依赖

| API | 用途 | 位置 |
|---|---|---|
| `backend.NewContext()` | 创建 prep context | ml/backend.go |
| `tensor.Cast(ctx, dtype)` | f32 → quantized 转换 | ml/backend/ggml/ggml.go:1550 |
| `ctx.Forward(t)` | 构建 graph | ml/backend/ggml/ggml.go |
| `ctx.ComputeOnBackend(0, t)` | 执行 graph 并 materialize | ml/backend/ggml/ggml.go:1117 |
| `tensor.Bytes()` | 读回 GPU tensor 数据到 CPU | ml/backend/ggml/ggml.go:1468 |
| `ctx.Input().FromBytes(dtype, bytes, shape...)` | 创建 quantized leaf tensor | ml/backend/ggml/ggml.go:1368 |

## 风险

1. **`Bytes()` 在 `ComputeOnBackend` 后是否正常工作？** — ✅ 已验证，Phase 1 PoC 通过。
2. **FromBytes 的 byte 数量是否匹配？** — ✅ 已验证，q4_0 64×32 = 1152 bytes 精确匹配。
3. **GPU context 资源耗尽？** — ✅ 无问题，Prep context 在 `defer Close()` 后立即释放。

## 验证结果

### Phase 1 验证 (2026-04-06)

**PoC 目标：** qwen3:1.7b decode 预测误差从 3.9x 降到 <2x。

**结果：** 预测 56.0ms/tok，实际 ~75ms/tok = **1.34x 误差** ✅（低估，因为只有 MUL_MAT/MUL_MAT_ADD 校准，其他 ops 跳过）

### Phase 2 验证 (2026-04-06)

**推广所有 ops 后 E2E 结果：**
- 预测 64.4ms/tok (16 tok/s)
- `ollama serve` 实测 decode: ~62ms/tok (via API)
- VK perf logger 实测 GPU 总时间: ~61.8ms/tok

**总体误差: 1.04x** ✅

### Per-Op 对比 (VK Perf Logger vs Estimate, decode N=1)

| Op | dtype | count | Estimate (us) | 实测 (us) | 比率 | 说明 |
|---|---|---|---|---|---|---|
| MUL_MAT_VEC | q4_K | 113x | 25,513 | 24,142 | 1.06x ✅ | 主力 op (39.6%) |
| MUL_MAT_VEC | q6_K (vocab) | 1x | 10,941 | 12,276 | 0.89x ⚠️ | M=151936 超出校准网格 (max 8192)，外推 |
| MUL_MAT_ADD | q4_K | 41x | 7,867 | 8,972 | 0.88x ⚠️ | |
| MUL_MAT_ADD | q6_K | 14x | 6,830 | 4,227 | 1.62x ❌ | q6_K→q8_0 映射误差 |
| MUL_MAT_VEC | f16 | 28x | 4,075 | 2,536 | 1.61x ❌ | M=1024 插值精度不足 |
| FLASH_ATTN | f16 | 28x | ~7,700 | 7,765 | ~1.0x ✅ | |
| RMS_NORM_MUL | f32 | 57x | ~800 | 814 | ~1.0x ✅ | |
| RMS_NORM_MUL_ROPE | f32 | 56x | ~600 | 615 | ~1.0x ✅ | |
| 未校准 ops | — | — | 0 | ~453 | — | SET_ROWS, GLU, CPY |
| **总计** | | | **64,396** | **~61,797** | **1.04x** | |

### 关键发现

1. **总体精度优秀**：1.04x 误差，远超 <2x 目标
2. **q4_K MUL_MAT（主力 39.6%）**：1.06x，非常准确
3. **vocab 外推 (M=151936)**：超出校准网格上限 20 倍，11% 误差，外推效果良好
4. **dtype 映射是剩余最大误差源**：q6_K→q8_0 和 f16 的映射导致 60% per-op 误差，但因方向相反互相抵消
5. **未校准 ops 影响极小**：SET_ROWS + GLU + CPY 总计 ~453us，占 0.7%
