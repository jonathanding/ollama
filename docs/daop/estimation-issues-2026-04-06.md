# Estimation Accuracy Issues — qwen3:1.7b (2026-04-06)

Model: qwen3:1.7b (num_heads=16, num_kv_heads=4, head_dim=128, sliding_window=256)
Hardware: Intel Arc Vulkan
Flash Attention: ON
Profile: post-NewForBench fix (fused kernel benchmark data)

Reference: [actual-vs-estimate-2026-04-06.md](actual-vs-estimate-2026-04-06.md)

## Overall Accuracy (input_length=300, post-fix)

| Phase | Estimate | Actual | Ratio | Note |
|---|---|---|---|---|
| **Prefill** | 3,999.8ms | 4,080.2ms | **0.98x** | 假象: FLASH_ATTN 高估 + MUL_MAT 低估互相抵消 |
| **Decode** | 70.4ms | 131.9ms | **0.53x** | 真实偏低，FLASH_ATTN + MUL_MAT 共同驱动 |

---

## Issue #1: FLASH_ATTN_EXT — GQA num_heads mismatch [CRITICAL]

**Severity: CRITICAL (Decode), HIGH (Prefill)**
**Affected: Prefill + Decode, all input lengths**

**Root cause**: Benchmark 只有 Q=KV=same heads 的曲线 ({4,8,16,32})。实际推理是 GQA: Q=16, KV=4。`estimate.go:45` 用 Q 的 num_heads=16 查找曲线，匹配到 Q=KV=16 的曲线（计算量远大于 Q=16,KV=4）。

| Phase | Input Len | Estimate (us) | Actual (us) | Ratio | Actual % |
|---|---|---|---|---|---|
| Prefill | 300 | 3,012,524 | 1,633,969 | **1.84x** HIGH | 40.0% |
| Decode | 300 | 6,793 | 41,440 | **0.16x** LOW | 31.4% |
| Decode | 150 | ~7,396 | 26,357 | **0.28x** LOW | 22.5% |
| Decode | 18 | ~7,396 | 13,148 | **0.56x** LOW | 16.2% |

**Why prefill overestimates but decode severely underestimates**:
- Prefill: Q=KV=16 曲线计算量大于 Q=16,KV=4 → 高估
- Decode: 可能还叠加了 decode 时 seq_kv 随 context 增长的问题，以及 captureGraph 用固定 context 导致 shapes 不准

**Fix options**:
1. Quick: 用 KV heads (4) 替代 Q heads (16) 做查找
2. Better: benchmark 增加 GQA 配置 (Q≠KV)
3. Explore: 几何均值 sqrt(Q*KV) heads

---

## Issue #2: MUL_MAT systematic underestimate [HIGH]

**Severity: HIGH (Prefill), MEDIUM (Decode)**
**Affected: All quantized types, worsens with input_length**

| Op | Phase | @18 | @150 | @300 | Actual % @300 |
|---|---|---|---|---|---|
| MUL_MAT q4_K | Prefill | 0.48x | 0.54x | 0.37x | 39.5% |
| MUL_MAT q6_K | Prefill | 0.45x | 0.48x | 0.41x | 10.8% |
| MUL_MAT f16 | Prefill | 0.73x | 0.72x | 0.38x | 5.5% |
| MUL_MAT q4_K | Decode | 0.85x | 0.62x | 0.63x | 33.1% |
| MUL_MAT q6_K | Decode | 0.70x | 0.50x | 0.50x | 13.3% |
| MUL_MAT f16 | Decode | 1.10x | 0.89x | 1.24x | 2.6% |

**Possible causes**:
- captureGraph 用固定 seq_len=512 / 2048，导致 shapes 与实际 input_length 不匹配
- benchmark 曲线在大 N 时外推不准
- MUL_MAT_ADD (fused) 与 MUL_MAT 之间的 routing 问题

**Note**: @300 的数据是修复前 estimate；修复后 snapshot 显示 MUL_MAT q4_K prefill 仍为 0.37x (601,569 vs 1,613,496 us)。

---

## Issue #3: Small ops systematic overestimate [LOW]

**Severity: LOW (individually <2% of actual)**
**Affected: Decode primarily**

| Op | Phase | Ratio Range | Actual % |
|---|---|---|---|
| RMS_NORM_MUL | Decode | 1.78x~2.97x | 1.1-1.4% |
| RMS_NORM_MUL_ROPE | Decode | 1.75x~3.62x | 0.9-1.2% |
| SET_ROWS | Both | 2.35x~6.43x | 0.1-0.4% |
| ADD | Both | 2.26x~3.00x | 0.0-0.4% |

Cumulative: ~10% 虚假延迟 in decode estimates。

---

## Issue #4: Ops with zero estimate [LOW]

| Op | Actual % | Note |
|---|---|---|
| GLU | 0.2-0.6% | 未在 registry 注册，prefill@300 有 21,867us |
| GET_ROWS | <0.1% | — |
| CPY | <0.1% | — |

---

## Priority Ranking

| Rank | Issue | Impact | Direction |
|---|---|---|---|
| **1** | FLASH_ATTN Decode underestimate (0.16x) | Decode 总体偏低主因 | ↓↓↓ |
| **2** | MUL_MAT q4_K Prefill underestimate (0.37x) | Prefill 最大 op; 与 FLASH_ATTN 高估巧合抵消 | ↓↓ |
| **3** | FLASH_ATTN Prefill overestimate (1.84x) | 掩盖 MUL_MAT 问题 | ↑↑ |
| **4** | MUL_MAT q6_K/f16 Prefill underestimate | 加剧 prefill 低估 | ↓ |
| **5** | MUL_MAT Decode underestimate (0.50-0.85x) | Decode 偏低次要原因 | ↓ |
| **6** | Small ops overestimate (~2-6x) | 部分抵消 MUL_MAT 低估 | ↑ |
| **7** | GLU/GET_ROWS zero estimate | 影响很小 | — |

## Issue #5: FLASH_ATTN_EXT — CachePadding 导致 seqKV 膨胀 [HIGH]

**Severity: HIGH (Prefill)**
**Affected: Prefill, 所有 input lengths**
**发现时间: 2026-04-06 Session 37**

**现象**: GQA 修复后 FLASH_ATTN prefill 仍然 2.08x 高估（之前 1.84x）。

**根因**: `estimate.go:captureGraph` 调用 `cache.StartForward(ctx, batch, reserve=true)`。`reserve=true` 把 `curCellRange` 设为整个 cache 尺寸。而 `cache.Init(capacity=300)` 后经过 `roundUp(300, CachePadding=256) = 512`，所以 K tensor 的 seqKV 维度 = 512 而非 300。

**完整链路**:
```
estimate.go:525  cache.Init(backend, DTypeF16, 1, inputLength=300, inputLength=300)
  → causal.go:170  cacheSize = 1 × 300 = 300
  → causal.go:175  roundUp(300, CachePadding=256) = 512
  → c.cells = [512]cacheCell

estimate.go:547  cache.StartForward(ctx, batch, reserve=true)
  → causal.go:233-241  reserve=true 分支:
      curCellRange = {min: 0, max: 511}

  → causal.go:362-366  buildMask():
      length = 512
  → causal.go:418  cache.Get():
      cachedSize = curMask.Dim(0) = 512
      K tensor shape: [128, 512, 8, 1]   ← seqKV = 512

nodeToQueryShape 提取: shape = [seqQ=300, seqKV=512, Q=16, KV=8]
```

**为什么实际推理也是 512 但延迟更低**: 实际推理的 buildMask 也做了 roundUp(300, 256)=512 对齐，K tensor 也是 512。但 mask 把多余 212 个位置标为 -inf，GPU flash attention kernel 对 masked 位置做了优化（early exit / skip），实际计算量接近 seqKV=300。

**Benchmark 对比**: benchmark 用精确 shape [n,n] 跑 kernel，没有 padding/masking，所以 benchmark 的 [300,512] 附近数据点反映的是真实的 512 计算量。

**修复**: `InterpolateFlashAttn` 的 prefill 分量改用 `seqQ` 而非 `seqKV` 插值（`interpolate.go:369`）。benchmark prefill 数据是方阵 [n,n]，GPU kernel 高效跳过 masked padding，所以 [seqQ, seqKV] 的有效计算量接近 [seqQ, seqQ]。同时添加单调性下限，确保 blend 结果不低于 decode 延迟。预期从 2.08x → ~0.84x。

---

## Key Insight

Prefill 总体 0.98x 是两个大偏差互相抵消的假象 (FLASH_ATTN 1.84x + MUL_MAT 0.37x)。
Decode 0.53x 是真实的系统性偏低，由 FLASH_ATTN (0.16x) 和 MUL_MAT (0.50-0.63x) 共同驱动。
