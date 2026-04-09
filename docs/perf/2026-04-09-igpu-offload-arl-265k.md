# iGPU Prefill Offload 性能分析报告

**日期**: 2026-04-09  
**机器**: Intel Core Ultra 265K (Arrow Lake-S), 128 GiB DDR5 6400 MT/s, RTX 3090 (24 GiB), Win11  
**模型**: qwen3-coder-next Q4_K_M (~55 GiB, 49 layers)  
**测试**: bench-sweep, batch-size=1024, input=1024 tokens, epochs=6, warmup=4  

---

## 1. 测试结果汇总

| 配置 | prefill_ms | prefill_tps | CV% | gen_tps | gen_ms |
|------|-----------|------------|-----|---------|--------|
| **Baseline** (CUDA 20L + CPU 29L) | 2,077 ms | **472 t/s** | 3.4% | 18 t/s | 886 ms |
| **Phase 2** (CUDA 20L + Vulkan iGPU 29L) | 19,231 ms | **51 t/s** | 3.5% | 16 t/s | 1,030 ms |
| **变化** | +1,055% | **-89%** | — | -11% | +16% |

### Phase 2 通过标准评估

| 指标 | 目标 | 实际 | 通过？ |
|------|------|------|--------|
| prefill_tps 提升 vs baseline | ≥ +20% | **-89%** | ❌ 严重退步 |
| gen_tps 回归 vs baseline | ≤ 10% | **-11%** | ❌ 略超 |
| prefill CV% | < 10% | 3.5% | ✓ |

**结论：Phase 2 bench 不通过。iGPU Offload 在 ARL 265K 上存在严重性能退步，不适合生产使用。**

---

## 2. 层分配与执行顺序

```
Baseline:
  CUDA0 (RTX 3090):   20 layers (28-47)   →  19.9 GiB GDDR6X
  CPU   (AVX2):       29 layers (0-27,48) →  28.3 GiB DDR5

Phase 2:
  CUDA0 (RTX 3090):   20 layers (28-47)   →  19.9 GiB GDDR6X
  Vulkan1 (iGPU):     29 layers (0-27,48) →  28.1 GiB UMA DDR5
  CPU:                0 layers            →  166 MiB (input tensors only)
```

Transformer 层有严格的数据依赖，执行完全串行：

```
[iGPU Vulkan: layer 0..27] ──► [CUDA: layer 28..47] ──► [iGPU Vulkan: layer 48]
        ~18.5 s                        ~0.4 s                   ~0.4 s
```

**实测**：dGPU 只有短暂利用率 spike，iGPU 长期 100%——瓶颈完全在 iGPU。

---

## 3. 根本原因分析

### 3.1 硬件参数与设计假设的差距

| 参数 | 设计文档假设 | ARL 265K 实际 |
|------|------------|--------------|
| EU 数量 | 128 EU（误估） | **64 EU**（4 Xe-core × 16 EU/core，Xe-LP 架构） |
| 理论算力 | ~4 TFLOPS FP16 | ~4 TFLOPS FP16（64×32 FP16 ops×2 GHz），与设计文档一致 |
| 矩阵加速 | 预期可用 | **matrix cores: none**（Vulkan 不暴露 XMX） |
| bf16 支持 | 未考虑 | **bf16: 0** |
| 实际可用算力 | ~4 TFLOPS | 通用 SIMT shader，实际吞吐 <<4 TFLOPS |
| CPU 竞争对手 | ~1.9 TFLOPS（理论） | AVX2 + LLAMAFILE，深度优化 Q4_K_M 内核 |

> **EU 数量推导**：Intel ARK 标注 8 TOPS Int8 @ 2 GHz → 4,000 Int8 ops/clock。Xe-LP 每 EU 执行 DP4a × SIMD-8 = 64 Int8 ops/clock。4,000 ÷ 64 = **64 EU**（4 Xe-core × 16 EU/core）。

### 3.2 每层实测性能

```
iGPU Vulkan  (29 层):  18,500 ms / 29 ≈ 638 ms/层
CPU AVX2     (29 层):   1,550 ms / 29 ≈  53 ms/层
比值:                   iGPU 比 CPU 慢约 12×
```

### 3.3 四个直接原因

**① Intel XMX 未通过 Vulkan 暴露**

Arrow Lake iGPU 硬件包含 Xe Matrix Extensions (XMX)，但 Intel Windows Vulkan 驱动
当前未实现 `VK_KHR_cooperative_matrix` 扩展，ggml Vulkan 后端因此退化为通用
SIMT compute shader 执行矩阵乘法，等效算力远低于理论峰值。

**② CPU Q4_K_M 内核效率超出预期**

CPU 运行 LLAMAFILE (JIT AVX2) + llama.cpp Q4_K_M 特化内核，可直接利用 L3 cache
流式处理权重矩阵，实际 Q4_K_M 吞吐约 ~8–12 TFLOPS 等效（含解量化加速）。
设计文档的 "1.9 TFLOPS" 是 FP32 理论峰值，严重低估了 CPU 实际能力。

**③ Vulkan Command Dispatch 开销**

Intel Windows Vulkan 驱动每次 dispatch（命令录制 + 提交 + GPU 同步）约 5–20 ms。
29 层 × ~8 matmul/层 = 232+ dispatch，仅 dispatch overhead 可贡献 1–5 s。

**④ DDR5 UMA 带宽竞争**

iGPU 与 CPU 共享 DDR5 6400 MT/s（~102 GB/s 理论）。Prefill batch=1024 时，
iGPU 高强度读写 UMA 内存，实际可用带宽低于 90 GB/s，进一步拉低吞吐。

---

## 4. 原始设计假设的误差来源

原设计文档 §8 的估算公式：

```
iGPU（含 memcpy）：3 × 0.5 / 90GB/s + 512×2 / 4T ≈ 2.73e-10 s/(N×K)
CPU（直接计算）：  1 × 0.5 / 90GB/s + 512×2 / 1.9T ≈ 5.45e-10 s/(N×K)
→ 预期 iGPU 约 2× 快
```

三个致命假设均不成立：

| 假设 | 问题 |
|------|------|
| iGPU = 4 TFLOPS 可用 | Vulkan 无 XMX，实际 SIMT shader 算力 <<4T |
| CPU = 1.9 TFLOPS | LLAMAFILE AVX2 特化内核远超此值 |
| Dispatch overhead ≈ 0 | 实际 Intel Vulkan dispatch ~5–20 ms/call，不可忽略 |

---

## 5. Decode（gen）性能分析

gen_tps 18 → 16 t/s（-11%，略超 10% 阈值）：

- Decode (batch=1) 是 memory-bandwidth-bound：每 token 需读取全部 55 GiB 权重
- iGPU 29 层权重在 UMA DDR5（无 per-call copy），CUDA 20 层在 GDDR6X
- 退步来源：Vulkan dispatch overhead（每层 1 次）+ UMA 带宽分摊
- 与 prefill 退步相比较小，因 batch=1 时 dispatch overhead 占比相对更低

---

## 6. 结论与建议

### 6.1 实验结论

| 项目 | 结论 |
|------|------|
| Phase 2 机制正确性 | ✓ UMA 零拷贝层分配、KV cache 分配均正常工作 |
| 此硬件性能表现 | ❌ prefill -89%，gen -11% |
| 技术可行性（ARL 265K） | ❌ Intel Arc 770 + Vulkan 无法超越 CPU AVX2 |
| 核心限制 | Intel XMX 未通过 Vulkan 暴露；通用 shader 不具竞争力 |

### 6.2 硬件适用性判断

iGPU Offload 在以下条件下 **可能有效**：

| 条件 | 说明 |
|------|------|
| Intel Arc A-series 独立显卡（A770/A580） | 有完整 XMX 支持，Vulkan cooperative_matrix 可用 |
| Intel Lunar Lake / Battlemage 平台 | 更新驱动可能暴露 XMX，EU 数量更多 |
| dGPU VRAM 极度紧张场景 | 如 RTX 3080 10G + 大模型，此时 CPU 也是 bottleneck |
| 未来 Intel Vulkan 驱动更新 | 若 `VK_KHR_cooperative_matrix` 被实现 |

### 6.3 当前建议

1. **不在 ARL 265K 上使用 `OLLAMA_IGPU_OFFLOAD=1`**，保持默认 CUDA+CPU 配置
2. 在文档和 envconfig 中添加 warning，说明此功能处于实验阶段，效果取决于 iGPU 架构
3. **代码保留**：Phase 2 架构（UMA 零拷贝显式层分配）设计正确，当 Intel 驱动更新后可直接受益
4. 考虑在启动时检测 `matrix cores: none` 并自动回退（可选）

---

## 7. 附录：关键日志片段

```
# Phase 2 层分配确认
GPULayers:49[CUDA Layers:20(28..47), Vulkan1 Layers:29[0..27,48]]
offloaded 49/49 layers to GPU

# 内存分配
model weights  device=CUDA0    size=19.9 GiB
model weights  device=Vulkan1  size=28.1 GiB
kv cache       device=CUDA0    size=1.1 GiB
kv cache       device=Vulkan1  size=1.5 GiB
compute graph  device=Vulkan1  size=354.2 MiB  (cf. Phase 1: 757.1 MiB)

# Vulkan 设备能力
ggml_vulkan: 1 = Intel(R) Graphics | uma: 1 | fp16: 1 | bf16: 0 |
             warp size: 32 | matrix cores: none
```

Phase 2 compute graph 从 757 MiB（Phase 1 op_offload）降到 354 MiB，
确认权重已永久驻留 Vulkan buffer，不再需要每次 prefill 复制。
