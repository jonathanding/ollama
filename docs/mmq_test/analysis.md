# MMQ vs Dequant+F16 A/B 测试分析

**测试日期**: 2026-04-08
**测试机器**: Intel Lunar Lake, Arc 140V GPU (16GB shared, Xe2 iGPU)
**模型**: Qwen3.5 4B Q4_K_M
**Ollama 版本**: 0.20.3
**Prompt size**: 1024 tokens
**Bench sweep**: 6 epochs, 4 warmup

## 0. Summary

### 关键结论

| # | 结论 | 证据 |
|---|------|------|
| 1 | **MMQ 与 Dequant+F16 稳态性能无差异** | 热启动后所有 Q4_K matmul GFLOPS/s 差异在 ±2.2% 内 ([§4.1](#41-q4_k-matmul-对比)) |
| 2 | **MMQ 有冷启动惩罚 ~1.76x** | m=8192 首次 prefill 均值: MMQ 18,377 us vs Dequant 10,466 us ([§2.2](#22-两条路径确实不同)) |
| 3 | **Matmul q4_K 占全 prefill 63.6%** | 两批合计 5,110 ms 中 q4_K 占 3,249 ms ([§0 Breakdown](#prefill-算子-breakdown), [§4.2](#42-全算子时间占比)) |
| 4 | **不是 memory-bound** | Arithmetic intensity 646-719 FLOP/byte，实测仅达带宽上限的 5-6% ([§5](#5-roofline-分析)) |
| 5 | **XMX 利用率极低 (1.5-6%)** | 实测 ~1,944 GFLOPS/s 远低于 XMX 理论峰值 ([§5.2](#52-带宽与计算上限)) |
| 6 | **需要硬件 profiling 确认根因** | ⚠️ 无法从性能数据区分 VE DP4A 映射 vs XMX 欠载 ([§5.3](#53-分析)) |

### Prefill Time-to-First-Token

来源: bench-sweep，6 epochs，4 warmup，prompt = 1024 tokens

| 指标 | MMQ (默认) | Dequant+F16 | 差异 |
|------|-----------|-------------|------|
| prefill_ms (服务端) | 5,023 ms | 4,976 ms | -0.9% |
| **ttft_ms (端到端)** | **5,174 ms** | **5,116 ms** | **-1.1%** |
| HTTP + 调度开销 | 151 ms | 140 ms | — |
| prefill_tps | 195 t/s | 197 t/s | +1.0% |

> ttft_ms = prefill_ms + HTTP round-trip + 调度延迟 (source: bench-sweep 注释)。两条路径的 TTFT 差异在噪声范围内 (prefill CV% ≈ 3.3-3.6%)。

### Prefill 算子 Breakdown

来源: `GGML_VK_PERF_LOGGER=1`，MMQ 路径

> **重要**: 1024 token 的 prompt 被拆为 **2 个 batch** (BatchSize=512) 处理。下表分别展示各 batch 以及合计。

| 算子 | Batch 1 (ms) ² | Batch 2 (ms) | 合计 (ms) | 占比 |
|------|---------------|-------------|----------|------|
| MUL_MAT q4_K | 1,653 | 1,596 | 3,249 | 63.6% |
| FLASH_ATTN_EXT | 130 | 437 | 567 | 11.1% |
| MUL_MAT q6_K | 116 | 113 | 229 | 4.5% |
| MUL_MAT f32 | 88 | 86 | 174 | 3.4% |
| CONT | 90 | 82 | 172 | 3.4% |
| 其余 | 349 | 335 | 684 | 13.4% |
| inter-batch ³ | — | — | 35 | 0.7% |
| **合计** | **2,426** | **2,649** | **5,110** | **100%** |

> ² Batch 1 数据取自第三次请求（全热启动, lines 920-963）。Batch 2 数据取自第一次请求的第二批（lines 167-210）。两者来自不同请求，因此合计 5,110 ms 与 bench-sweep 的 5,023 ms 有 ~2% 偏差。
>
> ³ inter-batch: 两个 batch 之间的单层过渡计算 (18 ms + 17 ms)。
>
> **FA 随 KV 长度增长**: Batch 1 FA (KV=512) = 130 ms, Batch 2 FA (KV=1024) = 437 ms。3.4x 增长，与 FA 的 O(n·d) 复杂度一致（KV 翻倍 → FA 时间翻倍+，因为 softmax 和 reduction 也增加）。

关键发现:
- **MUL_MAT q4_K** 占全 prefill 的 **63.6%**，是优化 ROI 最高的算子
- **FLASH_ATTN_EXT** 占 **11.1%**，但 Batch 2 中占比升至 16.5%（KV 长度效应）
- Top-2 合计 **74.7%**

---

## 1. 测试设计

| 运行 | 环境变量 | 目的 |
|------|---------|------|
| **baseline** | `OLLAMA_VULKAN=1 GGML_VK_PERF_LOGGER=1` | 默认路径（MMQ int8 dot product） |
| **disable-int-dot-product** | 同上 + `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` | 强制 Dequant+F16 coopmat 路径 |

## 2. 关键确认

### 2.1 环境变量生效确认

启动日志（release 构建可见）：

```
# 默认
ggml_vulkan: 0 = Intel(R) Arc(TM) 140V GPU (16GB) | int dot: 1 | matrix cores: KHR_coopmat

# 禁用后
ggml_vulkan: 0 = Intel(R) Arc(TM) 140V GPU (16GB) | int dot: 0 | matrix cores: KHR_coopmat
```

`int dot` 从 `1` 变为 `0`，确认 `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` 生效。

(source: `ggml-vulkan.cpp:4326-4329`, `getenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")`)

### 2.2 两条路径确实不同

冷启动（第一次 prefill）时 m=8192 的 MUL_MAT q4_K 耗时差异巨大：

| 路径 | 冷启动耗时 | GFLOPS/s |
|------|-----------|----------|
| MMQ (默认) | **18,377 us** | 1,168 |
| Dequant+F16 | 10,466 us | 2,051 |

MMQ 冷启动慢 1.76x，推测原因是 `dotPacked4x8EXT` 扩展指令的 Vulkan 驱动 JIT 编译成本更高。Dequant+F16 冷热一致（10,466 vs 10,842 us）。

## 3. Bench Sweep 端到端结果

| 指标 | MMQ (默认) | Dequant+F16 | 差异 |
|------|-----------|-------------|------|
| prefill_ms | 5,023 ms | 4,976 ms | -0.9% |
| prefill_tps | 195 t/s | 197 t/s | +1.0% |
| Prefill CV% | 3.6% | 3.3% | — |
| **ttft_ms** | **5,174 ms** | **5,116 ms** | **-1.1%** |
| TTFT CV% | 5.5% | 5.2% | — |
| gen_ms (16 tokens) | 933 ms | 846 ms | -9.3% |
| gen_tps | 17 t/s | 19 t/s | +11.8% |
| Decode CV% | 2.5% | 1.8% | — |

> ttft_ms 差异: bench-sweep 的两条 `⚠` 提示 (ttft CV > 5%) 表明 TTFT 方差较大，-1.1% 的差异在噪声范围内。

**端到端无显著差异**。Prefill 完全在噪声范围内，Decode 的 11.8% 差异需更多 epoch 确认（gen_ms 仅 ~900 ms，绝对波动 ~20 ms 即可产生此差异）。

## 4. Per-Op 热启动分析

取第二次 prefill（热启动，稳态）数据。

### 4.1 Q4_K Matmul 对比

> 差异 = (Dequant 耗时 - MMQ 耗时) / MMQ 耗时。正值 = Dequant 更慢。

| 矩阵尺寸 (m×n×k) | MMQ (us) | MMQ (GFLOPS/s) | Dequant (us) | Dequant (GFLOPS/s) | 差异 |
|---|---|---|---|---|---|
| 9216×512×2560 | 12,428 | 1,944 | 12,519 | 1,929 | +0.7% |
| 8192×512×2560 | 10,679 | 2,011 | 10,842 | 1,980 | +1.5% |
| 2560×512×9216 | 12,618 | 1,915 | 12,534 | 1,927 | -0.7% |
| 2560×512×4096 | 4,614 | 2,327 | 4,579 | 2,345 | -0.8% |
| 4096×512×2560 | 4,796 | 2,239 | 4,713 | 2,278 | -1.7% |
| 1024×512×2560 | 1,399 | 1,919 | 1,430 | 1,877 | +2.2% |

**所有差异均在 ±2.2% 内（噪声范围）。热启动后两条路径性能完全一致。**

### 4.2 全算子时间占比

| 算子 | MMQ 耗时 (us) | MMQ 占比 | Dequant 耗时 (us) | Dequant 占比 |
|------|-------------|---------|-------------------|-------------|
| **MUL_MAT q4_K** | 1,595,831 | **60.2%** | 1,601,994 | **60.3%** |
| **FLASH_ATTN_EXT** | 436,967 | **16.5%** | 426,995 | **16.1%** |
| MUL_MAT q6_K | 113,353 | 4.3% | 114,399 | 4.3% |
| MUL_MAT f32 | 85,510 | 3.2% | 86,348 | 3.3% |
| CONT | 81,531 | 3.1% | 84,696 | 3.2% |
| MUL | 55,653 | 2.1% | 55,371 | 2.1% |
| CONCAT | 52,874 | 2.0% | 51,170 | 1.9% |
| GLU | 40,936 | 1.5% | 44,839 | 1.7% |
| 其余 | ~186,455 | ~7.0% | ~191,058 | ~7.2% |
| **总计** | **2,649,110** | **100%** | **2,656,870** | **100%** |

Top 2 算子（q4_K matmul + FA）合计占比 76.7%，两条路径的占比分布几乎完全一致。

## 5. Roofline 分析

以最大 Q4_K matmul (m=9216, n=512, k=2560) 为例：

### 5.1 Arithmetic Intensity

| 路径 | 权重数据 | 激活数据 | 输出数据 | 总数据 | AI (FLOP/byte) |
|------|---------|---------|---------|--------|----------------|
| Dequant+F16 | 13.3 MB | 5.2 MB (fp32) | 18.9 MB | 37.4 MB | 646 |
| MMQ | 13.3 MB | 1.4 MB (Q8_1) | 18.9 MB | 33.6 MB | 719 |

n=512 时 arithmetic intensity 高达 646-719 FLOP/byte。

### 5.2 带宽与计算上限

Lunar Lake 内存带宽：LPDDR5X 双通道（具体频率取决于 SKU）。

> ⚠️ 以下使用 51.2 GB/s 作为保守估计。LPDDR5X 实际峰值可能更高（如 LPDDR5X-8533 128-bit 理论峰值 ~68 GB/s）。但即使使用更高的带宽值，"不是 memory-bound" 的结论只会更加确定（利用率更低）。

| 假设瓶颈 | 理论上限 | 实测 | 利用率 |
|----------|---------|------|--------|
| 内存带宽 (51.2 GB/s) | 33,000-37,000 GFLOPS/s | 1,944 | **5-6%** |
| XMX fp16 (~35-67 TFLOPS) | 35,000-67,000 | 1,944 | **3-6%** |
| XMX int8 (~70-134 TOPS) | 70,000-134,000 | 1,944 | **1.5-3%** |
| **VE DP4A (~4.2 TOPS)** ¹ | **4,200** | **1,944** | **46%** |

> ¹ VE DP4A 峰值估算: 8 Xe-cores × 8 EU/core × 8 int8-ops/DP4A × 1 DP4A/clock/EU × SIMD-8 = 4,096 int8-ops/clock。按 ~2.05 GHz 约为 4.2 TOPS。⚠️ 此估算假设每 EU 每周期执行 1 个 SIMD-8 DP4A 指令。如果 Xe2 VE 有双发射能力，峰值可能为 ~8.4 TOPS (对应 23% 利用率)。精确值需查阅 Intel Xe2 ISA 文档。

### 5.3 分析

- **不是 memory-bound**：实测仅达到带宽上限的 5-6%，差 20 倍
- **不是 XMX-saturated**：无论 fp16 还是 int8，XMX 利用率仅 1.5-6%
- **与 VE DP4A 峰值最匹配**：46% 利用率是唯一合理数值

但有两种可能的解释：

1. **`dotPacked4x8EXT` 映射到 VE DP4A（非 XMX）** — 46% 利用率合理，两条路径都受 VE 瓶颈限制
2. **`dotPacked4x8EXT` 映射到 XMX，但 XMX 严重欠载** — 32KB shared memory（典型值一半）导致 tile 尺寸受限，XMX 喂不饱；iGPU 系统内存延迟高

**无法仅从性能数据区分这两种情况。** 需要 Intel VTune / GPA 的硬件计数器（XMX active cycles）才能确认。

## 6. 硬件环境备注

从启动日志提取的设备参数：

| 参数 | 值 | 备注 |
|------|-----|------|
| GPU | Intel Arc 140V GPU (16GB) | Lunar Lake iGPU, Xe2 架构 |
| UMA | 1 (unified memory) | 与 CPU 共享系统内存 |
| fp16 | 1 | 支持 fp16 |
| bf16 | 0 | 不支持 bf16 |
| Warp size | 32 | subgroup size |
| **Shared memory** | **32,768** (32KB) | **仅为典型 64KB 的一半** |
| int dot | 1 (默认) / 0 (禁用) | `VK_KHR_shader_integer_dot_product` |
| Matrix cores | KHR_coopmat | KHR cooperative matrix (非 NV coopmat2) |
| 系统内存 | 31.5 GiB total | LPDDR5X |
| Flash Attention | Enabled | 自动启用 |
| 模型架构 | Qwen3.5 (hybrid, 含 SSM 层) | `architecture=qwen35`, 33 layers |

## 7. 结论

1. **MMQ 和 Dequant+F16 在 Xe2 Lunar Lake 上稳态性能无差异** — 所有 Q4_K matmul 尺寸的 GFLOPS/s 差异均在 ±2.2% 内
2. **MMQ 有冷启动惩罚** — 首次运行比 Dequant+F16 慢 ~1.76x（JIT 编译成本）
3. **Matmul 占 prefill 60%+** — 优化 matmul 路径的 ROI 最高
4. **实测吞吐 ~1,944 GFLOPS/s 远低于 XMX 理论峰值** — 无论哪条路径，XMX 利用率仅 1.5-6%
5. **瓶颈可能是 VE DP4A 映射或 XMX 欠载** — 32KB shared memory 和 iGPU 内存延迟是可能的限制因素
6. **需要硬件 profiling 工具才能确认根因** — Intel VTune / GPA 的 XMX active cycles 计数器

## 8. 代码修改方案: Pipeline 名称日志

> ✅ 以下方案**已在 Lunar Lake 机器上实测**。§8.9 为实测结果。

### 8.1 目标

确认 Q4_K matmul 在默认配置下使用的是 **MMQ pipeline** (`matmul_q4_k_q8_1_*`) 还是 **Dequant+F16 pipeline** (`matmul_q4_k_f32_*`)。

### 8.2 修改位置

**文件**: `ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp`
**函数**: `ggml_vk_dispatch_pipeline` (line 5871)

这是所有 Vulkan compute shader dispatch 的**唯一入口**。`pipeline->name` 是 `std::string`，包含 shader 的完整名称。

### 8.3 具体 Diff

在 line 5879 (`std::cerr << "}, (" << wg0 << ...`) 之后、line 5880 (`GGML_ASSERT(...)`) 之前，添加 3 行：

```diff
 static void ggml_vk_dispatch_pipeline(ggml_backend_vk_context* ctx, vk_context& subctx,
     vk_pipeline& pipeline, ...) {
     const uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
     const uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
     const uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);
     VK_LOG_DEBUG("ggml_vk_dispatch_pipeline(" << pipeline->name << ", {";
     for (auto& buffer : descriptor_buffer_infos) {
         std::cerr << "(" << buffer.buffer << ", " << buffer.offset << ", " << buffer.range << "), ";
     }
     std::cerr << "}, (" << wg0 << "," << wg1 << "," << wg2 << "))");
+    static const bool log_pipelines = (getenv("GGML_VK_LOG_PIPELINES") != nullptr);
+    if (log_pipelines) {
+        fprintf(stderr, "VK_DISPATCH: %s [wg=%u,%u,%u]\n", pipeline->name.c_str(), wg0, wg1, wg2);
+    }
     GGML_ASSERT(ctx->descriptor_set_idx < ctx->descriptor_sets.size());
     GGML_ASSERT(descriptor_buffer_infos.size() <= MAX_PARAMETER_COUNT);
```

### 8.4 编译

```bash
# 在 Ollama 项目根目录 (Windows)
cmake -B build
cmake --build build --config Release

# 运行（使用新编译的 binary）
go run . serve
# 或: go build . && .\ollama.exe serve
```

**前置条件**: CMake 3.21+, Vulkan SDK (含 glslc), MSVC 或 Clang, Go 1.24

### 8.5 运行

```bash
# 启用 pipeline 日志 + perf 日志
set OLLAMA_VULKAN=1
set GGML_VK_PERF_LOGGER=1
set GGML_VK_LOG_PIPELINES=1
ollama serve 2> vulkan_pipeline_log.txt
```

在另一个终端发送请求：

```bash
curl http://localhost:11434/api/generate -d "{\"model\":\"qwen3.5:4b\",\"prompt\":\"xHello world\",\"options\":{\"num_predict\":1}}"
```

> 注意: prompt 首字符每次不同以避免 prefix cache 命中 (see `docs/daop/how-to-run-actual-inference.md`)。

过滤 matmul 相关日志：

```bash
grep "q4_k" vulkan_pipeline_log.txt | head -20
```

### 8.6 预期输出与解读

**场景 A: MMQ 路径 (默认, `int dot: 1`)**
```
VK_DISPATCH: matmul_q4_k_q8_1_l [wg=288,16,1]
VK_DISPATCH: matmul_q4_k_q8_1_m [wg=128,16,1]
```

- 名称含 `_q8_1` → 使用了 integer dot product shader (`dotPacked4x8EXT` / `OpSDotKHR`)
- 激活被量化到 Q8_1 格式后与 Q4_K 权重做 int8 点积

**场景 B: Dequant+F16 路径 (`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`)**
```
VK_DISPATCH: matmul_q4_k_f32_l [wg=288,16,1]
VK_DISPATCH: matmul_q4_k_f32_f16acc_m [wg=128,16,1]
```

- 名称含 `_f32` 或 `_f16acc` → 使用了 dequantize + cooperative matrix f16 shader
- 权重反量化到 f16 后通过 `coopMatMulAdd` (KHR coopmat1) 做 f16×f16

**Pipeline 名称编码规则** (source: `ggml-vulkan.cpp` CREATE_MM / CREATE_MMQ 宏):

| 名称片段 | 含义 | 代码来源 |
|---------|------|---------|
| `_q8_1` | MMQ: 激活量化到 Q8_1, integer dot product | `CREATE_MMQ`, line 3295 |
| `_f32` | Dequant (f32 累加): 权重→f16, `coopMatMulAdd` f16×f16 | `CREATE_MM2`, line 3269 |
| `_f16acc` | Dequant (f16 累加): 同上但累加器用 f16 | `CREATE_MM2`, line 3269 |
| `_l` / `_m` / `_s` | 大 / 中 / 小 tile 尺寸 | 通用 |
| `_aligned_` | N 维度对齐到 tile 大小的优化变体 | 通用 |

### 8.7 Self-Review

| 检查项 | 结果 |
|--------|------|
| `getenv` 只调用一次？ | ✅ `static const bool` 在 C++11 中保证只初始化一次 |
| 需要额外 `#include`？ | ✅ 不需要，`fprintf` 已在文件中使用 (line 105, 1900 等) |
| `pipeline->name` 类型？ | ✅ `std::string`，`.c_str()` 返回 `const char*` |
| 不设置环境变量时有开销？ | ✅ 零开销，`static const bool` 为 `false` 时 branch prediction 跳过 |
| 线程安全？ | ✅ `static const` 初始化线程安全 (C++11)；`fprintf(stderr)` 线程安全 (POSIX/MSVC CRT) |
| 输出量？ | 每次 dispatch 一行。1024 token prefill 约 2000-3000 行，可用 `grep q4_k` 过滤 |

### 8.8 局限性

**此修改只能确认软件层面选择了哪个 shader，无法确认 GPU 硬件使用了哪个执行单元。**

- ✅ 可以确认: 使用了 `dotPacked4x8EXT` shader (MMQ) 还是 `coopMatMulAdd` shader (Dequant+F16)
- ❌ **无法确认**: `dotPacked4x8EXT` → DP4A (Vector Engine) 还是 DPAS (XMX)

原因: `dotPacked4x8EXT` 对应 SPIR-V 指令 `OpSDotKHR`。Intel Vulkan 驱动在 JIT 编译时决定将其映射到哪个硬件单元：

- **DP4A** — 在 Vector Engine (EU) 的 ALU pipeline 上执行
- **DPAS** — 在 XMX systolic array 上执行

这个映射决策对 Vulkan 应用层**完全不透明**，没有任何 API 或回调可以查询。即使确认了 shader 使用 `OpSDotKHR`，也无法从应用层区分 DP4A 和 DPAS。

确认硬件执行单元的方法：

1. **Intel VTune** — 查看 `XMX Active Cycles` 硬件计数器。非零 → 走了 XMX/DPAS
2. **IGC Shader Dump** — 设置环境变量 `IGC_ShaderDumpEnable=1`，导出驱动编译后的 GEN ISA 汇编。包含 `dpas` 指令 → XMX，包含 `dp4a` 指令 → VE。**注意**: 实测 `IGC_ShaderDumpEnable` 在 Windows Vulkan 驱动上**不生效**（仅适用于 OpenCL/Level Zero 驱动）
3. **`VK_KHR_pipeline_executable_properties`** — Vulkan 原生扩展，Intel 驱动已启用。实测可获取编译统计（指令数、寄存器数等），但 `InternalRepresentations`（即 GEN ISA 文本）**返回 0 条记录**，Intel Windows Vulkan 驱动未实现此部分
4. **Intel GPA** (Graphics Performance Analyzers) — 着色器级 profiling

### 8.9 实测结果

> 测试日期: 2026-04-09。模型: Qwen3 0.6B Q4_K_M（28 层, 16 Q heads, 8 KV heads, head_dim=128）。
> 机器与 §0 相同: Lunar Lake, Arc 140V, `OLLAMA_VULKAN=1`。
> 代码修改: 在 `ggml_vk_dispatch_pipeline` 添加 `GGML_VK_LOG_PIPELINES=1` 日志 + `GGML_VK_DUMP_ISA=1` ISA dump。

#### 8.9.1 Prefill 阶段: 单层 Dispatch 序列

每个 transformer 层在 prefill（batch_size > 1）时的完整 dispatch 序列：

**Attention 半区:**

| # | Pipeline | 精度 | 作用 |
|---|----------|------|------|
| 1 | `rms_norm_mul_f32` | f32 | RMSNorm + scale |
| 2 | `quantize_q8_1_x4` | f32→q8_1 | 激活量化（为 MMQ 准备） |
| 3 | `matmul_q4_k_q8_1_s` | int8 dot | **Q projection** (Q4_K 权重) |
| 4 | `matmul_q4_k_q8_1_s` | int8 dot | **K projection** (Q4_K 权重) |
| 5 | `matmul_f16_f32_f16acc_aligned_s` | f16 coopmat | **V projection** (F16 权重) |
| 6 | `rms_norm_mul_rope_f32_f32` | f32 | Q RoPE |
| 7 | `rms_norm_mul_rope_f32_f32` | f32 | K RoPE |
| 8 | `set_rows_f16_i32` | f32→f16 | K 写入 KV cache |
| 9 | `set_rows_f16_i32` | f32→f16 | V 写入 KV cache |
| 10 | `flash_attn_f32_f16_aligned_f32accf16` | Q=f32, KV=f16, acc=f32 | Flash Attention |
| 11 | `quantize_q8_1_x4` | f32→q8_1 | 激活量化 |
| 12 | `matmul_q4_k_q8_1_s` | int8 dot | **Output projection** (Q4_K 权重) |
| 13 | `add_f32_f32_f32_norepeat` | f32 | Residual add |

**FFN 半区:**

| # | Pipeline | 精度 | 作用 |
|---|----------|------|------|
| 14 | `rms_norm_mul_f32` | f32 | RMSNorm + scale |
| 15 | `quantize_q8_1_x4` | f32→q8_1 | 激活量化 |
| 16 | `matmul_q4_k_q8_1_s` | int8 dot | **Gate projection** (Q4_K 权重) |
| 17 | `matmul_q4_k_q8_1_s` | int8 dot | **Up projection** (Q4_K 权重) |
| 18 | `swiglu_f32_rte` | f32 | SwiGLU activation |
| 19 | `quantize_q8_1_x4` | f32→q8_1 | 激活量化 |
| 20 | `matmul_q6_k_q8_1_s` | int8 dot | **Down projection** (Q6_K 权重) |
| 21 | `add_f32_f32_f32_norepeat` | f32 | Residual add |

#### 8.9.2 Decode 阶段: 单层 Dispatch 序列

decode（n=1）使用 `mul_mat_vec` 变体替代 `matmul`：

**Attention 半区:**

| # | Pipeline | 精度 | 作用 |
|---|----------|------|------|
| 1 | `rms_norm_mul_f32` | f32 | RMSNorm + scale |
| 2 | `mul_mat_vec_q4_k_f32_f32` | dequant→f32 | **Q projection** |
| 3 | `mul_mat_vec_q4_k_f32_f32` | dequant→f32 | **K projection** |
| 4 | `mul_mat_vec_f16_f32_f32` | f16→f32 | **V projection** (F16 权重) |
| 5 | `rms_norm_mul_rope_f32_f32` | f32 | Q RoPE |
| 6 | `rms_norm_mul_rope_f32_f32` | f32 | K RoPE |
| 7 | `set_rows_f16_i32` | f32→f16 | K 写入 KV cache |
| 8 | `set_rows_f16_i32` | f32→f16 | V 写入 KV cache |
| 9 | `flash_attn_f32_f16_aligned_f32accf16` | Q=f32, KV=f16, acc=f32 | Flash Attention |
| 10 | `fa_split_k_reduce` | f32 | FA split-K 归约 |
| 11 | `quantize_q8_1_x4` | f32→q8_1 | 激活量化 |
| 12 | `mul_mat_vec_q4_k_q8_1_f32` | int8 dot | **Output projection** (MMQ) |

**FFN 半区:**

| # | Pipeline | 精度 | 作用 |
|---|----------|------|------|
| 13 | `rms_norm_mul_f32` | f32 | RMSNorm + scale |
| 14 | `mul_mat_vec_q4_k_f32_f32` | dequant→f32 | **Gate projection** |
| 15 | `mul_mat_vec_q4_k_f32_f32` | dequant→f32 | **Up projection** |
| 16 | `swiglu_f32_rte` | f32 | SwiGLU activation |
| 17 | `mul_mat_vec_q6_k_f32_f32` | dequant→f32 | **Down projection** |

#### 8.9.3 GGUF Tensor 量化类型 (Qwen3 0.6B Q4_K_M)

| Tensor | 量化类型 | Shape | 对应 Pipeline |
|--------|---------|-------|---------------|
| `attn_q.weight` | Q4_K | [1024, 2048] | `matmul_q4_k_q8_1` (prefill) / `mul_mat_vec_q4_k_f32_f32` (decode) |
| `attn_k.weight` | Q4_K | [1024, 1024] | 同上 |
| `attn_v.weight` | **F16** | [1024, 1024] | `matmul_f16_f32_f16acc` (prefill) / `mul_mat_vec_f16_f32_f32` (decode) |
| `attn_output.weight` | Q4_K | [2048, 1024] | `matmul_q4_k_q8_1` (prefill) / `mul_mat_vec_q4_k_q8_1_f32` (decode) |
| `ffn_gate.weight` | Q4_K | [1024, 3072] | `matmul_q4_k_q8_1` / `mul_mat_vec_q4_k_f32_f32` |
| `ffn_up.weight` | Q4_K | [1024, 3072] | 同上 |
| `ffn_down.weight` | Q6_K | [3072, 1024] | `matmul_q6_k_q8_1` / `mul_mat_vec_q6_k_f32_f32` |

#### 8.9.4 数据类型流总结

```
Layer 间主干: f32 (residual stream)
    │
    ├─ 进 matmul 前: f32 ──quantize_q8_1_x4──→ q8_1 (仅 prefill MMQ 需要)
    ├─ matmul 计算:  Q4_K/Q6_K × q8_1 → int8 dot product, 输出 f32
    ├─ V projection: F16 × f32 → f16 coopmat (prefill) / f16 dequant (decode), 输出 f32
    │
    ├─ 写 KV cache:  f32 ──set_rows_f16──→ f16
    ├─ Flash Attn:   Q=f32, K/V=f16(从 cache 读), 累加 f32, 输出 f32
    │
    └─ Residual add: f32 + f32 → f32 (传递给下一层)
```

关键观察:

1. **Layer 间传输全部是 f32** — `add_f32_f32_f32_norepeat` 输出 f32 作为下一层输入
2. **KV cache 存储 f16** — `set_rows_f16_i32` 写入时从 f32 转换为 f16
3. **Flash Attention** — Q 是 f32，K/V 从 f16 cache 读取，**累加器是 f32**
4. **V 权重是 F16** — Q4_K_M 量化格式中 V projection 不量化，保持 F16 精度
5. **Decode 多数 matmul 走 dequant 路径** — `mul_mat_vec_q4_k_f32_f32` (反量化到 f32 再计算)，不需要 `quantize_q8_1` 步骤。仅 Output projection 走 MMQ (`mul_mat_vec_q4_k_q8_1_f32`)

#### 8.9.5 编译统计 (VK_KHR_pipeline_executable_properties)

Intel 驱动通过 `getPipelineExecutableStatisticsKHR` 返回的编译统计:

| Pipeline | Instruction Count | Basic Blocks | Loops | Shared Memory |
|----------|------------------|--------------|-------|---------------|
| `matmul_q4_k_q8_1_s` | 3,127 | 264 | 1 | 7,168 B |
| `matmul_q6_k_q8_1_s` | 2,650 | 184 | 1 | 9,216 B |
| `matmul_f16_f32_f16acc_aligned_s` | 2,779 | 119 | 1 | 4,352 B |
| `flash_attn_f32_f16_aligned_f32accf16` | 25,107 | 609 | 1 | 7,680 B |
| `mul_mat_vec_q4_k_q8_1_f32` | 4,784 | 285 | 10 | 0 B |
| `mul_mat_vec_q4_k_f32_f32` | 1,632 | 141 | 7 | 512 B |
| `rms_norm_mul_f32` | 12,004 | 1,496 | 6 | 2,048 B |

注: `InternalRepresentations`（GEN ISA 文本）Intel Windows Vulkan 驱动**返回 0 条记录**，无法获取编译后的机器指令。

---

## 9. 后续建议

- [ ] 更大 batch size 测试（n=2048, 4096）看 XMX 利用率能否提升
- [ ] 纯 XMX fp16 micro-benchmark 作为基线，确认 Xe2 实际可达峰值
- [ ] Intel VTune profiling 确认 `dotPacked4x8EXT` → DP4A 还是 DPAS
- [ ] 对比 `GGML_VK_DISABLE_COOPMAT=1`（同时禁用 coopmat）看纯标量路径性能

---

**原始数据文件**:
- `bench_sweep_result.txt` — bench-sweep 端到端结果
- `default-1k-token-vulkan.txt` — 默认路径 (MMQ) 的完整 perf log
- `disable-int-dot-product-vulkan.txt` — 禁用 int dot product 的完整 perf log
