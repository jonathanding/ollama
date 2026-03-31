# CUDA + iGPU 下 80B MoE 模型的优化方向

以下分析基于 Qwen3-coder-next 80B Q4_K_M 在 NVIDIA 24GB + Intel iGPU (128GB RAM UMA) 上运行的场景。目标：**优化 prefill 速度（首 token 延迟）和 decode 速度（token/s 吞吐）**。

§6 的 Phase-Aware iGPU Offload 主攻 prefill。本章探讨与之互补的更多优化方向。

### 7.1 KV Cache 量化（已支持，立即可用）

#### 当前支持现状

Ollama **已完整支持** KV cache 量化：

- **配置方式**：环境变量 `OLLAMA_KV_CACHE_TYPE`（`envconfig/config.go:195`）
- **支持类型**（`fs/ggml/ggml.go:849-855, 908-919`）：

| 类型 | 每元素字节 | 相对 f16 | 条件 |
|------|-----------|---------|------|
| `f16` | 2.0 | 100% (默认) | 无 |
| `q8_0` | 1.0 | 50% | 需 Flash Attention |
| `q4_0` | 0.5 | 25% | 需 Flash Attention |

- **前提条件**：量化 KV cache 需要 Flash Attention 开启（`llm/server.go:224-258`，未开启时会 warning 并回退 f16）
- **模型支持**：Qwen3 全系列已支持（`fs/ggml/ggml.go:900-902`：`qwen3`, `qwen3moe`, `qwen35`, `qwen35moe`, `qwen3next`）

**注意**：Recurrent 层（GatedDeltaNet）的 state 始终用 f32（`fs/ggml/ggml.go:915`），不受 KV cache 类型影响。只有 full attention 层的 KV cache 被量化。

#### 对 CUDA + iGPU 场景的收益

以 32K context、Qwen3-coder-next 80B 为例（仅 full attention 层有 KV cache，约 20 层）：

```
KV cache f16: 20 层 × 32K × (K_dim + V_dim) × num_kv_heads × 2 bytes
KV cache q4_0: 同上 × 0.5 bytes → 节省 75%
```

GPU 24GB 中 KV cache 占比减少 → **能多放几层的权重到 GPU** → 减少溢出到 CPU 的层数。

#### 使用方式

```bash
OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve   # 保守：50% 节省，质量损失极小
OLLAMA_KV_CACHE_TYPE=q4_0 ollama serve   # 激进：75% 节省，可能有轻微质量影响
```

### 7.2 Speculative Decoding / EAGLE3（未支持，需实现）

#### 原理

Speculative decoding 把 K 次逐 token decode 变成 1 次 batch=K 的验证：

```
无 speculative decoding:
  大模型 decode: token₁ → token₂ → ... → tokenₖ
  每个 token 读一遍所有权重 → K 次全量读取

有 speculative decoding:
  1. Draft head 快速草拟 K 个候选 token（轻量）
  2. 大模型一次 forward pass (batch=K) 验证
  → 只读一遍权重，产出 ~K 个 token（取决于接受率）
```

对 CPU 上的 55 溢出层，decode 瓶颈是内存带宽（DDR5 ~90 GB/s）。Speculative decoding 让每次权重读取产出更多 token，**等效带宽利用率提高 K × acceptance_rate 倍**。

代码生成场景接受率通常 70-80%，K=8 时等效 decode 吞吐提升 **×5-6**。

EAGLE3（如 [Aurora-Spec-Qwen3-Coder-Next-FP8](https://huggingface.co/togethercomputer/Aurora-Spec-Qwen3-Coder-Next-FP8)）是一种高效 draft head，直接利用目标模型的 hidden states 预测下一组 token，不需要独立的小模型。

#### 与 iGPU offload 的协同

验证阶段 batch=K（K=8-16），batch >= 32 时 `op_offload` 触发。如果 K 足够大，验证阶段的溢出层也能 offload 到 iGPU 加速。即使 K < 32 不触发 offload，单次读取产出多 token 本身就是巨大收益。

#### 当前 Ollama 支持现状

**未实现。** 详细调查结果：

- **上游 llama.cpp**：已有完整 speculative decoding 基础设施
  - `common_params_speculative` 结构体（`llama.cpp/common/common.h`）：draft model path、n_max、p_min 等参数
  - `common_sampler_sample_and_accept_n()`：draft-verify 采样逻辑
- **Ollama 层面**：完全未暴露
  - `api/types.go`：无 draft model、speculative 相关字段
  - `runner/`（两种 runner）：无多模型加载、无 draft-verify 流水线
  - `envconfig/`：无 `OLLAMA_DRAFT_MODEL` 等环境变量
  - `server/routes.go`：单模型请求处理，无并行模型调度

#### 实现所需改动

```
1. API 层: 新增 draft_model 参数 (api/types.go)
2. Server 层: scheduler 支持同时加载 draft + target 模型 (server/sched.go)
3. Runner 层: 实现 draft-verify 循环 (runner/ollamarunner/runner.go)
4. 模型层: 支持 EAGLE3 head 作为 draft model 的一种
```

改动量较大，涉及 API、scheduler、runner 三层。

### 7.3 MoE 感知层内拆分（未支持，需实现）

#### 原理

当前层分配以**整层**为单位：

```
当前: Layer N → 全部 tensor (attention + MoE FFN) → 同一设备
                ↓
GPU 24GB ÷ ~3.2GB/层 ≈ 7 层（含 KV cache 后更少）
→ Layer 8-79 的 attention 全在 CPU 上
```

MoE 感知拆分：在**同一层内**，把 attention（小、dense）和 MoE FFN（大、sparse）分到不同设备：

```
优化: Layer N → attention tensor → GPU
                MoE FFN tensor  → CPU/iGPU

Attention: 80 层 × ~200MB = 16GB → 全部放 GPU 24GB ✓
MoE FFN: 80 层 × ~3GB (Q4_K_M) → CPU/iGPU (UMA ~100GB)
```

**效果**：所有层的 attention/recurrent 都在 GPU 上快速计算，MoE FFN 在 CPU/iGPU 上利用稀疏性（top-8/128 = 6.25% 权重活跃）。

#### 当前 Ollama 支持现状

**不支持。** 层分配是**原子的**——同一层的所有 tensor 必须在同一设备。

**关键代码证据**：

`ggml.go:207-223` — `assignLayer()` 返回**单一设备**给整层：

```go
assignLayer := func(layer int) deviceBufferType {
    for _, p := range params.GPULayers {
        for _, l := range p.Layers {
            if l == layer {
                return gpuDeviceBufferTypes[i]  // ← 整层一个设备
            }
        }
    }
    return cpuDeviceBufferType
}
```

`ggml.go:340-341` — 所有 tensor 用层级设备分配：

```go
if layerIndex >= 0 {
    createTensor(tensor{source: t}, layers[layerIndex].bts, layerIndex)
    // ↑ blk.25.attn_q 和 blk.25.ffn_gate 都用 layers[25].bts（同一设备）
}
```

Tensor 名到层号的映射（`ggml.go:334-337`）只提取数字索引，不区分 tensor 类型：

```go
// "blk.25.attn_q.weight" → layerIndex = 25
// "blk.25.ffn_gate.0.weight" → layerIndex = 25
// 两者分配到相同设备
```

内存追踪也是 per-layer（`ml/device.go:146-161`）：`Weights[layer]` 和 `Cache[layer]` 按层索引，不按 tensor 类型拆分。

#### 实现所需改动

需要将"层级分配"改为"tensor 级分配"：

```
1. ggml.go: assignLayer() → assignTensor()，根据 tensor 名中的 "attn_"/"ffn_" 前缀分配不同设备
2. llm/server.go: buildLayout() 需要 per-tensor-type 的内存统计（attention 权重 vs FFN 权重分开算）
3. ml/device.go: DeviceMemory 需要 per-tensor-type 跟踪
4. GGML scheduler: 已天然支持同一层 tensor 在不同 backend（5-pass split 按 tensor buffer type 路由）
```

**好消息**：GGML scheduler 底层**已支持**同一层的 tensor 在不同 backend——只要权重被分配到不同的 buffer type，split 算法自动处理跨 backend 计算和数据传输。**改动集中在 Go 层的分配逻辑**。

### 7.4 Hybrid 层类型感知分配（未支持，低成本可实现）

#### 原理

Qwen3-coder-next 是 hybrid 架构，混合两种层：

- **Recurrent 层 (GatedDeltaNet)**：60/80 层（75%），`full_attention_interval=4` → `isRecurrent[i] = (i+1)%4 != 0`
  - 无 KV cache（固定大小 state，用 f32）
  - 计算量较轻（线性注意力 + 1D 卷积 + state 更新）
  - 权重较小（~150MB vs attention 的 ~200MB）

- **Full Attention 层**：20/80 层（25%）
  - 有 KV cache（随上下文增长）
  - 计算量较重（scaled dot-product attention，O(seq_len²)）
  - 权重较大

当前 `buildLayout()` (`llm/server.go:939-952`) **不区分层类型**，所有层统一从后往前贪心填充 GPU：

```go
// llm/server.go:1143 — greedyFit 从最后一层往前塞
for i := len(layers) - 1; i >= 0; i-- {
    // 不管 layer i 是 recurrent 还是 full attention
    // 只看 layers[i] 大小是否装得下
}
```

#### 优化思路

优先把 **full attention 层放 GPU**（计算重、有 KV cache 受益于 GPU 带宽），recurrent 层放 CPU（计算轻、无 KV cache）。

#### 当前支持现状

**不支持**，但 per-layer size 已经是准确的。

`buildLayout()` 计算每层的实际大小（`server.go:943-952`），包含该层的 weights + cache。由于 recurrent 层没有 KV cache（只有固定 state），它们在 `memory.GPU[j].Cache[i]` 中的数值比 attention 层小。所以 `greedyFit()` 实际上已经"知道"不同层大小不同。

但问题是：`greedyFit()` 从**最后一层往前**贪心塞（`server.go:1143`），不考虑哪种层更应该上 GPU。如果 layer 78 是 recurrent（小），layer 79 是 attention（大），当前逻辑先放 79（正好是 attention），但这只是巧合。

#### 实现所需改动

两种思路：

**思路 A：修改 greedyFit 的层排序**

不再从后往前，而是按"GPU 收益"排序：full attention 层优先，recurrent 层最后。

```
改动: llm/server.go greedyFit() — 传入层优先级排序
约 20 行 Go 代码
```

**思路 B：与 MoE 感知拆分联合（§7.3）**

如果已实现 tensor 级分配，自然可以把 attention 类 tensor 放 GPU、recurrent 类 tensor 视情况放 CPU。两个优化合为一体。

### 7.5 优化方向对比与组合

#### 单项收益预估

| 方向 | Prefill 收益 | Decode 收益 | 实现状态 | 改动量 |
|------|-------------|-------------|---------|--------|
| KV cache 量化 (§7.1) | 间接（GPU 多放层） | 间接（GPU 多放层） | **已支持** | 0（配环境变量） |
| iGPU offload (§6.8) | **直接 ×2**（溢出层） | 无 | 未实现 | ~30 行 C + Go |
| Speculative decoding (§7.2) | 无 | **直接 ×5-6** | 未实现 | 大（API+scheduler+runner） |
| MoE 感知拆分 (§7.3) | **直接**（全层 attn 上 GPU） | **直接**（全层 attn 上 GPU） | 未实现 | ~50 行 Go |
| Hybrid 层感知 (§7.4) | 间接（更优分配） | 间接（更优分配） | 未实现 | ~20 行 Go |

#### 组合效果（理想情况）

```
Baseline (当前):
  GPU: 7 层完整，CPU: 73 层完整
  Prefill: CPU 算力瓶颈
  Decode: CPU 带宽瓶颈

+ KV cache q8_0 (§7.1):
  GPU: ~10 层完整，CPU: ~70 层完整（GPU 多放 3 层）

+ MoE 感知拆分 (§7.3):
  GPU: 全部 80 层的 attention/recurrent（~16GB）
  CPU/iGPU: 全部 80 层的 MoE FFN（~35GB）
  → 所有层的 dense 计算都在 GPU 上

+ iGPU offload (§6.8):
  Prefill: MoE FFN 部分 offload 到 iGPU（~4 TFLOPS vs CPU ~1.9 TFLOPS）

+ Speculative decoding (§7.2):
  Decode: 每次权重读取产出 ~6 个 token 而非 1 个

+ Hybrid 层感知 (§7.4):
  进一步微调：full attention 层的 MoE FFN 优先 iGPU（计算更重）
```

#### 推荐实施顺序

```
第 1 步: KV cache 量化          — 零成本，立即收益
第 2 步: MoE 感知层内拆分       — 根本性改变 GPU 利用效率
第 3 步: iGPU prefill offload  — 加速溢出层的 prefill
第 4 步: Hybrid 层感知          — 可与第 2 步合并实现
第 5 步: Speculative decoding  — 改动最大，decode 收益最显著
```

### 7.6 备选方案参考

以下方案作为参考记录，不作为主要优化方向。

**方案 A：`buildLayout()` 跨 Library 合并**

适用于多张不同厂商 discrete GPU（NVIDIA + AMD）共存。修改 `buildLayout()` 在 `ByLibrary` 循环后新增一轮跨组竞争。~20 行 Go。当前不需要。

**方案 D：`OLLAMA_CROSS_LIBRARY=1` 环境变量**

跳过 `ByLibrary` 分组，所有 GPU 统一分配层。最简单但不区分 prefill/decode 阶段。~10 行 Go。

---

