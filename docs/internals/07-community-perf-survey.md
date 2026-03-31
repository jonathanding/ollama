# 社区性能优化动向调研（2026-03-31）

> 来源：Ollama issues/PRs (`ollama/ollama`) + llama.cpp issues/PRs (`ggml-org/llama.cpp`)

### 8.1 Speculative Decoding / EAGLE3

#### llama.cpp

| PR/Issue | 标题 | 状态 | 要点 |
|----------|------|------|------|
| #18471 | Self-speculative decoding (n-gram) | **已合并** (2026-01) | 无需 draft model，搜索 token 历史中的 n-gram 模式作为 draft。GPT-OSS-120B 翻译代码注释 **2.5×**，Qwen3-235B 重度 offload **1.75×**。适合输出有重复模式的场景。 |
| #19164 | `spec: add ngram-mod` | **已合并** (2026-01) | n-gram speculative 的扩展模式。 |
| #18039 | EAGLE3 speculative decoding | **Draft PR** (进行中) | 三阶段 pipeline（feature extraction → fusion → draft generation）。LLaMA3.1-8B **2.85-3.28×**，Qwen3-8B **1.62-2.17×**。⚠️ **MoE 模型效果差**：GPT-OSS-120B 仅 0.83-1.08×，因为验证阶段多 expert 激活开销超过单 token decode 成本。maintainer ggerganov 要求重构后才合并。 |
| #15305 | EAGLE3 support request | **Open** | 社区请求 EAGLE3 支持，特别是 GPT-OSS-120B。 |
| #19712 | Speculative decoding for VL models | **Open** | 多模态 speculative decoding。 |

#### Ollama

| Issue | 标题 | 状态 | 要点 |
|-------|------|------|------|
| #5800 | Enable speculative decoding | **Open** (feature request, performance label) | 社区长期请求。contributor sammcj 提议 Modelfile 语法 `DRAFT qwen2:1.5-q4_k_m`。ExllamaV2/TabbyAPI 已有此功能，Ollama 用户因此迁移。有人自愿实现但合并流程缓慢。 |
| #1292 | Prompt lookup decoding | **Open** | 无需 draft model，字符串匹配实现 2-4× 加速。与 self-speculative 类似。 |

**关键结论**：
- **EAGLE3 对 MoE 模型无效** — 验证阶段 bottleneck 使其在 GPT-OSS-120B 上 < 1× 加速
- Self-speculative (n-gram) 已合并且对重复输出有效，但对通用 coding 场景提升有限
- Ollama 尚未暴露 llama.cpp 的 speculative decoding 接口

### 8.2 MoE 优化

#### llama.cpp

| PR/Issue | 标题 | 状态 | 要点 |
|----------|------|------|------|
| #20905 | Optimize MoE GEMV kernel (BS>1) | **已合并** (2026-03-29) | 新专用 `mul_mat_vec_q_moe` kernel。**1.06-1.77× decode 加速**（RTX 5090/4090/3090/MI60/P40/V100）。batch size 扩展到 8，RTX 5090 BS=8 达 1.80×。已测试 Qwen3MoE、GPT-OSS MoE。 |
| #19139 | Gate+expert weight fusion (`--fuse-gate-up-exps`) | **已合并** (2026-02-26) | GGUF 转换时 fuse gate+up tensor。**Qwen3-Next 80B Q2_K 提升 +12.4% prompt processing**，DeepSeek2 30B Q4_K +9.2%（RTX 5090）。 |
| #20757 | Two-tier GPU+RAM expert cache for MoE offload | **Open** (Issue) | 提议 3 层缓存系统（GPU VRAM / pinned CPU RAM / SSD）+ SLRU 驱逐策略。PoC 在 RTX PRO 2000 (8GB) 上冷启动 1.9-2.5 tok/s → 稳态 **12-14 tok/s**（vs 基线 CPU offload 0.5-1 tok/s）。**对消费级硬件跑 80B+ MoE 最具变革性。** |
| #20984 | Print `-n-cpu-moe` in llama-bench | **已合并** | 基准测试中显示 MoE CPU 核数参数。 |

#### Ollama

| Issue | 标题 | 状态 | 要点 |
|-------|------|------|------|
| #11772 | CPU offload MoE weights to reduce VRAM | **Open** (27 comments) | 请求暴露 llama.cpp 的 `--n-cpu-moe` 参数。有用户实测 qwen3.5:35b 从 10 tok/s → **57-70 tok/s**（GPU offload vs CPU MoE）。有人提议 `OLLAMA_MOE_OFFLOAD=FULL/PARTIAL/NONE`。也有人请求 per-model 配置（Modelfile）。**社区呼声极高。** |
| #14861 | Ollama vs llama.cpp speed gap (qwen3.5:35b) | **Open** (performance label) | Ollama engine 实现的 qwen3.5 比 llama.cpp engine 慢 ~2.5×。collaborator rick-github 确认是 ollamaengine 优化不足。llama.cpp PR #19504 大幅改善了 qwen3+ 性能。 |
| #14579 | Qwen3.5 much slower than llama.cpp | **Open** | 同上，用户 16k context 也非常慢。llama.cpp 60-120k context 仍有 35-40 tok/s。 |

**关键结论**：
- `--n-cpu-moe` 对 MoE 模型有**巨大**收益（实测 5-7× 提速），Ollama 未暴露此参数
- MoE GEMV kernel (#20905) 和 gate fusion (#19139) 已合并，对我们的 80B MoE 模型直接受益
- Expert caching (#20757) 是最有前景的方案，但需要大量 C++ 实现

### 8.3 Flash Attention 进展

#### llama.cpp

| PR | 标题 | 状态 | 要点 |
|----|------|------|------|
| #20525 | Native bf16 flash attention (CUDA vec kernel) | **已合并** (2026-03-22) | 消除 BF16 KV cache 的 CPU fallback。"HUGE improvement"。 |
| #19806 | CDNA3 MFMA flash attention (AMD MI300X) | **已合并** (2026-02-27) | MI300X pp512-pp4096 **+7% 到 +39%**。 |
| #20190 | SYCL flash attention | **已合并** (2026-03-08) | Intel Arc A770 + UHD 770 测试。**PP 提升 77%**，内存节省 38-463MB。 |
| #20589 | Vulkan FA dot product precision fix | **已合并** (2026-03-16) | Vulkan flash attention 精度修复。 |
| #19921 | Vulkan fp16 FA fix (Windows AMD RDNA2) | **已合并** (2026-02-26) | |
| #20998 | Flash Attention for head_dim=512 | **Open** | DeepSeek 需要的 head_k=512 支持。Blackwell **PP +5-7%**。 |
| #21029 | Vulkan FA dequant (q4_1, q5_0, q5_1, iq4_nl) | **Open** | 扩展 Vulkan FA 支持的量化类型。 |

**关键结论**：
- SYCL flash attention (#20190) 对 Intel iGPU 有 **77% PP 提升**
- 但 AMD iGPU (PHOENIX/RDNA3) 有 FA crash (#20889)，需修复
- Vulkan FA 不断完善中

### 8.4 KV Cache 优化 — TurboQuant

#### llama.cpp

| PR/Issue | 标题 | 状态 | 要点 |
|----------|------|------|------|
| #20977 | TurboQuant support | **Open** (41 comments) | Google 的 TurboQuant 算法用于 KV cache 压缩。社区积极实现中。 |
| #21089 | CPU TurboQuant KV types (TBQ3_0 / TBQ4_0) | **Open PR** | **TBQ4_0: 3.94× 压缩**，TBQ3_0: 5.22× 压缩，质量损失极小。目前仅 CPU。 |
| #21192 | TurboQuant Algo 1: random orthogonal rotation | **Draft PR** | 在 K/V 量化前应用随机正交旋转，恢复 Q4_0 KV ~39% 的困惑度损失。LLaMA-3.1-70B 在 8GB VRAM 上 KV 从 ~8GB 降到 ~2GB，释放 ~6GB 给 weight layers。仅 8MB 旋转矩阵开销。 |

#### Ollama

| Issue | 标题 | 状态 | 要点 |
|-------|------|------|------|
| #15051 | TurboQuant+RotorQuant for native Go engine | **Open** | 社区已有参考实现：[TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)（Python），[mudler/llama.cpp feat/turbo-quant](https://github.com/mudler/llama.cpp)。vllm 也在实现 (#38171)。 |

**关键结论**：
- TurboQuant 是 KV cache 的下一代压缩方案，**4-5× 压缩 + 近零质量损失**
- 对我们的 80B MoE + 24GB VRAM 场景意义重大：大幅减少 KV cache 占用
- 当前 Ollama 已支持 `OLLAMA_KV_CACHE_TYPE=q8_0|q4_0`（2× 和 4× 压缩），TurboQuant 将进一步提升

### 8.5 NVFP4 量化（NVIDIA 专项）

#### llama.cpp

| PR | 标题 | 状态 | 要点 |
|----|------|------|------|
| #20644 | NVFP4 dp4a kernel | **已合并** (2026-03-26) | Qwen3.5-27B 达 **1482 tok/s pp512**（vs CPU 468×）。初始 CUDA NVFP4 支持。 |
| #21074 | Generic NVFP4 MMQ kernel | **Open** | 在 MMVQ 基础上再提升 **+26.8% 到 +289.1%** prefill。RTX 5090 Qwen3.5-27B pp512: 1026 → **2988 tok/s**。 |

**关键结论**：
- NVFP4 是 NVIDIA 特有的 4-bit 浮点格式，prefill 提升巨大
- 但需要 NVIDIA GPU 支持，对我们 24GB GPU 上的层有直接收益
- Ollama #15157 提到 NVFP4 models 在 Windows 上的支持问题

### 8.6 Multi-GPU 与 Tensor Parallelism

#### llama.cpp

| PR/Issue | 标题 | 状态 | 要点 |
|----------|------|------|------|
| #19378 | Backend-agnostic tensor parallelism | **Open** (136 comments, very active) | `--split-mode tensor` 的跨 backend 实现。CUDA 可用（2× RTX 4090 测试），ROCm 可用，Vulkan 有 descriptor 验证错误，Metal 有非确定性输出。仅 1-2 GPU、dense model，MoE 未测试。大 context (d131072+) tensor 模式才有优势，常规场景 pipeline (layer split) 仍更快。 |
| #20518 | Vulkan async and event fixes | **已合并** (2026-03-17) | 修复多 GPU Vulkan 关键同步问题（gibberish 输出、device lost）。fence → timeline semaphore。 |
| #20551 | Vulkan: use graphics queue on AMD | **已合并** (2026-03-15) | RX 9070 XT **+4.8-10%** token generation。 |
| #21164 | Vulkan regression with 3 GPUs | **Open** | Qwen3.5 PP 从 ~3500 降到 ~350 tok/s，与 `--parallel` 有关。 |

**关键结论**：
- Tensor parallelism 尚未合并，且对 MoE 模型无支持
- Pipeline parallelism (layer split) 仍是多 GPU 的实际方案
- Vulkan 多 GPU 同步修复 (#20518) 是关键正确性修复

### 8.7 iGPU / Vulkan / Intel 相关

#### llama.cpp

| PR/Issue | 标题 | 状态 | 要点 |
|----------|------|------|------|
| #20190 | SYCL flash attention | **已合并** | Intel Arc A770 + iGPU UHD 770 测试，**PP +77%**。 |
| #20889 | Vulkan FA crash on AMD RADV PHOENIX (iGPU) | **Open** | AMD iGPU gfx1102 flash attention crash。 |
| #19887 | Low PP Q4/Q6 on Intel Arc A770 | **Open** | Intel Arc PP 性能低于预期。 |
| #20662 | Vulkan gated_delta_net sharding | **已合并** (2026-03-20) | Vulkan 下 GatedDeltaNet（Qwen3-next 的 recurrent 层）分片优化。**直接相关。** |
| #20672 | Vulkan disable mmvq on Intel Windows | **已合并** (2026-03-17) | Intel Windows 上禁用 mmvq（性能问题）。 |

#### Ollama

| Issue | 标题 | 状态 | 要点 |
|-------|------|------|------|
| #13086 | Vulkan on Intel iGPU = gibberish | **Closed** | 需要 `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`。修复后 iGPU 性能约等于 CPU +5%，但功耗低 3.6×。i7-12700 iGPU 2.48 tok/s vs CPU 4.58 tok/s（但 **0.118 tps/W vs 0.059 tps/W**）。 |
| #2169 | OpenVINO on Intel | **Open** (长期请求) | OpenVINO NPU 支持请求。有用户测试 NPU 比 CPU 快 64×（8B 模型）。但 NPU 有 static shape 限制。 |
| #15156 | Model doesn't fit Vulkan memory | **Open** | Vulkan 内存不足时的 fallback 问题。 |

**关键结论**：
- Intel iGPU Vulkan 需要 `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` 避免 gibberish
- iGPU decode 性能与 CPU 持平（共享内存带宽），但**功耗效率高 2×**
- SYCL backend 对 Intel GPU 有更好的 flash attention 支持
- GatedDeltaNet Vulkan 优化 (#20662) 对 Qwen3-coder-next 直接有益

### 8.8 Ollama Go Engine vs llama.cpp Engine 性能差距

#### 8.8.1 问题概述

Ollama 有两套 inference engine：

1. **ollamarunner (Go native engine)** — 模型推理逻辑用 Go 实现（`model/models/` 目录），调用底层 GGML C backend 做矩阵运算
2. **llamarunner (llama.cpp engine)** — 直接使用 llama.cpp 的 C++ 推理实现

新模型架构（qwen3.5、qwen3next、gpt-oss 等）被 Ollama **强制走 Go engine**，即使 llama.cpp 有完整实现。

#### 8.8.2 性能数据

**qwen3.5:35b MoE 模型** (ollama/ollama#14861)：

collaborator rick-github 的测试数据：

| 配置 | Engine | 速度 |
|------|--------|------|
| qwen3.5:35b（Ollama library 模型） | Go engine | **89 tok/s** |
| qwen3.5:35b-a3b-blind-ud（unsloth 模型） | Go engine | **86 tok/s** |
| qwen3.5:35b-a3b-ud（unsloth 模型） | llama.cpp engine | **219 tok/s** |

Go engine 慢 **~2.5×**。rick-github 评价："gpt-oss was similarly a lot slower when first implemented on the ollama engine"。

**GPT-OSS 120B MoE** (ollama/ollama#14579)：

用户 coder543 的数据（AMD R9 7950X + RTX 3090, 16k context）：

| 工具 | 配置 | 速度 |
|------|------|------|
| Ollama | 默认 | **8.5 tok/s** |
| llama.cpp | `--n-cpu-moe 24 --flash-attn` | **29.4 tok/s** |

差距 **3.5×**，主要因为 Ollama 未暴露 `--n-cpu-moe`。

**qwen3.5:27b dense 模型** (ollama/ollama#14579)：

用户 iChristGit 的数据：

| 工具 | Context | 速度 |
|------|---------|------|
| Ollama | 16k | **~12 tok/s** |
| llama.cpp | 60-120k | **35-40 tok/s** |

即使 Ollama context 更小，仍然慢 3×。

#### 8.8.3 Engine 选择机制

Runner 选择逻辑在 `llm/server.go:147-163`：

```go
// llm/server.go:147
if envconfig.NewEngine() || f.KV().OllamaEngineRequired() {
    if len(projectors) == 0 {
        tok, err = model.NewTextProcessor(modelPath)
    } else {
        err = errors.New("split vision models aren't supported")
    }
    if err != nil {
        // fallback to the old runner
        slog.Debug("model not yet supported by Ollama engine, switching to compatibility mode")
    }
}
if tok == nil {
    // tok == nil → 走 llama.cpp runner
    llamaModel, err = llama.LoadModelFromFile(modelPath, llama.ModelParams{VocabOnly: true})
}

// ...later...
if tok != nil {
    return &ollamaServer{...}   // Go engine
} else {
    return &llamaServer{...}    // llama.cpp engine
}
```

判定逻辑：
1. `envconfig.NewEngine()` → 环境变量 `OLLAMA_NEW_ENGINE=1` 强制所有模型走 Go engine
2. `f.KV().OllamaEngineRequired()` → **硬编码名单**，`fs/ggml/ggml.go:277-300`
3. 如果 Go engine 的 `NewTextProcessor` 失败 → fallback 到 llama.cpp runner

#### 8.8.4 qwen3next 被强制走 Go engine

`OllamaEngineRequired()` 的硬编码名单（`fs/ggml/ggml.go:278-300`）包括：

```go
func (kv KV) OllamaEngineRequired() bool {
    return slices.Contains([]string{
        "bert", "deepseek2", "deepseekocr",
        "gemma3", "gemma3n",
        "gptoss", "gpt-oss",
        "llama4", "mistral3", "mllama",
        "nemotron_h", "nemotron_h_moe",
        "nomic-bert", "olmo3",
        "qwen25vl",
        "qwen3", "qwen3moe",
        "qwen35", "qwen35moe",
        "qwen3next",              // ← 在名单中
        "qwen3vl", "qwen3vlmoe",
        "glm4moelite", "glmocr",
        "lfm2", "lfm2moe",
    }, kv.Architecture())
}
```

同时，qwen3next 的 Go 实现是**完整的**（`model/models/qwen3next/` 有 model.go、deltanet.go、attention.go、cache.go），`NewTextProcessor` **会成功**，因此 fallback 路径**不会触发**。

#### 8.8.5 llama.cpp 侧的 qwen3next 支持

llama.cpp runner **完整支持 qwen3next**：

| 文件 | 内容 |
|------|------|
| `llama/llama.cpp/src/llama-arch.cpp:36` | `LLM_ARCH_QWEN3NEXT` 注册 |
| `llama/llama.cpp/src/models/qwen3next.cpp` | `llm_build_qwen3next` 完整实现（含 GatedDeltaNet） |
| `llama/llama.cpp/src/llama-model.cpp:2276` | tensor 加载 |
| `llama/llama.cpp/src/llama-model.cpp:7677-7679` | graph build 分发 |

并且 llama.cpp 有两个已合并的 qwen3next 专项优化 PR：
- **#19504** — `GATED_DELTA_NET` 专用 op（替代逐步展开），graph nodes 从 14990 → 5342 (BS=1)
- **#20340** — chunked fused GDN path，进一步优化长序列

#### 8.8.6 切换方案

让 qwen3next 走 llama.cpp runner 只需**从 `OllamaEngineRequired()` 名单中删除 `"qwen3next"`**（1 行代码改动，`fs/ggml/ggml.go:294`）。删除后：

1. `OllamaEngineRequired()` 返回 false
2. `NewTextProcessor` 不会被调用 → tok 保持 nil
3. 走 `llamaModel = llama.LoadModelFromFile(...)` 路径
4. 返回 `&llamaServer{...}` → 使用 llama.cpp runner
5. 自动获得 `GATED_DELTA_NET` op 优化 + 后续所有 llama.cpp 性能改进

**收益预估**：基于 qwen3.5 的 2.5× 差距数据，qwen3next 切换到 llama.cpp runner 预期有类似提升。再叠加 `--n-cpu-moe` 可能达到 **5-7× 总提升**。

**风险**：Go engine 可能有 llama.cpp 没有的功能（如特定的 cache 策略、tokenizer 行为）。需要功能测试验证。

#### 8.8.7 其他性能问题

| Ollama Issue | 要点 |
|--------------|------|
| #12197 | GPT-OSS 随机 fallback 到 CPU，导致延迟从秒级变为分钟级。scheduler 问题。 |
| #14116 | 自动 context length 分级（≥24GB→32k）不考虑 `NUM_PARALLEL`，VRAM 溢出→model spilling→性能暴跌。 |
| #14579 | qwen3.5:27b 显示 "100% GPU" 但 CPU 100% 占用，可能 KV cache 溢出到系统内存。 |

#### 8.8.8 后续动态

| 动态 | 状态 |
|------|------|
| llama.cpp #19504 GATED_DELTA_NET op | **已合并**，qwen3next 直接受益 |
| llama.cpp #20340 chunked fused GDN | **已合并**，长序列进一步优化 |
| Ollama #14878 + #14884 MLX engine qwen3.5 优化 | **已合并**，M3 Max 35→96 tok/s (3×)，但仅 Mac MLX |
| Go engine qwen3.5 优化 | rick-github "expect it will be addressed"，**无明确时间表** |
| CUDA 支持 for MLX-style 优化 | pdevine 说 "coming really soon"，**无 PR** |

**关键结论**：
- qwen3next 被 `OllamaEngineRequired()` 强制走 Go engine，**1 行代码即可切换到 llama.cpp runner**
- llama.cpp runner 对 qwen3next 有完整支持 + 专用 op 优化
- Go engine 对 MoE/新架构优化不足，社区反馈 2.5-3.5× 慢于 llama.cpp
- 叠加 `--n-cpu-moe` 未暴露的问题，总差距可达 5-7×
- Go engine 优化无明确时间表，切换 runner 是当前最快的性能提升路径

### 8.9 DeepSeek Sparse Attention (DSA)

| llama.cpp PR | 标题 | 状态 | 要点 |
|--------------|------|------|------|
| #21149 | DeepSeekV32 with DSA | **Draft** | PoC 实现，引入 SCATTER、HADAMARD、FILL 等新 GGML op。CPU + CUDA。**目前尚无性能提升**，correctness-first。独立 KV cache 系统 `llama_kv_cache_dsa`。 |

### 8.10 综合分析：对我们场景的影响

**硬件：NVIDIA 24GB + Intel iGPU + 128GB DDR5，运行 Qwen3-coder-next 80B Q4_K_M**

#### 已合并 & 可立即受益

| 优化 | 来源 | 预期收益 | 备注 |
|------|------|----------|------|
| MoE GEMV kernel (BS>1) | llama.cpp #20905 | decode 1.06-1.77× | 需要同步上游 |
| Gate+expert weight fusion | llama.cpp #19139 | prefill +12% | 需要 `--fuse-gate-up-exps` 转换 GGUF |
| KV cache Q8/Q4 | 已支持 | KV 内存 2-4× 压缩 | `OLLAMA_KV_CACHE_TYPE=q8_0` |
| Self-speculative (n-gram) | llama.cpp #18471 | 1.75× (重复输出) | Ollama 未暴露 |
| Native BF16 flash attention | llama.cpp #20525 | 消除 CPU fallback | |
| Vulkan async fixes | llama.cpp #20518 | 多 GPU 正确性 | |
| GatedDeltaNet Vulkan sharding | llama.cpp #20662 | Qwen3-next recurrent 层优化 | 直接相关 |
| SYCL flash attention | llama.cpp #20190 | Intel iGPU PP +77% | 需 SYCL backend |

#### 高优先级待实现

| 优化 | 来源 | 预期收益 | 实现难度 |
|------|------|----------|----------|
| **`--n-cpu-moe` 暴露** | Ollama #11772 | **5-7× decode 加速** (MoE) | 低（已有 llama.cpp 支持） |
| **MoE 感知层内拆分** | 我们的 §7.3 | attention 全部上 GPU | 中（~50 行 Go） |
| **Phase-aware iGPU offload** | 我们的 §6.8 | prefill 加速 | 中（~30 行 C+Go） |
| **TurboQuant KV cache** | llama.cpp #21089 | KV 4-5× 压缩 | 高（等上游合并） |

#### 长期关注

| 优化 | 来源 | 预期收益 | 状态 |
|------|------|----------|------|
| Expert caching (3-tier) | llama.cpp #20757 | 稳态 12-14× vs 朴素 CPU offload | 提案阶段 |
| EAGLE3 speculative | llama.cpp #18039 | Dense model 2-3×，**MoE 无效** | Draft PR |
| Tensor parallelism | llama.cpp #19378 | 多 GPU 加速 | Open，MoE 未测试 |
| NVFP4 MMQ kernel | llama.cpp #21074 | prefill +289% (NVIDIA) | Open |
| DSA (Sparse Attention) | llama.cpp #21149 | 长 context 优化 | 早期 PoC |

#### 推荐优先级排序（更新）

```
第 1 步: --n-cpu-moe 暴露        — Ollama 已有上游支持，实测 5-7× MoE decode 加速
第 2 步: KV cache Q8 量化         — 零成本，立即收益
第 3 步: MoE 感知层内拆分         — 根本性改变 GPU 利用效率
第 4 步: Gate+expert fusion       — 需要重新转换 GGUF，prefill +12%
第 5 步: Phase-aware iGPU offload — 加速溢出层的 prefill
第 6 步: TurboQuant KV            — 等上游合并，4-5× KV 压缩
第 7 步: Expert caching (3-tier)  — 最有前景但实现复杂
```
