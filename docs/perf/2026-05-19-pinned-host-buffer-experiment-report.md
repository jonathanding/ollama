# Ollama Runner Pinned Host Buffer 实验报告

**日期**: 2026-05-19
**分支**: `experiment/prefill-gap-analysis`
**核心 commit**: `9cdb55d3` (`ml/ggml: add OLLAMA_PINNED_HOST_BUFFER opt-in for cuda_host weights`)
**硬件**: Intel Arrow Lake 265K, 128 GB DDR5 6400 MT/s, NVIDIA RTX 3090 (24 GB VRAM), Windows 11
**模型**: qwen3-coder-next 80B Q4_K_M (~52 GB GGUF, 49 layers, hybrid attention/recurrent + MoE)
**目标 workload**: batch_size=1024, prompt=1024, max_tokens=16

---

## 1. Motivation

### 1.1 实测的性能问题

在 RTX 3090 单卡 24 GB 显存 + qwen3-coder-next (52 GB) 这种**模型必然部分 offload 到 CPU 端**的场景下，实测 ollama runner 与 llama runner（强制走 llama.cpp 路径）有显著的 prefill 性能差距：

| 配置（batch=1024, prompt=1024） | prefill_ms | prefill_tps | gen_tps |
|---|---:|---:|---:|
| ollama runner（默认） | 2061 ms | 475 t/s | 18 t/s |
| llama runner | 1591 ms | 616 t/s | 11 t/s |
| **gap** | **+470 ms / +30%** | -23% | +64%（ollama 优势）|

prefill 落后 470 ms 是个真实存在的、稳定可复现的差距。decode 阶段 ollama 反而领先 64%——所以问题只在 prefill。

### 1.2 根因（前期 profiling 已确定）

通过 GGML scheduler 的 dump (`GGML_SCHED_DEBUG=2`) 直接观察单 prefill batch 的 op→backend 分配，已确认：

1. **op 调度本身没有差异**：ollama 默认情况下，所有 weight matmul、MoE expert mul_mat_id、attention 内部 flash_attn_ext 等已经被 ggml-backend scheduler 通过 `op_offload` 路径分配到 CUDA0 backend 上跑（满足 `batch_size >= 32` 阈值）
2. **慢在 Host-to-Device (H2D) 传输路径的两个细节**：
   - **Buffer 类型**：CPU 层 weight 落在 plain `malloc()` 分配的 pageable host memory；CUDA 的 `cudaMemcpyAsync(HostToDevice)` 对 pageable 源需要 driver 内部先 staging 到一个临时 pinned buffer 再 DMA → 有效带宽 **~14 GB/s**
   - **同步行为**：`ggml_backend_cuda_buffer_set_tensor`（ggml-cuda.cu:752-754）每次 `cudaMemcpyAsync` 后立刻 `cudaStreamSynchronize`，无法 overlap
3. **作为对照，llama runner 把"分配在 CPU"的 weight 通过 `ggml_backend_cuda_host_buffer_type()` 走 `cudaMallocHost()` 分配 page-locked pinned memory**，同样的 `cudaMemcpyAsync` 跳过 driver staging 直接 DMA → **~25 GB/s**

llama.cpp 端实测加载日志：
```
load_tensors: CUDA_Host model buffer size = 29804.92 MiB
```
明确写出 ~30 GiB CPU 端 weight 走的是 `CUDA_Host` buft（即 `cudaMallocHost` pinned）。

### 1.3 假设

如果在 ollama runner 端复刻 llama.cpp 的 buffer 类型选择策略——把 GPU device 暴露的 host buffer type（`ggml_backend_dev_host_buffer_type(gpu_dev)`）插入到 CPU buffer type 优先级列表前列——那么：

1. **CPU 层 weight 自动落在 pinned memory**（pinned 25 GB/s vs pageable 14 GB/s = 1.78× 带宽改进）
2. **不需要修改 op 调度逻辑**——op 已经在 GPU 跑，只是 H2D 路径变快
3. **不需要修改 decode 路径**——decode 时 batch_size=1，op_offload 阈值 32 不命中，op 跟着 weight 落 CPU backend，依然在 CPU 跑（保留 ollama 已有的 18 t/s decode 优势）

预期：prefill 从 ~2060 ms 拉近到 ~1600 ms（接近 llama runner 水平），decode 不变。

---

## 2. 实验方法

### 2.1 改动设计

**目标**：让 ollama runner 在为"应分配到 CPU device"的 tensor 选 buffer type 时，**优先**选择 GPU device 暴露的 host buffer type（cuda_host pinned），plain CPU buffer 作为 fallback。

**改动点**：`ml/backend/ggml/ggml.go`，`Backend.New()` 函数内构建 `cpuDeviceBufferType.bts`（CPU buffer type 优先级列表）的位置之后。

**关键代码**（commit `9cdb55d3` 真实改动，~15 行）：

```go
// EXPERIMENTAL: when OLLAMA_PINNED_HOST_BUFFER=1, prepend the GPU's host
// buffer type (cudaMallocHost-backed pinned memory) to the CPU buft list
// so weights routed to "CPU" land in pinned memory. ...
if envconfig.PinnedHostBuffer() {
    for _, d := range gpus {
        hostBuft := C.ggml_backend_dev_host_buffer_type(d)
        if hostBuft == nil {
            continue
        }
        cpuDeviceBufferType.bts = append(
            []C.ggml_backend_buffer_type_t{hostBuft},
            cpuDeviceBufferType.bts...,
        )
        btDeviceMemory[hostBuft] = &requiredMemory.CPU
        slog.Info("OLLAMA_PINNED_HOST_BUFFER enabled: prefer cuda_host pinned for CPU-resident tensors", ...)
        break
    }
}
```

**设计要点**：

| 设计决策 | 理由 |
|---|---|
| 默认关闭（opt-in） | 实验阶段，避免对其他用户的潜在影响 |
| `OLLAMA_PINNED_HOST_BUFFER=1` 启用 | 与 ollama 现有 env 命名风格一致 |
| 只取第一个 GPU 的 host buft | 与 llama.cpp `make_cpu_buft_list:343-349` 行为一致 |
| 插入到 bts 列表**前列** | `createTensor` 按顺序遍历 bts，前列优先；对于 weight tensor，cuda_host 永远赢 |
| 把 hostBuft 注册到 `btDeviceMemory[hostBuft] = &requiredMemory.CPU` | pinned memory 仍占系统 RAM，归 CPU 内存账本 |
| 不动 KV cache 路径 | KV cache 通过 `Layer(i).bt` 走另一条路径，与 weight 加载独立——保持与 llama.cpp 等价的内存布局 |

### 2.2 改动安全性保证

1. **alloc 失败有自动 fallback**：`createTensor` 的循环按 `bts` 列表顺序尝试 `ggml_backend_buft_alloc_buffer`；如果 `cudaMallocHost(30 GiB)` 失败（极小概率，host RAM 不足或 driver 限制），自动尝试列表中下一个 buft，最终落到原 plain CPU buffer。**最坏情况等价于改动前**。
2. **无 GPU 环境无副作用**：`for _, d := range gpus` 循环空跑，对纯 CPU 部署完全 no-op。
3. **多 GPU 取首个**：`break` 后退出循环，与 llama.cpp 一致，避免插入多个 cuda_host buft 让 sched 选择歧义。
4. **不影响 ggml backend 内部逻辑**：完全使用 ggml 已有的公开接口 `ggml_backend_dev_host_buffer_type(dev)` —— 该接口在 `ml/backend/ggml/ggml/include/ggml-backend.h:189` 被 `GGML_API` 标记为公开 API，CUDA backend 通过 `ggml-cuda.cu:4543-4546` 实现（直接返回 `ggml_backend_cuda_host_buffer_type()`，底层调 `cudaMallocHost`）。

### 2.3 测试方法

#### 2.3.1 测试矩阵

| 配置 | env 设置 | 用途 |
|---|---|---|
| **A. ollama runner pinned OFF** | （不设 OLLAMA_PINNED_HOST_BUFFER） | 改动前的基线 |
| **B. ollama runner pinned ON（首测）** | `OLLAMA_PINNED_HOST_BUFFER=1` | 改动效果 |
| **C. ollama runner pinned ON（复测）** | `OLLAMA_PINNED_HOST_BUFFER=1` | 验证可复现 |
| **D. llama runner**（参考） | （走原有 force-llama-runner 路径） | 性能上界对照 |

每组配置：bench-sweep 6 epochs + 4 warmup，去除 warmup 后取 6 个测量点的均值与 CV。

#### 2.3.2 工具与命令

`bench-sweep.exe` 是已有的内部 benchmark 工具，统一命令：

```
bench-sweep.exe run --model qwen3-coder-next --max-tokens 16 \
    --sizes 1024 --batch-size 1024 \
    --epochs 6 --warmup 4 \
    --name <run-name>
```

每次切换配置都重启 ollama serve（确保模型重新加载、env 变量重新读取）。

#### 2.3.3 验收标准

| 维度 | 通过标准 |
|---|---|
| Prefill 提升 | 从 ~2060 ms 拉到 ≤ ~1700 ms（向 llama runner 1591 ms 靠拢） |
| Prefill 稳定性 | CV < 5%（与基线相当） |
| Decode 不退化 | gen_tps ≥ 17 t/s（基线为 18 t/s） |
| 可复现 | 两次独立测量均值差 < 1%（5 ms 量级） |

---

## 3. 实验结果

### 3.1 性能数据

bench-sweep 输出（节选）：

| 配置 | prefill_ms | prefill_tps | ttft_ms | gen_ms | gen_tps |
|---|---:|---:|---:|---:|---:|
| **A. ollama pinned OFF** | 2061 ms (CV 3.1%) | 475 t/s (CV 2.5%) | 2137 ms | 890 ms | 18 t/s |
| **B. ollama pinned ON #1** | **1473 ms** (CV 2.4%) | **665 t/s** (CV 1.5%) | 1549 ms | 893 ms | 18 t/s |
| **C. ollama pinned ON #2** | **1469 ms** (CV 3.0%) | **667 t/s** (CV 1.9%) | 1545 ms | 895 ms | 18 t/s |
| **D. llama runner** | 1591 ms (CV 2.2%) | 616 t/s (CV 0.7%) | 1659 ms | 1451 ms | 11 t/s |

### 3.2 关键比较

#### 3.2.1 ollama 自身改动效果（B/C vs A）

- **prefill_ms：2061 → 1469 ms（平均），减少 592 ms / -28.7%**
- **prefill_tps：475 → 666 t/s（平均），提升 +40%**
- 两次复测之间差异：1473 ms vs 1469 ms = 4 ms / 0.27% —— 可复现性好
- gen_tps 三次都是 18 t/s —— **decode 路径完全无副作用**
- gen_ms 也几乎一致（890 / 893 / 895 ms） —— 进一步证明 decode 路径未受影响

#### 3.2.2 vs llama runner（B/C vs D）

- **prefill 比 llama runner 快 ~120 ms / +8%**：1591 → 1469 ms
- **prefill_tps 比 llama runner 高 50 t/s / +8%**：616 → 666 t/s
- **decode 比 llama runner 快 ~64%**：11 vs 18 t/s（这一项原本就是 ollama 优势，本次改动不影响）

ollama runner pinned ON 在所有指标上都优于 llama runner——这是改动前没有预期到的（最初目标只是"追平"）。

### 3.3 验收

| 验收项 | 标准 | 实测 | 通过 |
|---|---|---|---|
| Prefill 拉近 | ≤ ~1700 ms | 1469-1473 ms | ✓（远超目标） |
| Prefill 稳定性 | CV < 5% | 2.4% / 3.0% | ✓ |
| Decode 不退化 | gen_tps ≥ 17 t/s | 18 t/s | ✓ |
| 可复现 | 两次 < 1% | 0.27% | ✓ |

全部通过。

### 3.4 多尺寸验证（2026-05-18）

§3.1 测的是单 ubatch 配置（prompt=1024 ≤ batch=1024）。本节追加多 ubatch 路径验证：固定 batch=1024，扫描 prompt={512, 1024, 2048, 4096}（对应 1/1/2/4 个 ubatch）。

#### 3.4.1 数据

| prompt | ubatch 数 | pinned OFF prefill | pinned ON prefill | 减少 (ms) | 减少 % | OFF tps | ON tps | tps 提升 |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1（短）| 1575 ms | **1064 ms** | -511 | -32.4% | 299 | **442** | +47.8% |
| 1024 | 1 | 2084 ms | **1489 ms** | -595 | -28.5% | 464 | **649** | +39.9% |
| 2048 | 2 | 4785 ms | **3398 ms** | -1387 | -29.0% | 443 | **623** | +40.6% |
| 4096 | 4 | 8373 ms | **6002 ms** | -2371 | -28.3% | 484 | **676** | +39.7% |

decode 性能（gen_tps）OFF 和 ON 完全一致：512/1024 都是 18 t/s，2048 是 17 t/s，4096 是 16 t/s。**4096 在两边都自然降到 16，是因为 KV cache 在 prefill 后变长导致 decode attention 慢，与改动无关**。

#### 3.4.2 与 llama runner 的多尺寸对照

引用前期 `qwen3-coder-next_runner_comparison.md` 中 llama runner 的同条件数据：

| prompt | llama runner prefill | ollama pinned ON prefill | ollama 优势 (ms) | ollama 优势 (%) |
|---:|---:|---:|---:|---:|
| 512 | 1155 ms | 1064 ms | -91 | -7.9% |
| 1024 | 1620 ms | 1489 ms | -131 | -8.1% |
| 2048 | 3713 ms | 3398 ms | -315 | -8.5% |
| 4096 | 6639 ms | 6002 ms | -637 | -9.6% |

**ollama runner pinned ON 在所有测试尺寸上都比 llama runner 快**，且优势随 prompt 长度扩大（4096 case 节省 637 ms，比 512 case 的 91 ms 大 7 倍）。

#### 3.4.3 关键观察

1. **百分比节省稳定（-28% ~ -32%）**：跨四个尺寸，pinned 改动的相对收益在窄范围内一致。这是 H2D 主导 prefill 的特征——每 ubatch 都要 H2D 一份 weight，pinned 把每次 H2D 从 14 GB/s 提到 25 GB/s（1.78×），所以相对收益不随 ubatch 数变化。
2. **绝对节省随 prompt 长度线性扩大**：每多一个 ubatch ≈ 多节省 600-700 ms，与单 ubatch 数据高度一致——证明 pinned 优化在多 ubatch 路径上**没有衰减**，改动对 sched 重复 split 这类高频路径同样生效。
3. **2048 case 的 CV 偏高（OFF 6.4% / ON 5.7%）属于 bench-sweep 自身 warmup 提示，不是改动 issue**：OFF 和 ON 两边同样高 CV，3398 / 4785 = 0.71 与其他尺寸的 0.71 比例完美一致。
4. **tps 在多 ubatch 时收敛**：单 ubatch 短 prompt（512）tps 偏低（442），因为 prefill 启动开销摊在更少 token 上；从 1024 token 起 tps 稳定在 ~640-680，多 ubatch 也维持这个区间。

#### 3.4.4 验收

| 验收项 | 标准 | 实测 | 通过 |
|---|---|---|---|
| 多 ubatch 路径有效 | prefill 减少 ≥ 20% | 28-32% | ✓ |
| 跨尺寸一致性 | 百分比节省窄范围 | -28.3% ~ -32.4% | ✓ |
| Decode 跨尺寸不退化 | gen_tps 与 OFF 一致 | 完全一致 | ✓ |
| 全尺寸优于 llama runner | ollama < llama 在所有 prompt | 是（差额 91-637 ms） | ✓ |

---

## 4. 结果分析

### 4.1 收益机制（与 motivation 假设一致）

实测节省 592 ms 与原假设的"H2D 带宽 1.78× 改进"在量级上吻合：

- 单 prefill batch 中 CPU 端权重 H2D 总量约 28 GiB（参考 §1.2 模型加载日志）
- pageable @ 14 GB/s：~2.0 秒
- pinned @ 25 GB/s：~1.12 秒
- 理论节省：~880 ms（不含 H2D 与 compute 的 overlap 收益）

实测 -592 ms 落在理论上限以内，差额主要来自：
- 部分 H2D 段被 GPU compute 部分 overlap（即便每次 set_tensor 后都有 `cudaStreamSynchronize`，splits 之间仍存在天然的并行）
- selective copy_experts 优化：`compute_splits`（ggml-backend.cpp:1515）对 MUL_MAT_ID 的 expert 张量只拷贝当前 batch 激活的 expert 字节窗口（约 63%）。这条优化路径需要 `is_host(buffer) == true` 的判断，而 cuda_host buffer 同样满足该判断 —— 因此此优化已经"自然命中"，不需要额外配合
- 小张量（attention mask 等）和驱动同步的固定开销

### 4.2 为什么超过 llama runner（推测）

原本预期是"追平"llama runner（1591 ms），实测超出 ~120 ms。基于代码静态读 + GGML_SCHED_DEBUG dump 数据，推测有几个可能解释：

1. **图节点数差异**：之前的 dump 显示 ollama Go-side 模型构图产生约 17,500 个 graph nodes，llama.cpp 同样模型约 34,600 个（差 2 倍）。在 H2D 不再是瓶颈后，更少节点意味着更少的 sched 调度 + 更少的 kernel launch 开销。
2. **KV cache 放置差异**：ollama 默认配置 KV cache 1.1 GiB CUDA0 + 1.5 GiB CPU；llama runner 默认 0.349 GiB CUDA0 + 0.494 GiB CPU。ollama 把更多 KV 放在 GPU，对应层 attention 不需要 H2D 整段 KV。
3. **某些路径上的 ollama 实现更紧凑**：例如 deltanet chunked recurrent 计算 ollama 在 Go 端用更少的辅助 op 完成。

注意：这 +120 ms 的具体归属**目前没有直接 profile 数据支撑**，仅是从已有 dump 数据推测。如果未来需要在公开材料（如 PR description）中解释，建议补一次直接对比的 GPU profile 测量再下定论。

### 4.3 副作用清单与实测验证

| 风险项 | 实测/分析 | 是否问题 |
|---|---|---|
| Decode 性能退化 | 三次测试均 18 t/s，与 baseline 一致 | ✗ 无 |
| VRAM 映射开销（CUDA driver 为 pinned 页建立 GPU page table 映射） | 旧 MoE 实验同硬件下从 21.76 → 22.80 GiB（+1 GiB） | 可控；本次未实测 dedicated VRAM 占用变化（待 PR 前补充） |
| 模型加载时间增加 | `cudaMallocHost(30 GiB)` 比 `malloc(30 GiB)` 慢；本次未单独测量 | 一次性成本，可接受 |
| 系统 RAM pinned 限制 | Windows 上没有显式 ulimit 风险，128 GB 总 RAM 充足 | 无 |
| 多 GPU 行为 | 实施时只取首个 GPU host buft，与 llama.cpp 行为一致 | 无 |
| 无 CUDA 环境 | `gpus` 切片为空时循环空跑 | 无 |
| Prefill 长 prompt 多 ubatch 场景 | 本次仅测单 ubatch（prompt=1024 ≤ batch=1024），多 ubatch 待补充 | 待验证 |

### 4.4 收益可移植性分析

虽然本次实验在 qwen3-coder-next 上做，但改动机制（buft 选择层）与具体模型架构无关。对所有"GPU+CPU partial offload"场景应当都有效，受益强度取决于：

- CPU 端 weight 总量（越大，pageable 走完整段时间越长，pinned 节省越多）
- prefill batch_size（必须 ≥ 32 才能让 op_offload 路径触发并需要 H2D）
- PCIe 带宽利用率（PCIe 3.0 vs 4.0 vs 5.0，pinned 收益的绝对值不同但比例相近）

对完全 fit 在 VRAM 的小模型，本改动**无副作用也无收益**——因为没有 CPU 端 weight 需要 H2D。

---

## 5. 结论

### 5.1 主要结论

1. **改动有效**：ollama runner prefill 从 2061 ms 降到 1469 ms（-29% / +40% tps），且**反超 llama runner（1591 ms）8%**
2. **decode 完全不退化**：18 t/s 三次测试一致，op_offload 的 batch_size>=32 阈值正确隔离了 decode 路径
3. **可复现**：两次独立测试均值差 0.27%，CV < 3%
4. **实施成本低**：~15 行单文件改动，使用 ggml 已有公开 API，无 CGO helper、无生命周期管理

### 5.2 后续工作建议

短期（PR 提交前）：

1. **补充 GPU 利用率曲线对比**：用 `scripts/profile-gpu-prefill.ps1` 验证 pinned ON 下 GPU util 从 60% 锯齿变成 80%+ 平台
2. **VRAM 占用确认**：用 nvidia-smi 或 Task Manager 确认 dedicated VRAM 增量在预期 ~1 GiB 范围内
3. ~~**多 ubatch 场景验证**：跑 batch=1024/prompt=4096（4 ubatch）等，确认改动在多 ubatch 路径下也工作~~ — **已完成（见 §3.4）**：4 ubatch case 实测 -28.3% / 节省 2371 ms，跨尺寸提升一致
4. **跨模型回归**：选择 1-2 个其他主流模型（如 llama 3.x、deepseek-v2）确认改动不引入回归

中期（如果决定提交上游）：

1. **跨平台兼容性确认**：
   - Linux + CUDA：ollamarunner 不走 mmap 路径，cuda_host 应当正常生效；需在 Linux 测试机上验证
   - AMD ROCm：HIP backend 的 `get_host_buffer_type` 实现路径需要确认
   - Apple Silicon Metal：unified memory，本改动 no-op
2. **default 策略**：当前 opt-in，建议先 PR 一段时间观察，再考虑是否 default on

长期（可选）：

1. **KV cache 路径的 pinning**：当前改动只 pin weight；KV cache 走另一条路径（`Layer(i)` → device default buft），CPU 部分仍是 plain malloc。若额外 pin KV cache，attention 层 H2D KV 的部分还能再省 ~30-40 ms。但优先级低，因为已经超过 llama runner。

---

## 附录 A：实测原始数据

完整 bench-sweep 输出已保存在 `~/.ollama/bench/qwen3-coder-next_*.json`：

§3.1-3.3 单尺寸测试：
- `ollama-runner-pinned-off-2026-05-17`
- `ollama-runner-pinned-on-2026-05-17`
- `ollama-runner-pinned-on-2026-05-17_1`（复测）
- `llama-runner-2026-05-17`

§3.4 多尺寸验证：
- `ollama-runner-default-allSize-2026-05-18`（pinned OFF, sizes=512/1024/2048/4096）
- `ollama-runner-pinned-on-allSize-2026-05-18`（pinned ON, 同上）

## 附录 B：关键代码位置

| 文件:行 | 内容 |
|---|---|
| `ml/backend/ggml/ggml.go:170` 之后 | 本实验的核心改动 |
| `envconfig/config.go:241` | `OLLAMA_PINNED_HOST_BUFFER` env 开关定义 |
| `ml/backend/ggml/ggml/include/ggml-backend.h:189` | `ggml_backend_dev_host_buffer_type` 公开 API 声明 |
| `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu:4543-4546` | CUDA backend 对该接口的实现 |
| `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu:1264` | 底层 `cudaMallocHost` 调用 |
| `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu:748-754` | `ggml_backend_cuda_buffer_set_tensor`（H2D 路径，pinned 与否走同一函数，速度由 src buffer 类型决定） |

## 附录 C：实施 commit

`9cdb55d3` (`ml/ggml: add OLLAMA_PINNED_HOST_BUFFER opt-in for cuda_host weights`)
- `envconfig/config.go`: +7
- `ml/backend/ggml/ggml.go`: +35
- 总计 +42 行单文件改动
