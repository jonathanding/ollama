# 跨 Library GPU 混用的限制与优化机会

### 6.1 问题场景

考虑一台有以下配置的机器：

| 设备 | Backend | 显存/可用内存 | 计算速度 |
|------|---------|-------------|---------|
| NVIDIA RTX 4090 | CUDA | 24 GB VRAM | 极快 |
| Intel Arc / UHD iGPU | Vulkan | ~64 GB (UMA, 128GB RAM) | 慢但比 CPU 快 |

对于 80B MoE 模型（如 Qwen3-coder-next），**理想方案**是：

```
NVIDIA CUDA (24GB) → 高层（~25层，快速计算）
Intel iGPU Vulkan (64GB UMA) → 剩余层（~55层，比纯 CPU 快）
→ 全部 80 层上 GPU，无 CPU 回退
```

**但 Ollama 当前不支持这种混用。**

### 6.2 当前行为：ByLibrary 竞争选举

`buildLayout()` (`llm/server.go:954-996`) 按 Library 分组，各组独立竞争：

```go
// llm/server.go:954-996
gpuLayers := ml.GPULayersList{}
for _, gl := range ml.ByLibrary(gpus) {        // ← 按 Library 分组
    libraryGpuLayers := assignLayers(layers, gl, ...)
    if libraryGpuLayers.Sum() > gpuLayers.Sum() {
        gpuLayers = libraryGpuLayers             // ← 选层数最多的，其他全部丢弃
    }
}
```

对上述配置：

```
第1轮: CUDA 组 [NVIDIA 24GB]  → assignLayers → 25 层
第2轮: Vulkan 组 [iGPU 64GB]  → assignLayers → 80 层 (全部)
比较: 80 > 25 → Vulkan 组胜出，CUDA 组被整体丢弃
```

结果：**NVIDIA GPU 完全闲置，所有层跑 iGPU Vulkan**。这比 "NVIDIA 25层 + CPU 55层" 可能还慢，因为 iGPU Vulkan 的 FLOPS 远不如 NVIDIA CUDA。

### 6.3 限制作用的三个层次

**层次 1：设备发现去重** (`discover/runner.go:203-224`)

同一物理设备被多个 library 发现时（如 NVIDIA 同时被 CUDA 和 Vulkan 发现），会按 `PreferredLibrary()` 去重：

```go
// discover/runner.go:211-219
case ml.DuplicateDevice:
    if devices[i].PreferredLibrary(devices[j]) {
        droppedDevice = devices[j]     // CUDA 优先于 Vulkan
    }
```

`PreferredLibrary()` (`ml/device.go:552-559`): CUDA/ROCm 始终优先于 Vulkan。

> **注意**：NVIDIA 和 Intel iGPU 是**不同物理设备**（PCIID 不同），不会触发去重。两者都会保留在设备列表中，分别属于 CUDA 组和 Vulkan 组。

**层次 2：层分配分组** (`llm/server.go:955`)

如 6.2 所述，`ByLibrary()` 把设备按 Library 字段分成独立组，各组竞争，只保留一个胜者。

**层次 3：Backend 初始化过滤** (`ml/backend/ggml/ggml.go:367-372`)

未被分配到任何层的 backend 会被从 scheduler 中排除：

```go
// ggml.go:367-372
if !slices.Contains(cpuDeviceBufferType.bts, bt) {
    if c, ok := ctxs[bt]; !ok || C.ggml_get_first_tensor(c) == nil {
        continue  // ← 没有 tensor 分配到这个 backend，跳过
    }
}
```

### 6.4 底层 GGML Scheduler 支持混合 Backend

关键发现：**限制只在 Go 层（`buildLayout`），不在 C 层**。

GGML scheduler (`ggml-backend.cpp:1676-1706`) 接受任意多个不同 library 的 backend：

```c
// ggml-backend.cpp:1676
ggml_backend_sched_t ggml_backend_sched_new_ext(
    ggml_backend_t * backends,    // 可以是 [CUDA, Vulkan, CPU] 混合
    ggml_backend_buffer_type_t * bufts,
    int n_backends, ...
)
```

五遍 split 算法根据 tensor 的 buffer type 自动路由到正确的 backend，跨 backend 的数据拷贝在 `compute_splits()` 中自动完成（`ggml-backend.cpp:1515-1599`）。

即：如果上层把 CUDA 和 Vulkan backend 都传入 scheduler，并且把权重分别放在对应的 buffer 上，**scheduler 完全能正确处理混合执行**。

### 6.5 llamarunner 同样受限

两种 runner 都受 `buildLayout()` 约束，因为层分配在**它们之外**完成：

```
buildLayout() → GPULayersList → 传给 runner
```

**ollamarunner** (`runner/ollamarunner/runner.go:1321-1326`):

```go
params := ml.BackendParams{
    GPULayers: req.GPULayers,    // ← 来自 buildLayout()
}
```

→ 传给 `ggml.go:207-223` 的 `assignLayer()`，只给列表中的设备分配层。

**llamarunner** (`runner/llamarunner/runner.go:923-937`):

```go
gpuIDs := llama.EnumerateGPUs()
for _, layers := range req.GPULayers {    // ← 同样来自 buildLayout()
    for i := range gpuIDs {
        if gpuIDs[i].DeviceID == layers.DeviceID {
            numGPU += len(layers.Layers)
            tensorSplit = append(tensorSplit, float32(len(layers.Layers)))
            llamaIDs = append(llamaIDs, gpuIDs[i].LlamaID)
        }
    }
}
```

→ 传给 llama.cpp 的 `ModelParams.Devices`，只包含胜出组的设备 ID。

**测试用例也明确确认**了这个行为 (`llm/server_test.go:113-118`):

```go
{
    name:     "Multi GPU different libraries",
    gpus:     []ml.DeviceInfo{
        {DeviceID: ml.DeviceID{Library: "CUDA", ID: "gpu0"}, FreeMemory: 128MB},
        {DeviceID: ml.DeviceID{Library: "ROCm", ID: "gpu1"}, FreeMemory: 256MB},
    },
    expected: ml.GPULayersList{
        {DeviceID: ml.DeviceID{ID: "gpu1", Library: "ROCm"}, Layers: []int{0, 1}},
    },
    // ← 只有 ROCm gpu1 (256MB > 128MB)，CUDA gpu0 完全不用
}
```

### 6.6 iGPU (UMA) 的分阶段价值分析

直觉上 iGPU 和 CPU 共享同一块 RAM，没有独立显存优势。但需要分 prefill 和 decode 两个阶段分别讨论。

#### Decode 阶段（batch=1）：iGPU ≈ CPU，无优势

```
┌─────────────────────────────────────────────────┐
│              物理 RAM (128 GB, DDR5)             │
│         同一条内存总线，~90 GB/s 带宽            │
│                                                  │
│    CPU 直接访问 ←──────→ iGPU 通过 Vulkan 访问   │
└─────────────────────────────────────────────────┘
```

Decode 是**纯 memory-bandwidth bound**——每生成一个 token 要读一遍全部权重。iGPU 和 CPU 读同一块 RAM，带宽上限相同（DDR5 ~90 GB/s），**谁算都一样快**。

> 实测佐证 (llama.cpp #16230)：RTX 2060 上 Vulkan decode 164 t/s vs CUDA 163 t/s，几乎相同。Vulkan dispatch overhead（每次 ~10-50μs）对 decode 的毫秒级计算来说**微不足道**。

#### Prefill 阶段（batch=512）：iGPU 可能有优势

Prefill 是 **compute-bound**（大矩阵乘法）。这里算力对比才有意义：

| 设备 | FP32 TFLOPS | 来源 |
|------|-------------|------|
| CPU Arrow Lake 24核 (AVX2 only, **无 AVX-512**) | ~1.9 | 24c × 5GHz × 16 ops/cycle |
| CPU 12/13代 8P核 (AVX-512, 2×FMA) | ~2.8 | 8c × 5.5GHz × 64 ops/cycle |
| Intel UHD 770 (32 EU) | ~1.5 | Wikipedia: 1484-1588 GFLOPS |
| Intel Arc iGPU (128 EU, Xe-LPG) | ~4.0 | 按 EU 比例从 Xe-LP 数据估算 |

**关键发现：Arrow Lake 移除了 AVX-512，CPU 算力只有 ~1.9 TFLOPS。而 Arrow Lake 的 Arc iGPU (128 EU) 约 4 TFLOPS——是 CPU 的 2 倍。**

GPU 天生擅长大 batch 矩阵乘法的并行计算。ggml 的 Vulkan shader 使用 cooperative matrix / subgroup 操作，对 batch=512 的 matmul 效率较高。

> 实测佐证 (llama.cpp #19221)：Intel iGPU Vulkan 跑 DeepSeek-Qwen3 8B Q4_K_M 达到 ~965 t/s（含 prefill）。
> AMD Renoir APU (8 CU, 远弱于 Arrow Lake iGPU) 跑 Qwen3-30B Q8_0 prefill 达 101.88 t/s (llama.cpp #17715)。

#### 结论

| 阶段 | CPU vs iGPU | 谁更快 |
|------|-------------|--------|
| Decode (batch=1) | 相同带宽，CPU 零调度开销 | CPU 略优或持平 |
| Prefill (batch=512) | iGPU ~4 TFLOPS vs CPU ~1.9 TFLOPS (无 AVX-512) | **iGPU 可能 2× 快** |

这引出一个优化方向：**prefill 阶段让溢出层走 iGPU Vulkan，decode 阶段走 CPU**。

### 6.7 Vulkan UMA 零拷贝机制（上游已实现）

在讨论优化方案之前，需要理解一个关键事实：**Vulkan backend 在 UMA 系统上已实现零拷贝内存共享**。

#### UMA 检测

`ggml-vulkan.cpp:4436`：

```cpp
device->uma = device->properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
```

所有 iGPU 自动标记为 UMA。

#### 设备内存分配策略

`ggml_vk_create_buffer_device()` (`ggml-vulkan.cpp:2448-2451`)：

```cpp
} else if (device->uma) {
    // Fall back to host memory type
    buf = ggml_vk_create_buffer(device, size,
        {vk::MemoryPropertyFlagBits::eDeviceLocal,                              // 优先
         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});  // fallback
}
```

UMA 系统上 `eDeviceLocal` 和 `eHostVisible` 指向同一块物理 RAM。所有 Vulkan buffer 都 CPU 可见。

#### 计算时零拷贝：直接用 CPU 指针

**这是核心机制**。每个计算函数（matmul、mul_mat_id 等）在 UMA 系统上会尝试直接使用 tensor 的 CPU 内存地址：

`ggml_vk_tensor_subbuffer()` (`ggml-vulkan.cpp:5807-5814`)：

```cpp
if (ctx->device->uma) {
    ggml_vk_host_get(ctx->device, tensor->data, buffer, offset);
    // ↑ 用 tensor 的 CPU 指针 (tensor->data) 查找对应的 VkBuffer 映射
    //   如果这块内存是通过 Vulkan 的 pinned memory 分配的，直接返回 VkBuffer
}
if (!buffer) {
    // 找不到映射才用 Vulkan 自己的 buffer
    buffer = buf_ctx->dev_buffer;
}
```

`ggml_vk_mul_mat()` (`ggml-vulkan.cpp:6698-6702`)：

```cpp
if (ctx->device->uma) {
    ggml_vk_host_get(ctx->device, src0->data, d_Qx, qx_buf_offset);  // 权重
    ggml_vk_host_get(ctx->device, src1->data, d_Qy, qy_buf_offset);  // 激活值
    src0_uma = d_Qx != nullptr;  // 如果找到映射 → GPU 直接读这块内存，零拷贝
}
```

`ggml_vk_host_get()` (`ggml-vulkan.cpp:5787-5800`) 的实现：

```cpp
static void ggml_vk_host_get(const vk_device& device, const void * ptr,
                              vk_buffer& buf, size_t& buf_offset) {
    buf = nullptr;
    for (size_t i = 0; i < device->pinned_memory.size(); i++) {
        const uint8_t* addr = (const uint8_t*) std::get<0>(device->pinned_memory[i]);
        const uint8_t* endr = addr + std::get<1>(device->pinned_memory[i]);
        if (ptr >= addr && ptr < endr) {
            buf = std::get<2>(device->pinned_memory[i]);  // 找到包含此地址的 VkBuffer
            buf_offset = ((const uint8_t *)ptr) - addr;
            break;
        }
    }
}
```

#### 读回数据也零拷贝

`ggml-vulkan.cpp:6227-6230`：

```cpp
if (src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible && src->device->uma) {
    memcpy(dst, (uint8_t *) src->ptr + offset, size);  // 直接 CPU memcpy，不走 GPU
}
```

#### 物理内存视图

```
┌───────────────────────────────────────────────────────────┐
│                  物理 RAM (128 GB, DDR5)                   │
│                                                            │
│  ┌─ Layer 25 权重 ────────────────────────────────────┐   │
│  │  物理页: 0x7F00_0000 - 0x7F25_0000                 │   │
│  │                                                     │   │
│  │  VkBuffer (pinned_memory 表中注册)                  │   │
│  │    → Vulkan shader 直接读 (prefill 计算)            │   │
│  │                                                     │   │
│  │  CPU 指针 (tensor->data, 通过 vkMapMemory 映射)    │   │
│  │    → CPU AVX2 直接读 (decode 计算)                  │   │
│  │                                                     │   │
│  │  同一块物理内存，两种访问路径，零拷贝               │   │
│  └─────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────┘
```

**这意味着：在 UMA iGPU 上，CPU backend 和 Vulkan backend 之间切换计算不需要任何数据拷贝。** 权重、KV cache、激活值都可以被两个 backend 直接访问。

### 6.8 Phase-Aware Scheduling 方案设计

#### 理想行为

```
Prefill (compute-bound, batch=512):
  Layer 0-24  → NVIDIA CUDA (24GB VRAM, 最快)
  Layer 25-79 → Intel iGPU Vulkan (UMA, ~4 TFLOPS > CPU ~1.9 TFLOPS)

Decode (memory-bound, batch=1):
  Layer 0-24  → NVIDIA CUDA (24GB VRAM, 最快)
  Layer 25-79 → CPU (DDR5 90 GB/s, 与 iGPU 共享带宽，零调度开销)
```

**上游 Vulkan UMA 零拷贝已经解决了数据共享问题** (§6.7)，不需要担心 prefill/decode 切换时的数据拷贝。

#### 当前架构的限制

当前 `op_offload` (`ggml-backend.cpp:860-875`) 已经能按 batch_size 动态决定是否 offload：

```c
// ggml-backend.cpp:860-875 (现有逻辑)
if (op_offload && cur_backend_id > 0 && sched->batch_size >= 32) {
    // batch 大 → offload 到更高优先级的 backend (GPU)
}
```

但它只在已加入 scheduler 的 backend 之间做切换。当前由于 `ByLibrary` 竞争 (§6.2)，iGPU Vulkan backend 根本没有进入 scheduler，所以 `op_offload` 无从发挥。

#### 实现路径：两步改动

**第一步 (Go 层)：让 iGPU 进入 scheduler**

修改 `buildLayout()` (`llm/server.go`)，在 CUDA 组胜出后，将 UMA iGPU 附加为辅助设备。溢出层的权重仍分配到 CPU buffer（UMA 上 CPU 和 iGPU 访问同一物理内存），但 iGPU backend 被加入 scheduler。

```go
// 伪代码：buildLayout() 中，CUDA 组胜出后
for _, gpu := range vulkanGroup {
    if gpu.Integrated && gpu.UMA {
        // 不分配层给 iGPU，但确保 iGPU backend 进入 scheduler
        // op_offload 会在 prefill 时动态决定是否使用
    }
}
```

**第二步 (C 层)：`op_offload` 自动按 batch_size 切换**

现有 `op_offload` 逻辑 (`ggml-backend.cpp:860-875`) 已按 `batch_size >= 32` 做 offload。由于 UMA 零拷贝 (§6.7)，权重在 CPU buffer 上 → iGPU Vulkan 可以直接通过 `ggml_vk_host_get` 找到对应的 VkBuffer → 无需拷贝 → offload 到 iGPU 计算。

Decode 时 `batch_size = 1 < 32`，`op_offload` 不触发，计算留在 CPU。

**效果**：

```
Prefill (batch=512, batch >= 32 → op_offload 触发):
  溢出层: CPU buffer 上的权重 → iGPU 通过 UMA 零拷贝直接读 → Vulkan shader 计算

Decode (batch=1, batch < 32 → op_offload 不触发):
  溢出层: CPU buffer 上的权重 → CPU 直接读 → AVX2 计算
```

#### 需要验证的问题

1. **iGPU backend 未分配层时能否进入 scheduler？** — 当前 `ggml.go:367-372` 会跳过没有 tensor 的 backend。需要特殊处理让 UMA iGPU 即使没有直接分配的层也能作为 offload 目标。
2. **`op_offload` 的 UMA 路径是否正确？** — `ggml_vk_host_get` 能否找到 CPU 分配的权重？需要确认这些权重在 Vulkan pinned_memory 表中注册。
3. **实际 prefill 加速效果** — 理论上 iGPU ~4 TFLOPS vs CPU ~1.9 TFLOPS，但需要 llama-bench 实测。

---

