# iGPU Prefill Offload — 设计文档

**日期**: 2026-04-03
**机器**: Win11 Intel ARL 265K, 128GB DDR5, RTX 3090 (24GB VRAM)
**目标模型**: qwen3-coder-next Q4_K_M (~55 GB)
**目标**: 利用 iGPU Vulkan 加速 prefill 阶段，缓解溢出层在 CPU 上的计算瓶颈

---

## 1. 问题背景

### 1.1 当前运行状态

```
NAME                       SIZE     PROCESSOR          CONTEXT
qwen3-coder-next:latest    55 GB    58%/42% CPU/GPU    32768
```

- 约 33 层在 NVIDIA RTX 3090 (24 GB VRAM)
- 约 47 层溢出到 CPU，平均每层权重 ~0.69 GB（Q4_K_M, 80B MoE）
- iGPU (Intel Arc, Arrow Lake, 128 EU, ~4 TFLOPS) 利用率 ≈ 0

### 1.2 为什么 iGPU 闲置

`buildLayout()` (`llm/server.go:955`) 的 `ByLibrary` 竞选机制：每个 library 组独立分配层，取层数最多的组作为唯一胜者。在有 CUDA dGPU 的场景下，CUDA 以 24 GB VRAM 胜出，Vulkan iGPU 组整体丢弃。

### 1.3 优化机会

prefill 是 compute-bound 阶段（batch=512，大矩阵乘法）。Arrow Lake iGPU 算力约 4 TFLOPS，CPU（无 AVX-512）约 1.9 TFLOPS，iGPU 有约 2× 优势。

decode 是 memory-bandwidth bound（batch=1），iGPU 与 CPU 共享 DDR5（~90 GB/s），无优势。

---

## 2. 技术调研结论

### 2.1 阻塞点 A（硬性）— Scheduler 注入

`ml/backend/ggml/ggml.go:368-372`：

```go
if !slices.Contains(cpuDeviceBufferType.bts, bt) {
    if c, ok := ctxs[bt]; !ok || C.ggml_get_first_tensor(c) == nil {
        continue  // iGPU 无 tensor 分配 → ctxs[bt]=nil → 直接跳过
    }
}
```

iGPU Vulkan backend 无任何 tensor 分配 → `ctxs[bt]` 为 nil → 不进入 GGML scheduler → op_offload 无从触发。**必须修复。**

### 2.2 阻塞点 B（已验证可忽略）— UMA 零拷贝

文档 §6.7 称"UMA 零拷贝已解决数据共享问题"，**经代码核查此说法有条件成立**：

- CPU 权重通过 `ggml_backend_cpu_buffer_type()` → `malloc()` 分配，**不在** Vulkan `pinned_memory` 表中
- `ggml_backend_vk_host_buffer_type()` 硬编码 `vk_instance.devices[0]`，多 GPU 场景下只对 device[0] 有效（上游已知问题，有注释 `"Should be changed to return device-specific host buffer type"`）
- 实际执行路径：`compute_splits` 通过 `ggml_vk_buffer_write_2d()` 做 DDR5 内 `memcpy()`（UMA 下 dst 为 `eHostVisible` buffer），消耗约 3× 带宽（copy 读+写，shader 读）

**性能影响测算**（prefill B=512，per N×K）：

```
iGPU（含 memcpy 开销）: 3×0.5/90GB/s + 512×2/4T = 2.73e-10 s/(N×K)
CPU（直接计算）:          1×0.5/90GB/s + 512×2/1.9T = 5.45e-10 s/(N×K)
```

iGPU 仍约 2× 快，memcpy 开销约占收益的 6.5%，可忽略。

### 2.3 Vulkan op_offload 门控（内置，无需实现）

`ggml_backend_vk_device_offload_op()` (`ggml-vulkan.cpp:14407`)：

```cpp
return (op->ne[1] >= 32 && op->op != GGML_OP_GET_ROWS) ||
       (op->ne[2] >= 32 && op->op == GGML_OP_MUL_MAT_ID);
```

prefill（ne[1]=512）→ offload to iGPU；decode（ne[1]=1）→ 留 CPU。**phase-aware 行为完全由 Vulkan 后端内置机制保证。**

---

## 3. 方案设计

### 3.1 两阶段路径

```
Phase 1 (实验性): Scheduler Injection
  门控: OLLAMA_IGPU_OFFLOAD=1
  改动: ml/backend/ggml/ggml.go (~8 行)
  目标: 验证 iGPU 是否真实加速 prefill，不影响 decode
  成功标准: bench-sweep prefill_tps 提升 ≥ 20%，CV% < 10%

Phase 2 (生产): Cross-Library Layer Assignment
  前提: Phase 1 bench 满足成功标准
  改动: llm/server.go (~45 行) + envconfig/config.go (~3 行)
  目标: 显式层分配，可观测，decode 回归 ≤ 10%
```

---

## 4. Phase 1 实现

### 4.1 修改 `ml/backend/ggml/ggml.go`

在 scheduler 注入循环中，对 `GGML_BACKEND_DEVICE_TYPE_IGPU` 类型的设备绕过空 context 检查：

```go
// 原代码（line 368-372）
if !slices.Contains(cpuDeviceBufferType.bts, bt) {
    if c, ok := ctxs[bt]; !ok || C.ggml_get_first_tensor(c) == nil {
        continue
    }
}

// 修改后
if !slices.Contains(cpuDeviceBufferType.bts, bt) {
    isIGPU := C.ggml_backend_dev_type(d) == C.GGML_BACKEND_DEVICE_TYPE_IGPU
    if !isIGPU || !envconfig.IGPUOffload() {
        if c, ok := ctxs[bt]; !ok || C.ggml_get_first_tensor(c) == nil {
            continue
        }
    }
}
```

注入后 scheduler backend 优先级顺序：

```
index 0:   NVIDIA CUDA   (最高优先级，权重在 VRAM)
index 1:   Intel iGPU Vulkan (次高，op_offload 目标)
index N-1: CPU           (最低，op_offload 触发源)
```

op_offload 条件（`ggml-backend.cpp:862`）满足后自动路由溢出层 matmul 至 iGPU。

### 4.2 添加环境变量

`envconfig/config.go`：

```go
IGPUOffload = Bool("OLLAMA_IGPU_OFFLOAD", false)
```

### 4.3 可观测性

Phase 1 无内置日志，使用以下方式验证：
- **Windows Task Manager → GPU (Intel)**: prefill 期间 Compute 占用率出现 spike
- **`GGML_SCHED_DEBUG=2`**: 打印每个 op 的 backend 分配详情（日志量大）
- **bench-sweep**: prefill_tps 和 gen_tps 定量对比

---

## 5. Phase 2 实现

### 5.1 修改 `llm/server.go`

在 `buildLayout()` 的 ByLibrary 循环后追加第二轮分配：

```go
// 现有 ByLibrary 竞选（不改动）
gpuLayers := ml.GPULayersList{}
for _, gl := range ml.ByLibrary(gpus) {
    // ... winner-takes-all 逻辑保持不变 ...
}

// 新增：iGPU 溢出层分配
if envconfig.IGPUOffload() && gpuLayers.Sum() < len(layers) {
    igpuOverflow := assignOverflowToIGPU(layers, gpus, gpuLayers, memory, backoff)
    gpuLayers = append(gpuLayers, igpuOverflow...)
}
```

新增函数 `assignOverflowToIGPU()`：

1. 从 `gpus` 中筛选 `Integrated=true` 的设备，跳过 FreeMemory 不足的
2. 收集未被 CUDA 分配的层索引（由于 `greedyFit` 从末尾倒序填充，溢出层总是低索引端的连续区间 0...overflow-1）
3. 以溢出层大小数组调用 `assignLayers()`，获取相对索引
4. 将相对索引重映射为绝对层索引，返回 `GPULayersList`

### 5.2 Phase 2 与 Phase 1 的关系

Phase 2 部署后：
- iGPU 获得显式层分配 → `ctxs[iGPU_bt]` 非 nil → Phase 1 的 `ggml.go` bypass 不再需要，可移除
- 两阶段共用 `OLLAMA_IGPU_OFFLOAD=1` 开关

### 5.3 Decode 性能预期

Phase 2 中溢出层权重在 Vulkan device buffer（UMA），decode 时必须走 Vulkan，无法回退 CPU：

- Vulkan dispatch overhead：~10–50 µs/dispatch
- 47 层 decode：overhead ≈ 47 × 50 µs = 2.4 ms/token
- 当前 CPU decode 耗时：~20–50 ms/token（DDR5 带宽限制）
- 预期 decode 回归：**< 10%**（需 bench-sweep 实测确认）

### 5.4 测试

现有 `llm/server_test.go` "Multi GPU different libraries" 测试无需改动（Phase 2 门控在 env var 后，默认路径不变）。

新增测试 case：

```go
{
    name: "iGPU overflow assignment with OLLAMA_IGPU_OFFLOAD=1",
    // CUDA 24GB + iGPU UMA 80GB
    // 期望: CUDA 分配高索引层，iGPU 分配剩余低索引层，CPU 不分配
}
```

---

## 6. 验证方法

### 6.1 Phase 1 验收（A→B 的 gate）

```bash
# Baseline
bench-sweep run -model qwen3-coder-next -name baseline \
  -sizes 512,1024,2048,4096 -epochs 8 -warmup 4

# Phase 1（需要先以 OLLAMA_IGPU_OFFLOAD=1 启动 ollama serve）
bench-sweep run -model qwen3-coder-next -name igpu-phase1 \
  -sizes 512,1024,2048,4096 -epochs 8 -warmup 4

bench-sweep diff baseline igpu-phase1
```

**通过标准（继续 Phase 2 的条件）**：

| 指标 | 标准 |
|------|------|
| prefill_tps 提升（vs baseline） | ≥ 20% |
| gen_tps 回归（vs baseline） | ≤ 5%（Phase 1 应无 decode 回归） |
| CV% (prefill_tps) | < 10% |

### 6.2 Phase 2 验收

```bash
bench-sweep run -model qwen3-coder-next -name igpu-phase2 \
  -sizes 512,1024,2048,4096 -epochs 8 -warmup 4

bench-sweep diff baseline igpu-phase2
```

**通过标准**：

| 指标 | 标准 |
|------|------|
| prefill_tps 提升（vs baseline） | ≥ 20% |
| prefill_tps（vs phase1） | ≥ 0%（不回退） |
| gen_tps 回归（vs baseline） | ≤ 10% |

---

## 7. 文档更新

实现完成后需更正 `docs/internals/05-cross-library-gpu-mixing.md` §6.7 的描述：

> 当前描述"UMA 零拷贝已解决数据共享问题"不准确。正确描述：
> CPU 权重通过标准 CPU buffer type 分配，不在 Vulkan pinned_memory 中；
> op_offload 触发时 compute_splits 通过 DDR5 内 memcpy 将权重复制到 Vulkan device buffer，
> 在 UMA 上约消耗 3× 带宽，但对 prefill（B≥512）的加速效果影响约 6.5%，可忽略。

---

## 8. 风险与局限

| 风险 | 说明 |
|------|------|
| CUDA/Vulkan 注册顺序 | `initDevices()` 的 GPU 枚举顺序决定 scheduler backend 优先级。需验证 CUDA 始终在 Vulkan 之前注册 |
| 非 UMA iGPU | Thunderbolt 外接 GPU 可能被识别为 Integrated=true，但有 PCIe 带宽瓶颈。Phase 2 应额外校验 UMA 标记 |
| iGPU VRAM 上报 | Windows 下 iGPU 通过 DXGI/PDH 上报共享内存（可达 64–100 GB），`greedyFit` 可能把所有溢出层都分配给 iGPU。需确认与实际 DDR5 使用量的一致性 |
| MoE expert weights | op_offload 对 `GGML_OP_MUL_MAT_ID` 的处理路径（稀疏 expert copy）与 dense matmul 不同，需验证 MoE 层是否正确 offload |
