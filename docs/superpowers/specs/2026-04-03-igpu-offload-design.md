# iGPU Prefill Offload — 设计文档

**日期**: 2026-04-03（修订 2026-04-09）
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

### 1.2 为什么 iGPU 闲置（根因修订）

原始设计文档将原因归结为 `buildLayout()` 的 ByLibrary 竞选机制。这是**正确但不完整**的。

2026-04-09 bench-sweep 验证（Phase 1）显示 iGPU 利用率始终为 0。经深入代码审查，根本原因是：

**Ollama 使用 `GGML_BACKEND_DL ON` 的动态插件架构**（见下节），Phase 1 的假设前提不成立。

### 1.3 优化机会

prefill 是 compute-bound 阶段（batch=512，大矩阵乘法）。Arrow Lake iGPU 算力约 4 TFLOPS，CPU（无 AVX-512）约 1.9 TFLOPS，iGPU 有约 2× 优势。

decode 是 memory-bandwidth bound（batch=1），iGPU 与 CPU 共享 DDR5（~90 GB/s），无优势。

---

## 2. Ollama 后端架构（关键）

理解 iGPU 为何无法通过简单注入激活，需要先了解 Ollama 的库加载机制。

### 2.1 GGML 动态插件架构 (`GGML_BACKEND_DL ON`)

`CMakeLists.txt` 设置 `GGML_BACKEND_DL ON`，ggml 后端以**独立 DLL** 方式构建并动态加载：

```
dist/lib/ollama/               ← LibOllamaPath（CPU 基础库）
├── ggml-base.dll
├── ggml-cpu-*.dll
├── cuda_v13/                  ← CUDA 子目录 (OLLAMA_RUNNER_DIR=cuda_v13)
│   ├── ggml-cuda.dll
│   └── cublas*.dll, cudart*.dll
└── vulkan/                    ← Vulkan 子目录 (OLLAMA_RUNNER_DIR=vulkan)
    ├── ggml-vulkan.dll
    └── vulkan-1.dll
```

每个子目录通过 CMake preset 的 `OLLAMA_RUNNER_DIR` 变量生成：

| Preset | OLLAMA_RUNNER_DIR |
|--------|------------------|
| CPU | （无，安装到根目录） |
| CUDA 13 | `cuda_v13` |
| Vulkan | `vulkan` |

### 2.2 GPU 发现流程（`discover/runner.go`）

```go
// discover/runner.go:55
files, err := filepath.Glob(filepath.Join(ml.LibOllamaPath, "*", "*ggml-*"))
```

对每个找到的子目录，Ollama **启动子进程**探测 GPU 能力。`DeviceInfo.LibraryPath` 记录对应目录：

- CUDA NVIDIA → `LibraryPath: [lib/ollama, lib/ollama/cuda_v13]`
- Vulkan NVIDIA → `LibraryPath: [lib/ollama, lib/ollama/vulkan]` ← 仅当 `OLLAMA_VULKAN=1`
- Vulkan Intel iGPU → `LibraryPath: [lib/ollama, lib/ollama/vulkan]` ← 仅当 `OLLAMA_VULKAN=1`

**关键门控** (`discover/runner.go:105`)：

```go
} else if !envconfig.EnableVulkan() && strings.Contains(filepath.Base(dir), "vulkan") {
    slog.Info("experimental Vulkan support disabled.  To enable, set OLLAMA_VULKAN=1")
    continue  // ← 未设置 OLLAMA_VULKAN=1 时，vulkan/ 目录整体跳过
}
```

### 2.3 模型运行时库路径（`llm/server.go`）

ByLibrary 竞选后，CUDA 胜出，只有 CUDA 设备进入 `gpus`：

```go
// llm/server.go:262
gpuLibs := ml.LibraryPaths(gpus)  // → [lib/ollama, lib/ollama/cuda_v13]
```

`OLLAMA_LIBRARY_PATH` 被设为 `lib/ollama;lib/ollama/cuda_v13`，传给 runner 子进程。

### 2.4 Runner 子进程的后端加载（`ml/backend/ggml/ggml/src/ggml.go`）

```go
// OnceLoad — 只执行一次
C.ggml_backend_load_all_from_path(cpath)
// 对每个 OLLAMA_LIBRARY_PATH 路径调用，扫描该目录下所有 ggml-*.dll
```

`ggml_backend_load_all_from_path` 依次尝试加载 `cuda`、`vulkan`、`cpu` 等后端：
- 在 `cuda_v13/` 目录找到并加载 `ggml-cuda.dll` ✓
- 在 `cuda_v13/` 目录找不到 `ggml-vulkan.dll` → Vulkan 后端**从未注册**

### 2.5 根本原因链

```
OLLAMA_VULKAN=0 (默认)
    → discover/runner.go 跳过 vulkan/ 目录
    → iGPU 从未被发现，不在 systemGPUs 列表中

OLLAMA_VULKAN=1 但未修复 LibraryPaths
    → iGPU 被发现，但 ByLibrary 竞选中 Vulkan 输给 CUDA
    → LibraryPaths(gpus) 只返回 cuda_v13/ 目录
    → runner 子进程 OLLAMA_LIBRARY_PATH 不含 vulkan/
    → ggml-vulkan.dll 从未加载
    → iGPU 从未出现在 ggml.go 的 initDevices() 中
    → Phase 1 注入代码（ggml.go:368）永远不命中任何设备
    → iGPU 利用率 = 0
```

---

## 3. 旧方案修订

### 3.1 Phase 1 原方案的问题

原 Phase 1（已实现，commit `1154ed2b`）修改 `ggml.go` 注入循环：

```go
isIGPU := envconfig.IGPUOffload() && C.ggml_backend_dev_type(d) == C.GGML_BACKEND_DEVICE_TYPE_IGPU
```

**这段代码本身是正确的**，但前提条件不满足：`ggml-vulkan.dll` 从未被加载，iGPU 设备从未出现在 `gpus` 列表中，代码永远无法执行到。

### 3.2 需要增加的两个修复

在 Phase 1 注入代码生效之前，需要修复两个上游门控：

**修复 A — Discovery 门控**（`discover/runner.go`）：
当 `OLLAMA_IGPU_OFFLOAD=1` 时，允许探测 `vulkan/` 目录，即使未设置 `OLLAMA_VULKAN=1`。

**修复 B — LibraryPaths 注入**（`llm/server.go`）：
当 `OLLAMA_IGPU_OFFLOAD=1` 时，将所有已发现 iGPU 设备的 LibraryPath 加入 `gpuLibs`，
即使它们不在 ByLibrary 胜出的 `gpus` 列表中。

---

## 4. 三阶段方案（修订版）

```
Phase 0 (快速实验验证，零代码改动)
  目标: 验证 CUDA+Vulkan 双后端共存机制是否可行
  方法: 手动设置 OLLAMA_LIBRARY_PATH 包含 cuda_v13/ 和 vulkan/ 两个目录
  成功标准: GGML_SCHED_DEBUG=2 日志显示 op 被路由到 Vulkan 设备，Task Manager 显示 iGPU spike

Phase 1 (修复版): Discovery + LibraryPaths 双门控修复
  门控: OLLAMA_IGPU_OFFLOAD=1
  改动: discover/runner.go (~3 行) + llm/server.go (~10 行)
  已有改动: ml/backend/ggml/ggml.go (Phase 1 注入代码，commit 1154ed2b，继续保留)
  目标: 验证 iGPU 是否真实加速 prefill，不影响 decode
  成功标准: bench-sweep prefill_tps 提升 ≥ 20%，CV% < 10%

Phase 2 (生产): Cross-Library Layer Assignment
  前提: Phase 1 bench 满足成功标准
  改动: llm/server.go (~45 行)
  目标: 显式层分配，可观测，decode 回归 ≤ 10%
```

---

## 5. Phase 0 — 快速实验验证（零代码）

### 5.1 目标

验证 CUDA + Vulkan 双后端共存时，Phase 1 注入代码能否激活 iGPU。无需写代码，几分钟即可完成。

### 5.2 前提

1. 已构建包含 `cuda_v13/` 和 `vulkan/` 子目录的 Ollama（使用 `rebuild_windows.ps1`）
2. 或使用已安装的 Ollama（通常在 `C:\Users\<user>\AppData\Local\Programs\Ollama\lib\ollama\`）

### 5.3 操作步骤

```powershell
# Step 1: 找到 lib/ollama 路径
# 已安装 Ollama:
$LIB = "C:\Users\lingyun\AppData\Local\Programs\Ollama\lib\ollama"
# 本地构建:
# $LIB = "C:\Users\lingyun\Desktop\projects\ollama\dist\lib\ollama"

# Step 2: 验证 vulkan/ 子目录存在
ls "$LIB\vulkan\ggml-vulkan.dll"   # 必须存在
ls "$LIB\cuda_v13\ggml-cuda.dll"   # 或 cuda_v12\, 视构建版本

# Step 3: 启动 ollama serve，手动注入双后端路径
$env:OLLAMA_VULKAN       = "1"
$env:OLLAMA_IGPU_OFFLOAD = "1"
$env:OLLAMA_LIBRARY_PATH = "$LIB;$LIB\cuda_v13;$LIB\vulkan"
$env:GGML_SCHED_DEBUG    = "2"    # 打印每个 op 的 backend 分配（日志量大）

ollama serve
# 或用本地构建: go run . serve
```

```powershell
# Step 4: 另一终端，加载模型
ollama run qwen3-coder-next:latest

# Step 5: 发送一个长 prompt 触发 prefill
# 输入一段 1000 token 以上的文本，观察：
# - Windows Task Manager → GPU (Intel) → Compute 是否有 spike
# - ollama serve 终端的 GGML_SCHED_DEBUG=2 输出，找 "Vulkan" 字样的 op 分配
```

### 5.4 预期结果

- **成功**：Task Manager 显示 Intel GPU Compute 使用率在 prefill 期间出现明显 spike（>10%）
- **失败**：如果 iGPU 仍为 0，需检查 `ggml-vulkan.dll` 是否加载（日志中应有 `loaded vulkan backend` 类似信息）

### 5.5 如果 Phase 0 失败的调试路径

检查 ollama serve 日志：
- `ggml backend load all from path: .../cuda_v13` — CUDA 后端加载
- `ggml backend load all from path: .../vulkan` — Vulkan 后端加载
- `[Vulkan] Found X devices` — Vulkan 后端设备枚举

如果日志中没有 Vulkan 相关条目，说明 `ggml-vulkan.dll` 未被加载，检查 `OLLAMA_LIBRARY_PATH` 设置。

---

## 6. Phase 1 修复版实现

**前提**: Phase 0 证明机制可行（iGPU 在 CUDA+Vulkan 双后端场景下确实参与计算）。

### 6.1 修复 A — `discover/runner.go`

允许 `OLLAMA_IGPU_OFFLOAD=1` 时探测 Vulkan 目录：

```go
// 原代码（line 105）
} else if !envconfig.EnableVulkan() && strings.Contains(filepath.Base(dir), "vulkan") {
    slog.Info("experimental Vulkan support disabled.  To enable, set OLLAMA_VULKAN=1")
    continue
}

// 修改后：OLLAMA_IGPU_OFFLOAD=1 也允许探测 vulkan/ 目录
} else if !envconfig.EnableVulkan() && !envconfig.IGPUOffload() && strings.Contains(filepath.Base(dir), "vulkan") {
    slog.Info("experimental Vulkan support disabled.  To enable, set OLLAMA_VULKAN=1 or OLLAMA_IGPU_OFFLOAD=1")
    continue
}
```

### 6.2 修复 B — `llm/server.go`

在构建 runner 的 `gpuLibs` 时，将 iGPU 设备的 Vulkan 目录注入：

```go
// 现有代码（line 262）
gpuLibs := ml.LibraryPaths(gpus)

// 新增：当 OLLAMA_IGPU_OFFLOAD=1 时，注入 Vulkan 目录以让 runner 加载 ggml-vulkan.dll
if envconfig.IGPUOffload() {
    for _, dev := range systemGPUs {
        if dev.Integrated {
            for _, dir := range dev.LibraryPath {
                if !slices.Contains(gpuLibs, dir) {
                    gpuLibs = append(gpuLibs, dir)
                    slog.Debug("igpu offload: injecting vulkan libdir", "dir", dir)
                }
            }
        }
    }
}
```

> `systemGPUs` 是 `createLayout/NewLLMServer` 的参数，包含所有已发现设备（不仅仅是 ByLibrary 胜出的设备）。需要在调用 `StartRunner` 之前完成注入。

### 6.3 已有修复 — `ml/backend/ggml/ggml.go`（保留）

commit `1154ed2b` 的注入代码继续有效：

```go
isIGPU := envconfig.IGPUOffload() && C.ggml_backend_dev_type(d) == C.GGML_BACKEND_DEVICE_TYPE_IGPU
```

当 `ggml-vulkan.dll` 被加载后，Intel iGPU 出现在 `gpus` 列表中，此代码将其注入 scheduler。

### 6.4 修复后的数据流

```
OLLAMA_IGPU_OFFLOAD=1
    → discover/runner.go 允许探测 vulkan/ 目录
    → iGPU 被发现，DeviceInfo 含 LibraryPath=[lib/ollama, lib/ollama/vulkan]
    → ByLibrary 竞选：CUDA 仍胜出（gpus 只含 CUDA 设备）
    → llm/server.go 新增代码注入 iGPU 的 vulkan/ 目录
    → gpuLibs = [lib/ollama, lib/ollama/cuda_v13, lib/ollama/vulkan]
    → OLLAMA_LIBRARY_PATH 含三个路径
    → runner 子进程 OnceLoad 对每个路径调用 ggml_backend_load_all_from_path
    → ggml-cuda.dll 和 ggml-vulkan.dll 都被加载
    → initDevices() 枚举到: CUDA dGPU + Vulkan iGPU + CPU
    → ggml.go:368 Phase 1 注入代码命中 iGPU
    → iGPU 进入 scheduler backends 列表
    → op_offload 将 ne[1]>=32 的 matmul 路由到 iGPU Vulkan ✓
```

---

## 7. Phase 2 实现（与原方案相同）

Phase 2 为显式层分配，在 Phase 1 bench 通过后执行。

### 7.1 修改 `llm/server.go`

在 `buildLayout()` 竞选结束后追加 iGPU 溢出层分配（见原设计 §5.1，逻辑不变）。

Phase 2 部署后：
- iGPU 获得显式层分配 → `ctxs[iGPU_bt]` 非 nil → Phase 1 的 `ggml.go` bypass 不再需要，可移除
- Phase 2 中 iGPU 权重在 Vulkan device buffer（UMA），decode 时走 Vulkan dispatch

### 7.2 Decode 性能预期

- Vulkan dispatch overhead：~10–50 µs/dispatch
- 47 层 decode：overhead ≈ 47 × 50 µs = 2.4 ms/token
- 当前 CPU decode 耗时：~20–50 ms/token（DDR5 带宽限制）
- 预期 decode 回归：**< 10%**（需 bench-sweep 实测确认）

---

## 8. UMA 数据共享机制（已校正）

CPU 上的权重通过 `ggml_backend_cpu_buffer_type()` → `malloc()` 分配，**不在**
Vulkan `pinned_memory` 表中。`ggml_backend_vk_host_buffer_type()` 当前硬编码为
`vk_instance.devices[0]`（上游已知问题，有注释 `"Should be changed to return
device-specific host buffer type"`），在多 GPU 场景下不提供 per-device 零拷贝。

实际执行路径：当 op_offload 路由 matmul 至 iGPU 时，通过
`ggml_vk_buffer_write_2d()` 将权重从 CPU buffer 复制到 Vulkan device buffer。
在 UMA（`eHostVisible`）系统上，这是一次 DDR5 内部 `memcpy()`，消耗约 3×
带宽（copy 读 + copy 写 + shader 读）。

**性能影响**（prefill B=512，per N×K 元素）：

| 路径 | 时间 (s/N×K) |
|------|-------------|
| iGPU（含 memcpy 开销） | 3×0.5/90GB/s + 512×2/4T ≈ 2.73e-10 |
| CPU（直接计算）         | 1×0.5/90GB/s + 512×2/1.9T ≈ 5.45e-10 |

iGPU 仍约 2× 快；memcpy 开销约占收益的 6.5%，可忽略。

---

## 9. Vulkan op_offload 门控（内置，无需修改）

`ggml_backend_vk_device_offload_op()` (`ggml-vulkan.cpp:14407`)：

```cpp
return (op->ne[1] >= 32 && op->op != GGML_OP_GET_ROWS) ||
       (op->ne[2] >= 32 && op->op == GGML_OP_MUL_MAT_ID);
```

prefill（ne[1]=512）→ offload to iGPU；decode（ne[1]=1）→ 留 CPU。**phase-aware 行为完全由 Vulkan 后端内置机制保证。**

---

## 10. 验证方法

### 10.1 Phase 0 验收（进入 Phase 1 代码修改的门控）

| 指标 | 标准 |
|------|------|
| Task Manager Intel GPU Compute spike | 在 prefill 期间可见（>10%） |
| GGML_SCHED_DEBUG=2 日志含 Vulkan op 分配 | 至少有 matmul op 被路由到 Vulkan |

### 10.2 Phase 1 验收（进入 Phase 2 的门控）

```bash
bench-sweep diff baseline igpu-phase1
```

| 指标 | 标准 |
|------|------|
| prefill_tps 提升（vs baseline） | ≥ 20% |
| gen_tps 回归（vs baseline） | ≤ 5%（Phase 1 应无 decode 回归） |
| CV% (prefill_tps) | < 10% |

### 10.3 Phase 2 验收

| 指标 | 标准 |
|------|------|
| prefill_tps 提升（vs baseline） | ≥ 20% |
| prefill_tps（vs phase1） | ≥ 0%（不回退） |
| gen_tps 回归（vs baseline） | ≤ 10% |

---

## 11. 风险与局限（更新）

| 风险 | 说明 |
|------|------|
| CUDA/Vulkan 设备冲突 | 同一物理 NVIDIA dGPU 可能既被 CUDA 后端枚举，也被 Vulkan 后端枚举。`initDevices()` 会将二者都加入 gpus，scheduler 可能重复使用同一 GPU。需验证或在 Phase 1 中过滤：只对 `GGML_BACKEND_DEVICE_TYPE_IGPU` 类型设备注入，不影响 dGPU |
| vulkan/ 目录不存在 | 如果 Vulkan 后端未构建（vulkan/ 子目录缺失），Phase 1 修复 A 会探测失败但静默跳过，不影响正常使用 |
| iGPU VRAM 上报 | Windows 下 iGPU 通过 DXGI/PDH 上报共享内存（可达 64–100 GB），`greedyFit` 可能把所有溢出层都分配给 iGPU（Phase 2 风险）。Phase 1 op_offload 模式不涉及此问题 |
| MoE expert weights | op_offload 对 `GGML_OP_MUL_MAT_ID` 的处理路径（稀疏 expert copy）与 dense matmul 不同，需验证 MoE 层是否正确 offload（ne[2]>=32 条件） |
| 非 UMA iGPU | Thunderbolt 外接 GPU 可能被识别为 Integrated=true，但有 PCIe 带宽瓶颈。Phase 2 应额外校验 UMA 标记 |
