# Ollama + llama.cpp 在 Windows 上的完整技术笔记

> 整理自对话记录，2026-03-18 / 2026-03-19

---

## 1. 架构总览

ollama 对接 llama.cpp 分两层：

```
┌─────────────────────────────────────────────┐
│                  ollama.exe                  │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │   层1：编译期 CGO（llama 核心）       │   │
│  │   llama/llama.go                    │   │
│  │   import "C" + #cgo                 │   │
│  │   → 直接调用 llama.cpp C 函数        │   │
│  │   → 静态编译进 ollama.exe            │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │   层2：运行期 DLL 动态加载（GPU后端） │   │
│  │   ggml-backend-reg.cpp              │   │
│  │   LoadLibraryW → ggml-*.dll         │   │
│  │   路径：<exe>\lib\ollama\<后端名>\  │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
         ↓ 运行时加载
  ggml-cuda.dll / ggml-vulkan.dll / ggml-cpu.dll
```

**关键点**：
- llama.cpp 以 **vendor 模式**内嵌在 `ollama/llama/llama.cpp/`，不是 submodule
- `git clone` 不需要 `--recursive`，普通 clone 即包含完整 llama.cpp
- cmake 只编译 GPU 后端 DLL，`ollama.exe` 本体由 `go build` 编译

---

## 2. 完整调用链

| 步骤 | 发生了什么 | 文件:行号（近似） |
|------|-----------|----------------|
| 1 | `ollama serve` CLI 入口 | `cmd/cmd.go ~1821` → `server.Serve()` |
| 2 | 收到模型推理请求 | `llm/server.go` `NewLlamaServer()` ~142 |
| 3 | 启动 runner 子进程，注入 PATH + OLLAMA_LIBRARY_PATH | `llm/server.go` `StartRunner()` ~320 |
| 4 | 子进程分发到具体 runner | `runner/runner.go` `Execute()` ~10 → `llamarunner` 或 `ollamarunner` |
| 5 | 调用 `llama.BackendInit()` | `runner/llamarunner/runner.go ~967` |
| 6 | 触发 DLL 扫描 | `llama/llama.go ~62` → `ggml.OnceLoad()` → `C.ggml_backend_load_all_from_path()` |
| 7 | `LoadLibraryW` 加载 ggml-*.dll | `ggml-backend-reg.cpp ~624`，扫描指定目录 |
| 8 | 调用 `C.llama_backend_init()` | `llama/llama.go ~64` |
| 9 | 父进程发 `/load` 加载模型 | `llamarunner/runner.go ~882` → `C.llama_model_load_from_file` |
| 10 | 推理运行 | `llama/llama.cpp/src/llama.cpp ~955` |

---

## 3. Build 流程（cmake vs go build 分工）

`scripts/build_windows.ps1` 主流程（行 523-535）：

```powershell
cpu        # cmake → ggml-cpu.dll
cuda12     # cmake → ggml-cuda.dll
cuda13     # cmake → ggml-cuda.dll
rocm6      # cmake → ggml-hip.dll
vulkan     # cmake → ggml-vulkan.dll
ollama     # go build → ollama.exe   ← 独立步骤，不是 cmake
```

**cmake 只负责编译 GPU 后端 DLL，ollama.exe 由 go build 单独编译。两个步骤完全分开。**

### ollama 函数（行 317-323）

```powershell
function ollama {
    go build -trimpath -ldflags "..." .   # ← Go 编译器，不是 cmake
    cp .\ollama.exe "${script:DIST_DIR}\"
}
```

### go build vs go run

```powershell
# 开发时正确姿势：先编译，反复运行
go build .
$env:OLLAMA_VULKAN = "1"
.\ollama.exe serve

# go run . serve = go build + 立即执行临时 exe，每次都重新编译，调试时低效
```

`serve` 是传给 ollama 程序的子命令参数，不是 `go` 的参数。`go run . serve` 里 `.` 指当前目录所有 `.go` 文件，`serve` 是传给生成的临时 exe 的 argv。

### cmake 编的 vs CGO 编的：不是重复，是职责分离

**关键点**：cmake 已经编出了 DLL，`go build` 仍然会触发 C++ 编译——因为两者编出来的东西职责完全不同。

```
┌─────────────────────────────────────────────────────┐
│  cmake -B build && cmake --build build              │
│                                                      │
│  → ggml-vulkan.dll      ← Vulkan GPU kernel         │
│  → ggml-cpu-haswell.dll ← AVX2 优化的矩阵乘法       │
│  → ggml-base.dll                                    │
│                                                      │
│  ★ 运行时被 LoadLibraryW 动态加载的后端插件          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  go build .  （通过 CGO）                           │
│                                                      │
│  → llama/llama.cpp/src/llama.cpp          ┐         │
│  → llama/llama.cpp/src/llama-context.cpp  ├ 静态    │
│  → llama/llama.cpp/common/common.cpp      ┘ 链接    │
│  → ml/backend/ggml/ggml/src/ggml.cpp               │
│  → ml/backend/ggml/ggml/src/ggml-backend-reg.cpp   │
│    （DLL 搜索加载逻辑就在这里）                      │
│                                                      │
│  ★ llama 推理核心，静态烧进 ollama.exe              │
└─────────────────────────────────────────────────────┘
```

整体结构：

```
ollama.exe  （go build 生成）
├── llama 推理核心       ← CGO 静态链接：llama.cpp, llama-context.cpp...
├── ggml 后端管理        ← CGO 静态链接：ggml-backend-reg.cpp
│     └── LoadLibraryW("ggml-vulkan.dll")  ← 运行时才去加载 DLL
└── Go 代码              ← HTTP server, 模型管理...

         运行时动态加载 ↓

ggml-vulkan.dll     （cmake 生成）← Vulkan shader, GPU kernel
ggml-cpu-haswell.dll（cmake 生成）← AVX2 矩阵乘法
```

**`ggml-backend-reg.cpp` 被编译了两次，但用途不同：**
- cmake 版：进入 DLL，是插件自我注册的逻辑
- CGO 版：进入 exe，是宿主程序搜索并加载 DLL 的逻辑

`llama.cpp` 核心只有 CGO 版，不在任何 DLL 里。

### CGO 触发机制

`llama/llama.go` 顶部注释块是 CGO 编译指令（不是普通注释）：

```go
/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++17
#cgo CPPFLAGS: -I${SRCDIR}/llama.cpp/include
...
*/
import "C"
```

`go build` 遇到 `import "C"` 时，会调用系统 C++ 编译器（Windows 上是 MSVC `cl.exe`）编译同目录下所有 `.cpp` 文件。

Blank import 触发子包的 CGO 编译：

```go
import (
    _ "github.com/ollama/ollama/llama/llama.cpp/common"  // 触发 common/*.cpp 编译
    _ "github.com/ollama/ollama/llama/llama.cpp/src"     // 触发 src/*.cpp 编译
    ...
)
```

`llama/llama.cpp/common/common.go`（整个文件就这几行，纯 CGO 触发器）：

```go
package common

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/../include
import "C"
```

---

## 4. Vulkan Build 步骤

### 前置条件

1. 安装 [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)，记下安装路径
2. 设置环境变量：`$env:VULKAN_SDK = "C:\VulkanSDK\<version>"`
3. 安装 Visual Studio Build Tools（含 MSVC + CMake）

### CMakePresets.json 中的 Vulkan preset

```json
{ "name": "Vulkan", "cacheVariables": { "OLLAMA_RUNNER_DIR": "vulkan" } }
```

### build_windows.ps1 中的 vulkan() 函数（行 245-257）

```powershell
function vulkan {
    if ($env:VULKAN_SDK) {
        cmake -B build\vulkan --preset Vulkan --install-prefix $script:DIST_DIR
        cmake --build build\vulkan --target ggml-vulkan --config Release
        cmake --install build\vulkan --component Vulkan --strip
    }
}
```

### 手动执行（在 ollama 仓库根目录）

```powershell
# 1. 设置 Vulkan SDK 路径
$env:VULKAN_SDK = "C:\VulkanSDK\1.4.304.1"  # 改成你的实际版本

# 2. CMake 配置
cmake -B build\vulkan --preset Vulkan --install-prefix dist

# 3. 编译 Vulkan 后端
cmake --build build\vulkan --target ggml-vulkan --config Release

# 4. 安装 DLL 到 dist 目录
cmake --install build\vulkan --component Vulkan --strip
```

编译产物：`dist\lib\ollama\vulkan\ggml-vulkan.dll`

### 运行时启用 Vulkan（必须！）

Vulkan 后端默认被禁用，必须设置环境变量才能加载：

```powershell
$env:OLLAMA_VULKAN = "1"
ollama serve
```

对应代码（`discover/runner.go:103-105`）：

```go
if !envconfig.EnableVulkan() && strings.Contains(filepath.Base(dir), "vulkan") {
    slog.Info("experimental Vulkan support disabled. To enable, set OLLAMA_VULKAN=1")
    continue
}
```

---

## 5. DLL 发现机制

### 默认搜索路径

**Windows**（`ml/path.go:16-56`）：

```go
case "windows":
    libPath = filepath.Join(filepath.Dir(exe), "lib", "ollama")
// 开发时 fallback：
filepath.Join(filepath.Dir(exe), "build", "lib", "ollama")
filepath.Join(cwd, "build", "lib", "ollama")
```

即：`<ollama.exe 所在目录>\lib\ollama\`

### 环境变量覆盖

```powershell
# 覆盖默认 DLL 搜索路径
$env:OLLAMA_LIBRARY_PATH = "C:\my\custom\path"

# 加载第三方 ggml 后端
$env:GGML_BACKEND_PATH = "C:\third-party\ggml-backends"
```

### DLL 发现（`discover/runner.go:55-61`）

```go
files, _ := filepath.Glob(filepath.Join(ml.LibOllamaPath, "*", "*ggml-*"))
```

### DLL 命名规则（`ggml-backend-reg.cpp`）

```cpp
#ifdef _WIN32
    // 前缀: "ggml-"，后缀: ".dll"
    // 例: ggml-vulkan.dll, ggml-cuda.dll, ggml-cpu.dll
#else
    // 前缀: "libggml-"，后缀: ".so"
    // 例: libggml-vulkan.so
#endif
```

### StartRunner 注入环境变量（`llm/server.go:352-423`）

```go
case "windows": pathEnv = "PATH"
// gpuLibs 路径优先放最前面
cmd.Env[i] = pathEnv + "=" + pathEnvVal
cmd.Env[i] = "OLLAMA_LIBRARY_PATH=" + strings.Join(gpuLibs, ...)
```

---

## 6. 为什么不能用单独编译的 llama.cpp DLL 替换

**不可行，原因有两个：**

### 原因 1：ABI 版本检查（`ggml-backend-reg.cpp:305`）

```cpp
if (reg->api_version != GGML_BACKEND_API_VERSION) {
    // 版本不匹配 → 拒绝加载，直接跳过该 DLL
}
```

ollama 的 `GGML_BACKEND_API_VERSION` 与上游 llama.cpp 不同（因为打了补丁），ABI 不兼容会导致 DLL 被拒绝加载。

### 原因 2：ollama 打了 34 个上游没有的补丁

这些补丁修改了核心接口，特别是 GPU 发现相关的代码（+1005 行新增）。单独编译的 llama.cpp 没有这些接口，即使 ABI 版本侥幸匹配，GPU 发现功能也会缺失或崩溃。

---

## 7. ollama 对 llama.cpp 的 34 个补丁分类

ollama 在 `llama/patches/` 目录维护了 34 个补丁（基于 llama.cpp commit `ec98e2002`）：

| 类别 | 数量 | 代表补丁 | 性质 |
|------|------|---------|------|
| Windows 特有 bug 修复 | ~4 个 | `0001` malloc/free 跨编译器问题、`0031` exit 代替 abort | 不打就在 Windows 崩溃 |
| GPU 发现/显存信息上报 | ~5 个 | `0015` GPU UUID、`0024` GPU 发现增强(+1005行)、`0028` DXGI+PDH 显存检测 | ollama 独有功能，上游没有 |
| 模型兼容性修复 | ~7 个 | `0002` pretokenizer、`0004` Solar Pro、`0005` DeepSeek | 支持特定模型 |
| 性能优化 | ~6 个 | `0007` 设备排序、`0018` batch hint、`0029` CUDA 大 batch 跳过 | 推理加速 |
| 调试/稳定性 | ~5 个 | `0011` debug tensor、`0026` LoadLibrary 失败报告 | 生产环境需要 |
| CPU 变体/构建系统 | ~3 个 | `0008` CPU phony target、`0009` 移除 AMX | 多平台支持 |

---

## 8. Windows vs Linux 区别

| 方面 | Windows | Linux |
|------|---------|-------|
| DLL 命名 | `ggml-<name>.dll` | `libggml-<name>.so` |
| 加载 API | `LoadLibraryW` / `GetProcAddress` | `dlopen` / `dlsym` |
| 路径环境变量 | `PATH` | `LD_LIBRARY_PATH` |
| 默认库路径 | `<exe>\lib\ollama` | `<exe>\..\lib\ollama` |
| 错误抑制 | `SetErrorMode` 防弹窗 | 无需 |

---

## 9. 关键环境变量速查表

| 环境变量 | 作用 | 示例 |
|---------|------|------|
| `VULKAN_SDK` | Build 时指定 Vulkan SDK 路径 | `C:\VulkanSDK\1.4.304.1` |
| `OLLAMA_VULKAN` | 运行时启用 Vulkan（默认禁用）| `1` |
| `GGML_VK_VISIBLE_DEVICES` | 选择 Vulkan 设备 | `0` |
| `OLLAMA_LIBRARY_PATH` | 覆盖默认 DLL 搜索路径 | `C:\my\ollama\libs` |
| `GGML_BACKEND_PATH` | 加载额外的第三方 ggml 后端 | `C:\third-party\backends` |
| `OLLAMA_DEBUG` | 启用详细日志 | `1` |

---

## 10. 关键文件速查

```
ollama/
├── CMakeLists.txt                    # GGML_SHARED=ON、install 规则
├── CMakePresets.json                 # Vulkan/CUDA presets
├── Makefile.sync                     # vendor 同步，FETCH_HEAD=ec98e2002
├── scripts/build_windows.ps1        # 完整 Windows build 流程
├── llama/
│   ├── llama.go                      # CGO bindings，OnceLoad，BackendInit
│   ├── patches/                      # 34 个补丁
│   └── llama.cpp/                    # vendored llama.cpp 完整源码
├── ml/
│   ├── path.go                       # LibOllamaPath 默认路径计算
│   └── backend/ggml/ggml/src/
│       ├── ggml.go                   # OnceLoad，OLLAMA_LIBRARY_PATH 读取
│       └── ggml-backend-reg.cpp      # LoadLibraryW，DLL 扫描，ABI 检查
├── llm/server.go                     # StartRunner，注入 PATH+OLLAMA_LIBRARY_PATH
├── runner/
│   ├── runner.go                     # runner 分发逻辑
│   └── llamarunner/runner.go         # 推理循环
└── discover/runner.go                # DLL 发现，Vulkan 跳过逻辑
```

---

## 11. 日志与性能分析

> 详细内容见 `docs/debugging-and-profiling.md`，此处为快速参考。

### 能力边界

| 想看什么 | 能否开箱即用 | 方式 |
|---|---|---|
| 哪些层在 GPU / CPU，VRAM 用量 | ✅ | `OLLAMA_DEBUG=1` |
| Prefill / Decode 总速度 | ✅ | 默认输出 / API 响应字段 |
| **Per-op 计时**（Vulkan） | ✅ 运行时 | `GGML_VK_PERF_LOGGER=1` |
| **Per-op 计时**（CUDA/Metal） | ❌ 需外部工具 | Nsight Systems / Instruments |
| 计算图节点列表 + 张量形状 | ⚠️ 需插桩重编译 | `ggml_graph_print()` |

### `OLLAMA_DEBUG` 三档

```bash
OLLAMA_DEBUG=1   # DEBUG：层分配、VRAM 用量、KV cache 大小
OLLAMA_DEBUG=2   # TRACE：内部调度细节
```

### Vulkan Per-op 计时（最有价值的内置工具）

```bash
OLLAMA_VULKAN=1 GGML_VK_PERF_LOGGER=1 ollama run llama3
```

输出（stderr）：
```
Vulkan Timings:
MUL_MAT_VEC: 64 x 142.3 us (187.4 GFLOPS/s)
MUL_MAT:     32 x 8921.4 us (412.1 GFLOPS/s)
RMS_NORM:    64 x 12.1 us
Total time: 294821.0 us.
```

每 N 次推理打印一次：`GGML_VK_PERF_LOGGER_FREQUENCY=10`

### Ollama 内置 Bench 工具

```bash
go run ./cmd/bench -model gemma3 -epochs 6 -format csv
go run ./cmd/bench -model gemma3,llama3.2 -epochs 6 | benchstat -col /name
```

指标：`prefill`（token/s）、`generate`（token/s）、`ttft`（首 token 延迟）、`load`、`total`

---

## 12. 大模型超显存运行机制

### Layer-level Offloading（层级卸载）

Ollama 自动计算可以放进 GPU 的层数，剩余层放 CPU RAM：

```
GPU VRAM (12GB):  Layer 0 ~ 47   ← GPU 执行（快）
CPU RAM  (64GB):  Layer 48 ~ 79  ← CPU 执行（慢）
```

每个 token 生成时，数据必须经过 **PCIe 总线**在两侧传输，是性能瓶颈。

典型速度（Llama 70B Q4，12GB 显卡）：

| 配置 | 速度 |
|---|---|
| 全 GPU（多卡） | ~30 token/s |
| 部分 GPU + 部分 CPU | ~3–8 token/s |
| 全 CPU | ~0.5–2 token/s |

用 `OLLAMA_DEBUG=1` 验证实际分配：
```
llm_load_tensors: offloaded 30/81 layers to GPU
llm_load_tensors: VRAM used = 11843 MiB
llm_load_tensors: CPU  used = 28421 MiB
```

### NVIDIA WDDM GPU Memory Paging

Windows 11 + WDDM 3.0（驱动 ≥ 526.x）支持将物理 VRAM 页面换页到系统 RAM，让显存看起来比实际更大。

**Ollama 不感知这个机制。** 完整调用链（源码追踪）：

```
buildLayout()  [llm/server.go]
    → FreeMemory 字段
    → ggml_backend_dev_memory()  [ml/backend/ggml/ggml.go:732]
    → ggml_backend_cuda_device_get_memory()  [ggml-cuda.cu]
    → nvmlDeviceGetMemoryInfo()  [mem_nvml.cpp:230]
```

`nvmlDeviceGetMemoryInfo()` 只返回**物理 framebuffer**，不含 WDDM paging 的虚拟空间。

两件事发生在不同层：

| 层 | 谁决策 | 看到的内存 |
|---|---|---|
| Ollama 分层计算 | NVML | 只有物理 VRAM |
| CUDA `cudaMalloc` | WDDM 驱动 | 物理 VRAM + 可换页到 RAM 的虚拟空间 |

**实际效果**：WDDM paging 在 Ollama 计算完 ngl 之后，在 CUDA 分配层面透明地兜底溢出的部分——是"意外地帮忙"，而非主动利用。严重超额时（paging 频繁），速度可能比纯 CPU 推理还慢。

---

## 13. Vulkan Backend 的 Layer Offloading 机制

### 结论先行

Vulkan backend 和 CUDA backend 在 **layer offloading 逻辑上完全一样**——两者共用同一套 `buildLayout()` / `assignLayers()` 代码（`llm/server.go`）。差异只在**内存查询**这一步：Vulkan 有更复杂的多级 fallback 链。

---

### Layer Offloading 主流程（两个后端共用）

```
buildLayout()  [llm/server.go ~920]
    → ggml_backend_dev_memory()  [ml/backend/ggml/ggml.go:732]
        ↓ 返回 free/total
    → assignLayers()  [llm/server.go]
        → 按 ngl 计算每个 layer 放哪里
        → 优先塞满 GPU，超出部分放 CPU RAM
```

这段 Go 代码对所有后端（CUDA、Vulkan、Metal、CPU）一视同仁，区别只在于如何回答"GPU 还剩多少空闲内存"这个问题。

---

### Vulkan 内存查询的完整决策树（源码：`ggml-vulkan.cpp` 第 13680–13757 行）

```
ggml_backend_vk_get_device_memory()
│
├─ 1. Windows DXGI + PDH（最优先，Windows 独有）
│   ggml_dxgi_pdh_init() → ggml_dxgi_pdh_get_device_memory(luid)
│   ✅ 成功 → 返回
│   ❌ 失败 → 继续
│
├─ 2. 厂商专用库（仅限独立显卡）
│   ├─ AMD → ggml_hip_get_device_memory(pci_id / uuid)
│   │         ✅ 成功 → 返回
│   │         ❌ 失败 → 继续
│   └─ NVIDIA → ggml_nvml_get_device_memory(uuid)
│                ✅ 成功 → 返回
│                ❌ 失败 → 继续
│
└─ 3. Vulkan 原生 Memory Budget（通用 fallback）
    vkGetPhysicalDeviceMemoryProperties2()
    + VkPhysicalDeviceMemoryBudgetPropertiesEXT
    → 遍历所有 heap，只统计 DeviceLocal（显存）堆
    → free = heapBudget[i] - heapUsage[i]
```

---

### 与 CUDA backend 的对比

| 比较项 | CUDA backend | Vulkan backend（NVIDIA GPU） |
|---|---|---|
| 内存查询方式 | NVML（直接） | DXGI+PDH → NVML（fallback链） |
| Windows 优先策略 | ❌ 无 DXGI 路径 | ✅ DXGI+PDH 优先（更准确） |
| AMD 支持 | ❌ 不支持 | ✅ HIP mgmt 库 |
| 通用 fallback | cudaMemGetInfo / Linux UMA | Vulkan heapBudget |
| Layer offloading 逻辑 | `buildLayout()` 共用 | `buildLayout()` 共用 |

**关键差异**：Vulkan 在 Windows 上会优先用 DXGI（DirectX Graphics Infrastructure），这是微软的标准 GPU 枚举 API，对所有 GPU 厂商都适用，通常比 NVML 更准确地反映 Windows 下的实际可用显存。

---

### WDDM Paging 对 Vulkan 的影响

与 CUDA 路径同理：

- Vulkan backend 查到的内存量（无论 DXGI 还是 heapBudget）都是**当前物理 VRAM 的视角**
- Vulkan 分配（`VkDeviceMemory`）在 WDDM 3.0 下同样受 GPU Memory Paging 影响，驱动可以在超额时换页
- Ollama 的 `buildLayout()` 同样不感知这个 paging 扩展

因此，Vulkan backend 的"意外溢出容忍"机制与 CUDA 相同——是 WDDM 在分配层透明兜底，而非 Ollama 主动规划。

---

### 集成显卡（iGPU）的完整机制（以 Intel iGPU 为例）

> 以下分析基于三个问题的源码追踪：
> 1. Windows UMA 不会把全部系统内存给 iGPU，Ollama 考虑了吗？
> 2. Ollama 是否仍然计算"放多少层进 iGPU"？
> 3. 混合分配（部分层在 iGPU、部分在 CPU）时，中间激活值需要拷贝吗？

---

#### 问题 1：Windows 给 iGPU 分配的是共享内存子集，Ollama 知道吗？

**是的，Ollama 通过 DXGI + PDH 精确感知这个数值，不是全部系统 RAM。**

Windows 上内存查询的最高优先级路径是 `mem_dxgi_pdh.cpp`（第 264–268 行）：

```cpp
// ggml_dxgi_pdh_get_device_memory() — mem_dxgi_pdh.cpp:264
if (is_integrated_gpu) {
    // iGPU: 共享内存（Windows 分配给 iGPU 的那部分）+ 可能有的少量 dedicated 内存
    *free  = (sharedTotal - sharedUsage) + (dedicatedTotal - dedicatedUsage);
    *total = sharedTotal + dedicatedTotal;
} else {
    // dGPU: 只看独立显存
    *free  = dedicatedTotal - dedicatedUsage;
    *total = dedicatedTotal;
}
```

`sharedTotal` 来自 `DXGI_ADAPTER_DESC1::SharedSystemMemory`——这正是 Windows 分配给该 iGPU 的共享内存上限（例如 16 GB 系统 RAM 中给 iGPU 分配的 2–4 GB），**不是**全部系统 RAM。

当前已用量则通过 PDH 性能计数器实时采样：
```
\GPU Adapter Memory(luid_0x...)\Dedicated Usage
\GPU Adapter Memory(luid_0x...)\Shared Usage
```

这与任务管理器中"GPU 共享内存"显示的数值完全一致。如果 DXGI 路径成功，直接返回，跳过后续所有 fallback（NVML、Vulkan heapBudget 等）。

**Vulkan heapBudget fallback 的问题（仅当 DXGI 失败时走到这里）：**

```cpp
// ggml-vulkan.cpp:13748 — 仅作为最后的 fallback
if (is_integrated_gpu || (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal)) {
    *total += heap.size;
    *free  += heapBudget[i] - heapUsage[i];
}
```

iGPU 下统计所有堆（含共享系统内存堆），**可能**会看到更大的数值，取决于驱动如何上报 heapBudget。因此 DXGI 路径在 Windows 上更准确，这也是它优先级最高的原因。

---

#### 问题 2：Ollama 是否仍然对 iGPU 计算"放多少层"？

**是的，和独显完全一样的代码路径，一行都没省略。**

iGPU 在 Go 层与独显合并处理：

```go
// ml/backend/ggml/ggml.go:62
case C.GGML_BACKEND_DEVICE_TYPE_GPU,
    C.GGML_BACKEND_DEVICE_TYPE_IGPU:
    gpus = append(gpus, d)  // iGPU 进入同一个 gpus 列表
```

`buildLayout()` → `assignLayers()` → `greedyFit()` 对 iGPU 完整执行，用问题 1 查到的 `FreeMemory` 决定放多少层。

**iGPU 的优先级低于 dGPU：**

```go
// ml/device.go:362 — ByFreeMemory 排序，iGPU 排在前面（表示"更小"）
// Reverse() 后 iGPU 排在最后，即最低优先级
if a[i].Integrated && !a[j].Integrated {
    return true
}
```

`ByPerformance`（第 376 行）把 iGPU 和 dGPU 分成不同的性能组，`greedyFit` 先把层塞满 dGPU，再考虑 iGPU。混用场景下（机器同时有独显和核显），独显优先。

---

#### 问题 3：混合分配时（部分层在 iGPU、部分在 CPU），激活值需要拷贝吗？

**混合分配会发生，但 iGPU 的"拷贝"开销远小于独显。**

**混合分配的触发条件：** 模型总大小超过 iGPU 的共享内存上限（问题 1 查到的 `free`），`greedyFit` 放不下的层留在 CPU。

**iGPU 下的传输开销：** UMA 架构下，`device->uma == true`，Vulkan 有两处关键优化：

1. **Tensor 直接复用 CPU 指针**（`ggml-vulkan.cpp:5807`）：
```cpp
if (ctx->device->uma) {
    // UMA 下直接拿 CPU 侧的 host buffer，无需 GPU←→CPU 数据搬运
    ggml_vk_host_get(ctx->device, tensor->data, buffer, offset);
}
```

2. **缓冲区分配优先用可 CPU/GPU 共同访问的内存**（`ggml-vulkan.cpp:2448`）：
```cpp
} else if (device->uma) {
    buf = ggml_vk_create_buffer(device, size, {
        vk::MemoryPropertyFlagBits::eDeviceLocal,          // 先试 GPU 最优
        vk::MemoryPropertyFlagBits::eHostVisible | ...     // fallback 到 CPU/GPU 共享可见
    });
}
```
iGPU 的 `DeviceLocal` 和 `HostVisible` 堆在物理上就是同一块系统 RAM，没有 PCIe 总线，这个 fallback 几乎零开销。

**与独显（dGPU）混合分配的对比：**

| 项目 | dGPU 混合推理 | iGPU 混合推理 |
|---|---|---|
| 层间激活值传输路径 | PCIe 总线（CPU RAM ↔ VRAM） | 同一块系统 RAM，指针复用 |
| 传输延迟 | 高（PCIe 带宽 16–32 GB/s） | 接近零（内存带宽级，50–80 GB/s）|
| 混合推理实用性 | 很差，速度剧降 | 尚可，开销主要来自调度而非传输 |
| 性能瓶颈 | PCIe 带宽 | iGPU shader 单元少 + 与 CPU 共享内存带宽 |

**实际结论：** Intel iGPU 做混合推理，"层在 GPU 还是 CPU"的界限比独显模糊得多——物理内存本来就是同一块，主要开销是调度切换和 iGPU 计算单元的算力上限，而不是数据搬运。
