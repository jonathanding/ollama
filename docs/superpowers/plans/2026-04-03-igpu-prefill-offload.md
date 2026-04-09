# iGPU Prefill Offload Implementation Plan

> **修订记录**: 2026-04-09 — Phase 1 bench-sweep 验证失败（iGPU 利用率=0）。根因分析发现原始计划存在架构假设错误（见设计文档 §2）。本计划据此修订，新增 Phase 0 快速验证步骤，并在 Phase 1 中增加两个必要的代码修复。

**Goal:** Enable Intel Arc iGPU (Arrow Lake, UMA) to accelerate prefill for models that overflow the dGPU VRAM, using dGPU(CUDA) + iGPU(Vulkan) combination.

**Architecture:** Ollama uses `GGML_BACKEND_DL ON` — each backend (CUDA, Vulkan) is a separate DLL in its own subdirectory (`cuda_v13/`, `vulkan/`). To enable iGPU alongside CUDA, both DLL directories must appear in the runner's `OLLAMA_LIBRARY_PATH`. Phase 0 validates this manually; Phase 1 automates it with two code fixes; Phase 2 adds explicit layer assignment.

**Tech Stack:** Go, CGo (ggml C bindings), `discover/runner.go`, `ml/backend/ggml/ggml.go`, `llm/server.go`, `envconfig/config.go`

---

## 已完成 (pre-existing commits)

| Commit | 内容 |
|--------|------|
| `881fb5dd` | `envconfig`: 添加 `OLLAMA_IGPU_OFFLOAD` 环境变量 |
| `1154ed2b` | `ml/backend/ggml/ggml.go`: Phase 1 scheduler 注入代码（当 iGPU 被枚举后生效） |

这两个改动**保留不动**。Phase 1 的代码修复在它们的基础上解决"iGPU 从未被枚举"的问题。

---

## File Map

| File | Change | Phase |
|------|--------|-------|
| `envconfig/config.go` | 已完成（添加 `IGPUOffload`） | pre-existing |
| `ml/backend/ggml/ggml.go` | 已完成（Phase 1 scheduler 注入） | pre-existing |
| `discover/runner.go` | 修复 A：`OLLAMA_IGPU_OFFLOAD=1` 时允许探测 vulkan/ | Phase 1 |
| `llm/server.go` | 修复 B：注入 iGPU Vulkan libdir 到 gpuLibs | Phase 1 |
| `llm/server.go` | Phase 2：`assignOverflowToIGPU()` + `createLayout` 调用 | Phase 2 |
| `llm/server_test.go` | Phase 2：`TestAssignOverflowToIGPU` 测试 | Phase 2 |

---

## Task 0: Phase 0 — 零代码快速验证

**目标**: 手动注入双后端路径，验证 CUDA+Vulkan 共存时 iGPU 是否参与计算。

**前提**: 需要有包含 `cuda_v13/` 和 `vulkan/` 的已构建 Ollama。

- [ ] **Step 1: 确认构建产物存在**

```powershell
# 选择你的 lib/ollama 路径（已安装版本）
$LIB = "C:\Users\lingyun\AppData\Local\Programs\Ollama\lib\ollama"

# 验证两个后端 DLL 都存在
ls "$LIB\vulkan\ggml-vulkan.dll"
ls "$LIB\cuda_v13\ggml-cuda.dll"   # 注意版本号可能是 cuda_v12
```

如果 vulkan/ 不存在，需要先用 `scripts/rebuild_windows.ps1` 重新构建。

- [ ] **Step 2: 启动 ollama serve 并手动注入双后端路径**

```powershell
# 停止现有 ollama 进程
taskkill /IM ollama.exe /F 2>$null

$env:OLLAMA_VULKAN       = "1"
$env:OLLAMA_IGPU_OFFLOAD = "1"
$env:OLLAMA_LIBRARY_PATH = "$LIB;$LIB\cuda_v13;$LIB\vulkan"
$env:GGML_SCHED_DEBUG    = "2"

ollama serve
# 或本地构建: 先 go build -o ollama.exe . 再 .\ollama.exe serve
```

- [ ] **Step 3: 验证双后端加载日志**

观察 ollama serve 输出，应包含类似：
```
ggml backend load all from path: .../cuda_v13
ggml backend load all from path: .../vulkan
...loaded vulkan backend...
```

如果只看到 cuda 相关条目，检查 OLLAMA_LIBRARY_PATH 设置是否正确。

- [ ] **Step 4: 触发 prefill 并观测 iGPU**

另一终端：
```bash
ollama run qwen3-coder-next:latest
```

向模型输入一段长文本（1000 token 以上）触发 prefill。同时：
- 打开 Windows Task Manager → Performance → GPU (Intel Arc) → Compute
- 观察 prefill 期间是否出现使用率 spike（>10%）

- [ ] **Step 5: 检查 GGML_SCHED_DEBUG 输出**

在 ollama serve 终端中搜索含 "Vulkan" 的 op 分配日志：
```
# 应该看到类似: backend 1: Vulkan [...] GGML_OP_MUL_MAT
# 表示 prefill matmul 被路由到 Vulkan backend
```

**Phase 0 通过标准**（进入 Phase 1 代码修改的 Gate）：
- Task Manager Intel GPU Compute 在 prefill 期间有明显 spike，或
- GGML_SCHED_DEBUG 日志显示 matmul op 路由到 Vulkan

**Phase 0 失败时**：检查 Vulkan 设备枚举是否正常（日志中是否有 Intel GPU 相关行），
可能需要更新 Vulkan 驱动或检查 Vulkan SDK 安装。

---

## Task 1: Phase 1 修复 A — Discovery 门控 (`discover/runner.go`)

**背景**: 未设置 `OLLAMA_VULKAN=1` 时，`vulkan/` 目录被完全跳过，iGPU 从未被发现。
当 `OLLAMA_IGPU_OFFLOAD=1` 时应允许探测 `vulkan/` 目录。

**前提**: Task 0 Phase 0 验证通过。

- [ ] **Step 1: 定位代码**

读取 `discover/runner.go` 第 100-110 行，确认当前的 Vulkan 门控逻辑：

```go
} else if !envconfig.EnableVulkan() && strings.Contains(filepath.Base(dir), "vulkan") {
    slog.Info("experimental Vulkan support disabled.  To enable, set OLLAMA_VULKAN=1")
    continue
}
```

- [ ] **Step 2: 应用修复**

修改为：当 `OLLAMA_IGPU_OFFLOAD=1` 时也允许探测 `vulkan/` 目录：

```go
} else if !envconfig.EnableVulkan() && !envconfig.IGPUOffload() && strings.Contains(filepath.Base(dir), "vulkan") {
    slog.Info("experimental Vulkan support disabled.  To enable, set OLLAMA_VULKAN=1 or OLLAMA_IGPU_OFFLOAD=1")
    continue
}
```

- [ ] **Step 3: 验证编译**

```bash
go build ./discover/...
```

Expected: 无错误。

- [ ] **Step 4: Commit**

```bash
git add discover/runner.go
git commit -m "discover: allow vulkan dir probe when OLLAMA_IGPU_OFFLOAD=1"
```

---

## Task 2: Phase 1 修复 B — LibraryPaths 注入 (`llm/server.go`)

**背景**: ByLibrary 竞选后，只有 CUDA 设备进入 `gpus`。`LibraryPaths(gpus)` 只返回
`cuda_v13/`，runner 永远不加载 `ggml-vulkan.dll`，iGPU 无法在 runner 中被枚举。

- [ ] **Step 1: 定位 `StartRunner` 调用点**

读取 `llm/server.go` 第 260-280 行，找到：

```go
gpuLibs := ml.LibraryPaths(gpus)
status := NewStatusWriter(os.Stderr)
cmd, port, err := StartRunner(
    tok != nil,
    modelPath,
    gpuLibs,
    ...
```

确认 `systemGPUs`（或类似名称的全量 GPU 列表参数）在此处可用。
如果当前函数没有 `systemGPUs`，需要追溯调用链找到全量设备列表的来源。

- [ ] **Step 2: 确认 systemGPUs 的可用性**

读取 `llm/server.go` 的函数签名（`NewLLMServer` 或类似入口），确认参数列表中有全量
GPU 设备列表（包含 ByLibrary 竞选未胜出的 iGPU 设备）。

如果当前没有，需要将其从调用链传入（通常是 `llm/server.go` 里某个上层函数参数）。

- [ ] **Step 3: 应用注入代码**

在 `gpuLibs := ml.LibraryPaths(gpus)` 之后、`StartRunner` 之前，添加：

```go
// When OLLAMA_IGPU_OFFLOAD=1, inject the Vulkan library directory so the runner
// also loads ggml-vulkan.dll. This allows the iGPU to be enumerated by ggml
// even though it lost the ByLibrary competition to the CUDA dGPU.
if envconfig.IGPUOffload() {
    for _, dev := range systemGPUs {
        if dev.Integrated {
            for _, dir := range dev.LibraryPath {
                if !slices.Contains(gpuLibs, dir) {
                    gpuLibs = append(gpuLibs, dir)
                    slog.Debug("igpu offload: injecting vulkan libdir into runner path",
                        "dir", dir, "device", dev.Description)
                }
            }
        }
    }
}
```

需要确保 `slices` 包已导入（Go 1.21+，`import "slices"`）。

- [ ] **Step 4: 验证编译**

```bash
go build ./llm/...
```

Expected: 无错误。

- [ ] **Step 5: Commit**

```bash
git add llm/server.go
git commit -m "llm: inject iGPU Vulkan libdir into runner path when OLLAMA_IGPU_OFFLOAD=1"
```

---

## Task 3: Phase 1 集成验证

- [ ] **Step 1: 运行测试套件**

```bash
go test ./discover/...
go test ./llm/...
go test ./ml/...
```

Expected: all PASS.

- [ ] **Step 2: 端到端验证（有代码，无手动 OLLAMA_LIBRARY_PATH）**

只设置功能开关，不再手动指定 OLLAMA_LIBRARY_PATH：

```powershell
$env:OLLAMA_IGPU_OFFLOAD = "1"
# 不设置 OLLAMA_LIBRARY_PATH（由代码自动注入）
# 不需要 OLLAMA_VULKAN=1（由修复 A 自动允许 vulkan/ 探测）

go run . serve
```

观察日志是否包含 Vulkan 后端加载信息，Task Manager 是否显示 Intel GPU 使用。

- [ ] **Step 3: Commit 修复**

（已在 Task 1 和 Task 2 单独提交）

---

## Task 4: Phase 1 bench-sweep 验证（进入 Phase 2 的 Gate）

此任务定量验证 Phase 1 的加速效果。**Phase 2 必须在此 bench 通过后才能实施。**

| Metric | Pass criterion |
|--------|----------------|
| `prefill_tps` improvement vs baseline | ≥ 20% |
| `gen_tps` regression vs baseline | ≤ 5% |
| `prefill_tps` CV% | < 10% |

- [ ] **Step 1: 运行 baseline bench（无 IGPU_OFFLOAD）**

```bash
# 停止 ollama serve，不设置 OLLAMA_IGPU_OFFLOAD
go run . serve &

./ollama-bench-sweep.exe run \
  -model qwen3-coder-next \
  -name baseline \
  -sizes 512,1024,2048,4096 \
  -epochs 8 \
  -warmup 4
```

- [ ] **Step 2: 运行 Phase 1 bench（OLLAMA_IGPU_OFFLOAD=1）**

```bash
OLLAMA_IGPU_OFFLOAD=1 go run . serve &

./ollama-bench-sweep.exe run \
  -model qwen3-coder-next \
  -name igpu-phase1 \
  -sizes 512,1024,2048,4096 \
  -epochs 8 \
  -warmup 4
```

- [ ] **Step 3: 对比结果**

```bash
./ollama-bench-sweep.exe diff baseline igpu-phase1
```

**评估通过标准**，记录结果到 `docs/superpowers/bench-results.md`：
- prefill_tps 提升 ≥ 20% 且 CV% < 10%: **进入 Task 5 (Phase 2)**
- 未满足: 停止，分析原因，不实施 Phase 2

---

## Task 5: Phase 2 — 显式 iGPU 层分配 (`llm/server.go`)

**前提**: Task 4 bench 通过。

Phase 2 为 iGPU 分配明确的溢出层，使其成为 ggml scheduler 中的一等成员，
而非仅依赖 op_offload 的隐式路由。

- [ ] **Step 1: 读取 `createLayout` / `buildLayout` 代码**

读取 `llm/server.go` 第 920-1000 行，确认函数签名和调用链。

- [ ] **Step 2: 添加 `assignOverflowToIGPU` 函数**

在 `buildLayout` 闭合括号后（约 line 999）添加（详见原设计文档 §5.1 的完整代码）：

```go
func assignOverflowToIGPU(allLayers []uint64, gpus []ml.DeviceInfo,
    assigned ml.GPULayersList, systemInfo ml.SystemInfo) ml.GPULayersList {
    // ... 见设计文档 §5.1 完整实现 ...
}
```

- [ ] **Step 3: 在 `createLayout` 中调用**

```go
gpuLayers, layers := s.buildLayout(systemGPUs, memory, requireFull, backoff)
if envconfig.IGPUOffload() && gpuLayers.Sum() < len(layers) {
    igpuOverflow := assignOverflowToIGPU(layers, systemGPUs, gpuLayers, systemInfo)
    gpuLayers = append(gpuLayers, igpuOverflow...)
}
err := s.verifyLayout(systemInfo, systemGPUs, memory, requireFull, gpuLayers, layers)
```

- [ ] **Step 4: 验证编译**

```bash
go build ./llm/...
```

- [ ] **Step 5: Commit**

```bash
git add llm/server.go
git commit -m "llm: add iGPU cross-library overflow layer assignment for OLLAMA_IGPU_OFFLOAD (Phase 2)"
```

---

## Task 6: Phase 2 测试 (`llm/server_test.go`)

（详见原设计文档的 Task 6，测试代码和步骤不变）

- [ ] 写 `TestAssignOverflowToIGPU` 和 `TestAssignOverflowToIGPUDisabled`
- [ ] `go test ./llm/... -run TestAssignOverflowToIGPU -v` 全部通过
- [ ] Commit

---

## Task 7: Phase 2 bench-sweep 验证

```bash
OLLAMA_IGPU_OFFLOAD=1 go run . serve &

./ollama-bench-sweep.exe run \
  -model qwen3-coder-next \
  -name igpu-phase2 \
  -sizes 512,1024,2048,4096 \
  -epochs 8 \
  -warmup 4

./ollama-bench-sweep.exe diff baseline igpu-phase2
./ollama-bench-sweep.exe diff igpu-phase1 igpu-phase2
```

**通过标准**：

| Metric | Criterion |
|--------|-----------|
| `prefill_tps` vs baseline | ≥ 20% improvement |
| `prefill_tps` vs phase1 | ≥ 0% (no regression) |
| `gen_tps` vs baseline | ≤ 10% regression |

---

## Self-Review Checklist

- [x] 架构假设已修正：GGML_BACKEND_DL → 动态 DLL 子目录结构
- [x] Phase 0 零代码验证步骤完整，可独立执行
- [x] Phase 1 修复 A（discovery）和 B（libpath 注入）分别独立，可分开提交
- [x] 已有 commits（881fb5dd, 1154ed2b）保留，无需修改
- [x] Phase 2 设计不变，仍在 Phase 1 bench 通过后执行
- [x] bench Gate 明确：Phase 0 通过 → Phase 1 代码修复，Phase 1 bench 通过 → Phase 2
