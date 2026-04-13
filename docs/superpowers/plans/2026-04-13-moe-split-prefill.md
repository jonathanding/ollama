# MoE Split Prefill Optimization (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Qwen3-Next-80B 在 RTX 3090 上的 prefill 延迟从 ~2.0s 降至 ~1.55s，通过把所有层的 attention/dense 权重常驻 GPU，MoE expert 权重按需拷贝。

**Architecture:** 在 probe 阶段追踪每层 MoE 权重大小（`MoEWeights`），在 `buildLayout()` 中用两轮分配（先为所有 dense 预留 VRAM，再贪心填充 MoE），在 `ggml.go` 中按 tensor 名称路由 MoE expert tensors 到不同设备。

**Tech Stack:** Go (server/runner层), C/CGo (ggml.go tensor分配), 影响路径: `ml/` → `llm/` → `runner/`

**设计规格:** `docs/superpowers/specs/2026-04-13-moe-split-prefill-design.md`

**关键架构说明：**
- `ollamaServer.Load()` 使用 Go-native runner（Qwen3-Next 走此路径），从 probe 结果中获取内存数据
- `buildLayout()` 返回两个 layer list：`gpuLayers`（MoE 在 GPU 的 K 层，用于迭代收敛）和 `denseGPULayers`（dense 在 GPU 的全 48 层，发送给 backend）
- `BackendParams.GPULayers = denseGPULayers`（non-MoE tensor 路由）
- `BackendParams.MoEGPULayers = gpuLayers`（MoE tensor 路由）

---

## File Map

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `ml/device.go:154` | Modify | `DeviceMemory` 新增 `MoEWeights []uint64` |
| `ml/backend.go:70` | Modify | `BackendParams` 新增 `MoEGPULayers GPULayersList` |
| `envconfig/config.go` | Modify | 新增 `Int()` 函数 + `MoeGpuLayers` var |
| `ml/backend/ggml/ggml.go` | Modify | MoE tensor 识别、追踪、路由、日志 |
| `llm/server.go` | Modify | `LoadRequest`、`buildLayout()`、`createLayout()`、`ollamaServer.Load()` |
| `runner/ollamarunner/runner.go:1308` | Modify | `BackendParams` 中传入 `MoEGPULayers` |

---

## Task 1: 数据模型 — DeviceMemory + BackendParams

**Files:**
- Modify: `ml/device.go:153-161`
- Modify: `ml/backend.go:69-74`

- [ ] **Step 1: 在 `DeviceMemory` 中添加 `MoEWeights` 字段**

  在 `ml/device.go` 第 154 行（`Weights []uint64` 之后）插入：

  ```go
  // Weights is the per-layer memory needed for the model weights.
  Weights []uint64

  // MoEWeights is the per-layer memory for MoE expert tensors only
  // (matching \.ffn_(up|down|gate)_(ch_)?exps$). Always a subset of
  // Weights. Zero for non-MoE models or layers with MoE fully on GPU.
  MoEWeights []uint64

  // Cache is the per-layer memory needed for the KV cache.
  Cache []uint64
  ```

- [ ] **Step 2: 在 `BackendParams` 中添加 `MoEGPULayers` 字段**

  在 `ml/backend.go` 第 70 行（`GPULayers GPULayersList` 之后）插入：

  ```go
  // GPULayers is the set of layers to offload to GPUs
  GPULayers GPULayersList

  // MoEGPULayers is the subset of GPULayers where MoE expert weights
  // are also resident on GPU. Layers in GPULayers but not here have
  // their MoE expert tensors on CPU (copied on demand via op_offload).
  // Nil means no MoE split is active.
  MoEGPULayers GPULayersList

  // FlashAttention indicates that we should use a fused flash attention kernel
  FlashAttention FlashAttentionType
  ```

- [ ] **Step 3: 编译验证**

  ```bash
  cd C:/Users/lingyun/Desktop/projects/ollama
  go build ./ml/...
  ```
  预期：编译通过，无报错。

- [ ] **Step 4: Commit**

  ```bash
  git add ml/device.go ml/backend.go
  git commit -m "ml: add MoEWeights to DeviceMemory and MoEGPULayers to BackendParams"
  ```

---

## Task 2: 配置 — envconfig.MoeGpuLayers

**Files:**
- Modify: `envconfig/config.go`

- [ ] **Step 1: 添加 `Int()` 函数**

  在 `envconfig/config.go` 中 `Uint()` 函数（第 259 行）之后插入：

  ```go
  // Int returns a function that parses an integer environment variable.
  // Returns defaultValue if the variable is unset or invalid.
  func Int(key string, defaultValue int) func() int {
  	return func() int {
  		if s := Var(key); s != "" {
  			if n, err := strconv.ParseInt(s, 10, 64); err != nil {
  				slog.Warn("invalid environment variable, using default", "key", key, "value", s, "default", defaultValue)
  			} else {
  				return int(n)
  			}
  		}
  		return defaultValue
  	}
  }
  ```

- [ ] **Step 2: 添加 `MoeGpuLayers` 变量**

  在 `envconfig/config.go` 的 `var (` 块（第 214 行附近，`KvCacheType` 之后）插入：

  ```go
  // MoeGpuLayers controls how many layers have MoE expert weights resident on GPU.
  //   -1 (default) = auto-compute from remaining VRAM after dense allocation
  //    0           = disable MoE split (all MoE stays on CPU, useful for baseline comparison)
  //   >0           = force this many layers to have MoE on GPU
  // Complementary to llama.cpp's --n-cpu-moe: n_cpu_moe = total_layers - MoeGpuLayers
  MoeGpuLayers = Int("OLLAMA_MOE_GPU_LAYERS", -1)
  ```

- [ ] **Step 3: 编译验证**

  ```bash
  go build ./envconfig/...
  ```
  预期：编译通过。

- [ ] **Step 4: Commit**

  ```bash
  git add envconfig/config.go
  git commit -m "envconfig: add Int() helper and OLLAMA_MOE_GPU_LAYERS variable"
  ```

---

## Task 3: ggml.go — MoE tensor 追踪与路由

**Files:**
- Modify: `ml/backend/ggml/ggml.go`

该 task 有 4 个独立改动点，全部在同一文件中。

### 3a: 初始化 MoEWeights 切片

- [ ] **Step 1: CPU DeviceMemory 初始化（line 177-178 附近）**

  将：
  ```go
  requiredMemory.CPU.Weights = make([]uint64, blocks+1)
  requiredMemory.CPU.Cache = make([]uint64, blocks+1)
  ```
  改为：
  ```go
  requiredMemory.CPU.Weights    = make([]uint64, blocks+1)
  requiredMemory.CPU.MoEWeights = make([]uint64, blocks+1)
  requiredMemory.CPU.Cache      = make([]uint64, blocks+1)
  ```

- [ ] **Step 2: GPU DeviceMemory 初始化（line 196-197 附近）**

  将：
  ```go
  requiredMemory.GPUs[i].Weights = make([]uint64, blocks+1)
  requiredMemory.GPUs[i].Cache = make([]uint64, blocks+1)
  ```
  改为：
  ```go
  requiredMemory.GPUs[i].Weights    = make([]uint64, blocks+1)
  requiredMemory.GPUs[i].MoEWeights = make([]uint64, blocks+1)
  requiredMemory.GPUs[i].Cache      = make([]uint64, blocks+1)
  ```

### 3b: isMoEExpertTensor + 追踪

- [ ] **Step 3: 在 `New()` 函数顶部（`initDevices()` 调用之前，line ~147）添加 regexp**

  在 `func New(...)` 函数体开头插入：
  ```go
  // moeExpertRE matches MoE expert weight tensors within a block.
  // Pattern mirrors llama.cpp's LAYER_FRACTION_MOE pattern.
  var moeExpertRE = regexp.MustCompile(`\.ffn_(up|down|gate)_(ch_)?exps$`)
  isMoEExpertTensor := func(name string) bool {
  	return moeExpertRE.MatchString(name)
  }
  ```

  确认 `regexp` 已在 import 中，如没有则添加至 import 列表。

- [ ] **Step 4: 在 `createTensor` 闭包中追踪 MoEWeights（line 282-286 附近）**

  找到以下代码：
  ```go
  if layer == -1 {
  	requiredMemory.InputWeights += uint64(size)
  } else {
  	btDeviceMemory[bt].Weights[layer] += uint64(size)
  }
  ```

  改为：
  ```go
  if layer == -1 {
  	requiredMemory.InputWeights += uint64(size)
  } else {
  	btDeviceMemory[bt].Weights[layer] += uint64(size)
  	if isMoEExpertTensor(name) {
  		btDeviceMemory[bt].MoEWeights[layer] += uint64(size)
  		logutil.Trace("moe split: tracked MoE tensor",
  			"name", name,
  			"layer", layer,
  			"size", format.HumanBytes2(uint64(size)),
  			"buffer", C.GoString(C.ggml_backend_buft_name(bt)))
  	}
  }
  ```

  注意：`name` 变量需确认在作用域内。在 `createTensor` 闭包中 `name` 是根据 `t.source.Name` 和 `t.target` 计算的，确认使用正确变量（搜索 `cname := C.CString(name)` 附近代码，`name` 在其之前计算）。

### 3c: assignMoELayer + moeLayers

- [ ] **Step 5: 在 `assignLayer` 函数定义之后（line ~219）添加 `assignMoELayer`**

  找到 `assignLayer` 函数结束处（`return cpuDeviceBufferType` 后的 `}`），在其后插入：

  ```go
  // assignMoELayer returns the buffer type for MoE expert tensors of a given layer.
  // Uses params.MoEGPULayers: layers in this list get GPU, others get CPU.
  assignMoELayer := func(layer int) deviceBufferType {
  	for _, p := range params.MoEGPULayers {
  		for _, l := range p.Layers {
  			if l == layer {
  				for i := range requiredMemory.GPUs {
  					if requiredMemory.GPUs[i].DeviceID == p.DeviceID {
  						return gpuDeviceBufferTypes[i]
  					}
  				}
  				return cpuDeviceBufferType
  			}
  		}
  	}
  	return cpuDeviceBufferType
  }
  ```

- [ ] **Step 6: 在 `layers` 数组初始化之后（line ~222-225）添加 `moeLayers`**

  找到：
  ```go
  layers := make([]deviceBufferType, blocks)
  for i := range layers {
  	layers[i] = assignLayer(i)
  }
  ```

  在其后插入：
  ```go
  // moeLayers holds the buffer type for MoE expert tensors per layer.
  // Only populated when MoEGPULayers is non-empty (MoE split active).
  moeLayers := make([]deviceBufferType, blocks)
  for i := range moeLayers {
  	moeLayers[i] = assignMoELayer(i)
  }

  // Log routing summary on formal allocation (AllocMemory=true) when MoE split is active
  if params.AllocMemory && len(params.MoEGPULayers) > 0 {
  	moeGPUSet := make(map[int]bool)
  	for _, p := range params.MoEGPULayers {
  		for _, l := range p.Layers {
  			moeGPUSet[l] = true
  		}
  	}
  	for i := range layers {
  		moeLocation := "cpu"
  		if moeGPUSet[i] {
  			moeLocation = "gpu"
  		}
  		slog.Info("moe split: tensor routing",
  			"layer", i,
  			"dense", "gpu",
  			"moe", moeLocation)
  	}
  }
  ```

### 3d: 修改 tensor 路由

- [ ] **Step 7: 在 `switch default` 分支（line ~336）修改路由逻辑**

  找到（在 `default:` 分支内）：
  ```go
  if layerIndex >= 0 {
  	createTensor(tensor{source: t}, layers[layerIndex].bts, layerIndex)
  } else {
  	createTensor(tensor{source: t}, input.bts, -1)
  }
  ```

  改为：
  ```go
  if layerIndex >= 0 {
  	bts := layers[layerIndex].bts
  	if isMoEExpertTensor(t.Name) && len(params.MoEGPULayers) > 0 {
  		// MoE expert tensor: route based on MoEGPULayers (subset of GPULayers)
  		bts = moeLayers[layerIndex].bts
  	}
  	createTensor(tensor{source: t}, bts, layerIndex)
  } else {
  	createTensor(tensor{source: t}, input.bts, -1)
  }
  ```

- [ ] **Step 8: 编译验证**

  ```bash
  go build ./ml/backend/ggml/...
  ```
  预期：编译通过。

- [ ] **Step 9: Commit**

  ```bash
  git add ml/backend/ggml/ggml.go
  git commit -m "ggml: add MoE tensor tracking, assignMoELayer, and split routing"
  ```

---

## Task 4: server.go — LoadRequest、buildLayout 两轮分配、ollamaServer.Load

**Files:**
- Modify: `llm/server.go`

### 4a: LoadRequest 新增 MoEGPULayers

- [ ] **Step 1: 在 `LoadRequest` 结构体中添加字段（line 482 附近）**

  找到：
  ```go
  GPULayers      ml.GPULayersList
  ```
  在其后插入：
  ```go
  GPULayers      ml.GPULayersList
  // MoEGPULayers is the subset of GPULayers where MoE expert weights
  // are also resident on GPU. Nil means no MoE split.
  MoEGPULayers   ml.GPULayersList
  ```

### 4b: buildLayout 两轮分配

- [ ] **Step 2: 修改 `buildLayout` 签名（line 941）**

  将：
  ```go
  func (s *llmServer) buildLayout(systemGPUs []ml.DeviceInfo, memory *ml.BackendMemory, requireFull bool, backoff float32) (ml.GPULayersList, []uint64) {
  ```
  改为：
  ```go
  // buildLayout computes layer assignments for GPU offload.
  // Returns:
  //   gpuLayers      - layers with MoE on GPU (used for iteration convergence)
  //   denseGPULayers - layers with dense (non-MoE) on GPU; nil if no MoE split
  //   layers         - total per-layer memory sizes
  func (s *llmServer) buildLayout(systemGPUs []ml.DeviceInfo, memory *ml.BackendMemory, requireFull bool, backoff float32) (gpuLayers ml.GPULayersList, denseGPULayers ml.GPULayersList, layers []uint64) {
  ```

- [ ] **Step 3: 在 `buildLayout` 函数体内，`layers` 计算完成后插入 MoE 分支**

  找到现有 `layers` 计算循环（line ~945-954）：
  ```go
  layers := make([]uint64, len(memory.CPU.Weights))
  for i := range layers {
  	for j := range memory.GPUs {
  		layers[i] += memory.GPUs[j].Weights[i]
  		layers[i] += memory.GPUs[j].Cache[i]
  	}
  	layers[i] += memory.CPU.Weights[i]
  	layers[i] += memory.CPU.Cache[i]
  	logutil.Trace("layer to assign", "layer", i, "size", format.HumanBytes2(layers[i]))
  }
  ```

  在此循环**之后**，`gpuLayers := ml.GPULayersList{}` **之前**插入以下 MoE split 逻辑：

  ```go
  // ── MoE split logic ───────────────────────────────────────────────
  // Compute per-layer dense and MoE sizes (summed across all devices)
  denseSize := make([]uint64, len(layers))
  moeSize   := make([]uint64, len(layers))
  for i := range layers {
  	var totalMoE uint64
  	for j := range memory.GPUs {
  		totalMoE += memory.GPUs[j].MoEWeights[i]
  	}
  	totalMoE += memory.CPU.MoEWeights[i]
  	moeSize[i]   = totalMoE
  	denseSize[i] = layers[i] - totalMoE
  }

  // Check if this is a MoE model (any layer has MoE weights)
  isMoEModel := slices.ContainsFunc(moeSize, func(s uint64) bool { return s > 0 })

  if isMoEModel {
  	// Compute total dense overhead (dense is always on GPU when split is active)
  	var totalDenseOverhead uint64
  	for i := range denseSize {
  		totalDenseOverhead += denseSize[i]
  	}

  	// Check if all dense fits in GPU (prerequisite for MoE split)
  	// Use the first GPU's free memory as a proxy; adjust for backoff/overhead
  	var availableVRAM uint64
  	for _, gpu := range systemGPUs {
  		reserved := uint64(float32(gpu.FreeMemory)*backoff) + gpu.MinimumMemory() + envconfig.GpuOverhead()
  		if gpu.FreeMemory > reserved {
  			availableVRAM += gpu.FreeMemory - reserved
  		}
  		// Only check once - single GPU assumption for Phase 1
  		break
  	}

  	if totalDenseOverhead > availableVRAM {
  		slog.Warn("moe split: dense weights exceed available VRAM, falling back to standard layout",
  			"dense_total", format.HumanBytes2(totalDenseOverhead),
  			"available_vram", format.HumanBytes2(availableVRAM))
  		// Fall through to existing assignLayers() below
  	} else {
  		slog.Info("moe split: dense weights fit, activating split",
  			"dense_total",   format.HumanBytes2(totalDenseOverhead),
  			"vram_for_moe",  format.HumanBytes2(availableVRAM-totalDenseOverhead))

  		// Determine how many layers' MoE can fit in remaining VRAM
  		moeGPUCount := envconfig.MoeGpuLayers()
  		source := "auto"
  		if moeGPUCount >= 0 {
  			source = "user-override"
  			if moeGPUCount > len(moeSize) {
  				slog.Warn("moe split: OLLAMA_MOE_GPU_LAYERS exceeds total layers, clamped",
  					"requested", moeGPUCount, "clamped", len(moeSize))
  				moeGPUCount = len(moeSize)
  			}
  		} else {
  			remainingVRAM := availableVRAM - totalDenseOverhead
  			moeGPUCount = 0
  			for i := range moeSize {
  				if moeSize[i] > remainingVRAM {
  					break
  				}
  				remainingVRAM -= moeSize[i]
  				moeGPUCount++
  			}
  		}

  		slog.Info("moe split: layer budget",
  			"moe_gpu_layers", moeGPUCount,
  			"moe_cpu_layers", len(moeSize)-moeGPUCount,
  			"source",         source)

  		for i := range layers {
  			loc := "cpu"
  			if i < moeGPUCount {
  				loc = "gpu"
  			}
  			slog.Debug("moe split: layer layout",
  				"layer",     i,
  				"dense_size", format.HumanBytes2(denseSize[i]),
  				"moe_size",   format.HumanBytes2(moeSize[i]),
  				"moe_loc",    loc)
  		}

  		// gpuLayers: first moeGPUCount layers have MoE on GPU (used for iteration)
  		// Build using the adjusted (MoE-only) layer sizes
  		adjustedLayers := make([]uint64, len(layers))
  		for i := range adjustedLayers {
  			// MoE-only cost per layer; dense overhead is pre-allocated
  			adjustedLayers[i] = moeSize[i]
  			// Add cache back
  			for j := range memory.GPUs {
  				adjustedLayers[i] += memory.GPUs[j].Cache[i]
  			}
  			adjustedLayers[i] += memory.CPU.Cache[i]
  		}
  		adjustedGPUs := make([]ml.DeviceInfo, len(systemGPUs))
  		copy(adjustedGPUs, systemGPUs)
  		for i := range adjustedGPUs {
  			if adjustedGPUs[i].FreeMemory > totalDenseOverhead {
  				adjustedGPUs[i].FreeMemory -= totalDenseOverhead
  			} else {
  				adjustedGPUs[i].FreeMemory = 0
  			}
  		}
  		// Apply backoff/overhead to adjustedGPUs
  		for i := range adjustedGPUs {
  			reserved := uint64(float32(adjustedGPUs[i].FreeMemory)*backoff) + adjustedGPUs[i].MinimumMemory() + envconfig.GpuOverhead()
  			if adjustedGPUs[i].FreeMemory > reserved {
  				adjustedGPUs[i].FreeMemory -= reserved
  			} else {
  				adjustedGPUs[i].FreeMemory = 0
  			}
  		}

  		gpuLayersMoE := assignLayers(adjustedLayers, adjustedGPUs, requireFull, s.options.NumGPU, 0)

  		// denseGPULayers: all layers (all dense on GPU)
  		// Assign all layers to the GPU(s) used by gpuLayersMoE, or first GPU
  		allLayerIndices := make([]int, len(layers))
  		for i := range allLayerIndices {
  			allLayerIndices[i] = i
  		}
  		var denseDeviceID ml.DeviceID
  		if len(gpuLayersMoE) > 0 {
  			denseDeviceID = gpuLayersMoE[0].DeviceID
  		} else if len(systemGPUs) > 0 {
  			denseDeviceID = systemGPUs[len(systemGPUs)-1].DeviceID
  		}
  		denseGPULayers = ml.GPULayersList{{
  			DeviceID: denseDeviceID,
  			Layers:   allLayerIndices,
  		}}

  		return gpuLayersMoE, denseGPULayers, layers
  	}
  }
  // ── End MoE split logic ───────────────────────────────────────────
  ```

- [ ] **Step 4: 修改 `buildLayout` 中现有的 return 语句**

  找到函数末尾：
  ```go
  return gpuLayers, layers
  ```
  改为：
  ```go
  return gpuLayers, nil, layers
  ```

### 4c: createLayout 签名更新

- [ ] **Step 5: 修改 `createLayout` 签名（line 926）**

  将：
  ```go
  func (s *llmServer) createLayout(systemInfo ml.SystemInfo, systemGPUs []ml.DeviceInfo, memory *ml.BackendMemory, requireFull bool, backoff float32) (ml.GPULayersList, error) {
  ```
  改为：
  ```go
  func (s *llmServer) createLayout(systemInfo ml.SystemInfo, systemGPUs []ml.DeviceInfo, memory *ml.BackendMemory, requireFull bool, backoff float32) (ml.GPULayersList, ml.GPULayersList, error) {
  ```

- [ ] **Step 6: 更新 `createLayout` 函数体**

  找到函数体内容：
  ```go
  if memory == nil {
  	memory = &ml.BackendMemory{CPU: ml.DeviceMemory{
  		Weights: make([]uint64, s.totalLayers),
  		Cache:   make([]uint64, s.totalLayers),
  	}}
  }
  gpuLayers, layers := s.buildLayout(systemGPUs, memory, requireFull, backoff)
  err := s.verifyLayout(systemInfo, systemGPUs, memory, requireFull, gpuLayers, layers)
  if err != nil {
  	return nil, err
  }
  return gpuLayers, nil
  ```
  改为：
  ```go
  if memory == nil {
  	memory = &ml.BackendMemory{CPU: ml.DeviceMemory{
  		Weights:    make([]uint64, s.totalLayers),
  		MoEWeights: make([]uint64, s.totalLayers),
  		Cache:      make([]uint64, s.totalLayers),
  	}}
  }
  gpuLayers, denseGPULayers, layers := s.buildLayout(systemGPUs, memory, requireFull, backoff)
  err := s.verifyLayout(systemInfo, systemGPUs, memory, requireFull, gpuLayers, denseGPULayers, layers)
  if err != nil {
  	return nil, nil, err
  }
  return gpuLayers, denseGPULayers, nil
  ```

### 4d: verifyLayout 更新

- [ ] **Step 7: 修改 `verifyLayout` 签名（line 1003）**

  将：
  ```go
  func (s *llmServer) verifyLayout(systemInfo ml.SystemInfo, systemGPUs []ml.DeviceInfo, memory *ml.BackendMemory, requireFull bool, gpuLayers ml.GPULayersList, layers []uint64) error {
  ```
  改为：
  ```go
  func (s *llmServer) verifyLayout(systemInfo ml.SystemInfo, systemGPUs []ml.DeviceInfo, memory *ml.BackendMemory, requireFull bool, gpuLayers ml.GPULayersList, denseGPULayers ml.GPULayersList, layers []uint64) error {
  ```

- [ ] **Step 8: 在 `verifyLayout` 的 `requireFull` 检查中使用 `denseGPULayers`**

  找到：
  ```go
  if requireFull {
  	if len(systemGPUs) > 0 && gpuLayers.Sum() < len(layers) && (s.options.NumGPU < 0 || gpuLayers.Sum() < s.options.NumGPU) {
  ```
  改为：
  ```go
  if requireFull {
  	// When MoE split is active, use denseGPULayers (all 48 layers) for the full-load check
  	checkLayers := gpuLayers
  	if len(denseGPULayers) > 0 {
  		checkLayers = denseGPULayers
  	}
  	if len(systemGPUs) > 0 && checkLayers.Sum() < len(layers) && (s.options.NumGPU < 0 || checkLayers.Sum() < s.options.NumGPU) {
  ```

### 4e: ollamaServer.Load 更新

- [ ] **Step 9: 更新 `ollamaServer.Load` 中所有 `createLayout` 调用**

  **Line 764** — 初始调用：
  ```go
  // 原来
  gpuLayers, err := s.createLayout(systemInfo, gpus, s.mem, requireFull, backoff)
  // 改为
  var denseGPULayers ml.GPULayersList
  gpuLayers, denseGPULayers, err := s.createLayout(systemInfo, gpus, s.mem, requireFull, backoff)
  ```

  **Line 777** — 设置 loadRequest：
  ```go
  // 原来
  s.loadRequest.GPULayers = gpuLayers
  // 改为（dense on GPU for non-MoE tensors; moe on GPU for first K layers）
  if len(denseGPULayers) > 0 {
  	s.loadRequest.GPULayers = denseGPULayers
  } else {
  	s.loadRequest.GPULayers = gpuLayers
  }
  s.loadRequest.MoEGPULayers = gpuLayers
  ```

  **Line 790** — 内层循环 createLayout：
  ```go
  // 原来
  newGPULayers, err := s.createLayout(systemInfo, gpus, s.mem, requireFull, backoff)
  // 改为
  newGPULayers, newDenseGPULayers, err := s.createLayout(systemInfo, gpus, s.mem, requireFull, backoff)
  ```

  **Line 802** — 更新 gpuLayers：
  ```go
  // 原来
  gpuLayers = newGPULayers
  // 改为
  gpuLayers = newGPULayers
  denseGPULayers = newDenseGPULayers
  ```

  **Line 823** — 中间探索 createLayout：
  ```go
  // 原来
  newGPULayers, err = s.createLayout(systemInfo, gpus, s.mem, requireFull, backoff)
  // 改为
  newGPULayers, newDenseGPULayers, err = s.createLayout(systemInfo, gpus, s.mem, requireFull, backoff)
  ```

  **Line 830** — 中间探索 loadRequest：
  ```go
  // 原来
  s.loadRequest.GPULayers = newGPULayers
  // 改为
  if len(newDenseGPULayers) > 0 {
  	s.loadRequest.GPULayers = newDenseGPULayers
  } else {
  	s.loadRequest.GPULayers = newGPULayers
  }
  s.loadRequest.MoEGPULayers = newGPULayers
  ```

  **Line 840** — verify createLayout（仅需 gpuLayers.Sum() 比较）：
  ```go
  // 原来
  verifyGPULayers, err := s.createLayout(systemInfo, gpus, &resp.Memory, requireFull, backoff)
  // 改为
  verifyGPULayers, _, err := s.createLayout(systemInfo, gpus, &resp.Memory, requireFull, backoff)
  ```

  **Line 848** — 更新 gpuLayers（在 intermediate exploration 中）：
  ```go
  // 原来
  gpuLayers = newGPULayers
  // 改为（newDenseGPULayers already updated at line 823）
  gpuLayers = newGPULayers
  denseGPULayers = newDenseGPULayers
  ```

- [ ] **Step 10: 编译验证**

  ```bash
  go build ./llm/...
  ```
  预期：编译通过。如果有其他 `createLayout` 调用（`llamaServer.Load` 等），同样更新签名调用处，第二返回值用 `_` 忽略。

- [ ] **Step 11: Commit**

  ```bash
  git add llm/server.go
  git commit -m "llm: add MoE split two-pass layout in buildLayout, update createLayout/verifyLayout signatures"
  ```

---

## Task 5: runner — 传递 MoEGPULayers 至 BackendParams

**Files:**
- Modify: `runner/ollamarunner/runner.go:1308`

- [ ] **Step 1: 在 `BackendParams` 构造中添加 `MoEGPULayers`**

  找到（line 1308-1313）：
  ```go
  params := ml.BackendParams{
  	AllocMemory:    req.Operation != llm.LoadOperationFit,
  	NumThreads:     req.NumThreads,
  	GPULayers:      req.GPULayers,
  	FlashAttention: req.FlashAttention,
  }
  ```
  改为：
  ```go
  params := ml.BackendParams{
  	AllocMemory:    req.Operation != llm.LoadOperationFit,
  	NumThreads:     req.NumThreads,
  	GPULayers:      req.GPULayers,
  	MoEGPULayers:   req.MoEGPULayers,
  	FlashAttention: req.FlashAttention,
  }
  ```

- [ ] **Step 2: 全量编译验证**

  ```bash
  go build ./...
  ```
  预期：整个项目编译通过，无报错。

- [ ] **Step 3: Commit**

  ```bash
  git add runner/ollamarunner/runner.go
  git commit -m "runner: pass MoEGPULayers from LoadRequest to BackendParams"
  ```

---

## Task 6: 验证 MoE Split 是否正确生效

本 task 通过日志验证新机制是否按预期工作，无代码改动。

- [ ] **Step 1: 确认 MoE tensor 被正确识别（probe 阶段）**

  ```bash
  OLLAMA_DEBUG=1 ollama serve 2>&1 | grep "moe split: tracked MoE tensor"
  ```
  **预期**：加载 qwen3-coder-next 时，出现 144 条左右日志（48 层 × 3 个 expert tensor）。示例：
  ```
  moe split: tracked MoE tensor name=blk.0.ffn_up_exps layer=0 size=288MiB buffer=CUDA0
  moe split: tracked MoE tensor name=blk.0.ffn_down_exps layer=0 size=420MiB buffer=CUDA0
  moe split: tracked MoE tensor name=blk.0.ffn_gate_exps layer=0 size=288MiB buffer=CUDA0
  ```

- [ ] **Step 2: 确认 MoE split 被激活及 dense/MoE 预算**

  ```bash
  OLLAMA_DEBUG=1 ollama serve 2>&1 | grep "moe split:"
  ```
  **预期**（按顺序出现）：
  ```
  moe split: dense weights fit, activating split dense_total=1.2GiB vram_for_moe=~22GiB
  moe split: layer budget moe_gpu_layers=22 moe_cpu_layers=26 source=auto
  ```

- [ ] **Step 3: 确认逐层 MoE 位置分配**

  ```bash
  OLLAMA_DEBUG=1 ollama serve 2>&1 | grep "moe split: layer layout"
  ```
  **预期**：48 行日志，前 ~22 行 `moe_loc=gpu`，后 ~26 行 `moe_loc=cpu`，所有层 `dense_size=~25MiB`。

- [ ] **Step 4: 确认正式分配时 tensor 路由**

  ```bash
  OLLAMA_DEBUG=1 ollama serve 2>&1 | grep "moe split: tensor routing"
  ```
  **预期**：48 行日志，`dense=gpu` 全部，`moe` 字段与 Step 3 一致。

- [ ] **Step 5: Baseline vs Phase 1 Benchmark**

  先运行 baseline（禁用 MoE split）：
  ```bash
  OLLAMA_MOE_GPU_LAYERS=0 ollama run qwen3-coder-next --verbose ""
  ```
  记录 `eval rate`（prefill tokens/s）。

  再运行 Phase 1：
  ```bash
  OLLAMA_MOE_GPU_LAYERS=-1 ollama run qwen3-coder-next --verbose ""
  ```
  **预期**：prefill 速度提升约 22%（从 ~500 tokens/s 提升至 ~640 tokens/s，或延迟从 2.0s 降至 1.55s）。

---

## 已知边界情况

| 情况 | 行为 |
|------|------|
| 非 MoE 模型 | `isMoEModel=false`，跳过两轮分配，行为与今天完全相同 |
| dense 超过 VRAM | `Warn` 日志，fall through 到原有 `assignLayers()`，无 MoE split |
| `OLLAMA_MOE_GPU_LAYERS=0` | `moeGPUCount=0`，所有 MoE 在 CPU，可用于 baseline 对比 |
| 多 GPU | `denseGPULayers` 仅分配到第一个 GPU（Phase 1 简化），多 GPU MoE split 为后续工作 |
| `requireFull=true` | `verifyLayout` 用 `denseGPULayers` 做检查，正确处理 |

## 成功标准

| 指标 | 目标 |
|------|------|
| 所有 48 层 attention/dense 在 GPU | Step 4 日志 `dense=gpu` 全部确认 |
| ~22 层 MoE 在 GPU，~26 层在 CPU | Step 3 日志确认 |
| Prefill 1k tokens 延迟 | ≤ 1.6s（较基线 2.0s 改善 ≥ 20%）|
| 非 MoE 模型行为不变 | 无 `moe split:` Info 日志输出 |
