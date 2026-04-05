# Runner 架构

## 概览

Runner 是独立的**子进程**，由 Ollama 主服务器通过 `exec.Command` 启动，通过 localhost HTTP 通信。
有三种 runner：`ollamarunner`（新引擎）、`llamarunner`（llama.cpp）、`mlxrunner`（MLX）。

> 图例：🟢 绿色粗边框 = Ollama Go ｜ 🟠 橙色粗边框 = llama.cpp C/C++

```mermaid
flowchart TD
    subgraph main ["🟢 Ollama 主进程"]
        CLI["CLI (cmd/)"] --> Server["HTTP Server (server/)"]
        Server --> LLM["LLM Manager<br/>llm/server.go"]
    end

    LLM -->|"exec.Command<br/>localhost:PORT"| Runner

    subgraph sub ["Runner 子进程 (独立进程)"]
        Runner["🟢 runner/runner.go<br/>入口分发"]
        Runner -->|"--ollama-engine"| OR["🟢 ollamarunner<br/>Go 构图"]
        Runner -->|"默认 (无 flag)"| LR["🟠 llamarunner<br/>llama.cpp 构图"]
        Runner -->|"--mlx-engine"| MR["mlxrunner<br/>MLX 构图"]
    end

    OR -->|HTTP| Server
    LR -->|HTTP| Server

    classDef go stroke:#22c55e,stroke-width:3px
    classDef cpp stroke:#f97316,stroke-width:3px

    class CLI,Server,LLM,Runner,OR go
    class LR cpp

    style main stroke:#22c55e,stroke-width:2px,stroke-dasharray: 5 5
```

## Runner 选择逻辑

### 决策点：`llm/server.go` (Ollama Go)

```mermaid
flowchart TD
    Start["NewLlamaServer()"] --> Check{"envconfig.NewEngine()<br/>或<br/>f.KV().OllamaEngineRequired()?"}
    
    Check -->|"否"| LlamaRunner["🟠 使用 llamarunner"]
    Check -->|"是"| TryTok["尝试创建 tokenizer<br/>model.NewTextProcessor(modelPath)"]
    
    TryTok -->|"成功 (tok != nil)"| OllamaRunner["🟢 使用 ollamarunner"]
    TryTok -->|"失败"| Fallback["降级日志:<br/>'model not yet supported<br/>by Ollama engine'"]
    Fallback --> LlamaRunner

    classDef go stroke:#22c55e,stroke-width:3px
    classDef cpp stroke:#f97316,stroke-width:3px

    class OllamaRunner go
    class LlamaRunner cpp
    style Fallback fill:#fdd,stroke:#333
```

### 关键代码

```go
// llm/server.go (Ollama Go) — line 143
if envconfig.NewEngine() || f.KV().OllamaEngineRequired() {
    if len(projectors) == 0 {
        tok, err = model.NewTextProcessor(modelPath)
    } else {
        err = errors.New("split vision models aren't supported")
    }
    if err != nil {
        slog.Debug("model not yet supported by Ollama engine, switching to compatibility mode")
    }
}

// ... 启动子进程 ...

// 分叉点 — line 315
if tok != nil {
    return &ollamaServer{llmServer: s, tokenizer: tok}, nil   // → ollamarunner
} else {
    return &llamaServer{llmServer: s, ggml: f}, nil           // → llamarunner
}
```

### 子进程启动

```go
// llm/server.go (Ollama Go) — line 346
params := []string{"runner"}
if ollamaEngine {
    params = append(params, "--ollama-engine")  // 关键 flag
}
// ... 添加模型路径、GPU 参数 ...
cmd := exec.Command(exe, params...)
```

### 子进程入口分发

```go
// runner/runner.go (Ollama Go) — line 10
func Execute(args []string) error {
    switch args[0] {
    case "--ollama-engine":   return ollamarunner.Execute(args[1:])
    case "--imagegen-engine": return imagegen.Execute(args[1:])
    case "--mlx-engine":      return mlxrunner.Execute(args[1:])
    }
    return llamarunner.Execute(args)  // 默认
}
```

## Runner 职责

每个 runner 是一个**完整的推理服务器**：

```mermaid
flowchart LR
    subgraph "Runner 内部职责"
        A["🟢 HTTP API<br/>/completion<br/>/embedding<br/>/health"] --> B["🟢 请求调度<br/>多序列并发管理"]
        B --> C["🟢 批处理组装<br/>多序列 → 单个 batch"]
        C --> D["🟢/🟠 构图<br/>Forward() 或 build_graph()"]
        D --> E["🟠 执行<br/>GGML Backend"]
        E --> F["🟢 采样<br/>temperature/top-p"]
        F --> G["🟢 KV Cache 管理<br/>slot 分配/复用/eviction"]
    end

    classDef go stroke:#22c55e,stroke-width:3px
    classDef cpp stroke:#f97316,stroke-width:3px
    classDef mixed stroke:#22c55e,stroke-width:3px,stroke-dasharray: 5 5

    class A,B,C,F,G go
    class E cpp
    class D mixed
```

## ollamarunner vs llamarunner 详细对比

### 主循环

```mermaid
flowchart TD
    subgraph ollamaloop ["🟢 ollamarunner (流水线化)"]
        O1["run()"] --> O2["forwardBatch()<br/>构图 (主线程)"]
        O2 --> O3["computeBatch()<br/>执行 (★ 可异步)"]
        O3 --> O4["处理输出 + 采样"]
        O4 --> O2
        
        O2 -.->|"前一个 batch 的<br/>compute 可并行"| O3
    end

    classDef go stroke:#22c55e,stroke-width:3px
    class O1,O2,O3,O4 go
    style ollamaloop stroke:#22c55e,stroke-width:2px,stroke-dasharray: 5 5
```

```mermaid
flowchart TD
    subgraph llamaloop ["🟠 llamarunner (同步)"]
        L1["run()"] --> L2["processBatch()"]
        L2 --> L3["填充 batch"]
        L3 --> L4["lc.Decode(batch)<br/>构图 + 执行 (llama.cpp 内部)"]
        L4 --> L5["lc.Synchronize()"]
        L5 --> L6["处理输出 + 采样"]
        L6 --> L2
    end

    classDef cpp stroke:#f97316,stroke-width:3px
    classDef go stroke:#22c55e,stroke-width:3px
    class L1,L2,L3,L6 go
    class L4,L5 cpp
    style llamaloop stroke:#f97316,stroke-width:2px,stroke-dasharray: 5 5
```

### 核心差异

| 方面 | ollamarunner | llamarunner |
|------|-------------|-------------|
| **构图语言** | Go（调 GGML C API via CGo） | C++（llama.cpp 内部） |
| **模型抽象** | `model.Model` 接口 | `*llama.Model`（CGo 绑定） |
| **构图入口** | `model.Forward(ctx, batch)` | `lc.Decode(batch)` |
| **批处理** | **流水线化**（forward 和 compute 可重叠） | 同步顺序执行 |
| **Tokenizer** | Go 实现的通用 tokenizer | llama.cpp 内置 |
| **采样** | `sample.Sampler`（Go） | llama.cpp `SamplingContext` |
| **多模态** | `MultimodalProcessor` 接口 | `ImageContext`（llava 风格） |
| **支持架构数** | ~21（不支持的降级到 llamarunner） | ~120+（全量） |

## 从构图到执行的完整调用链

**两个 runner 最终都走到同一个 GGML backend scheduler**：

```mermaid
flowchart TD
    subgraph go_path ["🟢 ollamarunner 路径"]
        OA["forwardBatch()"]
        OA --> OB["model.Forward(ctx, batch)<br/><i>model/model.go:323</i>"]
        OB --> OC["m.Forward(ctx, batch)<br/><i>model/models/llama/model.go:183</i><br/>Go 代码逐层构建 GGML 节点"]
        OC --> OD["ctx.Forward(t)<br/><i>ml/backend/ggml/ggml.go:794</i><br/>→ ggml_build_forward_expand()"]
    end
    
    subgraph cpp_path ["🟠 llamarunner 路径"]
        LA["processBatch()<br/><i>(Go 入口)</i>"]
        LA --> LB["s.lc.Decode(batch)<br/><i>runner/llamarunner/runner.go:494</i><br/>(Go → CGo 边界)"]
        LB --> LC["llama_decode() 内部<br/><i>llama.cpp C++</i><br/>build_graph() → 架构 switch"]
    end
    
    OD --> Sched
    LC --> Sched
    
    subgraph shared ["🟠 共享路径: GGML Backend Scheduler (C)"]
        Sched["ggml_backend_sched_graph_compute_async()<br/><i>ggml-backend.cpp:1869</i>"]
        Sched --> Split["ggml_backend_sched_split_graph()<br/><i>ggml-backend.cpp:960</i><br/>按 backend 拆分图"]
        Split --> Opt["★ ggml_backend_graph_optimize()<br/><i>ggml-backend.cpp:1361</i><br/>Backend fusion 在此发生"]
        Opt --> Alloc["ggml_backend_sched_alloc_splits()<br/>为优化后的图分配内存"]
        Alloc --> Exec["ggml_backend_sched_compute_splits()<br/><i>ggml-backend.cpp:1480</i><br/>执行 GPU/CPU kernels"]
    end

    classDef go stroke:#22c55e,stroke-width:3px
    classDef cpp stroke:#f97316,stroke-width:3px

    class OA,OB,OC,OD go
    class LA,LB,LC,Sched,Split,Opt,Alloc,Exec cpp

    style go_path stroke:#22c55e,stroke-width:2px,stroke-dasharray: 5 5
    style cpp_path stroke:#f97316,stroke-width:2px,stroke-dasharray: 5 5
    style shared stroke:#f97316,stroke-width:2px,stroke-dasharray: 5 5
    style Opt fill:#ff9,stroke:#f97316,stroke-width:3px
```

### 时序细节

| 阶段 | ollamarunner | llamarunner | graph_optimize? |
|------|-------------|-------------|----------------|
| 1. 组装 batch | Go 代码 | Go 代码 | - |
| 2. 构图 | Go `Forward()` 逐层构建 | C++ `build_graph()` | - |
| 3. 触发执行 | `ComputeWithNotify()` (可异步) | `Decode()` → `Synchronize()` | - |
| 4. 图拆分 | `split_graph()` | `split_graph()` | - |
| 5. **Backend 融合** | `graph_optimize()` | `graph_optimize()` | **★ 在这里** |
| 6. 内存分配 | `alloc_splits()` | `alloc_splits()` | - |
| 7. Kernel 执行 | `compute_splits()` | `compute_splits()` | - |

> **关键结论**：步骤 4-7 是**完全相同的 C 代码路径**（`ggml-backend.cpp`）。
> 两个 runner 的区别仅在步骤 1-3（构图方式不同），backend fusion 对两者一视同仁。

## 通信协议

Runner 通过 HTTP JSON API 暴露服务：

| 端点 | 用途 |
|------|------|
| `POST /load` | 加载模型 |
| `POST /completion` | 文本生成（流式 SSE） |
| `POST /embedding` | 向量嵌入 |
| `GET /health` | 健康检查 + 进度 |

主进程通过 `llmServer` 封装 HTTP 客户端与 runner 通信。
