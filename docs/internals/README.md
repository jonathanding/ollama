# Internals 调查文档索引

Ollama / llama.cpp / GGML 内部机制的调查研究文档。

## 文档列表

### GGUF 格式与模型转换

- [GGUF 转换机制](gguf-conversion.md) — GGUF 文件结构（纯权重+元数据，无计算图）、PyTorch→GGUF 的 1:1 张量翻译流程、权重级拼接（MoE gate/up、QKV fusion）

### 计算图构建

- [模型计算图构建](model-graph-construction.md) — 每架构独立构图机制、Ollama Go (`model/models/`) 和 llama.cpp C++ (`llama.cpp/src/models/`) 两套并行实现、注册/分发方式对比、架构特异性示例（Llama / Qwen3 / Gemma3）
- [计算图与权重加载的分离](graph-without-weights.md) — 构图不需要权重数据、graph_optimize 也不需要、完整的数据加载时序、无权重获取计算图的方法（`ggml_graph_dump_dot`、Reserve API 等）

### 算子与执行

- [算子融合](operator-fusion.md) — 原语级 fused op（FlashAttention、GLU、RMSNorm）+ backend 级 graph_optimize pattern matching fusion（Metal / CUDA / Vulkan 各自实现）、融合触发时机、与 PyTorch 的对比
- [Runner 架构](runner-architecture.md) — Runner 选择逻辑与分叉点（`llm/server.go`）、子进程通信机制、ollamarunner vs llamarunner 对比、从构图→backend fusion→kernel 执行的完整调用链

## 架构全景

> 图例：🟢 绿色粗边框 = Ollama Go 代码 ｜ 🟠 橙色粗边框 = llama.cpp C/C++ 代码

```mermaid
flowchart TD
    GGUF["GGUF 文件<br/>(权重 + 元数据)"]

    subgraph "Ollama 主进程"
        CLI["CLI"] --> Server["HTTP Server"]
        Server --> LLM["LLM Manager<br/>选择 runner"]
    end

    LLM -->|"exec.Command"| Runner["Runner 子进程"]

    Runner -->|"--ollama-engine"| OR["🟢 ollamarunner<br/>Go 构图 (21 架构)"]
    Runner -->|"默认"| LR["🟠 llamarunner<br/>llama.cpp C++ 构图 (120+ 架构)"]

    OR -->|"model.Forward()"| Graph["GGML 计算图"]
    LR -->|"build_graph()"| Graph

    GGUF -->|"加载权重"| OR
    GGUF -->|"加载权重"| LR

    Graph --> Sched["🟠 GGML Backend Scheduler"]
    Sched --> Split["🟠 图拆分 (按 backend)"]
    Split --> Fuse["🟠 graph_optimize()<br/>Backend Fusion"]
    Fuse --> Exec["🟠 Kernel 执行"]

    Exec --> Metal["Metal"]
    Exec --> CUDA["CUDA"]
    Exec --> Vulkan["Vulkan"]
    Exec --> CPU["CPU (无 fusion)"]

    classDef go stroke:#22c55e,stroke-width:3px
    classDef cpp stroke:#f97316,stroke-width:3px

    class CLI,Server,LLM,Runner,OR go
    class LR,Sched,Split,Fuse,Exec,Metal,CUDA,Vulkan,CPU cpp

    click GGUF "gguf-conversion.md"
    click OR "model-graph-construction.md"
    click LR "model-graph-construction.md"
    click Fuse "operator-fusion.md"
    click Runner "runner-architecture.md"
```
