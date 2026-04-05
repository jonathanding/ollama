# GGUF 转换机制

## 核心结论

GGUF 文件是一个**纯数据容器**，只包含权重张量和元数据，**不包含计算图**。
从 PyTorch/HuggingFace 到 GGUF 的转换本质上是 **1:1 的张量重命名 + 可选量化**。

## GGUF 文件内容

```mermaid
graph LR
    subgraph "GGUF 文件结构"
        A["Header<br/>magic, version, tensor_count, kv_count"]
        B["Key-Value 元数据<br/>general.architecture = 'llama'<br/>llama.attention.head_count = 32<br/>llama.context_length = 4096<br/>tokenizer.ggml.tokens = [...]<br/>..."]
        C["张量数据<br/>blk.0.attn_q.weight<br/>blk.0.attn_k.weight<br/>blk.0.ffn_gate.weight<br/>output_norm.weight<br/>..."]
    end
    A --> B --> C

    style A fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
    style C fill:#bfb,stroke:#333
```

**不包含的内容：**
- ❌ 计算图 / 算子定义
- ❌ Forward pass 逻辑
- ❌ 算子融合信息
- ❌ 执行计划

## 转换流程

```mermaid
flowchart TD
    A["PyTorch / HuggingFace 模型<br/>(model.safetensors + config.json)"] --> B["convert_hf_to_gguf.py<br/><i>llama.cpp Python 工具</i>"]
    
    B --> C["Step 1: 读取 config.json<br/>提取架构参数写入 KV 元数据"]
    B --> D["Step 2: 张量名映射<br/>model.layers.0.self_attn.q_proj.weight<br/>→ blk.0.attn_q.weight"]
    B --> E["Step 3: 可选量化<br/>FP16 → Q4_0 / Q5_1 / Q8_0 等"]
    B --> F["Step 4: 可选权重拼接<br/>gate + up expert → fused tensor<br/>Q + K + V → ATTN_QKV (--fuse-qkv)"]
    
    C --> G["GGUF 文件"]
    D --> G
    E --> G
    F --> G
    
    style B fill:#ff9,stroke:#333
    style G fill:#9f9,stroke:#333
```

### 转换脚本的实际代码 (llama.cpp Python)

```python
# convert_hf_to_gguf.py — 核心写入逻辑
def write(self):
    self.prepare_tensors()                              # 张量重命名 + 量化
    self.prepare_metadata(vocab_only=False)              # KV 元数据
    self.gguf_writer.write_header_to_file(path=self.fname_out)
    self.gguf_writer.write_kv_data_to_file()
    self.gguf_writer.write_tensors_to_file(progress=True)
```

没有任何图分析、算子识别或 fusion 步骤。

## 张量名映射规则

| HuggingFace 原始名 | GGUF 标准名 |
|---|---|
| `model.embed_tokens.weight` | `token_embd.weight` |
| `model.layers.{i}.self_attn.q_proj.weight` | `blk.{i}.attn_q.weight` |
| `model.layers.{i}.self_attn.k_proj.weight` | `blk.{i}.attn_k.weight` |
| `model.layers.{i}.self_attn.v_proj.weight` | `blk.{i}.attn_v.weight` |
| `model.layers.{i}.self_attn.o_proj.weight` | `blk.{i}.attn_output.weight` |
| `model.layers.{i}.mlp.gate_proj.weight` | `blk.{i}.ffn_gate.weight` |
| `model.layers.{i}.mlp.up_proj.weight` | `blk.{i}.ffn_up.weight` |
| `model.layers.{i}.mlp.down_proj.weight` | `blk.{i}.ffn_down.weight` |
| `model.layers.{i}.input_layernorm.weight` | `blk.{i}.attn_norm.weight` |
| `model.norm.weight` | `output_norm.weight` |
| `lm_head.weight` | `output.weight` |

映射由 `gguf.TensorNameMap` 定义（llama.cpp Python 库）。

## 权重级别的"融合"

转换时唯一做的"融合"是**张量拼接**（不是算子融合）：

### 1. MoE Gate/Up Expert 拼接（已在主线）

```
gate_proj: (n_expert, n_ff, n_embd)  ─┐
                                       ├──→ ffn_gate_up_exps: (n_expert, n_ff*2, n_embd)
up_proj:   (n_expert, n_ff, n_embd)  ─┘
```

推理时用一次 `MUL_MAT` 代替两次。

### 2. QKV 权重拼接（PR #20628，opt-in `--fuse-qkv`）

```
Q: (n_embd, n_head * head_dim)     ─┐
K: (n_embd, n_head_kv * head_dim)   ├──→ ATTN_QKV: (n_embd, total_dim)
V: (n_embd, n_head_kv * head_dim)  ─┘
```

推理时一次大矩阵乘代替三次小矩阵乘，提速 1.3%~13.6%。

## 关键认知

> **Flash attention、RMS norm 等 fused op 不在 GGUF 文件里。**
> 它们由推理引擎在构图时根据架构代码硬编码决定，且部分（如 flash attention）是运行时根据 GPU 能力动态选择的。

GGUF 的 `general.architecture` 字段告诉推理引擎实例化哪个**预写好的模型图定义**，引擎再把 GGUF 中的命名张量加载到图的对应位置。

> 图例：🟢 绿色粗边框 = Ollama Go ｜ 🟠 橙色粗边框 = llama.cpp C/C++

```mermaid
flowchart LR
    GGUF["GGUF 文件<br/>architecture = 'llama'<br/>+ 权重张量"]
    
    GGUF -->|"读取 architecture"| Dispatch["模型分发"]
    
    Dispatch -->|"Ollama Go"| GoModel["🟢 model/models/llama/<br/>Forward() 构图"]
    Dispatch -->|"llama.cpp C++"| CppModel["🟠 src/models/llama.cpp<br/>llm_build_llama 构图"]
    
    GoModel -->|"加载权重"| GGUF
    CppModel -->|"加载权重"| GGUF
    
    GoModel --> Graph["GGML 计算图"]
    CppModel --> Graph
    
    classDef go stroke:#22c55e,stroke-width:3px
    classDef cpp stroke:#f97316,stroke-width:3px
    
    class GoModel go
    class CppModel cpp
    
    style GGUF fill:#9f9,stroke:#333
    style Graph fill:#f99,stroke:#333
```
