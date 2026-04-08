# qwen3-coder-next: ollama runner vs llama runner 性能对比

**日期**: 2026-04-08  
**模型**: qwen3-coder-next (Q4_K_M, 48.18 GiB, 79.67B 参数)  
**硬件**: NVIDIA GeForce RTX 3090 (24 GB)  
**基准 commit**: `f0cbddd8` (llama: support qwen3-coder-next in llama runner with ollama GGUF format)

---

## Layer Offload

| Runner        | dGPU 层数 | dGPU 占比 | CPU 层数 | CPU 占比 |
|---------------|-----------|-----------|----------|----------|
| ollama runner | 20        | 42%       | 29       | 58%      |
| llama runner  | 18        | 40%       | 31       | 60%      |

两种 runner 的层分配几乎相同（相差 2 层），保证了 VRAM 占用对等，对比有效。

---

## Prefill 吞吐量（tokens/sec）

### Batch Size = 1024

| prompt_tokens | ollama runner | llama runner | llama 相对提升 |
|---------------|--------------|-------------|--------------|
| 512           | 292 t/s      | 407 t/s     | **+39%**     |
| 1024          | 463 t/s      | 596 t/s     | **+29%**     |
| 2048          | 447 t/s      | 570 t/s     | **+27%**     |
| 4096          | 485 t/s      | 612 t/s     | **+26%**     |

### Batch Size = 512（ollama 默认）

| prompt_tokens | ollama runner | llama runner | llama 相对提升 |
|---------------|--------------|-------------|--------------|
| 512           | 289 t/s      | 403 t/s     | **+39%**     |
| 1024          | 303 t/s      | 416 t/s     | **+37%**     |
| 2048          | 305 t/s      | 411 t/s     | **+35%**     |
| 4096          | 321 t/s      | 436 t/s     | **+36%**     |

---

## Decode 吞吐量（tokens/sec）

### Batch Size = 1024

| prompt_tokens | ollama runner | llama runner | ollama 相对优势 |
|---------------|--------------|-------------|---------------|
| 512           | 18 t/s       | 12 t/s      | **+50%**      |
| 1024          | 18 t/s       | 12 t/s      | **+50%**      |
| 2048          | 17 t/s       | 11 t/s      | **+55%**      |
| 4096          | 16 t/s       | 11 t/s      | **+45%**      |

### Batch Size = 512（ollama 默认）

| prompt_tokens | ollama runner | llama runner | ollama 相对优势 |
|---------------|--------------|-------------|---------------|
| 512           | 18 t/s       | 12 t/s      | **+50%**      |
| 1024          | 18 t/s       | 11 t/s      | **+64%**      |
| 2048          | 17 t/s       | 11 t/s      | **+55%**      |
| 4096          | 16 t/s       | 11 t/s      | **+45%**      |

---

## Prefill 延迟（ms，batch size = 1024）

| prompt_tokens | ollama runner | llama runner | llama 相对提升 |
|---------------|--------------|-------------|--------------|
| 512           | 1609 ms      | 1155 ms     | **-28%**     |
| 1024          | 2088 ms      | 1620 ms     | **-22%**     |
| 2048          | 4737 ms      | 3713 ms     | **-22%**     |
| 4096          | 8368 ms      | 6639 ms     | **-21%**     |

---

## 关键发现

### Prefill：llama runner 领先 ~26–39%

llama runner 在所有 prompt 长度上的 prefill 吞吐量均明显优于 ollama runner。以 batch size 1024 为例，512 tokens prompt 提升 39%，4096 tokens 提升 26%。batch size 对效果影响显著：

- **ollama runner**：batch size 从 512 增至 1024，1024-token prompt 吞吐量提升 53%（303 → 463 t/s）
- **llama runner**：同等条件下提升 43%（416 → 596 t/s）

两者均能从更大 batch size 受益，但 ollama runner 改善幅度更大，说明其在小 batch 下的利用率偏低。

### Decode：ollama runner 领先 ~45–64%

decode 阶段（单 token 自回归生成）ollama runner 显著更快（18 t/s vs 11–12 t/s）。这一差距在两种 batch size 配置下均保持稳定，说明不是 batch 处理效率的差异，而是单 token 推理路径的根本性能差异。

### 性能权衡总结

| 指标           | 更优 runner   | 优势幅度 |
|----------------|--------------|--------|
| Prefill 吞吐量  | **llama runner** | +26–39% |
| Decode 吞吐量   | **ollama runner** | +45–64% |

---

## 原因分析

**Prefill 优势（llama runner）**

qwen3-coder-next 的 recurrent 层在 prefill 阶段采用分块矩阵运算（chunked delta net）。llama.cpp 对大规模 GEMM 操作有成熟的 CUDA kernel 优化路径（cuBLAS），而 ollama 的 Go ML 引擎对该混合架构的 prefill batch 路径优化较少。

**Decode 劣势（llama runner）**

单 token 生成时，recurrent 层退化为纯 recurrent 更新（O(1) per token），无法利用矩阵批处理加速。ollama runner 专门针对该路径做了优化（deltanet autoregressive path 经过 Go 层精细调度），而 llama.cpp 的 autoregressive 路径对 qwen3-coder-next 这类新型混合架构的单步效率尚未充分优化。

**Layer 分配差异的潜在影响**

llama runner 比 ollama runner 少 2 层在 GPU 上运行（18 vs 20 层），略微不利于 llama runner。若对齐 GPU 层数，预计 llama runner prefill 性能还可进一步小幅提升。

---

## 结论

对于 qwen3-coder-next 这类混合 attention/recurrent 架构，当前两种 runner 存在明确的性能权衡：

- **长文档处理、RAG 检索、代码补全（prompt 较长）** → llama runner 更优（prefill 快 ~30%）
- **多轮对话、流式输出（decode token 多）** → ollama runner 更优（decode 快 ~50%）

下一步可以考虑：
1. 分析 llama runner decode 慢的根本原因（kernel profiling）
2. 尝试在 llama.cpp 中优化 qwen3next autoregressive 路径
3. 测试不同 GPU 层数配置对两种 runner 的影响

---

## Reference 1: Raw Benchmark Data

### ollama runner — batch size 1024

```
Starting benchmark: model=qwen3-coder-next  sizes=512,1024,2048,4096  epochs=6  warmup=4  batch-size=1024
Warning: warmup may be insufficient — prefill_tps changed 88% between warmup iterations
  hint: increase -warmup (current: 4)

Model: qwen3-coder-next  |  Epochs: 6  |  Warmup: 4
note: prefill_ms is the server-side prompt processing time (Ollama internal metric); ttft_ms is the wall-clock time to first token (= prefill_ms + HTTP round-trip + scheduling, typically 80–200 ms longer on localhost).

prompt_tokens │ prefill_ms │   CV% │ prefill_tps │   CV% │ ttft_ms    │   CV% │ gen_ms    │   CV% │ gen_tps   │   CV% │ status
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
512           │    1609 ms │  3.7% │     292 t/s │  4.6% │    1706 ms │  3.6% │    887 ms │  1.3% │    18 t/s │  1.3% │ ✓
  note: 1 epoch(s) excluded from stats for size=1024 (early EOS)
1024          │    2088 ms │  4.0% │     463 t/s │  3.5% │    2186 ms │  3.9% │    896 ms │  1.3% │    18 t/s │  1.3% │ ✓
2048          │    4737 ms │  7.3% │     447 t/s │  5.9% │    4836 ms │  7.1% │    918 ms │  2.9% │    17 t/s │  1.1% │ ⚠
⚠ [size=2048] prefill_tps CV=5.9% exceeds threshold 5.0%
  hint: consider increasing -warmup (current: 4) or closing background processes
⚠ [size=2048] ttft_ms CV=7.1% exceeds threshold 5.0%
  note: 2 epoch(s) excluded from stats for size=4096 (early EOS)
4096          │    8368 ms │  4.1% │     485 t/s │  2.8% │    8471 ms │  4.0% │    991 ms │  1.1% │    16 t/s │  1.1% │ ✓
```

### ollama runner — batch size 512（default）

```
Model: qwen3-coder-next  |  Epochs: 6  |  Warmup: 4
note: prefill_ms is the server-side prompt processing time (Ollama internal metric); ttft_ms is the wall-clock time to first token (= prefill_ms + HTTP round-trip + scheduling, typically 80–200 ms longer on localhost).

prompt_tokens │ prefill_ms │   CV% │ prefill_tps │   CV% │ ttft_ms    │   CV% │ gen_ms    │   CV% │ gen_tps   │   CV% │ status
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
512           │    1626 ms │  4.1% │     289 t/s │  4.9% │    1724 ms │  3.8% │    893 ms │  1.3% │    18 t/s │  1.3% │ ✓
  note: 1 epoch(s) excluded from stats for size=1024 (early EOS)
1024          │    3186 ms │  3.5% │     303 t/s │  3.1% │    3286 ms │  3.4% │    899 ms │  0.6% │    18 t/s │  0.6% │ ✓
2048          │    6929 ms │  5.3% │     305 t/s │  3.7% │    7028 ms │  5.2% │    930 ms │  1.1% │    17 t/s │  1.1% │ ⚠
⚠ [size=2048] ttft_ms CV=5.2% exceeds threshold 5.0%
  note: 2 epoch(s) excluded from stats for size=4096 (early EOS)
4096          │   12627 ms │  2.6% │     321 t/s │  1.4% │   12731 ms │  2.6% │    984 ms │  0.8% │    16 t/s │  0.8% │ ✓
```

### llama runner — batch size 1024

```
Starting benchmark: model=qwen3-coder-next  sizes=512,1024,2048,4096  epochs=6  warmup=4  batch-size=1024

Model: qwen3-coder-next  |  Epochs: 6  |  Warmup: 4
note: prefill_ms is the server-side prompt processing time (Ollama internal metric); ttft_ms is the wall-clock time to first token (= prefill_ms + HTTP round-trip + scheduling, typically 80–200 ms longer on localhost).

prompt_tokens │ prefill_ms │   CV% │ prefill_tps │   CV% │ ttft_ms    │   CV% │ gen_ms    │   CV% │ gen_tps   │   CV% │ status
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
512           │    1155 ms │  2.3% │     407 t/s │  3.2% │    1257 ms │  2.2% │   1387 ms │  2.4% │    12 t/s │  2.3% │ ✓
  note: 1 epoch(s) excluded from stats for size=1024 (early EOS)
1024          │    1620 ms │  1.9% │     596 t/s │  2.2% │    1723 ms │  1.8% │   1386 ms │  0.3% │    12 t/s │  0.3% │ ✓
2048          │    3713 ms │  6.8% │     570 t/s │  5.4% │    3818 ms │  6.6% │   1400 ms │  0.4% │    11 t/s │  0.4% │ ⚠
⚠ [size=2048] prefill_tps CV=5.4% exceeds threshold 5.0%
  hint: consider increasing -warmup (current: 4) or closing background processes
⚠ [size=2048] ttft_ms CV=6.6% exceeds threshold 5.0%
4096          │    6639 ms │  2.9% │     612 t/s │  1.8% │    6749 ms │  2.8% │   1475 ms │  0.6% │    11 t/s │  0.6% │ ✓
```

### llama runner — batch size 512（ollama default）

```
Starting benchmark: model=qwen3-coder-next  sizes=512,1024,2048,4096  epochs=6  warmup=4

Model: qwen3-coder-next  |  Epochs: 6  |  Warmup: 4
note: prefill_ms is the server-side prompt processing time (Ollama internal metric); ttft_ms is the wall-clock time to first token (= prefill_ms + HTTP round-trip + scheduling, typically 80–200 ms longer on localhost).

prompt_tokens │ prefill_ms │   CV% │ prefill_tps │   CV% │ ttft_ms    │   CV% │ gen_ms    │   CV% │ gen_tps   │   CV% │ status
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
512           │    1167 ms │  2.7% │     403 t/s │  4.8% │    1255 ms │  2.2% │   1387 ms │  0.4% │    12 t/s │  0.4% │ ✓
  note: 1 epoch(s) excluded from stats for size=1024 (early EOS)
1024          │    2325 ms │  2.3% │     416 t/s │  2.1% │    2415 ms │  2.5% │   1402 ms │  0.9% │    11 t/s │  0.9% │ ✓
2048          │    5140 ms │  5.1% │     411 t/s │  3.5% │    5238 ms │  4.9% │   1428 ms │  0.7% │    11 t/s │  0.7% │ ✓
4096          │    9319 ms │  2.7% │     436 t/s │  1.8% │    9429 ms │  2.7% │   1489 ms │  0.5% │    11 t/s │  0.5% │ ✓
```

### Layer Offload（原始记录）

```
layer offload
|      | ollama_runner | llama_runner |
| ---- | ------------- | ------------ |
| dGPU | 20 (42%)      | 18 (40%)     |
| CPU  | 29 (58%)      | 31 (60%)     |
```

## Reference 2： 验证ollama runner 的 llama runner 的输出一致性

```
curl -s http://localhost:11434/api/generate -d "{""model"":""qwen3-coder-next"",""prompt"":""def fibonacci(n: int) -> int:"",""options"":{""temperature"":0,""seed"":42,""num_predict"":200},""stream"":false}" | python -c "import sys,json; print(json.load(sys.stdin)['response'])"
```

### Ollama runner 输出
```
Here's a clean and efficient implementation of the Fibonacci function using **iterative approach** (O(n) time, O(1) space):  
  
\`\`\`python  
def fibonacci(n: int) -> int:  
"""  
Calculate the nth Fibonacci number (0-indexed).  
  
Args:  
n: Non-negative integer (n >= 0)  
  
Returns:  
The nth Fibonacci number  
  
Raises:  
ValueError: If n is negative  
"""  
if n < 0:  
raise ValueError("n must be non-negative")  
  
if n <= 1:  
return n  
  
a, b = 0, 1 # F(0) = 0, F(1) = 1  
for _ in range(2, n + 1):  
a, b = b, a + b  
  
return b
\`\`\`
  
### Key Features:  
- **0-indexed**: `fibonacci(0) = 0`,
```

### Llama runner 输出

```
Here's a clean, efficient implementation of the Fibonacci function using **iterative approach** (O(n) time, O(1) space):

\`\`\`python
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number (0-indexed).

    Args:
        n: Non-negative integer index

    Returns:
        The nth Fibonacci number

    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    if n <= 1:
        return n

    a, b = 0, 1  # F(0) = 0, F(1) = 1
    for _ in range(2, n + 1):
        a, b = b, a + b

    return b
\`\`\`

### Key Features:
- **0-indexed**: `fibonacci(0) = 0`, `fibonacci(1)
```
