# KV Cache 量化评测

**日期：** 2026-04-10
**模型：** qwen3-coder-next
**实验条件：** Batch size 1024，Input tokens 1024，GPU layers 20/49（20 repeating layers on CUDA0，output layer on CPU）

```bash
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

---

## 内存占用对比

> 括号内百分比均以 **f16 的对应列数值**为基准。

| KV Cache 类型 | GPU KV Cache | CPU KV Cache | 总内存 |
|:---:|:---:|:---:|:---:|
| f16（默认，基准） | 1.1 GiB | 1.5 GiB | 51.9 GiB |
| q8_0 | 955.2 MiB (vs f16 GPU KV: ↓13%) | 1.3 GiB (vs f16 CPU KV: ↓13%) | 51.6 GiB (vs f16 总内存: ↓0.6%) |
| q4_0 | 875.2 MiB (vs f16 GPU KV: ↓20%) | 1.2 GiB (vs f16 CPU KV: ↓20%) | 51.4 GiB (vs f16 总内存: ↓1.0%) |

---

## 性能对比

| KV Cache 类型 | prefill_ms | prefill_tps | ttft_ms | gen_ms | gen_tps |
|:---:|:---:|:---:|:---:|:---:|:---:|
| f16 | 2096 ms | 467 t/s | 2168 ms | 895 ms | 18 t/s |
| q8_0 | 2064 ms (↓1.5%) | 475 t/s (↑1.7%) | 2141 ms (↓1.2%) | 896 ms | 18 t/s |
| q4_0 | 2075 ms (↓1.0%) | 472 t/s (↑1.1%) | 2154 ms (↓0.6%) | 872 ms (↓2.6%) | 18 t/s |

> `prefill_ms`：服务端 prompt 处理时间（Ollama 内部指标）；`ttft_ms`：首 token 墙钟时间（= prefill_ms + HTTP 往返 + 调度，localhost 通常多 80–200 ms）。

---

## 结论

1. **KV cache 量化对推理性能无明显影响**：prefill 和 gen 速度在三种配置下基本持平，差异均在统计误差范围内。

2. **内存节省有限**：相较 f16，q8_0 的 KV cache 大小节省约 13%，q4_0 节省约 20%（均指 KV cache 本身，非总内存），但换算到总内存仅减少 0.3–0.5 GiB，远不足以将更多层从 CPU 搬到 GPU，三种配置下 GPU layers 均维持 20/49 不变。

3. **量化收益受限于当前瓶颈**：本实验场景下内存瓶颈在模型权重（GPU 19.9 GiB + CPU 28.3 GiB），而非 KV cache，因此量化无法带来 offload 层数增加的收益。

---

## Reference: Raw Data

### f16（默认）

**Load log：**
```
time=2026-04-09T23:30:17.630-07:00 level=INFO source=runner.go:1284 msg=load request="{Operation:commit LoraPath:[] Parallel:1 BatchSize:1024 FlashAttention:Enabled KvSize:32768 KvCacheType: NumThreads:8 GPULayers:20[ID:GPU-223c73ff-f253-3c8c-d212-82b5d1ce2963 Layers:20(28..47)] MultiUserCache:false ProjectorPath: MainGPU:0 UseMmap:false}"
time=2026-04-09T23:30:17.630-07:00 level=INFO source=ggml.go:482 msg="offloading 20 repeating layers to GPU"
time=2026-04-09T23:30:17.630-07:00 level=INFO source=ggml.go:486 msg="offloading output layer to CPU"
time=2026-04-09T23:30:17.630-07:00 level=INFO source=ggml.go:494 msg="offloaded 20/49 layers to GPU"
time=2026-04-09T23:30:17.630-07:00 level=INFO source=device.go:240 msg="model weights" device=CUDA0 size="19.9 GiB"
time=2026-04-09T23:30:17.631-07:00 level=INFO source=device.go:245 msg="model weights" device=CPU size="28.3 GiB"
time=2026-04-09T23:30:17.631-07:00 level=INFO source=device.go:251 msg="kv cache" device=CUDA0 size="1.1 GiB"
time=2026-04-09T23:30:17.631-07:00 level=INFO source=device.go:256 msg="kv cache" device=CPU size="1.5 GiB"
time=2026-04-09T23:30:17.631-07:00 level=INFO source=device.go:262 msg="compute graph" device=CUDA0 size="884.1 MiB"
time=2026-04-09T23:30:17.631-07:00 level=INFO source=device.go:267 msg="compute graph" device=CPU size="270.6 MiB"
time=2026-04-09T23:30:17.631-07:00 level=INFO source=device.go:272 msg="total memory" size="51.9 GiB"
time=2026-04-09T23:30:17.631-07:00 level=INFO source=sched.go:561 msg="loaded runners" count=1
time=2026-04-09T23:30:17.631-07:00 level=INFO source=server.go:1352 msg="waiting for llama runner to start responding"
```

**Benchmark：**
```
Starting benchmark: model=qwen3-coder-next  sizes=1024  epochs=6  warmup=4  batch-size=1024
Warning: warmup may be insufficient — prefill_tps changed 94% between warmup iterations
  hint: increase -warmup (current: 4)

Model: qwen3-coder-next  |  Epochs: 6  |  Warmup: 4
note: prefill_ms is the server-side prompt processing time (Ollama internal metric); ttft_ms is the wall-clock time to first token (= prefill_ms + HTTP round-trip + scheduling, typically 80–200 ms longer on localhost).

prompt_tokens │ prefill_ms │   CV% │ prefill_tps │   CV% │ ttft_ms    │   CV% │ gen_ms    │   CV% │ gen_tps   │   CV% │ status
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
1024          │    2096 ms │  3.4% │     467 t/s │  2.6% │    2168 ms │  3.3% │    895 ms │  1.7% │    18 t/s │  1.7% │ ✓
```

---

### q8_0

**Load log：**
```
time=2026-04-09T23:17:19.964-07:00 level=INFO source=runner.go:1284 msg=load request="{Operation:alloc LoraPath:[] Parallel:1 BatchSize:1024 FlashAttention:Enabled KvSize:32768 KvCacheType:q8_0 NumThreads:8 GPULayers:20[ID:GPU-223c73ff-f253-3c8c-d212-82b5d1ce2963 Layers:20(28..47)] MultiUserCache:false ProjectorPath: MainGPU:0 UseMmap:false}"
time=2026-04-09T23:17:20.651-07:00 level=INFO source=runner.go:1284 msg=load request="{Operation:commit LoraPath:[] Parallel:1 BatchSize:1024 FlashAttention:Enabled KvSize:32768 KvCacheType:q8_0 NumThreads:8 GPULayers:20[ID:GPU-223c73ff-f253-3c8c-d212-82b5d1ce2963 Layers:20(28..47)] MultiUserCache:false ProjectorPath: MainGPU:0 UseMmap:false}"
time=2026-04-09T23:17:20.651-07:00 level=INFO source=device.go:240 msg="model weights" device=CUDA0 size="19.9 GiB"
time=2026-04-09T23:17:20.651-07:00 level=INFO source=ggml.go:482 msg="offloading 20 repeating layers to GPU"
time=2026-04-09T23:17:20.651-07:00 level=INFO source=ggml.go:486 msg="offloading output layer to CPU"
time=2026-04-09T23:17:20.651-07:00 level=INFO source=ggml.go:494 msg="offloaded 20/49 layers to GPU"
time=2026-04-09T23:17:20.652-07:00 level=INFO source=device.go:245 msg="model weights" device=CPU size="28.3 GiB"
time=2026-04-09T23:17:20.652-07:00 level=INFO source=device.go:251 msg="kv cache" device=CUDA0 size="955.2 MiB"
time=2026-04-09T23:17:20.652-07:00 level=INFO source=device.go:256 msg="kv cache" device=CPU size="1.3 GiB"
time=2026-04-09T23:17:20.652-07:00 level=INFO source=device.go:262 msg="compute graph" device=CUDA0 size="918.1 MiB"
time=2026-04-09T23:17:20.652-07:00 level=INFO source=device.go:267 msg="compute graph" device=CPU size="270.6 MiB"
time=2026-04-09T23:17:20.652-07:00 level=INFO source=device.go:272 msg="total memory" size="51.6 GiB"
time=2026-04-09T23:17:20.652-07:00 level=INFO source=sched.go:561 msg="loaded runners" count=1
time=2026-04-09T23:17:20.652-07:00 level=INFO source=server.go:1352 msg="waiting for llama runner to start responding"
```

**Benchmark：**
```
Starting benchmark: model=qwen3-coder-next  sizes=1024  epochs=6  warmup=4  batch-size=1024
Warning: warmup may be insufficient — prefill_tps changed 78% between warmup iterations
  hint: increase -warmup (current: 4)
  note: 1 epoch(s) excluded from stats for size=1024 (early EOS)

Model: qwen3-coder-next  |  Epochs: 6  |  Warmup: 4
note: prefill_ms is the server-side prompt processing time (Ollama internal metric); ttft_ms is the wall-clock time to first token (= prefill_ms + HTTP round-trip + scheduling, typically 80–200 ms longer on localhost).

prompt_tokens │ prefill_ms │   CV% │ prefill_tps │   CV% │ ttft_ms    │   CV% │ gen_ms    │   CV% │ gen_tps   │   CV% │ status
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
1024          │    2064 ms │  2.6% │     475 t/s │  0.8% │    2141 ms │  2.3% │    896 ms │  1.1% │    18 t/s │  1.1% │ ✓
```

---

### q4_0

**Load log：**
```
time=2026-04-09T23:38:35.344-07:00 level=INFO source=runner.go:1284 msg=load request="{Operation:commit LoraPath:[] Parallel:1 BatchSize:1024 FlashAttention:Enabled KvSize:32768 KvCacheType:q4_0 NumThreads:8 GPULayers:20[ID:GPU-223c73ff-f253-3c8c-d212-82b5d1ce2963 Layers:20(28..47)] MultiUserCache:false ProjectorPath: MainGPU:0 UseMmap:false}"
time=2026-04-09T23:38:35.344-07:00 level=INFO source=device.go:240 msg="model weights" device=CUDA0 size="19.9 GiB"
time=2026-04-09T23:38:35.344-07:00 level=INFO source=ggml.go:482 msg="offloading 20 repeating layers to GPU"
time=2026-04-09T23:38:35.344-07:00 level=INFO source=ggml.go:486 msg="offloading output layer to CPU"
time=2026-04-09T23:38:35.344-07:00 level=INFO source=ggml.go:494 msg="offloaded 20/49 layers to GPU"
time=2026-04-09T23:38:35.344-07:00 level=INFO source=device.go:245 msg="model weights" device=CPU size="28.3 GiB"
time=2026-04-09T23:38:35.344-07:00 level=INFO source=device.go:251 msg="kv cache" device=CUDA0 size="875.2 MiB"
time=2026-04-09T23:38:35.344-07:00 level=INFO source=device.go:256 msg="kv cache" device=CPU size="1.2 GiB"
time=2026-04-09T23:38:35.344-07:00 level=INFO source=device.go:262 msg="compute graph" device=CUDA0 size="893.1 MiB"
time=2026-04-09T23:38:35.344-07:00 level=INFO source=device.go:267 msg="compute graph" device=CPU size="270.6 MiB"
time=2026-04-09T23:38:35.344-07:00 level=INFO source=device.go:272 msg="total memory" size="51.4 GiB"
time=2026-04-09T23:38:35.344-07:00 level=INFO source=sched.go:561 msg="loaded runners" count=1
time=2026-04-09T23:38:35.344-07:00 level=INFO source=server.go:1352 msg="waiting for llama runner to start responding"
```

**Benchmark：**
```
Starting benchmark: model=qwen3-coder-next  sizes=1024  epochs=6  warmup=4  batch-size=1024
Warning: warmup may be insufficient — prefill_tps changed 92% between warmup iterations
  hint: increase -warmup (current: 4)
  note: 1 epoch(s) excluded from stats for size=1024 (early EOS)

Model: qwen3-coder-next  |  Epochs: 6  |  Warmup: 4
note: prefill_ms is the server-side prompt processing time (Ollama internal metric); ttft_ms is the wall-clock time to first token (= prefill_ms + HTTP round-trip + scheduling, typically 80–200 ms longer on localhost).

prompt_tokens │ prefill_ms │   CV% │ prefill_tps │   CV% │ ttft_ms    │   CV% │ gen_ms    │   CV% │ gen_tps   │   CV% │ status
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
1024          │    2075 ms │  2.9% │     472 t/s │  1.2% │    2154 ms │  2.8% │    872 ms │  1.3% │    18 t/s │  1.3% │ ✓
```
