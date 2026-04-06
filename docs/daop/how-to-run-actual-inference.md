# 如何运行实际推理并采集 per-op GPU 计时

## 环境要求

- Windows 11，Intel Arc GPU (Vulkan)
- ollama 已编译 (`go build .`)

## 环境变量

```bash
export OLLAMA_VULKAN=1                # 启用 Vulkan 后端
export GGML_VK_PERF_LOGGER=1          # 启用 per-op GPU 计时输出到 stderr
export GGML_VK_PERF_LOGGER_FREQUENCY=1  # 每次 compute 都输出（默认 1）
export OLLAMA_FLASH_ATTENTION=1       # 启用 flash attention（与 estimate 一致）
```

## 步骤

### 1. 启动 server

```bash
OLLAMA_VULKAN=1 GGML_VK_PERF_LOGGER=1 OLLAMA_FLASH_ATTENTION=1 ollama serve 2>perf_log.txt
```

stderr 重定向到文件以捕获 per-op 计时。

### 2. 运行推理

```bash
# 第一次（注意每次改变 prompt 首字符，避免 prefix cache）
curl -s http://localhost:11434/api/generate \
  -d '{"model":"qwen3:1.7b","prompt":"A quick test of performance","stream":false,"options":{"num_predict":20}}' \
  | jq '{prompt_eval_count, prompt_eval_duration, eval_count, eval_duration}'

# 第二次（改首字符）
curl -s http://localhost:11434/api/generate \
  -d '{"model":"qwen3:1.7b","prompt":"B quick test of performance","stream":false,"options":{"num_predict":20}}' \
  | jq '{prompt_eval_count, prompt_eval_duration, eval_count, eval_duration}'
```

### 3. 停止 server

```bash
# Ctrl+C 或 kill
```

### 4. 解析 perf_log.txt

`GGML_VK_PERF_LOGGER` 输出格式（每次 graph compute 输出一段）：

```
----------------
Vulkan Timings:
MUL_MAT_VEC q4_K m=1536 n=1 k=1536: 24 x 5.123 us
FLASH_ATTN_EXT: 12 x 100.456 us (xxx GFLOPS/s)
RMS_NORM: 48 x 1.234 us
...
Total time: 12345.678 us.
```

含义：
- `op_name: count x avg_time_us` — 该 op 执行了 count 次，平均每次 avg_time_us 微秒
- `Total time` — 所有 op 的 GPU 时间总和

### 5. API 返回的计时字段

```json
{
  "prompt_eval_count": 7,         // prefill token 数
  "prompt_eval_duration": 856000000,  // prefill 总时间 (纳秒)
  "eval_count": 20,               // decode token 数
  "eval_duration": 1632000000     // decode 总时间 (纳秒)
}
```

- prefill tok/s = prompt_eval_count / (prompt_eval_duration / 1e9)
- decode tok/s = eval_count / (eval_duration / 1e9)

## 注意事项

- 每次运行修改 prompt 首字符以避免 prefix cache
- PERF_LOGGER 输出在 stderr，需要重定向
- prefill 和 decode 的 per-op 分开在不同的 "Vulkan Timings" 段中
- 第一次推理会有模型加载开销，通常取第二次及以后的结果
