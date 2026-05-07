# GTPin ISA Dump 指南: 验证 dotPacked4x8EXT 映射到 dp4a 还是 dpas

## 背景

Ollama Vulkan 后端的 MMQ (Matrix Multiply Quantized) 路径在 GLSL 中使用 `dotPacked4x8EXT` 做 int8 点积。这个函数对应 SPIR-V 指令 `OpSDotKHR`，但 Intel Vulkan 驱动在 JIT 编译时可以将其映射到两种不同的硬件指令：

| GEN ISA 指令 | 执行单元 | 含义 |
|-------------|---------|------|
| `dp4a` | Vector Engine (EU ALU) | Dot Product of 4 elements, Accumulate |
| `dpas` | XMX (systolic array) | Dot Product Accumulate Systolic |

**核心问题**: 在有 XMX 的 Intel GPU (如 Lunar Lake, Arc 独显) 上，`dotPacked4x8EXT` 会被编译成 `dp4a` 还是 `dpas`？

**已知**: 在 Meteor Lake (无 XMX) 上已确认编译为 `dp4a`。需要在有 XMX 的硬件上验证。

## 工具

- **GTPin** (Graphics Technology Pin) — Intel GPU 二进制插桩框架，支持 Vulkan
- 下载: https://www.intel.com/content/www/us/en/download/730596/gtpin.html
- 版本: 4.7.1+
- 平台: Windows 10/11, Linux

## 前提条件

1. 已编译的 Ollama (带 Vulkan 后端)
2. 一个已下载的 GGUF 模型 (如 `qwen3:0.6b`)
3. GTPin 解压到某个目录 (以下假设 `C:\gtpin`)

## 步骤

### 1. 确认 GTPin 目录结构

```
C:\gtpin\Profilers\
  Bin\gtpin.exe              ← 主程序
  Examples\intel64\           ← 预编译工具 DLL
    opcodeprof.dll
    ...
  Lib\intel64\               ← 运行时库
    gtpin.dll, gtpin_core.dll, ged.dll, iga_wrapper.dll
```

### 2. 通过 GTPin 启动 Ollama

**关键**: GTPin 需要启动目标程序，不能 attach 到已运行的进程。

```bash
# Windows CMD
set OLLAMA_VULKAN=1
"C:\gtpin\Profilers\Bin\gtpin.exe" ^
  --dump_isa ^
  --profile_dir gtpin_out ^
  -t "C:\gtpin\Profilers\Examples\intel64\opcodeprof.dll" ^
  -- ollama.exe serve
```

```bash
# Linux / bash
OLLAMA_VULKAN=1 \
/path/to/gtpin/Profilers/Bin/gtpin \
  --dump_isa \
  --profile_dir gtpin_out \
  -t /path/to/gtpin/Profilers/Examples/intel64/opcodeprof.so \
  -- ./ollama serve
```

参数说明:
- `--dump_isa` — dump 编译后的 GEN ISA 汇编到 `profile_dir/ISA/`
- `--profile_dir gtpin_out` — 输出目录 (相对路径)
- `-t opcodeprof.dll` — 使用 opcode profiling 工具 (最轻量的内置工具之一)
- `-- ollama.exe serve` — 被分析的目标程序

### 3. 触发推理

在另一个终端:

```bash
curl http://127.0.0.1:11434/api/generate \
  -d '{"model":"qwen3:0.6b","prompt":"Hello world","options":{"num_predict":2}}'
```

> 注意: GTPin 插桩会显著拖慢 GPU 执行，推理可能比正常慢 5-10x。这是正常的。

### 4. 检查 ISA dump

推理完成后，ISA 文件出现在:

```
gtpin_out/GTPIN_PROFILE_OPCODEPROF_PID_<pid>_0/ISA/
```

每个 Vulkan compute shader 对应一个 `.asm` 文件，文件名格式:
```
CS_asm<hash>_simd<width>_<id>_<seq>_orig.asm
```

### 5. 搜索 dp4a / dpas

**Windows CMD:**
```cmd
REM 搜索所有 ISA 文件中的 dp4a 和 dpas
findstr /s /c:"dp4a" gtpin_out\GTPIN_PROFILE_*\ISA\*.asm > nul && echo Found dp4a
findstr /s /c:"dpas" gtpin_out\GTPIN_PROFILE_*\ISA\*.asm > nul && echo Found dpas

REM 按文件统计 dp4a 数量
for /r gtpin_out %%f in (*.asm) do @findstr /c:"dp4a" "%%f" > nul 2>&1 && echo %%~nxf: dp4a found

REM 按文件统计 dpas 数量
for /r gtpin_out %%f in (*.asm) do @findstr /c:"dpas" "%%f" > nul 2>&1 && echo %%~nxf: dpas found
```

**PowerShell (推荐):**
```powershell
Get-ChildItem -Recurse gtpin_out\*.asm | ForEach-Object {
    $dp4a = (Select-String -Path $_ -Pattern "dp4a" -SimpleMatch).Count
    $dpas = (Select-String -Path $_ -Pattern "dpas" -SimpleMatch).Count
    if ($dp4a -gt 0 -or $dpas -gt 0) {
        "$($_.Name): dp4a=$dp4a, dpas=$dpas"
    }
}
```

**Linux / Git Bash:**
```bash
for f in gtpin_out/GTPIN_PROFILE_*/ISA/*.asm; do
  dp4a=$(grep -c "dp4a" "$f")
  dpas=$(grep -c "dpas" "$f")
  if [ "$dp4a" -gt 0 ] || [ "$dpas" -gt 0 ]; then
    echo "$(basename $f): dp4a=$dp4a, dpas=$dpas"
  fi
done
```

### 6. 解读结果

**如果只有 `dp4a`**: `dotPacked4x8EXT` 映射到 Vector Engine，未使用 XMX。

**如果有 `dpas`**: `dotPacked4x8EXT` 映射到 XMX systolic array。

**如果两者都有**: 不同 shader 可能有不同的映射策略。

## 验证 ISA 对应关系

GTPin 文件名只有哈希值（如 `CS_asm04ef51056e138723_simd8_...`），不包含 Ollama 的 pipeline 名称（如 `matmul_q4_k_q8_1_s`）。有两种方法确认对应关系。

### 方法 A: ISA 汇编特征匹配（推荐）

直接阅读汇编，通过指令模式识别 shader。这是最可靠的方法。

#### Q4_K mmq_dot_product 的 ISA 特征

GLSL 源码 (`mul_mmq_funcs.glsl`, Q4_K `mmq_dot_product`):
```glsl
for (iqs = 0; iqs < 8; iqs++) {
    qs_a = (cache_a[ib_a].qs[iqs/2] >> ((iqs%2)*4)) & 0x0F0F0F0F;
    q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
}
return ds.x * dm.x * q_sum - dm.y * ds.y;
```

对应 GEN ISA 特征:
1. **`and ... 252645135:d`** — 即 `& 0x0F0F0F0F` (十进制 252645135)，提取 4-bit 量化值
2. **`shr ... 4:w`** — 右移 4 位，提取高 4 bit
3. **8 条连续 `dp4a`** — 循环展开的 8 次 int8 点积
4. **`dp4a` 第一条以 `0:w` 初始化** — `q_sum = 0` 然后累加
5. **最后 `mul` + `mad` 带取反** — scale 恢复: `ds.x * dm.x * q_sum - dm.y * ds.y`

示例 (Meteor Lake 实测):
```asm
and  r25.0:d  r11.0:d  252645135:d        // & 0x0F0F0F0F
shr  r12.0:ud r12.0:ud 4:w                // >> 4
and  r24.0:d  r12.0:d  252645135:d        // & 0x0F0F0F0F (shifted)

dp4a acc2.0:d  0:w       r26.0:d  r7.0:d  // q_sum = dp4a(0, qs_a[0], qs_b[0])
dp4a acc2.0:d  acc2.0:d  r25.0:d  r8.0:d  // q_sum += dp4a(qs_a[1], qs_b[1])
dp4a acc2.0:d  acc2.0:d  r24.0:d  r9.0:d  // q_sum += ...
dp4a acc0.0:d  acc2.0:d  r23.0:d  r10.0:d
dp4a acc2.0:d  acc0.0:d  r22.0:d  r31.0:d
dp4a acc2.0:d  acc2.0:d  r21.0:d  r32.0:d
dp4a acc2.0:d  acc2.0:d  r20.0:d  r33.0:d
dp4a r11.0:d   acc2.0:d  r19.0:d  r34.0:d // 第8次，结果写入 r11

mov  r11.0:f  r11.0:d                      // int → float
mul  r13.0:f  r17.0:f  r115.0:f            // dm.y * ds.y
mul  acc3.0:f r116.0:f r18.0:f             // dm.x * ds.x
mad  acc3.0:f -r13.0:f acc3.0:f r11.0:f   // dm.x*ds.x*q_sum - dm.y*ds.y
```

> 如果在有 XMX 的机器上，这些 `dp4a` 被替换为 `dpas`，则说明驱动利用了 XMX。

### 方法 B: Dispatch 顺序推断（辅助验证）

GTPin 日志（`--msgon all --msg_file gtpin_log.txt`）中记录了 kernel 的创建和 dispatch 顺序。结合我们在 Meteor Lake 上用修改版 Ollama（添加了 pipeline 名称日志）实测得到的 dispatch 序列，可以推断哪个 kernel ID 对应哪个 pipeline。

**GTPin 日志中的 dispatch 序列示例:**
```
D3DDevice::CreateKernel Kernel Name main, SIMD 16, Type CS    Kernel Id 1
D3DDevice::CreateKernel Kernel Name main, SIMD 8, Type CS     Kernel Id 2
D3DDevice::CreateKernel Kernel Name main, SIMD 32, Type CS    Kernel Id 3
...
D3DDevice::KernelBind KernelId 1 in Draw 0    ← 第1个dispatch
D3DDevice::KernelBind KernelId 3 in Draw 1    ← 第2个
D3DDevice::KernelBind KernelId 2 in Draw 2    ← 第3个
D3DDevice::KernelBind KernelId 2 in Draw 3    ← 第4个 (同一shader)
```

**Qwen3 0.6B Q4_K_M prefill 首层 dispatch 序列** (在 Meteor Lake 上实测确认):

| 顺序 | Pipeline 名称 | SIMD | 作用 |
|------|--------------|------|------|
| 1 | `rms_norm_mul_f32` | 16 | RMSNorm |
| 2 | `quantize_q8_1_x4` | 32 | 激活量化 f32→q8_1 |
| 3 | `matmul_q4_k_q8_1_s` | 8 | Q projection (int8 dot) |
| 4 | `matmul_q4_k_q8_1_s` | 8 | K projection (同一shader) |
| 5 | `matmul_f16_f32_f16acc_aligned_s` | 8 | V projection (F16权重) |
| 6 | `rms_norm_mul_rope_f32_f32` | 16 | Q RoPE |
| 7 | `rms_norm_mul_rope_f32_f32` | 16 | K RoPE |
| 8 | `set_rows_f16_i32` | 32 | K 写入 KV cache |
| 9 | `set_rows_f16_i32` | 32 | V 写入 KV cache |
| 10 | `flash_attn_f32_f16_aligned_f32accf16` | 32 | Flash Attention |

> 这个序列在 Meteor Lake 上通过修改 `ggml-vulkan.cpp` 添加 `GGML_VK_LOG_PIPELINES=1` 日志实测得到。不同硬件/驱动上的编译和 dispatch 顺序**可能不同**，此表仅供参考。以方法 A 的 ISA 特征匹配为准。

**注意**: Vulkan 所有 compute shader 入口函数都叫 `main`，因此 GTPin 日志中 Kernel Name 始终是 `main`，无法直接区分。SIMD width 可以作为辅助线索（如 MMQ matmul 通常是 SIMD-8）。

## Meteor Lake 实测结果 (无 XMX, 作为基线)

| 指标 | 值 |
|------|-----|
| GPU | Intel Arc 140V (Meteor Lake-P iGPU, Xe-LPG) |
| XMX | 无 |
| GTPin 版本 | 4.7.1 |
| 总 shader 数 | 18 |
| 含 dp4a 的 shader | 3 个 (256 + 512 + 152 条) |
| 含 dpas 的 shader | 0 |
| matmul_q4_k_q8_1_s 的 ISA | 3512 行, SIMD-8, 256 条 dp4a |

## 附加选项

### 开启详细日志 (获取 kernel 编译信息)

```bash
gtpin.exe --dump_isa --msgon all --msg_file gtpin_log.txt ...
```

日志中包含 kernel 创建顺序、IR hash、SIMD width 等信息。

### Dump 所有调试信息

```bash
gtpin.exe -d ...   # 等价于 --msgon all --dump_cfg --dump_isa --dump_debug_data --dump_bin
```

### 过滤特定 SIMD width

```bash
gtpin.exe --filter I:simd:8 ...   # 只插桩 SIMD-8 的 shader (MMQ matmul 通常是 SIMD-8)
```

## 故障排除

| 问题 | 解决方案 |
|------|---------|
| GTPin 报 arch mismatch | 确保 GTPin 和目标程序都是 64-bit |
| ISA 目录为空 | 确认推理确实执行了 (检查 curl 响应) |
| 程序崩溃 | 尝试不加 `--msgon all`；确认 GPU 驱动是最新版 |
| profile_dir 路径错误 | 使用相对路径，不要用绝对路径 (GTPin 会拼接 CWD) |
| 推理极慢 | 正常现象，GTPin 插桩有 5-10x 开销 |
