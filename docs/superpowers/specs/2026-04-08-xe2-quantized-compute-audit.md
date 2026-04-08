# Spec: Xe2/Xe3 上 GGUF 量化模型推理的计算精度与 XMX 利用现状审计

> **类型**：现状审计报告 spec
> **日期**：2026-04-08
> **目标读者**：Intel 内部工程师（熟悉 XMX/DPAS，关心 ggml 对 Xe2 的利用程度）
> **语言**：中英混合（中文主体，技术术语保持英文）
> **范围**：聚焦 Q4_K_M 量化格式，Vulkan backend，Xe2 (Lunar Lake/Battlemage) 硬件
> **代码基线**：Ollama main branch (ggml Vulkan backend)，2026-04-08 快照

## 1. 报告目标

客观记录 ggml Vulkan 后端在 Intel Xe2 GPU 上运行 Q4_K_M 量化模型时：

- 每个推理算子的**存储精度、计算精度、累加精度**
- 三条计算路径（MMQ / Dequant+F16 Coopmat / Flash Attention Coopmat1）的**精度流水线**
- XMX (DPAS) 矩阵引擎的**实际利用情况**
- Prefill 与 Decode 两个阶段的**路径差异**

报告性质为"是什么"（现状审计），不包含性能优化建议或行动提案。

## 1.1 写作原则

- **严格基于事实**：所有技术断言必须有代码引用（文件路径 + 行号）或实测日志佐证，不得凭空推测
- **主动验证**：实现时应去代码中逐行验证每个断言，或通过网络搜索确认硬件规格等外部事实；不能仅依赖记忆或推断
- **不确定即标注**：对于无法从代码或文档确认的内容，必须显式标注为"⚠️ 待确认"并说明原因，绝不以确定语气描述不确定的事实
- **Mermaid 图表辅助**：关键数据流和路径选择使用 Mermaid 图表可视化（见下方图表要求）

## 2. 报告结构

### 第 0 节：前置概念 — W4A16 与量化计算范式

**目的**：建立读者的概念框架，解释为什么存在多条计算路径。

**内容**：
- GGUF 量化的本质：所有 GGUF 格式都是权重-only 的训练后量化（PTQ），激活始终是 fp16/fp32
- W4A16 的含义：4-bit 权重 + 16-bit 激活，这是 Q4_K_M 的原始语义
- 计算矛盾：硬件没有 int4 × fp16 指令，无法直接做混合精度整数-浮点乘法
- 两种解法：
  - 反量化权重适配浮点（Dequant+F16 路径）→ 保持 W4A16 语义
  - 量化激活适配整数（MMQ 路径）→ 运行时变为 W4A8
- 这就是后文三条路径存在的根本原因

**篇幅**：3-5 段，约 300 字

### 第 1 节：Summary 总表

**目的**：一张表看全貌。读者看完此表即可掌握核心结论。

**表格设计**：

- 纵轴：推理中的每个算子
  - Embedding lookup (GET_ROWS)
  - RMS Norm + RoPE (融合 op)
  - QKV 投影 (MUL_MAT q4_K)
  - Flash Attention (FLASH_ATTN_EXT)
  - Attention Output 投影 (MUL_MAT q4_K)
  - 残差加法 (ADD)
  - FFN gate/up 投影 (MUL_MAT q4_K)
  - SwiGLU 激活 (GLU)
  - FFN down 投影 (MUL_MAT q4_K 或 q6_K，取决于 `use_more_bits()` 层位置启发式)
  - Output Head (MUL_MAT/MUL_MAT_VEC — output.weight 升级为 q6_K；实际可能为 f16 tied embedding)

- 横轴：
  - 权重存储精度
  - 计算路径（MMQ / Coopmat / Scalar）
  - 计算精度（输入端）
  - 累加精度
  - XMX 参与（是/否）
  - Prefill vs Decode 差异

**篇幅**：表格 + 2-3 段阅读指引

### 第 2 节：三条计算路径

**目的**：在机制层面讲清楚"为什么这样算"。

#### 2.1 MMQ 路径（Matrix Multiply Quantized，整数点积）

- 触发条件：
  - 编译时：`GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT` 宏启用（取决于 glslc 版本）
  - 运行时：`integer_dot_product=true` 且激活为 fp32 且维度对齐
  - 未被环境变量禁用：`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT` 未设为 1
- 完整数据流图：
  - 权重侧：q4_K block → 提取 4-bit nibble → 零扩展到 int8 容器（`& 0x0F0F0F0F`）
  - 激活侧：fp32 → `quantize_q8_1` shader → int8 值 + fp16 (scale, bias)
  - 计算：`dotPacked4x8EXT`（int8 × int8 → int32）
  - Scale 还原：int32 × fp16 scale → fp32
  - 累加：fp32（`ACC_TYPE=float`）
- XMX 利用：通过 `VK_KHR_shader_integer_dot_product` → DPAS int8 单元
- 关键代码引用：`mul_mmq.comp`、`mul_mmq_funcs.glsl`

#### 2.2 Dequant+F16 Coopmat 路径（反量化 + fp16 矩阵乘）

- 触发条件：`coopmat_support=true` 且 MMQ 不可用时的 fallback
- 完整数据流图：
  - 权重侧：q4_K block → shader 内完全反量化到 f16（`fma(d, float(nibble), m)` → f16）
  - 激活侧：fp32 → 转 f16
  - 计算：`coopmatMulAdd` f16 × f16
  - 累加：f32（默认）或 f16（f16acc 模式）
  - Coopmat 维度：16×16×16（Xe2 运行时上报）
- XMX 利用：通过 `VK_KHR_cooperative_matrix` → DPAS fp16 单元
- **说明**：Q4_K 在 Xe2 上不走此路径（MMQ 优先级更高），此处作为对照记录
- **Fallback 行为**：当 MMQ 被禁用时，Q4_K 是否走 coopmat 路径还是标量 dequant 路径，需在实现时从代码验证确认
- 关键代码引用：`mul_mm.comp`、`mul_mm_funcs.glsl`

#### 2.3 Flash Attention Coopmat1 路径

- 与前两条路径的根本区别：FA 的输入是 f16 的 Q/K/V（来自上一步 matmul 的输出和 KV cache），不涉及量化权重
- 触发条件：`coopmat1_fa_support=true` 且序列行数 > 16
- 完整数据流图：
  - Q：fp32 buffer → 缩放后存为 f16 到 shared memory
  - K/V：f16 buffer → 直接加载（如量化 KV cache 则 shader 内反量化到 f16）
  - QK^T：`coopmatMulAdd` f16 × f16 → f32/f16 累加
  - Softmax：始终 fp32（max、exp、sum 三步）
  - PV：`coopmatMulAdd` f16 × f16 → f32/f16 累加
  - 输出归一化（÷L）：fp32
- Decode (n=1) 回退：coopmat1 要求最少 16 行，decode 回退到 scalar shader（全 fp32，无 XMX）
- 关键代码引用：`flash_attn_cm1.comp`、`flash_attn_base.glsl`

#### 2.4 如何确认运行时走的哪条路径

三种方法，按易用性排序：

1. **启动日志关键字段**：`int dot: 1` 和 `matrix cores: none/coopmat` 可间接判断路径可用性
2. **环境变量排除法**：设置 `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` 等禁用特定路径，对比 Vulkan Timings 差异
3. **Debug build**：编译时开启 `GGML_VULKAN_DEBUG`，日志会输出每次 dispatch 的 pipeline 名字（如 `matmul_q4_k_q8_1_m` = MMQ，`matmul_q4_k_f32_m` = Dequant+F16）

### 第 3 节：推理走读

**目的**：以 Q4_K_M Qwen3 1.7B 在 Xe2 上的一次推理为实例，按数据流顺序走一遍。

#### 3.1 Prefill（n>1，例如 150 tokens）

按顺序列出每个计算步骤（步骤数和层数需从模型结构和代码实际确认），每步一行注明：
- 算子名
- 走哪条路径（引用 2.x 节编号）
- XMX 是否参与
- 一句话说明

示例格式：
```
3. QKV 投影 — MUL_MAT q4_K → MMQ 路径 (§2.1) → XMX int8 ✅
4. Flash Attention — FLASH_ATTN_EXT → Coopmat1 路径 (§2.3) → XMX fp16 ✅
```

#### 3.2 Decode（n=1）

与 Prefill 的关键差异（约 3-4 点）：
- MUL_MAT → MUL_MAT_VEC
- Flash Attention 回退到 scalar
- 瓶颈从计算变为访存

**篇幅**：每个子节约 15-20 行

### 第 4 节：精度细节

**目的**：面向想深挖 shader 实现的读者，逐层剥开精度。

#### 4.1 存储精度层
- Q4_K_M 混合量化格式：通过 `use_more_bits(i_layer, n_layers)` 启发式（前 1/8 层 + 后 1/8 层 + 中间每 3 层取 1 层）决定哪些层的 `ffn_down` 和 `attn_v` 升级为 Q6_K，其余用 Q4_K；`attn_qkv` 合并张量始终用 Q5_K；`output.weight` 升级为 Q6_K (source: `llama-quant.cpp:185-186, 225-226, 302-303, 358-364, 405`)
- q4_K block 结构：256 values + fp16 (d, dmin) + 6-bit sub-scales + 4-bit values
- KV Cache：默认 f16（`OLLAMA_KV_CACHE_TYPE` 未设置时）

#### 4.2 反量化 / 量化层
- MMQ 路径：权重不反量化（4-bit → int8 容器），激活做运行时量化（fp32 → Q8_1）
- Dequant+F16 路径（对照）：权重完全反量化到 f16，激活转 f16
- Q8_1 量化细节：`amax/127` 缩放、round、pack32、fp16 scale/bias 存储
- 关键 shader：`quantize_q8_1.comp`

#### 4.3 计算精度层
- `dotPacked4x8EXT`：4 对 int8×int8 → int32，循环 8 次/block
- `coopmatMulAdd`：f16×f16 → f32（FA 和 dequant matmul）
- 标量 ops：RMS_NORM（fp32 归约）、GLU（fp32 sigmoid+乘法）、ADD（fp32）
- 关键 shader：`mul_mmq_funcs.glsl`、`flash_attn_cm1.comp`

#### 4.4 累加精度层
- MMQ：int32 点积 → fp32 scale 还原 → fp32 跨 block 累加
- FA coopmat1：f32 累加（默认），f16 累加可选（取决于 `coopmat_acc_f16_support`）
- Softmax：始终 fp32

#### 4.5 输出精度层
- MUL_MAT / MUL_MAT_VEC 输出：fp32（`D_TYPE=float`）
- FA 输出：fp32
- 中间激活层间传递：fp32

## 3. Mermaid 图表要求

报告中应包含以下 Mermaid 图表，按项目惯例（CLAUDE.md）用颜色区分代码归属：
- 🟢 绿色粗边框 (`stroke:#22c55e,stroke-width:3px`) = Ollama Go 代码
- 🟠 橙色粗边框 (`stroke:#f97316,stroke-width:3px`) = llama.cpp C/C++ 代码
- subgraph 用同色虚线边框，每张图顶部加图例

### 必须包含的图表：

1. **第 0 节 — W4A16 分叉图**：从 "GGUF 权重 int4 + 激活 fp32" 出发，分叉到两种解法（反量化权重 vs 量化激活），对应到 Dequant+F16 路径 和 MMQ 路径
2. **第 2 节 — 每条路径的精度流水线图**（至少 3 张）：
   - MMQ 路径：存储(int4) → 零扩展(int8) → dotPacked4x8EXT(int8×int8→int32) → scale还原(fp32) → 累加(fp32)
   - Dequant+F16 路径：存储(int4) → 反量化(f16) → coopmatMulAdd(f16×f16) → 累加(f32)
   - Flash Attention 路径：Q(f16) + K(f16) → coopmatMulAdd → softmax(fp32) → PV(f16×f16→f32)
3. **第 3 节 — Prefill vs Decode 路径选择流程图**：体现 MUL_MAT vs MUL_MAT_VEC 分叉、FA coopmat1 vs scalar 回退

### 可选图表：
4. 第 2.4 节 — 路径判定决策树（启动日志 → 环境变量 → debug build）
5. 第 4 节 — Q4_K block 内存布局示意图

## 4. 已知的不确定项

在报告正文中需标注为"不确定"的点：

1. **Xe3 (Panther Lake)**：代码中只有 `INTEL_XE2` 架构枚举，Xe3 行为预期与 Xe2 一致但无法从代码确认
2. **`coopmat_acc_f16_support` 在 Xe2 上的实际值**：取决于驱动上报的 cooperative matrix properties，代码中无硬编码
3. **Decode MUL_MAT_VEC 的 XMX 映射**：`dotPacked4x8EXT` 在 n=1 场景下，驱动是否映射到 XMX 还是标量 EU，代码层面无法确定
4. **`GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT` 编译宏**：取决于构建环境的 glslc 版本，非所有构建一定启用

## 5. 代码引用索引

报告中引用的关键源文件（相对路径）：

| 文件 | 内容 |
|------|------|
| `ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp` | 路径选择逻辑、设备能力检测、pipeline 创建 |
| `ml/backend/ggml/ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp` | MMQ shader 主逻辑 |
| `ml/backend/ggml/ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq_funcs.glsl` | 各量化类型的整数点积实现 |
| `ml/backend/ggml/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp` | Dequant+F16 coopmat shader |
| `ml/backend/ggml/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_funcs.glsl` | 各量化类型的反量化实现 |
| `ml/backend/ggml/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp` | Flash Attention coopmat1 shader |
| `ml/backend/ggml/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl` | FA 基础逻辑 |
| `ml/backend/ggml/ggml/src/ggml-vulkan/vulkan-shaders/quantize_q8_1.comp` | 激活量化 shader |
| `ml/backend/ggml/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` | Shader 编译与精度配置 |

## 6. 不包含的内容

- 性能优化建议或行动提案（纯现状审计）
- 非 Vulkan 后端（CUDA、Metal、CPU）的分析
- 非 Q4_K_M 格式的详细分析
- Xe2 以外硬件的实测数据
- 模型质量/准确性评估（只关注计算精度，不关注推理质量）

## 7. 输出件

- 一份 markdown 文档，存放于 `docs/daop/` 或 `docs/internals/`
- 包含 Mermaid 精度流水线图（按项目惯例用颜色区分 Ollama Go / llama.cpp C++ 代码归属）
- 预计篇幅：2000-3000 字 + 表格 + 图
