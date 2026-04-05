# TODO

未来要做的工作。

<!-- 格式：- [ ] 描述 (来源: YYYY-MM-DD) -->
<!-- 用户主动指定优先级时加 [HIGH] 标记 -->

## Performance Prediction (perf-predict)

### 近期 Roadmap (2026-04-04)

**Phase 1A: 测量质量提升**
- [ ] [HIGH] 收敛早停: CV < 5% on trimmed samples，统一应用到 measureOp + hwchar (来源: 2026-04-04)
- [ ] [HIGH] 4-dtype MUL_MAT 参考曲线: f32, f16, q4_0, q8_0 各跑一条，提取独立效率常量 (来源: 2026-04-04)
- [ ] K-quant dtype 映射: estimate 时 q4_K→q4_0, q6_K→q8_0 等 (来源: 2026-04-04)
- [ ] 验证优化后 benchmark 总时间 (目标 ~12-14 min) (来源: 2026-04-04)

**Phase 1B: 完整算子覆盖**
- [ ] 实现剩余 element-wise 算子: GELU, SOFTMAX, ADD, MUL, RMS_NORM 等 (~22 个) (来源: 2026-04-04)
- [ ] 每个新算子的测试覆盖 (来源: 2026-04-04)

**Phase 1C: 端到端验证**
- [ ] [HIGH] daop-estimate 端到端测试: 加载真实模型 (如 llama3:8b-q4_0)，生成预测 (来源: 2026-04-04)
- [ ] [HIGH] 预测 vs 实际对比: 对比 estimate 结果与模型实际运行 tokens/sec (来源: 2026-04-04)

**Phase 1C: 性能优化**
- [ ] estimate 路径避免分配权重 buffer: 当前 `split_graph` 依赖 `buffer->usage == WEIGHTS` 判断 backend，导致 estimate 必须分配 ~1-2GB GPU 内存 + 大量 DXGI 查询。需要研究轻量替代方案（dummy buffer / C 端修改），让 estimate 不需要真正分配内存 (来源: 2026-04-04)

**Phase 1D: Accuracy Redesign（18x 偏差修正）**
- [x] GPU Timestamp C API: `ggml_vk_enable_timestamps()` + `ggml_vk_get_op_timings()` (来源: 2026-04-05, 完成: 2026-04-05)
- [x] Go CGO bindings: `ml.Backend.EnableGPUTimestamps()` + `GetOpTimings()` (来源: 2026-04-05, 完成: 2026-04-05)
- [x] benchmark `measureOpGPU()` 替代 wall-clock (来源: 2026-04-05, 完成: 2026-04-05)
- [x] MUL_MAT_VEC 区分: estimate 根据 N≤8 路由 (来源: 2026-04-05, 完成: 2026-04-05)
- [x] Op fusion 模拟: `ApplyFusion()` 3 条核心规则 (来源: 2026-04-05, 完成: 2026-04-05)
- [x] Fused op benchmark entries (来源: 2026-04-05, 完成: 2026-04-05)
- [x] CPU orchestration overhead benchmark (来源: 2026-04-05, 完成: 2026-04-05)
- [x] 共享基础设施 `perf/common.go` (来源: 2026-04-05, 完成: 2026-04-05)
- [ ] 端到端验证: estimate qwen3:1.7b 误差 < 2x vs 实际 75ms/tok (来源: 2026-04-05)
  - 当前状态: 272ms 预测 vs 75ms 实际 = 3.6x 误差（从 18x 降至 3.6x）
  - 主要剩余误差来源: benchmark bw_eff 过低（见 Phase 1F）

**Phase 1E: Direct Backend Execution（benchmark 在 CPU 而非 GPU 运行的修正）**
- [x] `ComputeOnBackend(backendIdx)` CGO 方法 + `transfer_ctx_tensors` C helper (来源: 2026-04-05, 完成: 2026-04-05)
- [x] 重构 `measureOpGPU`: 使用 `ComputeOnBackend` 替代 `ctx.Compute()` (来源: 2026-04-05, 完成: 2026-04-05)
- [x] 重构 `CharacterizeHardware` (peak TOPS/BW): 确保在 GPU 上执行 (来源: 2026-04-05, 完成: 2026-04-05)
- [x] benchmark 控制流统一: work plan 模式, `--ops` 完全控制 fused/overhead (来源: 2026-04-05, 完成: 2026-04-05)
- [x] 清理临时 debug 日志和 hack flags (来源: 2026-04-05, 完成: 2026-04-05)
- [x] `--skip-hwchar` CLI flag 恢复 (来源: 2026-04-05, 完成: 2026-04-05)
- [x] INT8 peak TOPS (q8_0, q4_0) + randomTensor 替代 Zeros (来源: 2026-04-05, 完成: 2026-04-05)
- [x] `tensor_can_set` guard: 修复 estimate 路径 GPU buffer 写入崩溃 (来源: 2026-04-05, 完成: 2026-04-05)
- [x] FLASH_ATTN_EXT: 自定义 CreateInputs (Q=f32, K/V=f16) 适配 Vulkan (来源: 2026-04-05, 完成: 2026-04-05)
- [x] benchPeakTOPS: activation 始终 f32 (修复 q8_0 Vulkan 断言崩溃) (来源: 2026-04-05, 完成: 2026-04-05)
- [x] 去除 per-op OverheadUs: dispatch overhead 是 per-batch 而非 per-op (来源: 2026-04-05, 完成: 2026-04-05)

**Phase 1F: Benchmark 精度提升（3.6x → <2x 误差）**
- [ ] [HIGH] benchmark bw_eff 严重偏低 (q4_0: 0.096, q8_0: 0.095): reference curve 只在 M=K=4096 采样，小 N 带宽利用率被低估 ~10x (来源: 2026-04-05)
- [ ] MUL_MAT_VEC 独立 calibration: N=1~8 走 VEC shader，需独立效率常数 (来源: 2026-04-05)
- [ ] 多 shape 覆盖: 覆盖模型常见的 (M,K) 对，不只是 4096x4096 (来源: 2026-04-05)
- [ ] 小 N 直接测量: 对 N=1,2,4,8 直接 benchmark 延迟，不用 roofline 外推 (来源: 2026-04-05)

**Phase 1G: Estimate 速度优化**
- [ ] [HIGH] SkipWeightAlloc: model.New() 跳过 ggml_backend_alloc_ctx_tensors_from_buft，不分配 GPU 权重 buffer。tensor metadata 仍正常创建（shape/dtype），只是 buffer=NULL (来源: 2026-04-05)
- [ ] [HIGH] Go 层 backend assignment: 替代 split_graph，根据 schedule + tensor name (blk.{i}.*) 在 Go 层标记每个 graph node 的 backend。不需要 weight buffer 归属 (来源: 2026-04-05)
- [ ] 合并两次 model.New() 为一次: discoverModelSchedule 不再需要单独加载模型，schedule 在 Go 层构建 (来源: 2026-04-05)
- [ ] Server API 路径: daop-estimate 作为 server 内 API 时，直接复用已加载模型的 backend，零额外开销 (来源: 2026-04-05)

**Phase 1H: Partial Offload Estimate**
- [ ] partial offload 的 backend assignment: 支持部分层在 GPU、部分在 CPU 的 schedule (来源: 2026-04-05)
- [ ] CPU↔GPU 数据搬运延迟建模: 跨 backend 的 tensor transfer overhead (来源: 2026-04-05)
- [ ] 多 schedule 对比: 输入不同 offload 方案，对比预测性能，支持自动寻优 (来源: 2026-04-05)

### 未来工作

- [ ] [HIGH] Model ID 下载前预估: 只下载 GGUF header（几十KB）而非完整文件，实现真正的"下载前预估"。当前 MVP 在本地无模型时会下载完整 GGUF (来源: 2026-04-02)
- [x] ~~η piecewise 扩展~~ — 已被经验模型方案取代，不再使用 η (来源: 2026-04-02, 关闭: 2026-04-03)
- [ ] C 层直接 benchmark K-quant 类型 (q4_K, q5_K, q6_K)，绕过 Go DType 限制 (来源: 2026-04-04)
- [ ] 实时 GPU/CPU utilization 感知，根据当前负载（如玩游戏）调整预测 (来源: 2026-04-02)
- [ ] 多用户并发推理的性能预测 (来源: 2026-04-02)
- [ ] llamarunner (C++) 架构的性能预测支持 (来源: 2026-04-02)
- [ ] 自动推荐最优量化方案（如 "建议用 Q5_K_M 而不是 Q4_0"）(来源: 2026-04-02)
- [ ] 多模态模型 vision encoder 的性能预测 (来源: 2026-04-02)
- [ ] HTML Viewer: 交互式二维/三维可视化算子 benchmark 数据（log-space 曲线、balance point、采样点分布等）— 已纳入 v2 spec Phase 1 (来源: 2026-04-02)
