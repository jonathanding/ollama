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
