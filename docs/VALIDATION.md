# 验证记录

更新日期：2026-10-13。

## 已执行

- 完整 CPU 测试：29 项。
- 显式命令行参数、JSON 与路径覆盖优先级、含空格的 Shell 路径、错误参数拒绝。
- 八进程 `torch.distributed.run main.py --print_config`：参数解析成功，仅 rank 0 打印配置。
- Time-R1 / Charades 标注读取，以及自动数据准备的缓存命中和变更失效。
- 微型 Qwen 的真实 Trainer：按数据 epoch 边界保存 checkpoint；`save_only_model=true` 不生成优化器状态。
- 八进程真实 Gloo/DDP：偏好损失的梯度平均与四微步累积，与单进程直接计算一致。
- Qwen2.5-VL 真实处理器：768 × 84 × 140 输入得到 5760 个视觉 token，没有发生第二次缩放。
- PyAV 实际生成视频、读取 PTS、区间采样、缩放和处理器衔接。
- 无需下载权重的微型 Qwen2.5-VL：自回归生成、参考适配器切换、梯度检查点和 LoRA backward。
- HF Trainer：两步训练中间保存后恢复，日程依次为 g0/g1/g2/g3；恢复后的 actor 权重与不中断训练逐元素比较。
- 动态像素预算、G8 优势标准化、完整 epoch、N1/N2 门控、评估分母与分片完整性。
- 源码 token 扫描：无 Python 分号拼接语句。
- Black 格式化检查、Python 编译检查、Shell 语法检查、命令行帮助检查。
- 独立 wheel 构建。
- 使用旧实验的真实训练 manifest 与 E5 checkpoint 做只读 dry-run：2444 条 TVG、387 条辅助问题、3056 个问题组、764 个更新、8 个 rank、G8。

测试使用隔离的工程临时副本和已有 Python 3.10 环境；主要版本为 PyTorch 2.6.0+cu124、Transformers 4.51.3、Accelerate 1.7.0、PEFT 0.15.2、DeepSpeed 0.18.9。测试不启动真实 7B 训练，不改变旧实验权重。

## ZeRO-3 GPU 集成验证

2026-10-13 在两张 RTX A6000 上运行 `scripts/test_zero3.sh`，完整通过：

- 真正的 DeepSpeed ZeRO Stage 3、参数/优化器 CPU offload、DeepSpeedCPUAdam。
- 微型 Qwen 的自回归生成、参考 log-prob、不同 rank 判别次数与缓存命中、GRPO + 非零偏好损失。
- reentrant 梯度检查点、两微步累积和实际 LoRA 参数更新。
- N1 暂停后重新启动进程恢复：最终 actor 与连续训练逐元素完全一致。
- E5 基础模型参考路径、`save_only_model` 和不经过 Trainer.train 的覆盖率路径。
- 冻结 C0 保持不变，聚合后的跨 rank actor 最大误差为 0。
- 完整 checkpoint 的每个 rank 分片和 RNG 检查；调度器由 DeepSpeed 内嵌保存/恢复。BF16 与 FP32 文件名前缀分别校验。

原始状态与实际后端配置见 [zero3_validation.json](zero3_validation.json)。GPU 测试使用随机微型模型、可控奖励/偏好对和替代视频加载器，用于验证工程执行链路，不代表真实训练指标。视频解码与 768 帧处理另由 CPU 集成测试验证。

ZeRO-3 使用 DeepSpeed 0.18.9 与上述 HF/Accelerate 版本；本次验证不代表其他版本组合已通过。独立 wheel 已重新构建，源码包包含 `scripts/zero3_offload.json` 和 GPU 验证脚本。

## 尚未验证

- 目标八卡 A100 的 NCCL 拓扑、驱动及实际训练峰值显存。
- 真实 7B 模型在新 768 帧/G8 协议下的完整收敛和 IoU 收益。
- A100 40GB / 80GB 上所有长宽比、最长视频与最长生成组合的吞吐。

在目标服务器先执行 README 中的 doctor、dry-run 和 `--stop-after 2`，确认后使用同一配置恢复继续训练。此前 128 帧/G4 的成绩不代表新协议的评测结果。
