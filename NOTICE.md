# 来源与分发说明

本仓库从原 VTG 项目的任务、奖励、解析器、偏好筛选与冻结适配器实现重构而来。新的包结构、训练组织、动态视频处理、入口和测试可独立运行，不需要访问原工程目录。

ZeRO-3 CPU offload 配置、动态像素预算和单次 rollout GRPO 公式参考 [Invert4TVG](https://github.com/nian1125/Invert4TVG)。模型与训练接口来自 Qwen2.5-VL、Transformers、PEFT、Accelerate、DeepSpeed 和 PyTorch。各依赖的许可分别适用。本仓库未捆绑这些项目的模型权重或训练数据，也未整体复制 vendor trainer。

仓库暂未替作者选择新的开源许可证。公开代码不自动赋予别人重新分发的许可；维护者确认来源与授权后可添加所选 LICENSE。视频、benchmark 数据和模型权重应从各自合法来源获取。

ZeRO-3 生成上下文采用 DeepSpeed/TRL 的权重聚合与临时移除参数 hooks 的方法，并增加异常恢复；相关接口固定在 DeepSpeed 0.18.9 上验证。
