# Invert4TVG 风格训练入口

参考：[官方 grpo_3b.sh](https://github.com/nian1125/Invert4TVG/blob/main/qwen-vl-finetune/scripts/grpo_3b.sh) 与 [train_grpo.py](https://github.com/nian1125/Invert4TVG/blob/main/qwen-vl-finetune/qwenvl/train/train_grpo.py)，核对日期 2026-10-12。

新入口沿用其 `torchrun + Python 入口 + 显式参数` 的使用方式，训练目标仍为本项目的 E5 与偏好学习协议。DeepSpeed 后端采用官方 ZeRO-3 + CPU 参数/优化器 offload 方案；训练目标、奖励与候选组协议保留本项目定义。

## 文件和优先级

- `scripts/train.sh`：直接修改顶部路径、GPU 和超参数的八卡入口。
- `scripts/zero3_offload.json`：与官方方案对齐的 ZeRO-3 配置。
- `main.py`：普通 Python 入口，也可自行编写 `torchrun ... main.py ...`。
- `scripts/train_from_config.sh`：原来的 JSON 配置入口。
- `src/vtg/cli/arguments.py`：参数定义与映射，未知参数直接报错。
- `src/vtg/training/options.py`：配置到 HF TrainingArguments 的实际映射。

直接调用 `main.py` 时，优先级是实验默认值 < `--paths` 文件 < 显式命令行参数。`train.sh` 会先提供一组显式参数，再附加用户参数，因此命令末尾的同名参数覆盖脚本默认值。希望完全使用 JSON 时调用 `train_from_config.sh`。

## 可以直接修改的参数

| 参数 | 作用 |
|---|---|
| `--deepspeed` | DeepSpeed ZeRO-3 JSON 路径；`none` 显式切换 DDP |
| `--branch` | E5、N0_E5_cont、N1_IoU_pair、N2_Event_pair、N3_Swap、N4_Event_swap、A_matched、A_verb_gate、A_sem_add |
| `--model_name_or_path` | 本地 Qwen2.5-VL 基础模型目录 |
| `--parent_checkpoint` | 继续训练时的 E5 adapter，冻结参考与 actor 均从这里初始化 |
| `--train_data_path` / `--video_folder` | 原始标注和训练视频根目录 |
| `--dataset_name` | `timer1` / `timer1_train_2k5` 列表格式，或 `charades` 视频字典格式 |
| `--artifact_root` / `--spacy_model` | manifest 输出与语义解析资源 |
| `--prepare_data auto` | 加锁准备、检查来源并复用相同训练 manifest |
| `--prepare_data never` | 使用已有 manifest，不自动覆盖 |
| `--output_dir` / `--run_name` / `--logging_dir` | 模型输出、运行名称和 TensorBoard 日志 |
| `--learning_rate` / `--weight_decay` | 优化器参数 |
| `--warmup_ratio` / `--lr_scheduler_type` | warmup 比例与学习率日程 |
| `--num_train_epochs` | 正整数，按精确 TVG epoch 计算完整任务日程 |
| `--max_steps` | 固定更新数，与显式 `--num_train_epochs` 互斥 |
| `--max_prompt_length` / `--max_completion_length` | 输入及输出 token 上限，超长视频输入不截断 |
| `--num_generations` | 同一问题的全局候选数 G |
| `--per_device_train_batch_size` | 每卡生成的候选数 |
| `--gradient_accumulation_steps` | 每次更新的不同问题数，同时更新内部日程配置 |
| `--seed` | 模型及采样种子，各 rank 使用不同随机流 |
| `--data_seed` | HF 数据种子，并默认设置任务日程种子 |
| `--schedule_seed` | 单独覆盖任务日程种子 |
| `--temperature` / `--top_p` | rollout 随机采样 |
| `--beta` | GRPO 的 KL 系数 |
| `--bf16` / `--torch_dtype` | 支持 BF16 或 FP32，矛盾的参数会报错 |
| `--gradient_checkpointing true/false` | 真实控制梯度检查点 |
| `--attn_implementation` | sdpa / flash_attention_2 / eager |
| `--logging_steps` / `--report_to` | 日志频率，支持 tensorboard / none |
| `--save_strategy` | steps / epoch / no |
| `--save_steps` / `--save_total_limit` | 保存间隔与保留数量 |
| `--save_only_model true/false` | 是否省略优化器、调度器和 RNG 状态 |
| `--resume_from_checkpoint` | 恢复本次实验目录内的完整 checkpoint |
| `--max_frames` / `--min_frames` / `--fps` | 帧数和采样率 |
| `--max_pixels` / `--min_pixels` / `--total_pixels` | 动态像素分配参数 |
| `--visual_token_cap` | 实际视觉 token 的硬上限 |
| `--lambda_pair` / `--beta_dpo` / `--min_iou` / `--min_gap` | 偏好损失和筛选参数 |
| `--reward_funcs iou format` | TVG 的 IoU、格式分，可单独启用其中之一；辅助任务奖励保持原定义 |

标注格式选择真实影响读取器。Charades 接受官方视频字典中的 `duration`、`timestamps`、`sentences`；额外的 `Missing`、`verb` 不替代本项目的语义解析。Time-R1 接受 `video`、`timestamp`、`sentence`、`duration` 等字段。

## 八卡设置

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=12371 \
  main.py \
  --deepspeed scripts/zero3_offload.json \
  --branch N1_IoU_pair \
  --model_name_or_path /data/models/Qwen2.5-VL-7B-Instruct \
  --parent_checkpoint /data/checkpoints/E5/checkpoint-3819 \
  --train_data_path /data/timer1/annotations/train_2k5.json \
  --video_folder /data/timer1/videos \
  --dataset_name timer1 --artifact_root artifacts/timer1 \
  --prepare_data auto --output_dir outputs/n1_768 \
  --num_generations 8 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 --num_train_epochs 1 \
  --bf16 --torch_dtype bfloat16 --gradient_checkpointing true \
  --max_frames 768 --max_completion_length 256 \
  --save_strategy epoch --save_only_model false
```

约束为 `G = world_size × per_device_train_batch_size`。不同 rank 共同组成同一候选组；不能把其他训练器中“单卡 batch=1、G8”的内部重复采样约定直接套过来。GPU 进程数由 torchrun 的 WORLD_SIZE 获取，显式 `--world_size` 与之冲突时立即报错。

默认使用 ZeRO-3：参数、梯度和优化器状态在各卡分片，参数和优化器状态卸载到 CPU，优化器由 DeepSpeed 创建 CPUAdam。冻结参考通过第二个 LoRA adapter 共享基础权重。生成和冻结判别在所有 rank 同时进入的权重聚合上下文中执行，退出后恢复 ZeRO hooks，再进行 actor forward/backward。

配置对应 [官方 zero3_offload.json](https://github.com/nian1125/Invert4TVG/blob/main/qwen-vl-finetune/scripts/zero3_offload.json)。唯一字段名兼容调整：官方的 `gather_16bit_weights_on_model_save` 写为等价的 `stage3_gather_16bit_weights_on_model_save`，以便 Accelerate 1.7 正确识别；命令行读取官方原文件时也会自动转换。`auto` 字段由 HF 根据 batch、累积、精度和学习率填充。

生成阶段临时聚合全模型，显存不等于基础模型权重简单除以八；CPU offload 也需要充足的主机内存及 CPU/GPU 传输带宽。`--deepspeed none` 保留 DDP 对照。JSON 配置中的 DeepSpeed 文件路径相对该 JSON 文件，命令行路径相对启动目录；解析后的配置内容进入恢复身份哈希，修改文件会阻止混用旧训练状态。

## 尚未支持的原脚本开关

`--is_early_stopping true`、`--slide_window true`、`--is_curriculum_learning true`、训练时 `--eval_data_path` 会报错。它们为 false 时不改变协议。`--fix_vit` 必须为 true。提示词当前只支持 `--prompt_type event_core_v1`。Time-R1 的 `iou_v2` 不是本项目 `iou` 的别名，不会静默替换。

早停需要额外建立独立验证集和指标回调；当前遵循固定训练日程和训练后独立评估。`--stop_after` 仅用于主动暂停，不等于基于验证指标的 early stopping。
