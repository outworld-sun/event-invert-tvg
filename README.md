# VTG Training

独立的视频时间定位训练仓库：Qwen2.5-VL-7B、语言侧 LoRA、GRPO、多任务 E5，以及 N0–N4 / A_matched / A_verb_gate / A_sem_add 偏好学习分支。

默认配置面向单机 **8 张 A100**：最多 **768 帧**、2 FPS、按帧数动态分配像素。数据准备、训练、偏好覆盖率检查和评估都有进度条。Python 使用正常换行和 Black 格式，不使用分号拼接语句。

这是新训练协议的独立实现。旧实验是 128 帧、G4；新配置是动态 768 帧、**G8（每卡 1 条候选）**。两者结果需要分别报告。默认使用 **DeepSpeed ZeRO-3 + 参数/优化器 CPU offload**，配置与 Invert4TVG 官方方案一致。生成与冻结判别阶段会临时聚合模型权重，因此仍需要为每卡的生成峰值预留显存。A100 的 40GB / 80GB 容量尚未在目标服务器上实测，应先运行两步 smoke test。

## 1. 安装

要求 Linux、Python 3.10/3.11/3.12、支持所选 CUDA wheel 的 NVIDIA 驱动，以及已下载的 Qwen2.5-VL-7B-Instruct。

```bash
bash scripts/install.sh
source .venv/bin/activate
```

安装脚本默认 PyTorch 2.6.0 + CUDA 12.4、DeepSpeed 0.18.9。CPU offload 使用 DeepSpeedCPUAdam，服务器需要可用的 C++ 编译器和 Ninja；首次启动会编译扩展。默认注意力实现是 `sdpa`，无需编译 FlashAttention。服务器已有可用环境时也可以直接 `pip install -e '.[dev]'`。如需 FlashAttention，安装时设置 `INSTALL_FLASH_ATTENTION=1`，并在本地配置中设置 `training.attn_implementation` 为 `flash_attention_2`。

<<<<<<< HEAD
## 2. 八卡脚本
=======
## 2. 八卡训练脚本
>>>>>>> d87746dd879fe061cd6b52d8ad222ba7bfce9f62

主入口是 [`scripts/train.sh`](scripts/train.sh)。参考官方的“脚本变量 + torchrun + 显式训练参数”结构，现在直接修改脚本顶部即可，无需先编辑 JSON：

```bash
GPU_LIST="0,1,2,3,4,5,6,7"
LOAD_MODEL_PATH="/data/models/Qwen2.5-VL-7B-Instruct"
DATA_PATH="/data/timer1/annotations/train_2k5.json"
VIDEO_FOLDER="/data/timer1/videos"
BRANCH="E5"
NUM_TRAIN_EPOCHS=5
LEARNING_RATE=1e-5
RUN_NAME="e5_768_g8"
```

路径改好后运行：

```bash
bash scripts/train.sh --print_config   # 查看最终参数，不读取数据、不启动 GPU
bash scripts/train.sh --prepare_only   # 可选：单独准备数据，带进度条
bash scripts/train.sh --stop_after 2   # 八卡执行两步，保留完整学习率日程
```

首次正式启动默认自动准备训练 manifest；八个进程通过文件锁保证只准备一次。注释文件或配置改变后自动重新准备。已有手工生成的 manifest 可设置 `PREPARE_DATA=never`。

不修改脚本也可通过环境变量和命令行覆盖。命令行参数优先级最高：

```bash
BRANCH=N1_IoU_pair \
PARENT_CHECKPOINT=/data/checkpoints/E5/checkpoint-3819 \
RUN_NAME=n1_768_g8 \
bash scripts/train.sh --learning_rate 5e-6 --num_train_epochs 1 --save_strategy epoch
```

恢复时重复相同配置，并加 `--resume_from_checkpoint outputs/n1_768_g8/checkpoint-2`。`LOAD_MODEL_PATH` 始终指向基础模型，`PARENT_CHECKPOINT` 指向 E5 LoRA，两者不要混用。

新脚本显式使用 `torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12371 main.py ...`。端口可通过 `MASTER_PORT` 改动。默认八卡各生成一条候选，组成 G8。脚本显式传入 `--deepspeed scripts/zero3_offload.json`，由 HF/Accelerate 创建真正的 DeepSpeed 引擎。参数、梯度和优化器状态分片，参数与优化器状态卸载到 CPU。需要 DDP 对照时使用 `--deepspeed none`。

`logging_steps`、`logging_dir`、`run_name`、`gradient_checkpointing`、`save_strategy`、`save_only_model` 均已接入训练。`save_strategy=epoch` 根据真实任务日程保存，在跨越数据 epoch 后的第一个完整 optimizer update 落盘；不会把 HF 的合成数据集 epoch 误当成真实训练 epoch。`save_only_model=true` 不保存可用于优化器恢复的状态。默认完整 checkpoint 同时保存独立可加载的 actor LoRA、`global_stepN/` 下的 DeepSpeed 分片、调度器及各卡 RNG；恢复必须保留整个 checkpoint 目录。旧 DDP checkpoint 不能当作 ZeRO-3 完整状态继续，但其中的 LoRA 可作为新实验的父模型。

完整参数说明见 [训练脚本参数](docs/training_scripts.md)。原来的 JSON 启动方式保留在 `scripts/train_from_config.sh`。

## 3. 可选：使用 JSON 配置资源路径

```bash
cp configs/paths.example.json configs/paths.local.json
# 修改 paths.local.json 中的模型、视频、标注、artifact、E5 checkpoint 路径。
```

`*.local.json` 已被 Git 忽略。资源路径支持 `${ENV_VAR}` 和 `~`；相对资源路径相对于执行目录，启动脚本会先切换到仓库根目录。配置的 `extends` 相对于当前配置文件。

训练和评估使用的资源：

| 资源 | 配置位置 |
|---|---|
| 本地基础模型目录 | `paths.model_path` |
| Time-R1 训练视频和标注 | `paths.video_root`、`paths.train_annotations` |
| spaCy 包名或模型目录 | `paths.spacy_model` |
| 预处理后的 JSONL manifests | `paths.artifact_root` |
| 冻结 E5 LoRA checkpoint | `continuation.parent_checkpoint` |
| Charades / TVGBench 标注、视频、manifest | `benchmarks` |

40GB 卡可将 `configs/a100_40gb.overrides.example.json` 中的 `training` 项合并到 `paths.local.json`；它在每步后释放缓存，像素预算保持相同。80GB 示例保留缓存。两者均使用每卡一条候选，不能仅凭卡数保证任何输入都不 OOM。

## 4. 准备数据（JSON 入口）

```bash
bash scripts/prepare.sh --config configs/e5_8xa100.json --paths configs/paths.local.json
```

也可迁移旧项目的 `fit.jsonl`、`aux-fit.jsonl`、`charades.jsonl`、`tvgbench.jsonl`，然后更新路径。训练 manifest 中的 `split` 必须是 `fit`；辅助数据必须来自同一训练集。视频帧会按新配置重新读取，旧的判别缓存不复用。

```bash
vtg-doctor --config configs/n1_8xa100.json --paths configs/paths.local.json
vtg-train --config configs/n1_8xa100.json --paths configs/paths.local.json --dry-run
```

`--dry-run` 检查数据、父 checkpoint 和日程，不加载模型，也不代表显存测试通过。

## 5. 训练（JSON 入口）

先检查完整代码路径能否在新机器运行。`--stop-after` 只是暂停，不改变学习率和完整训练日程：

```bash
bash scripts/train_from_config.sh --config configs/n1_8xa100.json --paths configs/paths.local.json \
  --output outputs/n1_768_seed3407 --stop-after 2
```

确认日志、checkpoint 和显存后，从第 2 步继续：

```bash
bash scripts/train_from_config.sh --config configs/n1_8xa100.json --paths configs/paths.local.json \
  --output outputs/n1_768_seed3407 --resume outputs/n1_768_seed3407/checkpoint-2
```

直接从原始模型重新训练 E5：

```bash
bash scripts/train_from_config.sh --config configs/e5_8xa100.json --paths configs/paths.local.json \
  --output outputs/e5_768_seed3407
```

已有旧 E5 checkpoint 时，N1/N2 可以直接载入其 LoRA 权重；优化器与学习率日程重新开始。若先完成新的 768 帧 E5，再将 `parent_checkpoint` 指向新的 E5 checkpoint，所有对照分支使用同一个父模型。

| 配置 | 方法 |
|---|---|
| `e5_8xa100.json` | TVG + SCC/SER/ED 联合奖励，基础模型初始化 |
| `n0_8xa100.json` | E5 继续训练，不加偏好损失 |
| `n1_8xa100.json` | GT 语义门控 + IoU 偏好对 |
| `n2_8xa100.json` | GT 与候选语义门控 + IoU 偏好对 |
| `n3_8xa100.json` | 角色交换惩罚 |
| `n4_8xa100.json` | N2 + 角色交换惩罚 |
| `a_matched_8xa100.json` | 按几何分桶匹配有效偏好数量 |
| `a_verb_gate_8xa100.json` | 用动作门控替代完整事件门控 |
| `a_sem_add_8xa100.json` | 语义判别直接加入奖励 |

默认 E5 是 5 个精确 TVG epoch，继续训练是 1 个精确 TVG epoch。2444 条训练问题分别对应 3819 / 764 个更新；数据量改变时按实际数量重新计算。每个更新包含 4 个不同问题、每个问题 G8 候选。若要固定更新数，在本地配置中设置 `training.num_train_epochs: null` 和 `training.max_optimizer_steps`。

进度条来自 HF Trainer，显示更新数和耗时。终端日志及 TensorBoard 包含 loss、reward、KL、pair_rate、实际帧数和 visual_tokens。`pair_rate` 是最近一个微步是否获得有效对；全程比例应统计 `pair_diagnostics.jsonl`。完整训练产物说明见 [架构文档](docs/architecture.md)。

```bash
tensorboard --logdir outputs
```

## 6. 偏好覆盖率与 matched 对照

用同一父 checkpoint、同一随机种子，先在固定训练问题上检查偏好对产率：

```bash
bash scripts/train_from_config.sh --config configs/n1_8xa100.json --paths configs/paths.local.json \
  --output outputs/coverage_n1 --coverage-size 128
bash scripts/train_from_config.sh --config configs/n2_8xa100.json --paths configs/paths.local.json \
  --output outputs/coverage_n2 --coverage-size 128
vtg-calibrate --ordinary outputs/coverage_n1/pair_diagnostics.jsonl \
  --event outputs/coverage_n2/pair_diagnostics.jsonl \
  --target-questions 2444 --output artifacts/matched_calibration.json
```

运行 `A_matched` 前，将 `continuation.matched_calibration` 设置为校准文件。覆盖率阶段只做生成、判别和统计，不更新权重。覆盖率是诊断结果，不被用作测试集上的方法选择条件。

## 7. 评估

八卡分别处理互不重复的问题，每卡显示自己的进度条；最后核对完整样本集合，再计算 `R1_03`、`R1_05`、`R1_07`、`mIoU` 和非法区间比例。

```bash
bash scripts/evaluate.sh --config configs/n1_8xa100.json --paths configs/paths.local.json \
  --checkpoint outputs/n1_768_seed3407/model --split charades --output outputs/eval_n1_charades
bash scripts/evaluate.sh --config configs/n1_8xa100.json --paths configs/paths.local.json \
  --checkpoint outputs/n1_768_seed3407/model --split tvgbench --output outputs/eval_n1_tvgbench
```

中断后原命令重跑即可从已有预测继续。权重、manifest、视频配置或实现发生变化时会拒绝混用旧输出。推理失败的输入保留在分母中并计零分；OOM 等运行错误会中止运行，不会伪装成完整评估结果。

单卡评估使用 `NPROC_PER_NODE=1 bash scripts/evaluate.sh ...`，不需要改训练配置。基础模型评估显式使用 `--base` 替代 `--checkpoint`。调试时可加 `--limit 16`，但必须使用独立输出目录，结果会标记为 `partial_benchmark`。

<<<<<<< HEAD
=======
## 8. 验证
>>>>>>> d87746dd879fe061cd6b52d8ad222ba7bfce9f62

## 数据集和评估集

<<<<<<< HEAD
### TimeLens
https://huggingface.co/datasets/TencentARC/TimeLens-100K
https://huggingface.co/datasets/TencentARC/TimeLens-Bench
### Time-R1
https://huggingface.co/datasets/Boshenxx/TimeR1-Dataset
=======
CPU 测试包含动态预算、768 帧真实处理器、PyAV 视频解码、严格偏好筛选、八进程 CPU DDP 梯度、微型 Qwen 反向传播和 checkpoint 恢复。另有独立的真实 GPU ZeRO-3 测试，使用临时随机微型模型，不下载权重、不读取训练数据：

```bash
GPU_LIST=0,1 NPROC_PER_NODE=2 bash scripts/test_zero3.sh
# 目标八卡服务器可验证完整进程布局：
GPU_LIST=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 bash scripts/test_zero3.sh
```

该脚本检查实际 CPUAdam、分片参数、生成、冻结判别、梯度累积、保存、跨进程重启恢复，以及恢复/连续训练的权重逐元素一致性。测试奖励与偏好对为可控 fixture，不作为 IoU 结果。真实训练输出 `backend.json` 可核查实际引擎和配置。验证范围与结果见 [VALIDATION.md](docs/VALIDATION.md)。

>>>>>>> d87746dd879fe061cd6b52d8ad222ba7bfce9f62
