# PI05 脚本索引

以下命令在服务器的 `pi05_seer_robot_train/` 目录执行。脚本保留原来的文件名，按用途分目录；
以前的 `bash train_rtc.sh` 等命令需要改成 `bash scripts/train/train_rtc.sh`。
Python 入口、`config/`、`checkpoints/` 和 `inference_results/` 仍在 `pi05_seer_robot_train/` 下。

## 训练：`train/`

| 脚本 | 用途 |
|---|---|
| [train.sh](train/train.sh) | 普通全量微调，不使用 RTC 训练前缀 |
| [train_rtc.sh](train/train_rtc.sh) | 全量微调，使用 RTC 训练前缀 |
| [train_multitask_full.sh](train/train_multitask_full.sh) | 从基础模型开始多任务全量微调，启用 RTC |
| [train_multitask_full_v1.sh](train/train_multitask_full_v1.sh) | 使用合并后的 `lerobot_multitask_dataset_v1` 从基础模型开始全量微调，启用 RTC |
| [train_multitask_from_rtc.sh](train/train_multitask_from_rtc.sh) | 从已有 RTC checkpoint 的模型权重开始全量微调，重新初始化优化器和学习率计划 |
| [train_right_flap_only.sh](train/train_right_flap_only.sh) | 只选择 `Fold the right flap of the box.` 对应 episode，全量微调并启用 RTC |
| [run.sh](train/run.sh) | 共享训练启动器：环境检查、训练参数、torchrun 启动 |

训练入口支持 `[check|smoke|full] [steps] [batch-per-GPU] [save-frequency]`。
`check` 只做预检查；`smoke` 默认训练两步；`full` 正式训练。
共享启动器的 full 默认 10,000 步、每卡 batch 64；right-flap 脚本默认按选中数据量计算步数。

先在 `config/server.env` 设置 `NUM_GPUS`、`LOG_FREQ`、MLflow 等参数，然后执行，例如：

```bash
cd /mnt/cfs/usr/jixiang/vla_learning/lerobot/pi05_seer_robot_train

# 合并数据集：预检查
ENV_FILE=config/server.env bash scripts/train/train_multitask_full_v1.sh check

# 合并数据集：10,000 步，每卡 batch 64，每 2,000 步保存
ENV_FILE=config/server.env bash scripts/train/train_multitask_full_v1.sh full 10000 64 2000

# 原始训练集：普通 / RTC 全量微调
ENV_FILE=config/server.env bash scripts/train/train.sh full 10000 64 2000
ENV_FILE=config/server.env bash scripts/train/train_rtc.sh full 10000 64 2000

# 右侧折盖单任务
ENV_FILE=config/server.env bash scripts/train/train_right_flap_only.sh full
```

环境文件中的赋值会覆盖同名外部环境变量；步数、batch、保存间隔以命令行位置参数优先。
多任务脚本用 `MULTITASK_DATASET_ROOT` / `MULTITASK_DATASET_REPO_ID` 覆盖数据集路径和名称，
用 `MULTITASK_OUTPUT_BASE` 覆盖输出目录。
`config/.mlflow.env` 由训练脚本自动加载，和 `config/server.env` 一样被 Git 忽略。

## 真机推理服务：`inference/`

| 脚本 | 用途 |
|---|---|
| [run_pi05_inference_server.sh](inference/run_pi05_inference_server.sh) | 通用服务入口，默认使用训练过 RTC 的 checkpoint |
| [run_pi05_multitask_inference_server.sh](inference/run_pi05_multitask_inference_server.sh) | 多任务 checkpoint 服务 |
| [run_pi05_inference_server_guided.sh](inference/run_pi05_inference_server_guided.sh) | 使用未训练 RTC 的 checkpoint，推理时启用 guided RTC |
| [run_pi05_inference_server_no_rtc.sh](inference/run_pi05_inference_server_no_rtc.sh) | 使用未训练 RTC 的 checkpoint，关闭 RTC |

```bash
# 使用脚本默认 checkpoint
bash scripts/inference/run_pi05_multitask_inference_server.sh

# 使用指定 checkpoint（替换为实际路径）
CHECKPOINT=/path/to/checkpoints/010000/pretrained_model \
  bash scripts/inference/run_pi05_multitask_inference_server.sh
```

可通过 `CHECKPOINT`、`PORT`、`EXECUTION_HORIZON` 等变量调整。服务只接收推理请求，不读取训练数据集。

## 离线验证：`eval/`

[infer_offline.sh](eval/infer_offline.sh) 对指定数据集、episode 和 checkpoint 做离线推理及轨迹对比。
在 `pi05_seer_robot_train/` 下执行，例如（替换 checkpoint 和数据集路径）：

```bash
DATASET_ROOT=/path/to/dataset \
  bash scripts/eval/infer_offline.sh /path/to/training-run \
  --episodes 12 --max-frames-per-episode 0 \
  --output-dir inference_results/episode-12
```

省略 `--output-dir` 时，结果仍按 RTC 类型保存在包根目录的 `inference_results/` 下。
