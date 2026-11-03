# Seer PI05 end-effector training

This folder contains an isolated training entry point for the LeRobot v3 dataset at
`/mnt/cfs/usr/jixiang/train_datasets/lerobot_ee`. It uses the stock PI05 model and LeRobot
training loop; the adapter only handles the Seer vector schema, normalization statistics,
and end-effector action representation.

## Dataset mapping

The expected v3 dataset has 120 episodes, 20 FPS, and three cameras (`head`, `left_wrist`,
`right_wrist`). The adapter reads state and action parquet columns only; source files and videos
are never changed.

- `observation.state`: 64D to 30D by keeping the 16 joint/base position values, the two gripper
  values, and both end-effector XYZ plus quaternion converted to a rotation vector.
- `action`: 16D to 14D by converting each end-effector quaternion target to a rotation vector and
  retaining both gripper targets.
- The two end-effector targets are trained as SE(3) deltas in the current end-effector frame. The
  saved postprocessor converts predictions back to absolute XYZ and rotation-vector targets.
- Action normalization statistics are computed on the selected training episodes and exact
  50-step chunks. Chunks repeat the final action within their own episode, matching LeRobot's
  episode-boundary padding.

## Server setup and launch

Run from a clean checkout of `feat/pi-train`. The server paths below match the current server
environment. The example environment file selects `/root/venvs/lerobot-pi05/bin/python` and
`/root/venvs/lerobot-pi05/bin/torchrun`; override them through environment variables or `ENV_FILE`
when needed.

```bash
cd /mnt/cfs/usr/jixiang/vla_learning/lerobot_pi05_train
ENV_FILE=pi05_seer_robot_train/config/server.env.example bash pi05_seer_robot_train/train.sh full 10000 8 0
ENV_FILE=pi05_seer_robot_train/config/server.env.example bash pi05_seer_robot_train/train_rtc.sh full 10000 8 0
```

Both scripts accept `[check|smoke|full] [steps] [batch-per-GPU] [save-frequency]`. `check` validates
dataset metadata, local model/tokenizer assets, GPU count, and bfloat16 support. `smoke` runs two
optimizer steps. `full` defaults to 10,000 steps, batch size 8 per GPU, and saves the final checkpoint
only. A periodic checkpoint interval can be passed as the fourth argument or through `SAVE_FREQ`.
Full PI05 checkpoints require substantial disk space.

`train.sh` disables trained RTC prefixes. `train_rtc.sh` uses the previous branch's RTC training
setting: `RTC_TRAIN_MAX_DELAY=10`, with `chunk_size=50` and `n_action_steps=10` by default. At 20 FPS,
10 steps represent up to 0.5 seconds of action-prefix conditioning. Set `RTC_TRAIN_MAX_DELAY` to adjust
the RTC prefix range.

Both scripts fully fine-tune the vision encoder and action expert (`freeze_vision_encoder=false`,
`train_expert_only=false`, `use_peft=false`). Training settings can be overridden with environment
variables, including `BATCH_SIZE`, `NUM_GPUS`, `GRAD_ACCUM_STEPS`, `STEPS`, `SAVE_FREQ`, `CHUNK_SIZE`,
`N_ACTION_STEPS`, `RTC_TRAIN_MAX_DELAY`, `OPTIMIZER_LR`, `SCHEDULER_WARMUP_STEPS`,
`SCHEDULER_DECAY_STEPS`, `SCHEDULER_DECAY_LR`, `LOG_FREQ`, `EVAL_STEPS`, `EVAL_SPLIT`,
`NUM_WORKERS`, and `IMAGE_AUGMENTATION`. The four positional arguments override mode, steps, batch per
GPU, and save frequency. `OUTPUT_BASE` and `OUTPUT_DIR` control checkpoint locations.

Visual augmentation is photometric only: brightness and contrast vary from 0.9 to 1.1, sharpness
from 0.8 to 1.2, with up to two transforms per image. It does not alter camera geometry.

## Resume

Use the checkpoint's `train_config.json` with the standard LeRobot resume flow, for example:

```bash
torchrun --standalone --nproc-per-node=1 pi05_seer_robot_train/train.py \
  --resume=true \
  --config_path=/path/to/checkpoints/last/pretrained_model/train_config.json \
  --output_dir=/path/to/new-resume-run
```

The launcher patch restores the saved custom processor pipelines on resume, so the normalization
statistics bundled with the checkpoint remain authoritative.

For inference through the stock LeRobot CLI, load the custom processor registry before opening the
checkpoint by adding:

```bash
--policy.discover_packages_path=pi05_seer_robot_train.seer_pi05
```

## Layout

- `seer_pi05/geometry.py`: quaternion, rotation-vector, and relative-pose conversions.
- `seer_pi05/data.py`: v3 dataset wrapper and train-split statistics.
- `seer_pi05/processors.py`: serializable PI05 preprocessing and postprocessing steps.
- `train.py`: thin hook into the stock `lerobot-train` loop.
- `preflight.py`, `run.sh`, `config/`: server launch and validation.
- `tests/`: vector conversion and episode-boundary checks.
