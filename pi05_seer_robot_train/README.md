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
environment; override them through environment variables or `ENV_FILE` when needed.

```bash
cd /mnt/cfs/usr/jixiang/vla_learning/lerobot_pi05_train
ENV_FILE=pi05_seer_robot_train/config/server.env.example bash pi05_seer_robot_train/run.sh check
ENV_FILE=pi05_seer_robot_train/config/server.env.example bash pi05_seer_robot_train/run.sh smoke
ENV_FILE=pi05_seer_robot_train/config/server.env.example bash pi05_seer_robot_train/run.sh full
```

`check` validates dataset metadata, local model/tokenizer assets, GPU count, and bfloat16 support.
`smoke` runs two optimizer steps. `full` defaults to 10,000 steps, batch size 8 per GPU, and saves
the final checkpoint only. A periodic checkpoint interval can be passed as the fourth argument or
through `SAVE_FREQ`; keep in mind that full PI05 checkpoints require substantial disk space.

The full run explicitly leaves the vision encoder and action expert unfrozen and disables PEFT.
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
