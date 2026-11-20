# Seer PI05 end-effector training

This folder contains an isolated training entry point for the LeRobot v3 dataset at
`/mnt/cfs/usr/jixiang/train_datasets/lerobot_ee`. It uses the stock PI05 model and LeRobot
training loop; the adapter only handles the Seer vector schema, normalization statistics,
and end-effector action representation.

## Script entry points

Shell launchers are grouped by purpose under `scripts/`. See the [中文脚本索引](scripts/README.md)
for the purpose of every launcher and commands to run on the server.

| Directory | Purpose |
|---|---|
| `scripts/train/` | Full fine-tuning, RTC, multi-task and single-task training; shared `run.sh` |
| `scripts/inference/` | Real-robot inference servers: trained RTC, guided RTC, no RTC and multi-task |
| `scripts/eval/` | Offline inference and trajectory comparison |

The former root-level shell paths have moved to these directories. Update saved launch commands
and DLC job commands accordingly. Python entry points, `config/`, `checkpoints/` and
`inference_results/` stay at the package root.

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
- Before training, the adapter checks each camera's decoded frame count against parquet timestamps.
  If a camera is missing terminal frames, it excludes those episode-tail observation rows and
  reports the affected episodes. The original parquet and video files remain unchanged.

## Server setup and launch

Run from a clean checkout of `feat/pi-train`. The launch scripts default to the server virtual
environment at `/root/venvs/lerobot-pi05`; override `VENV_ROOT`, `PYTHON_BIN`, or `TORCHRUN` when
needed. `ENV_FILE` remains available for dataset and training settings.

```bash
cd /mnt/cfs/usr/jixiang/vla_learning/lerobot
ENV_FILE=pi05_seer_robot_train/config/server.env bash pi05_seer_robot_train/scripts/train/train.sh full
ENV_FILE=pi05_seer_robot_train/config/server.env bash pi05_seer_robot_train/scripts/train/train_rtc.sh full
```

Both scripts accept `[check|smoke|full] [steps] [batch-per-GPU] [save-frequency] [num-gpus]`. Their
editable configuration blocks default to 60,000 steps, 4 GPUs, and batch size 64 per GPU. `check`
validates dataset metadata, local model/tokenizer assets, GPU count, and bfloat16 support. `smoke`
runs two optimizer steps with batch size 1 per GPU. `full` saves a checkpoint
every 5,000 steps by default, as well as at the final step. Set `SAVE_FREQ` in `config/server.env`
to change the interval; `SAVE_FREQ=0` disables periodic saves. Checkpoints are kept under the run's
`checkpoints/` directory, and `checkpoints/last` points to the newest completed save. No old
checkpoints are pruned. Full PI05 checkpoints require substantial disk space.

`train.sh` disables trained RTC prefixes. `train_rtc.sh` uses the previous branch's RTC training
setting: `RTC_TRAIN_MAX_DELAY=10`, with `chunk_size=50` and `n_action_steps=10` by default. At 20 FPS,
10 steps represent up to 0.5 seconds of action-prefix conditioning. Set `RTC_TRAIN_MAX_DELAY` to adjust
the RTC prefix range.

Both scripts fully fine-tune the vision encoder and action expert (`freeze_vision_encoder=false`,
`train_expert_only=false`, `use_peft=false`). Training settings can be overridden with environment
variables, including `GRAD_ACCUM_STEPS`, `SAVE_FREQ`, `CHUNK_SIZE`,
`N_ACTION_STEPS`, `RTC_TRAIN_MAX_DELAY`, `OPTIMIZER_LR`, `SCHEDULER_WARMUP_STEPS`,
`SCHEDULER_DECAY_STEPS`, `SCHEDULER_DECAY_LR`, `LOG_FREQ`, `EVAL_STEPS`, `EVAL_SPLIT`,
`NUM_WORKERS`, and `IMAGE_AUGMENTATION`. The positional arguments override mode, steps, batch per
GPU, save frequency, and GPU count. The generic `run.sh` defaults to 10,000 steps and one GPU when invoked
directly; recipe launchers override these with their top-of-file configuration. `OUTPUT_BASE` and
`OUTPUT_DIR` control checkpoint locations.

## Multi-task training and MLflow

`train_multitask_full.sh` starts from the PI05 base model with RTC training enabled;
`train_multitask_from_rtc.sh` starts a new optimizer and schedule from the saved RTC policy weights.
Both use RTC action-prefix conditioning (`RTC_TRAIN_MAX_DELAY=10` by default) and all 243 episodes in
`lerobot_ee_multitask_v3` (`EVAL_SPLIT=0`), keep full-model fine-tuning enabled, and write training
metrics to MLflow every `LOG_FREQ` steps. There is no held-out evaluation loss in these runs.

```bash
ENV_FILE=pi05_seer_robot_train/config/server.env bash pi05_seer_robot_train/scripts/train/train_multitask_full.sh full
ENV_FILE=pi05_seer_robot_train/config/server.env bash pi05_seer_robot_train/scripts/train/train_multitask_from_rtc.sh full
```

MLflow logging requires `mlflow>=3.0,<4` in `/root/venvs/lerobot-pi05` on the training server. Training
uses the tested CleanRL server at `https://mlflow.seer-robotics.cn` and its existing experiment
`openpi/pi05` (ID `20`). The server URL and experiment settings are in
`pi05_seer_robot_train/config/server.env`. Credentials are copied from CleanRL's ignored
`configs/.mlflow.env` to `pi05_seer_robot_train/config/.mlflow.env`; launchers load that file
automatically. Both private files are ignored by Git. Pass the settings file with
`ENV_FILE=pi05_seer_robot_train/config/server.env`.

The private server config pins experiment ID `20`, `openpi/pi05`, matching the tested CleanRL setup;
the launch preflight checks that the configured server and experiment are reachable before starting
distributed workers. Each run gets a unique name from the job name and timestamp;
`MLFLOW_RUN_NAME` can set an explicit name, and `MLFLOW_TAGS` accepts comma-separated `key=value` tags.

MLflow records loss, gradient norm, learning rate, throughput, timing, GPU memory, and training
configuration. It does not store or manage model checkpoints; those remain in the normal checkpoint
folder. The former CFS SQLite store is no longer used by new runs, and the official server's UI is
opened directly at its configured URL.

Visual augmentation is photometric only: brightness and contrast vary from 0.9 to 1.1, sharpness
from 0.8 to 1.2, with up to two transforms per image. It does not alter camera geometry.

## Offline inference

`infer_offline.sh` loads the checkpoint's saved PI05 config and custom Seer processors, then runs
teacher-forced inference on the held-out episodes from that run's `eval_split`. It does not connect
to a robot or simulator. The default checks up to 100 frames per episode; set
`--max-frames-per-episode 0` to process every valid frame.

```bash
bash pi05_seer_robot_train/scripts/eval/infer_offline.sh \
  pi05_seer_robot_train/checkpoints/rtc/full-20261103-185442 \
  --max-frames-per-episode 100

bash pi05_seer_robot_train/scripts/eval/infer_offline.sh \
  pi05_seer_robot_train/checkpoints/no-rtc/full-20261103-185856 \
  --max-frames-per-episode 100
```

Each run writes one organized result directory under `pi05_seer_robot_train/inference_results/rtc/`
or `pi05_seer_robot_train/inference_results/no-rtc/`, matching the checkpoint type.
It contains `predictions.csv`, `summary.json`, a compressed `episode_<index>.npz` for each tested
episode, and a `trajectory_episode_<index>.png` action comparison plot. The plot follows the GR00T
validation layout: one panel per action dimension, dataset action overlaid with PI05 prediction, and
red markers where PI05 refreshes its action chunk. Pass `--output-dir` to choose a readable folder
name; otherwise a timestamped `validation-*` folder is created automatically. Pass `--split all` to
inspect training and held-out episodes together, or `--episodes` with comma-separated dataset indices
to select episodes directly. For this dataset the valid indices are `0` through `119`; for example:

```bash
bash pi05_seer_robot_train/scripts/eval/infer_offline.sh \
  pi05_seer_robot_train/checkpoints/no-rtc/full-20261103-185856 \
  --episodes 114 \
  --max-frames-per-episode 0 \
  --output-dir pi05_seer_robot_train/inference_results/no-rtc/episode-114
```

`DATASET_ROOT`, `DATASET_REPO_ID`, `PYTHON_BIN`, and `VENV_ROOT` can be overridden in the shell
environment or through `ENV_FILE`.

## Real-robot inference server

The HTTP server keeps the request/response protocol used by the earlier Seer PI05 server and
loads the v3 checkpoint processors. It does not open the training dataset. It validates incoming
state and images, converts quaternion EEF state to rotation vectors, applies the saved processor
pair, and returns absolute XYZ+rotation-vector targets in `action_chunk`.

Start one server on the GPU host:

```bash
bash pi05_seer_robot_train/scripts/inference/run_pi05_inference_server.sh
```

That script defaults to the trained-RTC checkpoint and RTC mode `trained`. The alternate launchers
select `guided` RTC or plain inference:

```bash
bash pi05_seer_robot_train/scripts/inference/run_pi05_inference_server_guided.sh
bash pi05_seer_robot_train/scripts/inference/run_pi05_inference_server_no_rtc.sh
```

The service listens on `0.0.0.0:18080`; the robot controller should use
`http://<inference-server-ip>:18080/v1/pi05/action` and `/health` for readiness. The action endpoint
uses the existing `seer.model_inference.request.v1` request schema and returns a 50-step `action_chunk`.
For trained RTC, the client must pass the previous absolute action chunk, measured
`inference_delay`, and the matching `execution_horizon` in `pi05_rtc`. Defaults are a 25-step
execution horizon for trained RTC and 15 for guided RTC. Override `CHECKPOINT`, `PORT`,
`EXECUTION_HORIZON`, `WARMUP_STEPS`, or other launcher settings through the environment or `ENV_FILE`.

The server detects task changes from each request's `task` field. On the first request for a task,
it resets policy and processor state and ignores any supplied RTC prefix (`inference_delay=0`). Its
response sets `reset_action_queue=true`; the robot-side client must clear already buffered actions
before executing the returned chunk. Subsequent requests for the same task may continue RTC using the
new task's action chunks.

The server validates checkpoint processors and warms up with a synthetic request before opening
the HTTP listener. It does not connect to or command the robot; the existing robot-side client remains
responsible for connecting the service to the arm controller.

## Resume

The low-level LeRobot resume command is shown below. For recovery with the checkpoint's saved GPU
topology and configured save interval, use the dedicated launcher described after it:

```bash
torchrun --standalone --nproc-per-node=1 pi05_seer_robot_train/train.py \
  --resume=true \
  --config_path=/path/to/checkpoints/last/pretrained_model/train_config.json \
  --output_dir=/path/to/new-resume-run
```

The launcher patch restores the saved custom processor pipelines on resume, so the normalization
statistics bundled with the checkpoint remain authoritative. For a convenience launcher that
automatically selects the latest checkpoint and preserves the original run directory, use:

```bash
ENV_FILE=pi05_seer_robot_train/config/server.env bash pi05_seer_robot_train/scripts/train/resume_training.sh pi05_seer_robot_train/checkpoints/multitask-full-v1/full-20261118-071558 60000
```

The target is the total training step, not the additional number of steps. The optional third
argument selects `last` (the default) or a saved step, such as `010000`. The script uses the
checkpoint's recorded GPU topology and the configured `SAVE_FREQ`. An interruption leaves the
newest completed checkpoint available; work since that save must be repeated. Earlier checkpoints
are retained.

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
- `preflight.py`, `config/`: server configuration and validation.
- `scripts/train/`: training launchers and shared `run.sh`.
- `scripts/inference/`: real-robot inference launchers.
- `scripts/eval/`: offline validation launcher.
- `tests/`: vector conversion and episode-boundary checks.
