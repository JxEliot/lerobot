# Pi0.5 full finetuning: pick and place

This setup finetunes every Pi0.5 parameter on the local dual-arm, end-effector-action dataset. It is separate from `run_finetune.sh`, which trains GR00T.

## Paths and defaults

- Environment: `/root/venvs/lerobot-pi05`
- LeRobot: `/mnt/cfs/vyi1fr/usr/jixiang/lerobot`
- Source dataset: `/mnt/cfs/vyi1fr/usr/jixiang/Isaac-GR00T/seer_robot/datasets/pick_and_place_lerobotv2/pick_and_place_v3.0`
- Training dataset view: `/mnt/cfs/vyi1fr/usr/jixiang/Isaac-GR00T/seer_robot/datasets/pick_and_place_lerobotv2/pick_and_place_v3_pi05_gr00t_eef`
- Base policy bundle: `/mnt/cfs/vyi1fr/usr/jixiang/hf_models/pi05_base_local` (symlinks the original weights and uses the local tokenizer in its processor config)
- Local tokenizer: `/mnt/cfs/vyi1fr/usr/jixiang/hf_models/paligemma-3b-pt-224-tokenizer`
- Outputs: `/mnt/cfs/vyi1fr/usr/jixiang/lerobot/pi05_finetune/checkpoints`

The dataset has 103 episodes, 24,652 frames at 20 FPS, three cameras, 30 state dimensions, and 14 action dimensions. Its state/action quantile statistics are present. Pi0.5 pads state/action vectors to 32 dimensions internally.

The action field is an absolute end-effector target, not a delta: `action[:6]` and `action[6:12]` are the commanded left/right XYZ+rotvec targets, while `action[12:14]` are absolute gripper targets. In the 30-D state, the matching current end-effector poses are `observation.state[18:24]` and `observation.state[24:30]`.

The default Pi0.5 run uses the new GR00T-compatible SE(3) processor. For each arm it trains on `T_current^-1 @ T_target`; translations are expressed in the current EEF frame and rotations are composed rather than subtracted as rotvec values. Gripper targets remain absolute. The output processor applies the inverse transform, so robot deployment receives absolute EEF targets. `--policy.use_relative_actions=false` remains intentional because LeRobot's generic processor subtracts the first action-sized state dimensions and is not valid for this dataset.

## Train-Time RTC

The training script enables Pi0.5 trained RTC with `RTC_TRAIN_MAX_DELAY=10`. For each training sample it draws a clean prefix of 0-10 actions and trains the flow loss only over the remaining suffix. With this 20 FPS dataset, this covers up to 0.5 seconds of observed controller/inference delay. The policy still predicts a 50-step chunk.

Override the delay only after measuring deployment latency:

```bash
RTC_TRAIN_MAX_DELAY=6 bash run_pi05_finetune.sh full
```

For a checkpoint trained with delay 10, trained RTC inference must use an execution horizon from 10 to 40. The current `N_ACTION_STEPS=10` is therefore valid. Use `--inference.type=rtc --inference.rtc.mode=trained --inference.rtc.execution_horizon=10` in the rollout command. Do not use `mode=trained` with an older checkpoint trained with delay 0.

The training dataset view links the source dataset's `data`, `videos`, and episode metadata; it owns only `meta/stats.json`, whose action entry is calculated in GR00T relative-action space. The source dataset and its absolute action statistics are unchanged. Recreate the relative stats after replacing demonstrations:

```bash
cd /mnt/cfs/vyi1fr/usr/jixiang/lerobot/pi05_finetune
python compute_gr00t_eef_stats.py \
  --dataset-root /mnt/cfs/vyi1fr/usr/jixiang/Isaac-GR00T/seer_robot/datasets/pick_and_place_lerobotv2/pick_and_place_v3.0 \
  --output /mnt/cfs/vyi1fr/usr/jixiang/Isaac-GR00T/seer_robot/datasets/pick_and_place_lerobotv2/pick_and_place_v3_pi05_gr00t_eef/meta/stats.json
```

At deployment, the policy emits absolute end-effector targets. If the robot controller accepts absolute pose targets, send them directly. If it only accepts deltas, convert using the current EE pose from state indices 18:30 (and leave the two gripper dimensions absolute); do not enable the generic `use_relative_actions` flag unless a custom processor with this index mapping is added. Training predicts 50-frame chunks. `n_action_steps=10` only controls deployment-time execution: at 20 FPS it replans every 0.5 seconds instead of executing a full 2.5-second chunk open loop.

## Commands

Run metadata, tokenizer, dimension, quantile, and GPU checks without loading the model:

```bash
cd /mnt/cfs/vyi1fr/usr/jixiang/lerobot/pi05_finetune
bash run_pi05_finetune.sh check
```

Run two full forward/backward steps on both GPUs:

```bash
bash run_pi05_finetune.sh smoke
```

Start the default full run (2000 steps, batch 80 per GPU, save every 500 steps):

```bash
bash run_pi05_finetune.sh full
```

The four position arguments are mode, steps, batch per GPU, and save frequency:

```bash
bash run_pi05_finetune.sh full 10000 24 2000
```

With two GPUs and no accumulation, batch 80 means effective batch 160. This was validated with a two-step full-parameter smoke run: both GPUs reached 100% utilization, each peaked at 81.3 GiB of 97.9 GiB, and peak power was about 575 W. The effective batch is:

```text
batch_per_gpu * NUM_GPUS * GRAD_ACCUM_STEPS
```

For an out-of-memory error, lower `BATCH_SIZE`; use `GRAD_ACCUM_STEPS` to restore the desired effective batch:

```bash
BATCH_SIZE=8 GRAD_ACCUM_STEPS=2 bash run_pi05_finetune.sh full
```

W&B is enabled by default. Log in once on the server before starting training:

```bash
/root/venvs/lerobot-pi05/bin/wandb login
```

The default project is `pi05-pick-and-place`. Disable it for an isolated run with `WANDB_ENABLE=false`.

Each new run gets a timestamped directory, preventing accidental checkpoint replacement. To resume, point `RESUME_FROM` at a checkpoint's `train_config.json` or `pretrained_model` directory and optionally choose a new output directory:

```bash
RESUME_FROM=/path/to/checkpoint/pretrained_model \
OUTPUT_DIR=/path/to/resumed-output \
bash run_pi05_finetune.sh full 10000 16 2000
```

Useful overrides include `NUM_GPUS`, `NUM_WORKERS`, `OUTPUT_DIR`, `N_ACTION_STEPS`, `DATASET_ROOT`, `MODEL_ROOT`, and `TOKENIZER_ROOT`.

Pi0.5's configured 30,000-step learning-rate schedule is automatically scaled by this LeRobot version when a shorter run is requested. At the default 2000 steps, warmup is scaled from 1000 to 66 steps and cosine decay ends at step 2000. The learning rate remains the Pi0.5 preset `2.5e-5`: do not linearly scale it with the larger global batch on this 103-episode dataset without a validation comparison.
