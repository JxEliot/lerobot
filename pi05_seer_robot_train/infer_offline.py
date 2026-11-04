#!/usr/bin/env python
"""Run teacher-forced offline inference for a Seer PI05 checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import (
    bind_relative_anchor,
    policy_output_to_transition,
    transition_to_policy_action,
)

from pi05_seer_robot_train.seer_pi05.data import (
    ACTION_KEY,
    ACTION_NAMES,
    SeerDatasetView,
    _inspect_video_safe_rows,
    _safe_relative_indices,
)

def _resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "config.json").is_file():
        return path

    if (path / "pretrained_model" / "config.json").is_file():
        return path / "pretrained_model"

    checkpoints_dir = path / "checkpoints" if (path / "checkpoints").is_dir() else path
    candidates = [
        step_dir / "pretrained_model"
        for step_dir in checkpoints_dir.iterdir()
        if step_dir.is_dir()
        and step_dir.name.isdigit()
        and (step_dir / "pretrained_model" / "config.json").is_file()
    ] if checkpoints_dir.is_dir() else []
    if candidates:
        return max(candidates, key=lambda item: int(item.parent.name))

    raise FileNotFoundError(
        f"No PI05 config.json found under {path}. Pass a pretrained_model directory, "
        "a numbered checkpoint directory, or a training run directory."
    )


def _evaluation_episodes(meta: LeRobotDatasetMetadata, eval_split: float, split: str) -> list[int]:
    all_episodes = list(range(meta.total_episodes))
    if split == "all":
        return all_episodes

    episode_tasks = meta.episodes["tasks"]
    tasks_to_episodes: dict[str, list[int]] = defaultdict(list)
    for episode in all_episodes:
        tasks = episode_tasks[episode]
        task_key = tasks[0] if tasks else ""
        tasks_to_episodes[task_key].append(episode)

    train_episodes: list[int] = []
    eval_episodes: list[int] = []
    for episodes in tasks_to_episodes.values():
        n_eval = math.ceil(len(episodes) * eval_split)
        boundary = len(episodes) - n_eval
        train_episodes.extend(episodes[:boundary])
        eval_episodes.extend(episodes[boundary:])
    return eval_episodes if split == "eval" else train_episodes


def _rotvec_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half_angle = angle * 0.5
    scale = np.divide(
        np.sin(half_angle),
        angle,
        out=np.full_like(angle, 0.5),
        where=angle > 1e-8,
    )
    quaternion = np.concatenate((rotvec * scale, np.cos(half_angle)), axis=-1)
    return quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(min=1e-8)


def _rotation_error_degrees(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_quat = _rotvec_to_quaternion(prediction)
    target_quat = _rotvec_to_quaternion(target)
    dot = np.sum(pred_quat * target_quat, axis=-1).clip(-1.0, 1.0)
    return np.degrees(2.0 * np.arccos(np.abs(dot).clip(0.0, 1.0)))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_output_dir(output_dir: Path | None, checkpoint: Path) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return checkpoint.parent / f"offline_inference-{stamp}"


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {"num_frames": 0}
    metrics: dict[str, float | int] = {"num_frames": len(rows)}
    metric_names = (
        "left_position_error_m",
        "right_position_error_m",
        "left_rotation_error_deg",
        "right_rotation_error_deg",
        "left_gripper_abs_error",
        "right_gripper_abs_error",
    )
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        metrics[f"mean_{name}"] = float(values.mean())
        metrics[f"median_{name}"] = float(np.median(values))
        metrics[f"p90_{name}"] = float(np.quantile(values, 0.9))
    return metrics


def _run(args: argparse.Namespace) -> None:
    checkpoint = _resolve_checkpoint(args.checkpoint)
    checkpoint_train_config = checkpoint / "train_config.json"
    if not checkpoint_train_config.is_file():
        raise FileNotFoundError(f"Training config not found: {checkpoint_train_config}")

    train_config = json.loads(checkpoint_train_config.read_text())
    dataset_config = train_config.get("dataset", {})
    repo_id = args.repo_id or dataset_config.get("repo_id") or "seer_robot/lerobot_ee"
    dataset_root = args.dataset_root or dataset_config.get("root")
    if dataset_root is None:
        message = "Dataset root is missing; pass --dataset-root or use a checkpoint with train_config.json"
        raise ValueError(message)
    dataset_root = Path(dataset_root).expanduser().resolve()
    eval_split = float(dataset_config.get("eval_split", 0.0))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device}, but CUDA is not available")
    _set_seed(args.seed)

    logging.info("Checkpoint: %s", checkpoint)
    logging.info("Dataset: %s (%s)", dataset_root, repo_id)
    logging.info("Inference device: %s", device)

    policy_config = PreTrainedConfig.from_pretrained(checkpoint, cli_overrides=[f"--device={device}"])
    if policy_config.type != "pi05":
        raise ValueError(f"Expected PI05 checkpoint, got policy type {policy_config.type!r}")

    dataset_meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    split = args.split
    episodes = (
        [int(value) for value in args.episodes.split(",") if value.strip()]
        if args.episodes
        else _evaluation_episodes(dataset_meta, eval_split, split)
    )
    if not episodes:
        raise ValueError(
            f"No episodes selected for split={split!r} (checkpoint eval_split={eval_split}). "
            "Pass --split all or --episodes explicitly."
        )
    if any(episode < 0 or episode >= dataset_meta.total_episodes for episode in episodes):
        raise ValueError(f"Episode indices must be in [0, {dataset_meta.total_episodes - 1}]")
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=episodes,
        return_uint8=True,
    )
    valid_lengths, truncation_causes = _inspect_video_safe_rows(dataset, episodes)
    video_tail_rows = sum(
        int(dataset.meta.episodes[episode]["length"]) - length for episode, length in valid_lengths.items()
    )
    if args.max_frames_per_episode > 0:
        valid_lengths = {
            episode: min(length, args.max_frames_per_episode)
            for episode, length in valid_lengths.items()
        }
    valid_indices = _safe_relative_indices(dataset, episodes, valid_lengths)
    inference_dataset = SeerDatasetView(dataset, valid_indices, valid_lengths)
    logging.info(
        "Selected %d episode(s), %d safe frame(s); max frames/episode=%s",
        len(episodes),
        len(inference_dataset),
        args.max_frames_per_episode or "all",
    )
    if truncation_causes:
        logging.warning("Skipping missing video-tail frames in episodes: %s", truncation_causes)

    # Import registers Seer-specific serialized processor steps before loading pipelines.
    from pi05_seer_robot_train.seer_pi05 import processors as _seer_processors  # noqa: F401

    policy_class = get_policy_class(policy_config.type)
    policy = policy_class.from_pretrained(
        checkpoint,
        config=policy_config,
        local_files_only=True,
        strict=True,
    )
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    relative_step = bind_relative_anchor(policy, preprocessor)
    logging.info("Relative EEF action processor bound: %s", relative_step is not None)

    loader = DataLoader(
        inference_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
    )
    output_dir = _resolve_output_dir(args.output_dir, checkpoint)
    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_path = output_dir / "predictions.csv"
    all_rows: list[dict[str, Any]] = []
    previous_episode = None
    image_keys = dataset.meta.camera_keys

    with prediction_path.open("w", newline="") as csv_file:
        fieldnames = [
            "episode_index",
            "frame_index",
            "timestamp",
            "task",
            "left_position_error_m",
            "right_position_error_m",
            "left_rotation_error_deg",
            "right_rotation_error_deg",
            "left_gripper_abs_error",
            "right_gripper_abs_error",
            *[f"target_{name}" for name in ACTION_NAMES],
            *[f"pred_{name}" for name in ACTION_NAMES],
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for step, batch in enumerate(loader, start=1):
            episode = int(batch["episode_index"].item())
            if episode != previous_episode:
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                previous_episode = episode

            target = batch.pop(ACTION_KEY)[0].detach().cpu().numpy().astype(np.float32)
            for key in image_keys:
                image = batch.get(key)
                if isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
                    batch[key] = image.to(torch.float32) / 255.0

            processed = preprocessor(batch)
            with torch.inference_mode():
                action = policy.select_action(processed)
            action = transition_to_policy_action(policy_output_to_transition(action))
            prediction = postprocessor(action)[0].detach().cpu().numpy().astype(np.float32)
            if prediction.shape != (14,) or target.shape != (14,):
                raise ValueError(
                    f"Expected compact 14D actions, got prediction={prediction.shape}, target={target.shape}"
                )

            row: dict[str, Any] = {
                "episode_index": episode,
                "frame_index": int(batch["frame_index"].item()),
                "timestamp": float(batch["timestamp"].item()),
                "task": batch["task"][0],
                "left_position_error_m": float(np.linalg.norm(prediction[:3] - target[:3])),
                "right_position_error_m": float(np.linalg.norm(prediction[6:9] - target[6:9])),
                "left_rotation_error_deg": float(
                    _rotation_error_degrees(prediction[3:6], target[3:6])
                ),
                "right_rotation_error_deg": float(
                    _rotation_error_degrees(prediction[9:12], target[9:12])
                ),
                "left_gripper_abs_error": float(abs(prediction[12] - target[12])),
                "right_gripper_abs_error": float(abs(prediction[13] - target[13])),
            }
            row.update(
                {f"target_{name}": float(value) for name, value in zip(ACTION_NAMES, target, strict=True)}
            )
            row.update(
                {f"pred_{name}": float(value) for name, value in zip(ACTION_NAMES, prediction, strict=True)}
            )
            writer.writerow(row)
            all_rows.append(row)
            if step % args.log_every == 0:
                logging.info("Inferred %d frame(s)", step)

    summary = {
        "checkpoint": str(checkpoint),
        "policy_type": policy_config.type,
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "split": "explicit episodes" if args.episodes else split,
        "eval_split_from_training_config": eval_split,
        "episodes": episodes,
        "max_frames_per_episode": args.max_frames_per_episode,
        "skipped_video_tail_rows": video_tail_rows,
        "inference_mode": "teacher-forced dataset observations with PI05 action queue",
        "metrics": _metrics(all_rows),
        "predictions_csv": str(prediction_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    logging.info("Summary: %s", json.dumps(summary["metrics"], ensure_ascii=False))
    logging.info("Predictions: %s", prediction_path)
    logging.info("Summary file: %s", summary_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint_path",
        nargs="?",
        type=Path,
        help="optional positional checkpoint path",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=os.environ.get("CHECKPOINT"),
        help="pretrained_model directory, numbered checkpoint directory, or training run directory",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=os.environ.get("DATASET_ROOT"),
        help="defaults to DATASET_ROOT or the checkpoint train_config.json root",
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("DATASET_REPO_ID"),
        help="defaults to DATASET_REPO_ID or the checkpoint train_config.json dataset repo id",
    )
    parser.add_argument("--device", help="defaults to cuda when available")
    parser.add_argument("--split", choices=("eval", "train", "all"), default="eval")
    parser.add_argument("--episodes", help="comma-separated episode indices; overrides --split")
    parser.add_argument(
        "--max-episodes", type=int, default=0, help="0 means all episodes in the selected split"
    )
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=100,
        help="0 means all safe frames; the default is a short offline smoke evaluation",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint or args.checkpoint_path
    if args.checkpoint is None:
        parser.error("--checkpoint or CHECKPOINT is required")
    if args.max_episodes < 0 or args.max_frames_per_episode < 0 or args.num_workers < 0:
        parser.error("max episodes, max frames, and num workers must be non-negative")
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _run(_parse_args())


if __name__ == "__main__":
    main()
