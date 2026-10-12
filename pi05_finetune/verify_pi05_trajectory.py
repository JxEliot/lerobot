#!/usr/bin/env python3
"""Plot PI05 predictions against dataset actions for complete episodes.

The checkpoint processor converts the model's relative SE(3) output back to
absolute XYZ+rotvec targets.  This script plots those absolute targets, just
like the GR00T standalone evaluator, while reporting rotation geodesic error
instead of relying only on raw rotvec subtraction.
"""
from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lerobot.datasets import LeRobotDataset
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies import make_pre_post_processors


def rotation_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """SO(3) geodesic angle between two (..., 3) rotvec arrays."""
    ra = Rotation.from_rotvec(a.reshape(-1, 3))
    rb = Rotation.from_rotvec(b.reshape(-1, 3))
    return (ra.inv() * rb).magnitude().reshape(a.shape[:-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument(
        "--steps", type=int, default=0,
        help="Maximum frames per episode; 0 evaluates each complete episode.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.steps < 0:
        ap.error("--steps must be >= 0")
    ds = LeRobotDataset(repo_id="seer_robot/ee_v21", root=args.dataset)
    episode_count = len(ds.meta.episodes) if ds.meta.episodes is not None else 0
    invalid = [ep for ep in args.episodes if ep < 0 or ep >= episode_count]
    if invalid:
        ap.error(f"episodes out of range: {invalid}; dataset contains {episode_count} episodes")
    policy = PI05Policy.from_pretrained(args.checkpoint).to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.checkpoint,
        dataset_stats=ds.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # PI05 dataset action layout is [left_eef(6), right_eef(6), grippers(2)].
    # It is not the 16-D GR00T quaternion layout.
    labels = [
        "left x", "left y", "left z", "left rotvec x", "left rotvec y", "left rotvec z",
        "right x", "right y", "right z", "right rotvec x", "right rotvec y", "right rotvec z",
        "left gripper", "right gripper",
    ]
    reports = []
    for episode in args.episodes:
        episode_ds = LeRobotDataset(
            repo_id="seer_robot/ee_v21", root=args.dataset, episodes=[episode]
        )
        n = len(episode_ds) if args.steps == 0 else min(args.steps, len(episode_ds))
        if n == 0:
            raise RuntimeError(f"episode {episode} is empty")
        gt, pred = [], []
        policy.reset()
        for i in range(n):
            frame = episode_ds[i]
            obs = {
                k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim >= 1 else v)
                for k, v in frame.items()
                if k.startswith("observation.")
            }
            obs["task"] = [frame["task"]]
            processed = pre(obs)
            with torch.inference_mode():
                raw = policy.select_action(processed)
            # Dataset observations stay on CPU in this offline evaluator, so
            # keep the predicted action on CPU for the inverse EEF processor.
            absolute = post(raw.cpu()).detach().float().cpu().numpy().reshape(-1)
            pred.append(absolute)
            gt.append(frame["action"].detach().cpu().numpy())

        gt, pred = np.asarray(gt), np.asarray(pred)
        t = np.arange(n)
        episode_out = out.with_name(f"{out.stem}_episode{episode:03d}{out.suffix or '.png'}")
        fig, axes = plt.subplots(14, 1, figsize=(12, 32), sharex=True)
        fig.suptitle(f"PI05 episode {episode}: absolute action targets", fontsize=16)
        for j, ax in enumerate(axes):
            ax.plot(t, gt[:, j], label="ground truth", color="tab:blue", linewidth=1.5)
            ax.plot(t, pred[:, j], label="prediction", color="tab:orange", linewidth=1.2)
            ax.set_ylabel(labels[j])
            ax.grid(alpha=0.25)
            if j == 0:
                ax.legend(loc="upper right")
        axes[-1].set_xlabel("dataset frame (20 Hz)")
        fig.tight_layout()
        fig.savefig(episode_out, dpi=140)
        plt.close(fig)

        pos_err = np.linalg.norm(
            pred[:, [0, 1, 2, 6, 7, 8]].reshape(n, 2, 3)
            - gt[:, [0, 1, 2, 6, 7, 8]].reshape(n, 2, 3), axis=-1
        )
        rot_err = rotation_error(
            pred[:, [3, 4, 5, 9, 10, 11]].reshape(n, 2, 3),
            gt[:, [3, 4, 5, 9, 10, 11]].reshape(n, 2, 3),
        )
        grip_acc = np.mean(np.rint(pred[:, [12, 13]]) == np.rint(gt[:, [12, 13]]))
        report = {
            "episode": episode,
            "frames": n,
            "episode_frames": len(episode_ds),
            "complete_episode": n == len(episode_ds),
            "plot": str(episode_out),
            "position_error_median": np.median(pos_err, axis=0).tolist(),
            "position_error_p95": np.percentile(pos_err, 95, axis=0).tolist(),
            "rotation_error_rad_median": np.median(rot_err, axis=0).tolist(),
            "rotation_error_rad_p95": np.percentile(rot_err, 95, axis=0).tolist(),
            "gripper_rounded_accuracy": float(grip_acc),
            "raw_action_mae_not_used_as_primary_metric": float(np.mean(np.abs(pred - gt))),
        }
        episode_out.with_suffix(".json").write_text(json.dumps(report, indent=2))
        reports.append(report)
        print(json.dumps(report, indent=2))

    summary_path = out.parent / "summary.json"
    summary_path.write_text(json.dumps({"episodes": reports}, indent=2))
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
