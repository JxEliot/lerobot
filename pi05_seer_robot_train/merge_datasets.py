#!/usr/bin/env python3
"""Merge the three local Seer v3 datasets without modifying the sources.

Run on the server with:
    /root/venvs/lerobot-pi05/bin/python pi05_seer_robot_train/merge_datasets.py

The default output is /mnt/cfs/usr/jixiang/train_datasets/lerobot_ee_multitask_v3.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyarrow.dataset as pa_dataset

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata


SOURCE_NAMES = ("lerobot_ee", "lerobot_ee_add_data", "lerobot_ee_pnp_all")
DEFAULT_PARENT = Path("/mnt/cfs/usr/jixiang/train_datasets")
DEFAULT_OUTPUT_NAME = "lerobot_ee_multitask_v3"


def verify_merge(
    output: Path, source_metadata: list[LeRobotDatasetMetadata], repo_id: str
) -> tuple[int, int, Counter[str]]:
    merged = LeRobotDatasetMetadata(repo_id, root=output)
    expected_episodes = [episode for meta in source_metadata for episode in meta.episodes]
    expected_frames = sum(meta.total_frames for meta in source_metadata)
    if merged.total_episodes != len(expected_episodes) or merged.total_frames != expected_frames:
        raise RuntimeError("Merged episode or frame count differs from the sources")

    lengths = np.asarray([int(episode["length"]) for episode in expected_episodes], dtype=np.int64)
    task_names = [episode["tasks"] for episode in expected_episodes]
    if sum(lengths) != expected_frames:
        raise RuntimeError("Source episode lengths do not add up to the source frame count")

    counts: Counter[str] = Counter()
    task_by_episode = np.empty(len(expected_episodes), dtype=np.int64)
    for index, (expected, source_tasks) in enumerate(zip(expected_episodes, task_names, strict=True)):
        actual = merged.episodes[index]
        if int(actual["episode_index"]) != index or int(actual["length"]) != int(expected["length"]):
            raise RuntimeError(f"Episode {index} has an unexpected index or length")
        if actual["tasks"] != source_tasks or len(source_tasks) != 1:
            raise RuntimeError(f"Episode {index} has an unexpected task")
        task = source_tasks[0]
        counts[task] += 1
        task_by_episode[index] = int(merged.tasks.loc[task, "task_index"])

        paths = [merged.get_data_file_path(index)]
        paths.extend(merged.get_video_file_path(index, key) for key in merged.camera_keys)
        for relative_path in paths:
            path = output / relative_path
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Episode {index} references a missing or empty file: {path}")

    columns = pa_dataset.dataset(output / "data", format="parquet").to_table(
        columns=["index", "episode_index", "frame_index", "task_index"]
    )
    if columns.num_rows != expected_frames:
        raise RuntimeError("Merged parquet row count differs from the source frame count")
    global_indices = columns["index"].to_numpy()
    episode_indices = columns["episode_index"].to_numpy()
    frame_indices = columns["frame_index"].to_numpy()
    task_indices = columns["task_index"].to_numpy()
    if not np.array_equal(np.sort(global_indices), np.arange(expected_frames)):
        raise RuntimeError("Merged global frame indices are not consecutive")
    if np.any(episode_indices < 0) or np.any(episode_indices >= len(lengths)):
        raise RuntimeError("Merged parquet contains an invalid episode index")
    if not np.array_equal(np.bincount(episode_indices, minlength=len(lengths)), lengths):
        raise RuntimeError("Merged parquet frame counts do not match episode metadata")
    if not np.array_equal(task_indices, task_by_episode[episode_indices]):
        raise RuntimeError("Merged parquet task indices do not match episode task names")
    order = np.lexsort((frame_indices, episode_indices))
    expected_frames_per_episode = np.concatenate([np.arange(length) for length in lengths])
    if not np.array_equal(frame_indices[order], expected_frames_per_episode):
        raise RuntimeError("Merged episode frame indices are not consecutive")

    return len(expected_episodes), expected_frames, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--repo-id", default="seer/lerobot_ee_multitask_v3")
    args = parser.parse_args()

    parent = args.dataset_parent.resolve(strict=True)
    if Path(args.output_name).name != args.output_name or args.output_name in {"", ".", ".."}:
        parser.error("--output-name must be a single directory name")
    output = parent / args.output_name
    sources = [parent / name for name in SOURCE_NAMES]
    if output.exists():
        parser.error(f"Output already exists; refusing to overwrite: {output}")
    for source in sources:
        if not (source / "meta" / "info.json").is_file():
            parser.error(f"Source dataset metadata is missing: {source}")

    source_metadata = [
        LeRobotDatasetMetadata(f"seer/{name}", root=root) for name, root in zip(SOURCE_NAMES, sources, strict=True)
    ]
    expected_episodes = sum(meta.total_episodes for meta in source_metadata)
    expected_frames = sum(meta.total_frames for meta in source_metadata)
    print(f"Merging {expected_episodes} episodes and {expected_frames} frames into {output}", flush=True)

    staging = parent / f".{args.output_name}-{uuid4().hex}"
    try:
        aggregate_datasets(
            repo_ids=[f"seer/{name}" for name in SOURCE_NAMES],
            aggr_repo_id=args.repo_id,
            roots=sources,
            aggr_root=staging,
            concatenate_videos=False,
            concatenate_data=False,
        )
        episodes, frames, task_counts = verify_merge(staging, source_metadata, args.repo_id)
        if output.exists():
            raise RuntimeError(f"Output was created during the merge: {output}")
        staging.rename(output)
    except Exception:
        print(f"Merge failed; partial output retained for inspection at {staging}", file=sys.stderr)
        raise

    print(f"Merged dataset: {output}")
    print(f"Episodes: {episodes}, frames: {frames}")
    for task, count in task_counts.items():
        print(f"  {count} episodes: {task}")


if __name__ == "__main__":
    main()
