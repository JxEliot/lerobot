#!/usr/bin/env python3
"""Export exact Seer PI05 training-window stats without changing the dataset."""
import argparse
import json
from pathlib import Path
from lerobot.datasets.seer_pi05_stats import compute_seer_pi05_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted((args.dataset_root / "data").glob("**/episode_*.parquet"))
    if args.episodes is not None:
        selected = set(args.episodes)
        paths = [p for p in paths if int(p.stem.split("_")[-1]) in selected]
        if len(paths) != len(selected):
            parser.error("Some selected episodes are missing")
    if args.output.resolve().is_relative_to(args.dataset_root.resolve()):
        parser.error("Write statistics outside the original dataset")
    stats = compute_seer_pi05_stats(paths, args.chunk_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Wrote {args.output}: {len(paths)} episodes, chunk_size={args.chunk_size}")


if __name__ == "__main__":
    main()
