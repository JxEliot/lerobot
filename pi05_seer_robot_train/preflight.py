#!/usr/bin/env python
"""Validate local model assets and the remote/local dataset metadata before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()

    info_path = args.dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"Dataset metadata not found: {info_path}")
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    if info.get("codebase_version") != "v3.0":
        raise SystemExit(f"Expected LeRobot v3.0, got {info.get('codebase_version')!r}")
    expected = {"observation.state": [64], "action": [16]}
    for key, shape in expected.items():
        if key not in features or features[key].get("shape") != shape:
            raise SystemExit(f"Expected {key} shape {shape}, got {features.get(key, {}).get('shape')}")

    cameras = [
        key for key, feature in features.items() if feature.get("dtype") in ("image", "video")
    ]
    if not cameras:
        raise SystemExit("Dataset metadata contains no image/video observations")
    for camera in ("observation.images.head", "observation.images.left_wrist", "observation.images.right_wrist"):
        if camera not in cameras:
            raise SystemExit(f"Expected Seer camera feature is missing: {camera}")

    if not (args.model_root / "config.json").is_file():
        raise SystemExit(f"PI05 config not found: {args.model_root / 'config.json'}")
    weight_files = list(args.model_root.glob("*.safetensors"))
    if not weight_files:
        raise SystemExit(f"PI05 safetensors weights not found in {args.model_root}")
    if not (args.tokenizer_root / "tokenizer.json").is_file():
        raise SystemExit(f"PaliGemma tokenizer not found: {args.tokenizer_root / 'tokenizer.json'}")

    import torch
    from transformers import AutoTokenizer

    visible_gpus = torch.cuda.device_count()
    if args.num_gpus < 1 or args.num_gpus > visible_gpus:
        raise SystemExit(f"Requested {args.num_gpus} GPU(s), but PyTorch sees {visible_gpus}")
    for index in range(args.num_gpus):
        major, minor = torch.cuda.get_device_capability(index)
        if major < 8:
            raise SystemExit(f"GPU {index} does not support the requested bfloat16 training: sm_{major}{minor}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_root, local_files_only=True)
    if len(tokenizer) < 250_000:
        raise SystemExit(f"Unexpected PaliGemma tokenizer vocabulary size: {len(tokenizer)}")

    print(
        f"Preflight OK: episodes={info['total_episodes']}, frames={info['total_frames']}, "
        f"fps={info['fps']}, state=64->30, action=16->14"
    )
    print(f"Cameras: {', '.join(cameras)}")
    print(f"Tokenizer: {type(tokenizer).__name__}, vocab={len(tokenizer)}")
    print(f"CUDA: {args.num_gpus}/{visible_gpus} GPU(s)")


if __name__ == "__main__":
    main()
