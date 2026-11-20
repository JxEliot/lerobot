#!/usr/bin/env python
"""Launch the stock LeRobot trainer with the Seer v3 dataset adapter."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
LEROBOT_ROOT = Path(os.environ.get("LEROBOT_ROOT", REPOSITORY_ROOT)).resolve()
sys.path.insert(0, str(LEROBOT_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT))

from lerobot.policies.factory import make_pre_post_processors as _make_pre_post_processors
from lerobot.scripts import lerobot_train

from pi05_seer_robot_train.seer_pi05.data import make_seer_train_eval_datasets
from pi05_seer_robot_train.seer_pi05.processors import make_seer_pi05_pre_post_processors

_RESUME = False


def _make_processors(*, policy_cfg, pretrained_path=None, pretrained_revision=None, **kwargs):
    if _RESUME and pretrained_path is not None:
        return _make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=pretrained_path,
            pretrained_revision=pretrained_revision,
            **kwargs,
        )
    return make_seer_pi05_pre_post_processors(
        config=policy_cfg,
        dataset_stats=kwargs.get("dataset_stats"),
    )


def _make_datasets(cfg):
    global _RESUME
    _RESUME = bool(cfg.resume)
    return make_seer_train_eval_datasets(cfg)


def main() -> None:
    lerobot_train.make_train_eval_datasets = _make_datasets
    lerobot_train.make_pre_post_processors = _make_processors
    lerobot_train.main()


if __name__ == "__main__":
    main()
