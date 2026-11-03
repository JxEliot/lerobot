#!/usr/bin/env python3
"""Standalone PI05 HTTP inference server for Seer model requests."""
from __future__ import annotations

import argparse, base64, io, threading
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import uvicorn
from scipy.spatial.transform import Rotation

from lerobot.policies.pi05 import PI05Policy
from lerobot.policies import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference


class Request(BaseModel):
    schema: str = "seer.model_inference.request.v1"
    request_id: str
    task: str = ""
    state: list[float]
    state_names: list[str]
    images: dict[str, str]
    rtc_options: dict | None = None
    pi05_rtc: dict | None = None


class PI05Server:
    def __init__(self, checkpoint: str, dataset_root: str, device: str, action_horizon: int, execution_horizon: int = 15, rtc_mode: str = "trained"):
        from lerobot.datasets import LeRobotDataset
        self.server_info = {"checkpoint": checkpoint, "device": device}
        self.device = torch.device(device)
        startup_started = time.perf_counter()
        started = time.perf_counter()
        print(f"[startup] loading dataset metadata/sample: {dataset_root}", flush=True)
        self.dataset = LeRobotDataset(repo_id="seer_robot/ee_v21", root=dataset_root, episodes=[0])
        print(f"[startup] dataset loaded in {time.perf_counter() - started:.2f}s", flush=True)
        started = time.perf_counter()
        print(f"[startup] loading policy: {checkpoint}", flush=True)
        self.policy = PI05Policy.from_pretrained(checkpoint).to(self.device).eval()
        print(f"[startup] policy loaded in {time.perf_counter() - started:.2f}s", flush=True)
        self.server_info["num_inference_steps"] = self.policy.config.num_inference_steps
        training_max_delay = int(getattr(self.policy.config, "rtc_training_max_delay", 0))
        if rtc_mode not in {"trained", "guided", "off"}:
            raise ValueError("rtc_mode must be trained, guided or off")
        self.rtc_mode = rtc_mode
        if rtc_mode == "trained" and training_max_delay < 10:
            raise ValueError(
                "trained PI05 RTC requires checkpoint rtc_training_max_delay >= 10, "
                f"got {training_max_delay}"
            )
        if int(self.policy.config.chunk_size) != 50:
            raise ValueError(
                f"expected the Seer PI05 checkpoint chunk_size=50, got {self.policy.config.chunk_size}"
            )
        self.training_max_delay = training_max_delay
        if not (training_max_delay if rtc_mode == "trained" else 1) <= execution_horizon <= (self.policy.config.chunk_size - training_max_delay if rtc_mode == "trained" else self.policy.config.chunk_size):
            raise ValueError("execution_horizon is outside the trained RTC range")
        self.execution_horizon = execution_horizon
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        self.policy.config.rtc_config = RTCConfig(
            enabled=True,
            mode=rtc_mode,
            execution_horizon=self.execution_horizon,
        ) if rtc_mode != "off" else None
        self.policy.init_rtc_processor()
        if not 0 < action_horizon <= self.policy.config.chunk_size:
            raise ValueError(
                f"action_horizon must be in [1, {self.policy.config.chunk_size}], got {action_horizon}"
            )
        self.action_horizon = action_horizon
        self.action_dim = int(self.policy.config.output_features["action"].shape[0])
        if self.action_dim != 14:
            raise ValueError(f"Seer dual-arm PI05 expects 14 action values, got {self.action_dim}")
        started = time.perf_counter()
        print("[startup] loading observation/action processors", flush=True)
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint,
            # Deployment must retain the statistics saved with the trained weights.
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        preprocessor_config = json.loads(
            (Path(checkpoint) / "policy_preprocessor.json").read_text(encoding="utf-8")
        )
        normalizer = next(
            step for step in preprocessor_config["steps"]
            if step["registry_name"] == "normalizer_processor"
        )
        action_stats = load_file(Path(checkpoint) / normalizer["state_file"], device=str(self.device))
        self.action_q01 = action_stats["action.q01"]
        self.action_q99 = action_stats["action.q99"]
        if self.action_q01.numel() != self.action_dim or self.action_q99.numel() != self.action_dim:
            raise ValueError("checkpoint action normalization stats do not match PI05 action dimension")
        self.action_range = (self.action_q99 - self.action_q01).clamp_min(1e-8)
        # Fail startup if ordinary decoding and RTC prefix encoding disagree.
        for pipeline in (self.pre, self.post):
            normalizers = [step for step in pipeline.steps
                           if "action" in getattr(step, "_tensor_stats", {})]
            if len(normalizers) != 1:
                raise ValueError("expected one action normalizer per processor pipeline")
            effective = normalizers[0]._tensor_stats["action"]
            for name, expected in (("q01", self.action_q01), ("q99", self.action_q99)):
                actual = effective[name].to(device=expected.device, dtype=expected.dtype)
                if not torch.equal(actual, expected):
                    raise ValueError(f"processor action {name} differs from checkpoint RTC statistics")
        print(f"[startup] processors/statistics ready in {time.perf_counter() - started:.2f}s", flush=True)
        self.server_info["initialization_seconds"] = round(time.perf_counter() - startup_started, 3)
        self.server_info.update(
            normalization_source="checkpoint_saved",
            rtc_mode=self.rtc_mode,
            rtc_training_max_delay=training_max_delay,
            default_execution_horizon=self.execution_horizon,
        )
        print(f"[checkpoint] {json.dumps(self.server_info)}", flush=True)
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi05-inference")

    def warmup(self, steps: int = 3) -> None:
        """Exercise the real request path on the inference worker before listening."""
        if steps < 0:
            raise ValueError("warmup steps must be non-negative")
        if steps == 0:
            print("[warm-up] disabled", flush=True)
            return
        frame = self.dataset[0]
        state = frame["observation.state"].detach().cpu().numpy().reshape(-1)
        names = self.dataset.meta.features["observation.state"]["names"]
        # The Seer dataset adapter compacts the on-disk 64 quaternion fields
        # into the trained 30-value state while metadata retains disk names.
        if state.size == 30:
            values = state[:18].tolist()
            names = [
                *(f"{side}_arm_joint_{i}" for side in ("left", "right") for i in range(1, 8)),
                "waist_position", "lift_position", "left_gripper", "right_gripper",
            ]
            for side, offset in (("left", 18), ("right", 24)):
                values.extend(state[offset:offset + 3].tolist())
                values.extend(Rotation.from_rotvec(state[offset + 3:offset + 6]).as_quat().tolist())
                names.extend(f"{side}_ee.{axis}" for axis in ("x", "y", "z", "qx", "qy", "qz", "qw"))
            state = np.asarray(values)
        if len(names) != len(state):
            raise ValueError("warmup dataset state does not match its named fields")
        images = {}
        for camera in ("head", "left_wrist", "right_wrist"):
            pixels = frame[f"observation.images.{camera}"].detach().cpu().numpy()
            pixels = np.clip(pixels.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="JPEG", quality=75)
            images[f"{camera}_image"] = base64.b64encode(buffer.getvalue()).decode("ascii")
        req = Request(
            request_id="startup-warmup", task=frame.get("task", ""),
            state=state.tolist(), state_names=names, images=images,
        )
        timings = []
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        print(f"[warm-up] starting {steps} inference passes; HTTP is not listening yet", flush=True)
        try:
            with torch.random.fork_rng(devices=devices):
                previous = None
                for index in range(steps):
                    req.pi05_rtc = {
                        "enabled": self.rtc_mode != "off", "mode": self.rtc_mode, "execution_horizon": self.execution_horizon,
                        "inference_delay": 0 if previous is None or self.rtc_mode == "off" else 6,
                        "prev_chunk_left_over": previous if self.rtc_mode != "off" else None,
                    }
                    result = self.infer(req)
                    timing = result["server_timing_ms"]
                    timings.append(timing)
                    previous = result["action_chunk"][self.execution_horizon:]
                    print(
                        f"[warm-up] pass {index + 1}/{steps}: "
                        f"predict_ms={timing['predict']:.2f}, total_ms={timing['total']:.2f}, "
                        f"rtc_prefix={index > 0 and self.rtc_mode != 'off'}", flush=True,
                    )
        finally:
            self.policy.reset()
            self.pre.reset()
            self.post.reset()
            self.policy.init_rtc_processor()
        self.server_info["warmup_steps"] = steps
        self.server_info["warmup_timing_ms"] = timings
        print("[warm-up] complete; inference state reset; ready to start HTTP", flush=True)

    @staticmethod
    def _decode(value: str) -> np.ndarray:
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB"))

    @staticmethod
    def _get(values: dict[str, float], name: str, default: float = 0.0) -> float:
        return float(values.get(name, values.get(name + ".position", default)))

    def _state30(self, req: Request) -> np.ndarray:
        v = dict(zip(req.state_names, req.state, strict=False))
        required = [
            *(f"{side}_arm_joint_{i}" for side in ("left", "right") for i in range(1, 8)),
            "waist_position", "lift_position", "left_gripper", "right_gripper",
            *(f"{side}_ee.{axis}" for side in ("left", "right")
              for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")),
        ]
        missing = [
            name for name in required
            if name not in v and f"{name}.position" not in v
        ]
        if missing:
            raise ValueError(f"request state is missing required fields: {', '.join(missing)}")
        out = []
        for side in ("left", "right"):
            out.extend(self._get(v, f"{side}_arm_joint_{i}") for i in range(1, 8))
        out.extend([self._get(v, "waist_position"), self._get(v, "lift_position")])
        out.extend([self._get(v, "left_gripper"), self._get(v, "right_gripper")])
        for side in ("left", "right"):
            xyz = [self._get(v, f"{side}_ee.{x}") for x in ("x", "y", "z")]
            quat = np.asarray([self._get(v, f"{side}_ee.{x}") for x in ("qx", "qy", "qz", "qw")])
            if np.linalg.norm(quat) < 1e-8:
                quat = np.asarray([0, 0, 0, 1.0])
            quat = quat / np.linalg.norm(quat)
            out.extend(xyz + Rotation.from_quat(quat).as_rotvec().tolist())
        return np.asarray(out, dtype=np.float32)

    def infer(self, req: Request, received_at: float | None = None) -> dict:
        started = time.perf_counter()
        received_at = started if received_at is None else received_at
        required_images = ("head_image", "left_wrist_image", "right_wrist_image")
        missing = [name for name in required_images if name not in req.images]
        if missing:
            raise ValueError(f"missing required images: {', '.join(missing)}")
        decode_started = time.perf_counter()
        images = {name: self._decode(req.images[name]) for name in required_images}
        decode_done = time.perf_counter()
        obs = {
            "observation.state": self._state30(req),
            "observation.images.head": images["head_image"],
            "observation.images.left_wrist": images["left_wrist_image"],
            "observation.images.right_wrist": images["right_wrist_image"],
        }
        obs = prepare_observation_for_inference(obs, self.device, task=req.task, robot_type="seer_dual_arm")
        with self.lock, torch.no_grad():
            preprocess_started = time.perf_counter()
            processed = self.pre(obs)
            rtc = req.pi05_rtc or {}
            rtc_enabled = bool(rtc.get("enabled", False))
            if rtc_enabled and self.rtc_mode == "off":
                raise ValueError("server mode=off cannot accept trained RTC requests")
            if rtc.get("mode", self.rtc_mode) != self.rtc_mode:
                raise ValueError(f"request RTC mode does not match server mode={self.rtc_mode}")
            rtc_delay = int(rtc.get("inference_delay", 0))
            rtc_execution_horizon = int(rtc.get("execution_horizon", self.execution_horizon))
            if rtc_enabled and self.rtc_mode == "trained":
                if rtc.get("mode", "trained") != "trained":
                    raise ValueError("PI05 checkpoint requires RTC mode=trained")
                if not 0 <= rtc_delay <= self.training_max_delay:
                    raise ValueError("RTC inference_delay exceeds checkpoint training range")
                if not self.training_max_delay <= rtc_execution_horizon <= self.policy.config.chunk_size - self.training_max_delay:
                    raise ValueError("RTC execution_horizon is outside checkpoint training range")
            if rtc_enabled and self.rtc_mode == "guided":
                if not 0 <= rtc_delay < rtc_execution_horizon <= self.policy.config.chunk_size:
                    raise ValueError("guided RTC requires 0 <= delay < execution_horizon <= chunk_size")
            prev_chunk = rtc.get("prev_chunk_left_over")
            if rtc_enabled and rtc_delay > 0 and prev_chunk is None:
                raise ValueError("positive RTC inference_delay requires a previous chunk")
            rtc_prefix = None
            if rtc_enabled and prev_chunk is not None:
                absolute = np.array(prev_chunk, dtype=np.float64, copy=True)
                if absolute.ndim != 2 or absolute.shape[1] != 14:
                    raise ValueError("PI05 RTC previous chunk must have shape (steps, 14)")
                if absolute.shape[0] < rtc_delay or absolute.shape[0] > self.policy.config.chunk_size or not np.isfinite(absolute).all():
                    raise ValueError("RTC prefix length/values are invalid")
                state = self._state30(req).astype(np.float64)
                for arm, state_offset in ((0, 18), (6, 24)):
                    current_rotation = Rotation.from_rotvec(state[state_offset + 3:state_offset + 6])
                    target_rotation = Rotation.from_rotvec(absolute[:, arm + 3:arm + 6])
                    absolute[:, arm:arm + 3] = current_rotation.inv().apply(
                        absolute[:, arm:arm + 3] - state[state_offset:state_offset + 3]
                    )
                    absolute[:, arm + 3:arm + 6] = (
                        current_rotation.inv() * target_rotation
                    ).as_rotvec()
                rtc_prefix = torch.as_tensor(absolute, device=self.device, dtype=torch.float32)
                rtc_prefix = 2.0 * (rtc_prefix - self.action_q01) / self.action_range - 1.0
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            predict_started = time.perf_counter()
            if rtc_enabled:
                raw = self.policy.predict_action_chunk(
                    processed,
                    inference_delay=rtc_delay,
                    prev_chunk_left_over=rtc_prefix,
                    execution_horizon=rtc_execution_horizon,
                )
            else:
                raw = self.policy.predict_action_chunk(
                    processed,
                    inference_delay=0,
                    prev_chunk_left_over=None,
                    execution_horizon=rtc_execution_horizon,
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            predict_done = time.perf_counter()
            rtc_action_chunk = raw[0].detach().float().cpu().numpy()
            # The custom GR00T-compatible inverse EEF processor keeps the
            # current state on the inference device, so postprocess the raw
            # CUDA action before moving the result back to CPU.
            actions = self.post(raw).detach().float().cpu().numpy()
        post_done = time.perf_counter()
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != self.action_dim:
            raise ValueError(f"unexpected postprocessed action chunk shape: {actions.shape}")
        full_actions = actions[0]
        actions = full_actions[: self.action_horizon]
        if actions.shape[0] != self.action_horizon:
            raise ValueError(
                f"policy returned {actions.shape[0]} actions, expected {self.action_horizon}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("policy returned non-finite actions")
        names = [
            "left_ee_pose_target.x", "left_ee_pose_target.y", "left_ee_pose_target.z",
            "left_ee_pose_target.rx", "left_ee_pose_target.ry", "left_ee_pose_target.rz",
            "right_ee_pose_target.x", "right_ee_pose_target.y", "right_ee_pose_target.z",
            "right_ee_pose_target.rx", "right_ee_pose_target.ry", "right_ee_pose_target.rz",
            "left_gripper_command", "right_gripper_command",
        ]
        action_chunk = actions.tolist()
        return {
            "schema": "seer.model_inference.response.v1",
            "request_id": req.request_id,
            "backend": "pi05",
            "server_info": self.server_info,
            "server_timing_ms": {
                "queue": (started - received_at) * 1000,
                "image_decode": (decode_done - decode_started) * 1000,
                "preprocess_and_rtc": (predict_started - preprocess_started) * 1000,
                "predict": (predict_done - predict_started) * 1000,
                "postprocess": (post_done - predict_done) * 1000,
                "total": (time.perf_counter() - received_at) * 1000,
            },
            "status": "ok",
            "action": action_chunk[0],
            "action_chunk": action_chunk,
            "action_horizon": len(action_chunk),
            "action_names": names,
            "rtc_supported": bool((req.pi05_rtc or {}).get("enabled", False)),
            "rtc_action_chunk": rtc_action_chunk.tolist(),
            "rtc_mode": self.rtc_mode,
            "rtc_inference_delay": int((req.pi05_rtc or {}).get("inference_delay", 0)),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--action-horizon", type=int, default=32)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--execution-horizon", type=int, default=15)
    ap.add_argument("--rtc-mode", choices=("trained", "guided", "off"), default="trained")
    args = ap.parse_args()
    if args.warmup_steps < 0:
        ap.error("--warmup-steps must be non-negative")
    server = PI05Server(args.checkpoint, args.dataset_root, args.device, args.action_horizon, args.execution_horizon, args.rtc_mode)
    server.pool.submit(server.warmup, args.warmup_steps).result()
    app = FastAPI(title="Seer PI05 inference")

    @app.get("/health")
    def health(): return {"status": "ok", "backend": "pi05", "server_info": server.server_info}

    @app.post("/v1/pi05/action")
    def action(req: Request):
        if req.schema != "seer.model_inference.request.v1":
            raise HTTPException(400, "unsupported request schema")
        try:
            return server.pool.submit(server.infer, req, time.perf_counter()).result()
        except Exception as exc:
            raise HTTPException(500, f"inference failed: {type(exc).__name__}: {exc}") from exc

    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__": main()
