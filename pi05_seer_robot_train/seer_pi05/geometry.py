"""Geometry conversions for the Seer dual-arm end-effector dataset."""

from __future__ import annotations

import numpy as np


def _unit_quaternion_np(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if not np.isfinite(quaternion).all() or np.any(norm < 1e-8):
        raise ValueError("Seer end-effector quaternions must be finite and non-zero")
    return quaternion / norm


def _quat_multiply_np(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lv, lw = left[..., :3], left[..., 3:4]
    rv, rw = right[..., :3], right[..., 3:4]
    return np.concatenate(
        (lw * rv + rw * lv + np.cross(lv, rv), lw * rw - np.sum(lv * rv, axis=-1, keepdims=True)),
        axis=-1,
    )


def _quat_rotate_np(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    qv = quaternion[..., :3]
    first = np.cross(qv, vector)
    return vector + 2.0 * (quaternion[..., 3:4] * first + np.cross(qv, first))


def _rotvec_to_quat_np(rotvec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half_angle = angle * 0.5
    scale = np.divide(
        np.sin(half_angle),
        angle,
        out=np.full_like(angle, 0.5),
        where=angle > 1e-8,
    )
    return _unit_quaternion_np(np.concatenate((rotvec * scale, np.cos(half_angle)), axis=-1))


def _quat_to_rotvec_np(quaternion: np.ndarray) -> np.ndarray:
    quaternion = _unit_quaternion_np(quaternion)
    vector = quaternion[..., :3]
    first_nonzero = np.where(
        vector[..., 0:1] != 0,
        vector[..., 0:1],
        np.where(vector[..., 1:2] != 0, vector[..., 1:2], vector[..., 2:3]),
    )
    flip = (quaternion[..., 3:4] < 0) | ((quaternion[..., 3:4] == 0) & (first_nonzero < 0))
    quaternion = np.where(flip, -quaternion, quaternion)
    vector = quaternion[..., :3]
    sin_half = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, np.clip(quaternion[..., 3:4], 0.0, 1.0))
    scale = np.divide(angle, sin_half, out=np.full_like(angle, 2.0), where=sin_half > 1e-8)
    return vector * scale


def compact_state(state: np.ndarray) -> np.ndarray:
    """Map a 64D robot state to joints, grippers, and two 6D poses."""
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] == 30:
        return state.copy()
    if state.shape[-1] != 64:
        raise ValueError(f"Expected a 64D or compact 30D state, got {state.shape}")

    def pose(start: int) -> np.ndarray:
        position = state[..., start : start + 3]
        rotation = _quat_to_rotvec_np(state[..., start + 3 : start + 7])
        return np.concatenate((position, rotation), axis=-1)

    compact = np.concatenate((state[..., :16], state[..., 62:64], pose(48), pose(55)), axis=-1)
    if not np.isfinite(compact).all():
        raise ValueError("Seer state contains non-finite values")
    return compact.astype(np.float32, copy=False)


def compact_action(action: np.ndarray) -> np.ndarray:
    """Map a 16D absolute EEF action to two 6D poses and two grippers."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] == 14:
        return action.copy()
    if action.shape[-1] != 16:
        raise ValueError(f"Expected a 16D or compact 14D action, got {action.shape}")

    def pose(start: int) -> np.ndarray:
        position = action[..., start : start + 3]
        rotation = _quat_to_rotvec_np(action[..., start + 3 : start + 7])
        return np.concatenate((position, rotation), axis=-1)

    compact = np.concatenate((pose(0), pose(7), action[..., 14:16]), axis=-1)
    if not np.isfinite(compact).all():
        raise ValueError("Seer action contains non-finite values")
    return compact.astype(np.float32, copy=False)


def relative_eef_actions(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Express each absolute target pose in the corresponding current EEF frame."""
    state = compact_state(state)
    action = compact_action(action)
    converted = action.copy()
    for action_start, state_start in ((0, 18), (6, 24)):
        current = state[..., state_start : state_start + 6]
        target = action[..., action_start : action_start + 6]
        while current.ndim < target.ndim:
            current = np.expand_dims(current, axis=-2)
        current_quat = _rotvec_to_quat_np(current[..., 3:6])
        target_quat = _rotvec_to_quat_np(target[..., 3:6])
        inverse_current = current_quat.copy()
        inverse_current[..., :3] *= -1
        relative_position = _quat_rotate_np(inverse_current, target[..., :3] - current[..., :3])
        relative_quat = _quat_multiply_np(inverse_current, target_quat)
        converted[..., action_start : action_start + 3] = relative_position
        converted[..., action_start + 3 : action_start + 6] = _quat_to_rotvec_np(relative_quat)
    return converted


def absolute_eef_actions(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Restore relative EEF targets to absolute poses in the robot base frame."""
    state = compact_state(state)
    action = compact_action(action)
    converted = action.copy()
    for action_start, state_start in ((0, 18), (6, 24)):
        current = state[..., state_start : state_start + 6]
        relative = action[..., action_start : action_start + 6]
        while current.ndim < relative.ndim:
            current = np.expand_dims(current, axis=-2)
        current_quat = _rotvec_to_quat_np(current[..., 3:6])
        relative_quat = _rotvec_to_quat_np(relative[..., 3:6])
        absolute_position = current[..., :3] + _quat_rotate_np(current_quat, relative[..., :3])
        absolute_quat = _quat_multiply_np(current_quat, relative_quat)
        converted[..., action_start : action_start + 3] = absolute_position
        converted[..., action_start + 3 : action_start + 6] = _quat_to_rotvec_np(absolute_quat)
    return converted


def relative_action_windows(state: np.ndarray, action: np.ndarray, chunk_size: int) -> np.ndarray:
    """Create one relative action chunk per frame, repeating only within the episode."""
    state = compact_state(state)
    action = compact_action(action)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if state.ndim != 2 or action.ndim != 2 or len(state) != len(action) or not len(state):
        raise ValueError("An episode must contain matching, non-empty 2D state and action arrays")

    indices = np.minimum(np.arange(len(state))[:, None] + np.arange(chunk_size), len(state) - 1)
    absolute_windows = action[indices]
    return relative_eef_actions(state[:, None, :], absolute_windows)


def vector_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"Statistics require a finite, non-empty 2D array, got {values.shape}")
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(np.float32).tolist(),
        "count": [int(len(values))] * values.shape[-1],
    }
