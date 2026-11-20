from pi05_seer_robot_train.pi05_robot_inference_server import PI05Server, _rtc_context_for_task


class _ResetCounter:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class _PolicyResetCounter(_ResetCounter):
    def __init__(self):
        super().__init__()
        self.rtc_resets = 0

    def init_rtc_processor(self):
        self.rtc_resets += 1


def test_task_change_resets_inference_state_once_per_task():
    server = object.__new__(PI05Server)
    server.last_task = None
    server.policy = _PolicyResetCounter()
    server.pre = _ResetCounter()
    server.post = _ResetCounter()

    assert server._reset_for_task("Fold the left flap of the box.")
    server.last_task = "Fold the left flap of the box."
    assert not server._reset_for_task("Fold the left flap of the box.")
    assert server._reset_for_task("Fold the right flap of the box.")
    server.last_task = "Fold the right flap of the box."

    assert server.last_task == "Fold the right flap of the box."
    assert server.policy.resets == 2
    assert server.policy.rtc_resets == 2
    assert server.pre.resets == 2
    assert server.post.resets == 2


def test_task_change_discards_previous_rtc_prefix():
    rtc = {
        "enabled": True,
        "mode": "trained",
        "inference_delay": 6,
        "prev_chunk_left_over": [[1.0] * 14],
    }

    changed = _rtc_context_for_task(rtc, task_changed=True)
    unchanged = _rtc_context_for_task(rtc, task_changed=False)

    assert changed["inference_delay"] == 0
    assert changed["prev_chunk_left_over"] is None
    assert changed["enabled"] is True
    assert changed["mode"] == "trained"
    assert unchanged is rtc
    assert unchanged["inference_delay"] == 6
