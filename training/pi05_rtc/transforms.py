"""UR12e physical units and model slots shared by training and serving."""

import numpy as np

CAMERAS = {
    "third_left": "base_0_rgb",
    "wrist": "left_wrist_0_rgb",
    "third_right": "right_wrist_0_rgb",
}
CONTRACT = {
    "model": "pi05_rtc",
    "physical_dim": 7,
    "action_dim": 32,
    "horizon": 50,
    "joint_units": "rad",
    "gripper": "absolute_raw_0_255",
    "window": "current_row_end_clamped",
    "time": "retained_row_steps",
    "cameras": CAMERAS,
}


def relative(actions, state):
    """Subtract the current observation anchor from every joint target."""
    result = np.asarray(actions, dtype=np.float32).copy()
    result[..., :6] -= np.asarray(state)[..., :6]
    return result


def absolute(actions, state):
    """Invert deltas against the same observation, preserving gripper units."""
    result = np.asarray(actions, dtype=np.float32)[..., :7].copy()
    result[..., :6] += np.asarray(state)[..., :6]
    return result


def inputs(state, images, prompt, actions=None):
    """Map physical camera roles without Aloha joint or gripper conversions."""
    if set(images) != set(CAMERAS):
        raise ValueError("all three UR12e camera roles are required")
    mapped = {}
    for role, slot in CAMERAS.items():
        value = np.asarray(images[role])
        if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
            raise ValueError("server images must be HWC uint8 RGB")
        mapped[slot] = value
    state = np.asarray(state, dtype=np.float32)
    if (
        state.shape != (7,)
        or not np.isfinite(state).all()
        or not 0 <= state[6] <= 255
    ):
        raise ValueError("state must be seven finite physical dimensions")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("task prompt required")
    result = {
        "state": state,
        "image": mapped,
        "image_mask": {key: True for key in mapped},
        "prompt": prompt,
    }
    if actions is not None:
        result["actions"] = relative(actions, state)
    return result
