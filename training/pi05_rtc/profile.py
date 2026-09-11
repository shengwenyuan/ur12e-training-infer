"""Validated model-specific profiles; physical timing is configured
elsewhere.
"""

# Protocol numbers must exclude bool, which is an int subclass.
# pylint: disable=unidiomatic-typecheck


import math
from pathlib import Path
import re
import tomllib

DEFAULT = Path(__file__).parent / "config/profile.toml"


def load(path: str | Path = DEFAULT) -> dict:
    """Reject unsupported model semantics before initializing the GPU stack."""
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    with DEFAULT.open("rb") as stream:
        baseline = tomllib.load(stream)
    if set(raw) != set(baseline):
        raise ValueError("profile sections differ from the supported schema")
    for section in raw:
        if set(raw[section]) != set(baseline[section]):
            raise ValueError(f"unexpected keys in {section}")
    if raw["model"] != baseline["model"]:
        raise ValueError("pi05 model contract differs")
    rtc = dict(raw["rtc"])
    delay = rtc.pop("maximum_delay_exclusive")
    expected = dict(baseline["rtc"])
    expected.pop("maximum_delay_exclusive")
    if rtc != expected or type(delay) is not int or not 1 <= delay <= 50:
        raise ValueError("invalid training-time RTC contract")
    for section, keys in {
        "train": (
            "batch_size",
            "fsdp_devices",
            "num_train_steps",
            "log_interval",
        ),
        "checkpoint": ("save_interval", "keep_period"),
        "smoke": ("num_train_steps",),
        "norm_stats": ("batch_size",),
    }.items():
        for key in keys:
            if type(raw[section][key]) is not int or raw[section][key] < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    train = raw["train"]
    if (
        train["fsdp_devices"] not in (1, 8)
        or train["batch_size"] % train["fsdp_devices"]
    ):
        raise ValueError("batch size must divide 1 or 8 FSDP devices")
    if type(train["num_workers"]) is not int or train["num_workers"] < 0:
        raise ValueError("num_workers must be nonnegative")
    if type(train["wandb_enabled"]) is not bool:
        raise ValueError("wandb_enabled must be boolean")
    if not 0 < train["ema_decay"] < 1:
        raise ValueError("invalid EMA decay")
    _validate_optimizer_run(raw)
    return raw


def _validate_optimizer_run(raw):
    for section in ("lr_schedule", "optimizer"):
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in raw[section].values()
        ):
            raise ValueError(f"invalid {section}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", raw["run"]["name"]):
        raise ValueError("run name must be one safe path component")
    if type(raw["run"]["seed"]) is not int or raw["run"]["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
