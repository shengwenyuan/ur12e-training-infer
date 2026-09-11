from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from etils import epath

POLICY_TYPE_FILE = "policy_type.json"
POLICY_TYPE = "training_time_rtc"
_DIGEST_DOMAIN = b"ARX5_CHECKPOINT_TREE_SHA256_V1\0"


def initialize_checkpoint_dir(*args, **kwargs):
    from openpi.training import checkpoints as base_checkpoints

    return base_checkpoints.initialize_checkpoint_dir(*args, **kwargs)


def restore_state(*args, **kwargs):
    from openpi.training import checkpoints as base_checkpoints

    return base_checkpoints.restore_state(*args, **kwargs)


def save_state(
    checkpoint_manager,
    state,
    data_loader,
    step: int,
    *,
    policy_metadata: dict[str, Any],
) -> None:
    """Save an RTC checkpoint without changing the shared checkpoint API."""
    from openpi.shared import array_typing as at
    from openpi.shared import normalize

    def save_assets(directory: epath.Path) -> None:
        data_config = data_loader.data_config()
        if data_config.norm_stats is not None and data_config.asset_id is not None:
            normalize.save(directory / data_config.asset_id, data_config.norm_stats)
        metadata = dict(policy_metadata)
        (directory / POLICY_TYPE_FILE).write_text(json.dumps(metadata, sort_keys=True))

    with at.disable_typechecking():
        train_state, params = _split_params(state)
    checkpoint_manager.save(
        step,
        {
            "assets": save_assets,
            "train_state": train_state,
            "params": {"params": params},
        },
    )


def validate_policy_type(assets_dir: epath.Path | str, expected: str = POLICY_TYPE) -> dict[str, Any]:
    metadata_path = epath.Path(assets_dir) / POLICY_TYPE_FILE
    if not metadata_path.exists():
        raise ValueError(f"checkpoint is missing required policy type metadata: {expected}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("policy_type") != expected:
        raise ValueError(
            f"checkpoint policy type {metadata.get('policy_type')!r} does not match requested mode {expected!r}"
        )
    return metadata


def checkpoint_params_sha256(checkpoint: str | Path) -> str:
    """Hash the inference checkpoint view while excluding optimizer state."""
    root = Path(checkpoint)
    if not root.is_dir():
        raise ValueError(f"checkpoint is not a directory: {root}")
    entries = sorted(
        (entry for entry in root.rglob("*") if entry.relative_to(root).parts[0] != "train_state"),
        key=lambda entry: entry.relative_to(root).as_posix(),
    )
    if any(entry.is_symlink() for entry in entries):
        raise ValueError("checkpoint must not contain symbolic links")
    files = [entry for entry in entries if entry.is_file()]
    if not files:
        raise ValueError("checkpoint directory contains no files")

    digest = hashlib.sha256(_DIGEST_DOMAIN)
    for file in files:
        relative = file.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file.stat().st_size.to_bytes(8, "big"))
        with file.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _split_params(state) -> tuple[Any, Any]:
    if state.ema_params is not None:
        return dataclasses.replace(state, ema_params=None), state.ema_params
    return dataclasses.replace(state, params={}), state.params
