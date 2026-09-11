"""Fresh normalization assets bound to actual data and action semantics."""

# Optional backend imports keep the robot/CLI dependency boundary lightweight.
# pylint: disable=import-outside-toplevel


import json
from pathlib import Path
import numpy as np

from training.common.identity import digest, file_digest, write_json
from training.pi05_rtc import backend, transforms


def identity(dataset, profile):
    """Keep optimizer changes separate from the data normalization contract."""
    return {
        "dataset_sha256": dataset.identity,
        "contract": transforms.CONTRACT,
        "model": profile["model"],
        "rtc": profile["rtc"],
    }


def compute(dataset, profile, output):
    """Use inherited streaming statistics on exactly the training projection."""
    backend.activate()
    from openpi.shared import (
        normalize,
    )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    batch_size = profile["norm_stats"]["batch_size"]
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        states = dataset.states[start:stop]
        actions = np.stack(
            [
                transforms.relative(dataset.window(i, 50), dataset.states[i])
                for i in range(start, stop)
            ]
        )
        stats["state"].update(states)
        stats["actions"].update(actions)
    normalize.save(
        output, {key: value.get_statistics() for key, value in stats.items()}
    )
    marker = identity(dataset, profile) | {
        "norm_sha256": file_digest(output / "norm_stats.json"),
        "samples": len(dataset),
    }
    write_json(output / "validation.json", marker)
    return marker


def verify(dataset, profile, output):
    """Detect changed datasets, transforms, model semantics or asset
    contents.
    """
    output = Path(output)
    marker = json.loads(
        (output / "validation.json").read_text(encoding="utf-8")
    )
    expected = identity(dataset, profile)
    if any(marker.get(k) != v for k, v in expected.items()) or marker.get(
        "norm_sha256"
    ) != file_digest(output / "norm_stats.json"):
        raise ValueError("normalization identity mismatch")
    return digest(marker)
