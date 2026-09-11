"""Load the pinned vendor backend only inside training/model-server
processes.
"""

from pathlib import Path
import os
import sys

from training.common.identity import digest, file_digest


def activate():
    """Make preserved upstream packages importable without a global install."""
    root = Path(__file__).parent / "vendor" / "openpi"
    for path in (root, root / "src", root / "packages/openpi-client/src"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def local_assets(tokenizer: str | Path | None = None) -> Path:
    """Require mounted model assets and disable Hub network fallbacks."""
    configured = tokenizer or os.environ.get("UR12E_TOKENIZER_PATH")
    if configured is None or not Path(configured).is_file():
        raise ValueError(
            "provide --tokenizer or UR12E_TOKENIZER_PATH as a mounted file"
        )
    path = Path(configured).resolve()
    os.environ["UR12E_TOKENIZER_PATH"] = str(path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    return path


def algorithm_identity() -> str:
    """Bind checkpoints to the numerical implementation and physical adapter."""

    root = Path(__file__).parent
    names = [
        "transforms.py",
        "vendor/openpi/src/openpi/models/pi0.py",
        "vendor/openpi/src/openpi/models/pi0_config.py",
        "vendor/openpi/src/openpi/experiments/arx5_joint_rtc/model.py",
        "vendor/openpi/src/openpi/experiments/arx5_joint_rtc/model_config.py",
        "vendor/openpi/src/openpi/experiments/arx5_joint_rtc/gemma.py",
        "vendor/openpi/src/openpi/shared/normalize.py",
        "vendor/openpi/src/openpi/transforms.py",
    ]
    return digest({name: file_digest(root / name) for name in names})
