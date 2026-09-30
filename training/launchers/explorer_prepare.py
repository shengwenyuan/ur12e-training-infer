"""Validate the mounted runtime, data, normalization and one real CPU batch."""

# Heavy imports stay behind the scheduled preparation entry point.
# pylint: disable=import-outside-toplevel

import argparse
import importlib.metadata
from pathlib import Path
import platform

from training.common.identity import file_digest, write_json
from training.pi05_rtc import backend, normalization, profile


def main():
    """Prepare exactly the assets consumed by the two-GPU smoke/full run."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "dataset",
        "config",
        "norm-root",
        "checkpoint-root",
        "initialization",
        "tokenizer",
        "report",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    raw = profile.load(args.config)
    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise RuntimeError("the training runtime must use Python 3.12")
    lock = Path(__file__).parents[1] / "pi05_rtc/requirements-gpu.lock"
    if file_digest(lock) != file_digest(Path("/tmp/requirements.lock")):
        raise RuntimeError("image dependency lock differs from mounted source")
    backend.local_assets(args.tokenizer)
    backend.activate()
    import jax
    import numpy as np
    from training.common.dataset import ExportedDataset
    from training.pi05_rtc import loader, trainer

    if jax.default_backend() != "cpu":
        raise RuntimeError("preparation must use the CPU backend")
    dataset = ExportedDataset(args.dataset)
    if not args.norm_root.exists():
        normalization.compute(dataset, raw, args.norm_root)
    norm_identity = normalization.verify(dataset, raw, args.norm_root)
    config = trainer.configuration(
        raw,
        args.norm_root,
        args.checkpoint_root,
        args.initialization,
        "smoke",
        False,
    )
    batch = next(iter(loader.create(config, dataset)))
    if batch[1].shape != (raw["train"]["batch_size"], 50, 32):
        raise ValueError(f"unexpected action batch: {batch[1].shape}")
    if not all(np.isfinite(x).all() for x in jax.tree.leaves(batch)):
        raise ValueError("non-finite observation/action batch")
    report = {
        "status": "PASS",
        "initialization_restore": "NOT_RUN; required in GPU smoke",
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("jax", "flax", "torch", "lerobot", "transformers")
        },
        "dataset_sha256": dataset.identity,
        "episodes": dataset.info["total_episodes"],
        "frames": len(dataset),
        "norm_identity": norm_identity,
        "profile_sha256": file_digest(args.config),
        "dependency_lock_sha256": file_digest(lock),
        "tokenizer_sha256": file_digest(args.tokenizer),
        "algorithm_sha256": backend.algorithm_identity(),
        "action_shape": list(batch[1].shape),
        "image_shapes": {
            key: list(value.shape) for key, value in batch[0].images.items()
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, report)
    print(report)


if __name__ == "__main__":
    main()
