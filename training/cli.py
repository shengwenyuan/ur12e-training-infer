"""Model-selected training entry point; dry-runs never allocate a GPU."""

# Optional backend imports keep the robot/CLI dependency boundary lightweight.
# pylint: disable=import-outside-toplevel


import argparse
import json
from pathlib import Path

from training.backends import BACKENDS, component


def main(argv=None):
    """Validate data, compute fresh statistics, or run the selected backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("validate", "norm-stats", "smoke", "train")
    )
    parser.add_argument("--model", choices=tuple(BACKENDS), default="pi05_rtc")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--norm-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--initialization", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--allow-simulated", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    profile = component(args.model, "profile")
    raw = profile.load(args.config or profile.DEFAULT)
    if args.resume and args.command != "train":
        parser.error("--resume requires train")
    for key in (
        ("norm_root",)
        if args.command == "norm-stats"
        else (
            ("norm_root", "checkpoint_root", "initialization")
            if args.command in ("train", "smoke")
            else ()
        )
    ):
        if getattr(args, key) is None:
            parser.error(f'--{key.replace("_","-")} is required')
    summary = vars(args) | {"config": raw}
    if args.dry_run:
        print(json.dumps(summary, default=str, indent=2))
        return
    from training.common.dataset import (
        ExportedDataset,
    )

    dataset = ExportedDataset(
        args.dataset, allow_simulated=args.allow_simulated
    )
    if args.command == "norm-stats":
        compute = component(args.model, "normalization").compute

        summary["normalization"] = compute(dataset, raw, args.norm_root)
    elif args.command in ("train", "smoke"):
        run = component(args.model, "trainer").run
        local_assets = component(args.model, "backend").local_assets

        local_assets(args.tokenizer)
        if not args.initialization.is_dir():
            parser.error(
                "--initialization must be a mounted local params directory"
            )
        run(
            dataset,
            raw,
            args.norm_root,
            args.checkpoint_root,
            args.initialization,
            args.command,
            args.resume,
        )
    summary.update(
        dataset_identity=dataset.identity,
        frames=len(dataset),
        episodes=dataset.info["total_episodes"],
    )
    print(json.dumps(summary, default=str, indent=2))


if __name__ == "__main__":
    main()
