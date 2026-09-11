"""Construct UR runs using the inherited model, optimizer and checkpoint
code.
"""

# Optional backend imports keep the robot/CLI dependency boundary lightweight.
# pylint: disable=import-outside-toplevel


from dataclasses import dataclass, replace
from pathlib import Path
import json

from training.common.identity import file_digest
from training.pi05_rtc import backend, normalization, transforms


def configuration(profile, norm, checkpoint_root, initialization, mode, resume):
    """Resolve backend configuration only after lightweight validation."""
    backend.activate()
    from openpi.training import (
        config,
        optimizer,
        weight_loaders,
    )
    from openpi.shared import (
        normalize,
    )
    from openpi.experiments.arx5_joint_rtc.model_config import (
        Pi05RtcConfig,
    )

    @dataclass(frozen=True)
    class Data:
        """Only the adapter, not the upstream global registry, selects data."""

        def create(self, _assets, model):
            """Return UR transforms and the verified normalization assets."""
            return config.DataConfig(
                repo_id="ur12e",
                asset_id="ur12e",
                norm_stats=normalize.load(Path(norm)),
                use_quantile_norm=True,
                model_transforms=config.ModelTransformFactory()(model),
            )

    train = profile["train"]
    run_profile = profile["run"]
    return config.TrainConfig(
        name="pi05_rtc",
        exp_name=run_profile["name"],
        project_name=run_profile["project"],
        seed=run_profile["seed"],
        model=Pi05RtcConfig(
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            discrete_state_input=True,
            max_delay=profile["rtc"]["maximum_delay_exclusive"],
        ),
        data=Data(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(initialization)
        ),
        checkpoint_base_dir=str(checkpoint_root),
        assets_base_dir=str(norm),
        batch_size=train["batch_size"],
        fsdp_devices=train["fsdp_devices"],
        num_workers=train["num_workers"],
        num_train_steps=(
            profile["smoke"]["num_train_steps"]
            if mode == "smoke"
            else train["num_train_steps"]
        ),
        log_interval=1 if mode == "smoke" else train["log_interval"],
        ema_decay=train["ema_decay"],
        wandb_enabled=False if mode == "smoke" else train["wandb_enabled"],
        lr_schedule=optimizer.CosineDecaySchedule(**profile["lr_schedule"]),
        optimizer=optimizer.AdamW(**profile["optimizer"]),
        save_interval=profile["checkpoint"]["save_interval"],
        keep_period=profile["checkpoint"]["keep_period"],
        overwrite=False,
        resume=resume,
        policy_metadata={
            "policy_type": "training_time_rtc",
            "max_delay": profile["rtc"]["maximum_delay_exclusive"],
        },
    )


def run(
    dataset, profile, norm, checkpoint_root, initialization, mode, resume=False
):
    """Validate GPU and data identity before training can mutate a
    checkpoint.
    """
    tokenizer_sha256 = file_digest(backend.local_assets())
    if not Path(initialization).is_dir():
        raise ValueError(
            "initialization must be a mounted local params directory"
        )
    identity = normalization.verify(dataset, profile, norm)
    backend.activate()
    import jax
    from training.pi05_rtc.train_loop import (
        run_training,
    )

    if (
        jax.default_backend() != "gpu"
        or jax.device_count() != profile["train"]["fsdp_devices"]
    ):
        raise RuntimeError("selected GPU/FSDP device count unavailable")
    if mode == "smoke" and resume:
        raise ValueError("smoke cannot resume")
    config = configuration(
        profile, norm, checkpoint_root, initialization, mode, resume
    )
    config = replace(
        config,
        policy_metadata=config.policy_metadata
        | {
            "contract": transforms.CONTRACT,
            "norm_sha256": file_digest(Path(norm) / "norm_stats.json"),
            "tokenizer_sha256": tokenizer_sha256,
            "algorithm_sha256": backend.algorithm_identity(),
        },
    )
    run_dir = Path(config.checkpoint_dir)
    run_identity = {
        "algorithm_sha256": backend.algorithm_identity(),
        "normalization": identity,
        "tokenizer_sha256": tokenizer_sha256,
        "contract": transforms.CONTRACT,
        "profile": profile,
    }
    if resume:
        previous = json.loads(
            (run_dir / "run.json").read_text(encoding="utf-8")
        )
        # Extending the step budget is allowed; data and optimizer must match.
        previous["profile"]["train"]["num_train_steps"] = profile["train"][
            "num_train_steps"
        ]
        if previous != run_identity:
            raise ValueError("resume identity mismatch")
    elif run_dir.exists():
        raise FileExistsError("choose a new run name; overwrite is disabled")
    run_training(
        config,
        dataset,
        save_checkpoints=mode != "smoke",
        run_identity=run_identity,
    )
