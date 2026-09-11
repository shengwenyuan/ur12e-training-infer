"""Asset round-trip, profile construction and reproducible cloud job tests."""

# Named pytest cases define the behavior; imports keep core CI GPU-independent.
# pylint: disable=missing-function-docstring,import-outside-toplevel
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from training.common.identity import file_digest
from training.pi05_rtc import backend, profile, transforms
from training.launchers.job import render


def test_job_uses_structured_arguments_and_independent_mounts():
    settings = json.loads(
        Path("training/launchers/deployment.example.json").read_text(
            encoding="utf-8"
        )
    )
    command = [
        "python",
        "-m",
        "training.cli",
        "smoke",
        "--dataset",
        "/datasets/a b",
        "--norm-root",
        "/output/norm",
    ]
    job = render(settings, command)
    assert "'/datasets/a b'" in job["jobSpec"]["command"]
    assert job["jobSpec"]["replicas"] == 1
    assert [m["mountPath"] for m in job["datasources"]] == [
        "/datasets",
        "/weights",
        "/output",
        "/cache",
    ]
    assert [m["options"]["readOnly"] for m in job["datasources"]] == [
        True,
        True,
        False,
        False,
    ]


@pytest.mark.backend
def test_checkpoint_assets_preserve_norm_and_serving_contract(tmp_path):
    backend.activate()
    from openpi.shared import normalize
    from openpi.experiments.arx5_joint_rtc import checkpoints

    @dataclasses.dataclass
    class State:
        """Minimal checkpoint-owned parameter state."""

        params: dict
        ema_params: object = None

    stats = {
        "state": normalize.NormStats(
            mean=np.zeros(7), std=np.ones(7), q01=-np.ones(7), q99=np.ones(7)
        ),
        "actions": normalize.NormStats(
            mean=np.zeros(7), std=np.ones(7), q01=-np.ones(7), q99=np.ones(7)
        ),
    }
    source = tmp_path / "source"
    normalize.save(source, stats)
    metadata = {
        "policy_type": "training_time_rtc",
        "max_delay": 10,
        "contract": transforms.CONTRACT,
        "norm_sha256": file_digest(source / "norm_stats.json"),
    }

    class Manager:
        """Exercise actual asset callback without a GPU checkpoint run."""

        def save(self, step, items):
            assert step == 5
            (tmp_path / "assets").mkdir()
            items["assets"](tmp_path / "assets")

    loader = SimpleNamespace(
        data_config=lambda: SimpleNamespace(norm_stats=stats, asset_id="ur12e")
    )
    checkpoints.save_state(
        Manager(), State({"weight": 1}), loader, 5, policy_metadata=metadata
    )
    assert checkpoints.validate_policy_type(tmp_path / "assets") == metadata
    assert (
        file_digest(tmp_path / "assets/ur12e/norm_stats.json")
        == metadata["norm_sha256"]
    )


@pytest.mark.backend
def test_training_configuration_imports_without_loading_weights(tmp_path):
    from training.pi05_rtc.trainer import configuration

    config = configuration(
        profile.load(),
        tmp_path / "norm",
        tmp_path / "checkpoints",
        tmp_path / "base",
        "smoke",
        False,
    )
    assert config.model.action_horizon == 50 and config.model.action_dim == 32
    assert config.num_train_steps == 5 and config.wandb_enabled is False
    assert config.overwrite is False
