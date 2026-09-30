"""Check Slurm launch boundaries without containers, GPUs or submission."""

# Explicit pytest names document the launch contract.
# pylint: disable=missing-function-docstring,redefined-outer-name

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from training.pi05_rtc import profile


@pytest.fixture
def launch_environment(tmp_path):
    executable = tmp_path / "apptainer"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'args': sys.argv[1:], 'env': "
        "{k: v for k, v in os.environ.items() "
        "if k.startswith('APPTAINERENV_')}}))\n"
    )
    executable.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}")
    for name in ("IMAGE", "DATASET", "PRETRAINED", "OUTPUT", "CACHE"):
        path = tmp_path / f"mount {name}"
        path.mkdir()
        env[f"UR12E_{name}"] = str(path)
    env.update(SLURM_JOB_ID="test", CUDA_VISIBLE_DEVICES="3,5")
    return env


@pytest.mark.parametrize("mode", ["prepare", "smoke", "train"])
def test_explorer_mounts_and_visibility(launch_environment, mode):
    result = subprocess.run(
        ["bash", "training/launchers/explorer.sh", mode],
        env=launch_environment,
        capture_output=True,
        text=True,
        check=True,
    )
    captured = json.loads(result.stdout.splitlines()[-1])
    args, env = captured["args"], captured["env"]
    assert "--no-mount" in args and "tmp" in args
    assert launch_environment["UR12E_DATASET"] + ":/data:ro" in args
    assert env["APPTAINERENV_WANDB_MODE"] == (
        "offline" if mode == "train" else "disabled"
    )
    if mode == "prepare":
        assert "--nv" not in args
        assert env["APPTAINERENV_JAX_PLATFORMS"] == "cpu"
    else:
        assert "--nv" in args
        assert env["APPTAINERENV_CUDA_VISIBLE_DEVICES"] == "3,5"
        assert env["APPTAINERENV_JAX_PLATFORMS"] == "cuda"


def test_explorer_refuses_login_node_execution(launch_environment):
    launch_environment.pop("SLURM_JOB_ID")
    result = subprocess.run(
        ["bash", "training/launchers/explorer.sh", "smoke"],
        env=launch_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Slurm allocation" in result.stderr


def test_aligned_profile_uses_two_devices_and_zero_prefix():
    path = Path("training/pi05_rtc/config/ur12e_pick_place_2gpu.toml")
    raw = profile.load(path)
    assert raw["train"]["fsdp_devices"] == 2
    assert raw["train"]["batch_size"] == 16
    assert raw["rtc"]["maximum_delay_exclusive"] == 1


def test_external_profile_and_online_wandb(launch_environment, tmp_path):
    external = tmp_path / "profile.toml"
    external.write_text("# external configuration")
    secret = tmp_path / "test-key"
    secret.write_text("dummy-test-key\n")
    launch_environment.update(
        UR12E_PROFILE=str(external),
        UR12E_WANDB_MODE="online",
        UR12E_WANDB_ENTITY="test-team",
        UR12E_WANDB_KEY_FILE=str(secret),
    )
    result = subprocess.run(
        ["bash", "training/launchers/explorer.sh", "train", "--resume"],
        env=launch_environment,
        capture_output=True,
        text=True,
        check=True,
    )
    captured = json.loads(result.stdout.splitlines()[-1])
    assert str(external) + ":/config/profile.toml:ro" in captured["args"]
    assert "--resume" in captured["args"]
    assert "dummy-test-key" not in " ".join(captured["args"])
    assert captured["env"]["APPTAINERENV_WANDB_MODE"] == "online"
    assert captured["env"]["APPTAINERENV_WANDB_ENTITY"] == "test-team"
    assert captured["env"]["APPTAINERENV_WANDB_API_KEY"] == "dummy-test-key"


def test_online_wandb_requires_credential(launch_environment):
    launch_environment.update(
        UR12E_WANDB_MODE="online", UR12E_WANDB_ENTITY="test"
    )
    launch_environment.pop("UR12E_WANDB_KEY_FILE", None)
    result = subprocess.run(
        ["bash", "training/launchers/explorer.sh", "train"],
        env=launch_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "private credential file" in result.stderr
