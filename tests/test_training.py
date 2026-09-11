"""Data and model-contract tests independent of physical hardware."""

# Pytest fixture names and explicit case names supply the test interface.
# pylint: disable=missing-function-docstring,redefined-outer-name,import-outside-toplevel,duplicate-code


import json
import os
import numpy as np
import pytest
from training.pi05_rtc import profile, transforms, normalization, backend
from training.common.dataset import ExportedDataset


def test_delta_uses_observation_anchor_and_keeps_gripper():
    state = np.arange(7, dtype=np.float32)
    actions = np.stack([state + 2, state + 4])
    actions[:, 6] = [50, 200]
    result = transforms.relative(actions, state)
    np.testing.assert_array_equal(result[:, :6], [[2] * 6, [4] * 6])
    np.testing.assert_array_equal(result[:, 6], [50, 200])
    np.testing.assert_array_equal(transforms.absolute(result, state), actions)


def test_profile_preserves_rtc_contract():
    raw = profile.load()
    assert raw["model"]["action_dim"] == 32
    assert raw["model"]["action_horizon"] == 50
    assert raw["rtc"]["maximum_delay_exclusive"] == 10
    assert raw["rtc"]["loss"] == "postfix_only"


def test_camera_mapping_is_explicit():
    images = {
        role: np.full((10, 12, 3), i, dtype=np.uint8)
        for i, role in enumerate(transforms.CAMERAS)
    }
    result = transforms.inputs(np.zeros(7), images, "task")
    for role, slot in transforms.CAMERAS.items():
        np.testing.assert_array_equal(result["image"][slot], images[role])


@pytest.fixture
def dataset():
    path = os.environ.get("UR12E_TEST_DATASET")
    if not path:
        pytest.skip("set UR12E_TEST_DATASET to an official exporter fixture")
    return ExportedDataset(path, allow_simulated=True)


@pytest.mark.dataset
def test_official_v3_all_frames_and_three_cameras(dataset):
    reader = dataset.official()
    for i in range(len(dataset)):
        row = reader[i]
        np.testing.assert_array_equal(
            np.asarray(row["observation.state"]), dataset.states[i]
        )
        assert row["task"]
        for role in transforms.CAMERAS:
            image = np.asarray(row[f"observation.images.{role}"])
            assert image.shape == (3, 480, 640) and np.isfinite(image).all()


@pytest.mark.dataset
def test_window_current_row_terminal_repeat_and_no_cross_episode(dataset):
    for i in range(len(dataset)):
        window = dataset.window(i, 50)
        for k in (0, 1, 49):
            j = min(i + k, int(dataset.ends[i]))
            np.testing.assert_array_equal(window[k], dataset.actions[j])
            assert dataset.episodes[i] == dataset.episodes[j]


@pytest.mark.dataset
def test_normalization_identity_and_independent_numeric_mean(dataset, tmp_path):
    raw = profile.load()
    output = tmp_path / "norm"
    normalization.compute(dataset, raw, output)
    assert normalization.verify(dataset, raw, output)
    values = np.concatenate(
        [
            transforms.relative(dataset.window(i, 50), dataset.states[i])
            for i in range(len(dataset))
        ]
    )
    result = json.loads((output / "norm_stats.json").read_text())["norm_stats"][
        "actions"
    ]
    np.testing.assert_allclose(
        result["mean"], values.mean(axis=0), rtol=1e-4, atol=1e-4
    )
    raw["rtc"]["maximum_delay_exclusive"] = 9
    with pytest.raises(ValueError, match="identity"):
        normalization.verify(dataset, raw, output)


@pytest.mark.backend
def test_imported_rtc_primitives_match_independent_equations():
    backend.activate()
    import jax.numpy as jnp
    from openpi.experiments.arx5_joint_rtc import model

    actions = jnp.arange(30, dtype=jnp.float32).reshape(2, 5, 3)
    noise = -actions
    t = jnp.array([0.25, 0.75])
    delay = jnp.array([0, 3])
    mixed, target, tokens, mask = model.prepare_training_conditioning(
        actions, noise, t, delay
    )
    expected = np.asarray(t)[:, None, None] * np.asarray(noise) + (
        1 - np.asarray(t)[:, None, None]
    ) * np.asarray(actions)
    expected[1, :3] = np.asarray(actions)[1, :3]
    np.testing.assert_allclose(mixed, expected)
    np.testing.assert_allclose(target, np.asarray(noise - actions))
    np.testing.assert_array_equal(tokens[1, :3], 0)
    updated = model.hard_prefix_update(
        mixed, jnp.ones_like(mixed), actions, mask, -0.1
    )
    np.testing.assert_array_equal(updated[1, :3], actions[1, :3])
