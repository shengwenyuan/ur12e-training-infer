import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_config

from . import gemma
from . import model
from . import model_config


def test_training_conditioning_uses_clean_prefix_and_noisy_postfix():
    actions = jnp.arange(2 * 5 * 3, dtype=jnp.float32).reshape(2, 5, 3)
    noise = -actions
    time = jnp.array([0.25, 0.75])
    delay = jnp.array([0, 3])

    x_t, target, token_time, mask = model.prepare_training_conditioning(actions, noise, time, delay)

    np.testing.assert_array_equal(mask, [[False] * 5, [True, True, True, False, False]])
    np.testing.assert_array_equal(x_t[1, :3], actions[1, :3])
    np.testing.assert_array_equal(token_time[1, :3], 0.0)
    np.testing.assert_allclose(x_t[0], time[0] * noise[0] + (1 - time[0]) * actions[0])
    np.testing.assert_array_equal(target, noise - actions)


def test_postfix_loss_masks_and_normalizes_prefix():
    predicted = jnp.ones((2, 5, 3))
    target = jnp.zeros_like(predicted)
    mask = model.prefix_mask(jnp.array([0, 3]), 5)
    loss = model.postfix_only_loss(predicted, target, mask)

    np.testing.assert_array_equal(loss[1, :3], 0.0)
    assert float(jnp.mean(loss)) == pytest.approx(1.0)


def test_hard_prefix_update_never_integrates_committed_actions():
    x_t = jnp.zeros((1, 5, 2))
    velocity = jnp.ones_like(x_t)
    prefix = jnp.full_like(x_t, 7.0)
    mask = model.prefix_mask(jnp.array([2]), 5)

    updated = model.hard_prefix_update(x_t, velocity, prefix, mask, -0.1)

    np.testing.assert_array_equal(updated[:, :2], prefix[:, :2])
    np.testing.assert_allclose(updated[:, 2:], -0.1)


def test_rtc_config_has_identical_parameter_tree_to_pi05():
    common = {
        "action_horizon": 5,
        "action_dim": 3,
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
    }
    vanilla = nnx.eval_shape(pi0_config.Pi0Config(pi05=True, **common).create, jax.random.key(0))
    rtc = nnx.eval_shape(model_config.Pi05RtcConfig(max_delay=3, **common).create, jax.random.key(0))

    def shapes(value):
        return [(path, variable.value.shape) for path, variable in nnx.state(value, nnx.Param).flat_state().items()]

    assert shapes(rtc) == shapes(vanilla)


def test_rtc_config_rejects_invalid_delay():
    with pytest.raises(ValueError, match="max_delay"):
        model_config.Pi05RtcConfig(action_horizon=5, max_delay=6)


def test_local_adarms_accepts_scalar_or_per_token_condition_without_new_parameters():
    module = gemma.RMSNorm()
    x = jnp.ones((2, 5, 8))
    scalar_cond = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
    token_cond = jnp.broadcast_to(scalar_cond[:, None, :], x.shape)
    variables = module.init(jax.random.key(0), x, scalar_cond)

    scalar_out, scalar_gate = module.apply(variables, x, scalar_cond)
    token_out, token_gate = module.apply(variables, x, token_cond)

    assert scalar_out.shape == token_out.shape == x.shape
    assert scalar_gate.shape == (2, 1, 8)
    assert token_gate.shape == x.shape
    np.testing.assert_allclose(scalar_out, token_out)
    np.testing.assert_allclose(jnp.broadcast_to(scalar_gate, x.shape), token_gate)


def test_compute_loss_masks_conditioned_prefix():
    config = model_config.Pi05RtcConfig(
        action_horizon=5,
        action_dim=3,
        max_delay=4,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    rtc_model = config.create(jax.random.key(0))
    observation, actions = config.fake_obs(2), config.fake_act(2)

    loss = rtc_model.compute_loss(jax.random.key(1), observation, actions, delay=jnp.array([0, 3]))

    assert loss.shape == (2, 5)
    np.testing.assert_array_equal(loss[1, :3], 0.0)
    assert bool(jnp.all(jnp.isfinite(loss)))


def test_sample_actions_preserves_committed_prefix():
    config = model_config.Pi05RtcConfig(
        action_horizon=5,
        action_dim=3,
        max_delay=4,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    rtc_model = config.create(jax.random.key(0))
    observation = config.fake_obs(2)
    action_prefix = jnp.arange(30, dtype=jnp.float32).reshape(2, 5, 3)
    delay = jnp.array([0, 3])

    actions = rtc_model.sample_actions(
        jax.random.key(1),
        observation,
        action_prefix=action_prefix,
        delay=delay,
        num_steps=2,
    )

    assert actions.shape == action_prefix.shape
    np.testing.assert_array_equal(actions[1, :3], action_prefix[1, :3])
    assert bool(jnp.all(jnp.isfinite(actions)))
