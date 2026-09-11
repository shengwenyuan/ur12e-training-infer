"""Training-time action conditioning for the isolated ARX5 π0.5 experiment."""

from __future__ import annotations

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import siglip as _siglip
from openpi.shared import array_typing as at

from . import gemma as _gemma
from . import model_config


def prefix_mask(delay: at.Int[at.Array, " b"], action_horizon: int) -> at.Bool[at.Array, "b ah"]:
    return jnp.arange(action_horizon)[None, :] < delay[:, None]


def prepare_training_conditioning(actions, noise, time, delay):
    mask = prefix_mask(delay, actions.shape[-2])
    token_time = jnp.where(mask, 0.0, time[:, None])
    x_t = token_time[..., None] * noise + (1.0 - token_time[..., None]) * actions
    return x_t, noise - actions, token_time, mask


def postfix_only_loss(predicted_velocity, target_velocity, conditioned_prefix):
    token_loss = jnp.mean(jnp.square(predicted_velocity - target_velocity), axis=-1)
    postfix = jnp.logical_not(conditioned_prefix)
    return token_loss * postfix / jnp.maximum(jnp.mean(postfix), 1e-8)


def hard_prefix_update(x_t, velocity, action_prefix, conditioned_prefix, dt):
    updated = x_t + dt * velocity
    return jnp.where(conditioned_prefix[..., None], action_prefix, updated)


class Pi05ActionPrefixModel(pi0.Pi0):
    """π0.5 with per-action timesteps and exact action-prefix generation."""

    def __init__(self, config: model_config.Pi05RtcConfig, rngs: nnx.Rngs):
        _model.BaseModel.__init__(self, config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = True
        self.max_delay = config.max_delay
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        self.deterministic = True

    def embed_suffix(self, obs: _model.Observation, noisy_actions: _model.Actions, timestep):
        del obs
        if timestep.ndim != 2 or timestep.shape != noisy_actions.shape[:2]:
            raise ValueError(
                f"training-time RTC expects per-token timesteps {noisy_actions.shape[:2]}, got {timestep.shape}"
            )
        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = pi0.posemb_sincos(
            timestep.reshape(-1),
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
        ).reshape(*timestep.shape, self.action_in_proj.out_features)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        adarms_cond = nnx.swish(time_emb)
        input_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
        ar_mask = jnp.array([True] + ([False] * (self.action_horizon - 1)))
        return action_tokens, input_mask, ar_mask, adarms_cond

    def _full_velocity(self, observation, x_t, token_time):
        prefix_tokens, prefix_mask_, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, token_time)
        input_mask = jnp.concatenate([prefix_mask_, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        delay: at.Int[at.Array, " b"] | None = None,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_size = actions.shape[0]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (batch_size,)) * 0.999 + 0.001
        if delay is None:
            delay = jax.random.randint(delay_rng, (batch_size,), 0, self.max_delay)
        x_t, target, token_time, mask = prepare_training_conditioning(actions, noise, time, delay)
        velocity = self._full_velocity(observation, x_t, token_time)
        return postfix_only_loss(velocity, target, mask)

    def _cached_velocity(self, observation, x_t, token_time, prefix_mask_, kv_cache):
        batch_size = x_t.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, token_time)
        suffix_mask_self = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        suffix_mask_prefix = einops.repeat(prefix_mask_, "b p -> b s p", s=suffix_tokens.shape[1])
        full_mask = jnp.concatenate([suffix_mask_prefix, suffix_mask_self], axis=-1)
        positions = jnp.sum(prefix_mask_, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        return self.action_out_proj(suffix_out[:, -self.action_horizon :]).reshape(
            batch_size, self.action_horizon, self.action_dim
        )

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        action_prefix: _model.Actions | None = None,
        delay: at.Int[at.Array, " b"] | None = None,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        if (action_prefix is None) != (delay is None):
            raise ValueError("action_prefix and delay must be provided together")
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        expected_shape = (batch_size, self.action_horizon, self.action_dim)
        if noise is None:
            noise = jax.random.normal(rng, expected_shape)
        elif noise.shape != expected_shape:
            raise ValueError(f"noise must have shape {expected_shape}, got {noise.shape}")
        if action_prefix is None:
            action_prefix = jnp.zeros_like(noise)
        elif action_prefix.shape != expected_shape:
            raise ValueError(f"action_prefix must have shape {expected_shape}, got {action_prefix.shape}")
        if delay is None:
            delay = jnp.zeros((batch_size,), dtype=jnp.int32)
        elif delay.shape != (batch_size,):
            raise ValueError(f"delay must have shape {(batch_size,)}, got {delay.shape}")
        mask = prefix_mask(delay, self.action_horizon)

        prefix_tokens, prefix_mask_, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask_, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask_, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        dt = -1.0 / num_steps

        def step(carry):
            x_t, time = carry
            x_t = jnp.where(mask[..., None], action_prefix, x_t)
            token_time = jnp.where(mask, 0.0, jnp.broadcast_to(time, mask.shape))
            velocity = self._cached_velocity(observation, x_t, token_time, prefix_mask_, kv_cache)
            return hard_prefix_update(x_t, velocity, action_prefix, mask, dt), time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        actions, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return jnp.where(mask[..., None], action_prefix, actions)
