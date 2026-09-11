"""Experiment-local Gemma adapter with per-token adaRMS conditioning.

The implementation mirrors ``openpi.models.gemma`` at baseline 0c94293 while
reusing parameter-compatible primitives. It exists locally so vanilla Gemma
keeps its scalar-only conditioning contract.
"""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models import gemma as _base
import openpi.models.lora as lora
from openpi.shared import array_typing as at
from openpi.training import sharding

Config = _base.Config
Variant = _base.Variant
KVCache = _base.KVCache
get_config = _base.get_config


@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-6)))
        if cond is None:
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            return (normed_inputs * (1 + scale)).astype(dtype), None

        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        if modulation.ndim == x.ndim - 1:
            modulation = modulation[:, None, :]
        elif modulation.shape[:-1] != x.shape[:-1]:
            raise ValueError(f"adaRMS condition shape {cond.shape} does not match token shape {x.shape}")
        scale, shift, gate = jnp.split(modulation, 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift
        return normed_inputs.astype(dtype), gate


@at.typecheck
class Block(nn.Module):
    configs: tuple[Config, ...]
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):  # noqa: FBT002
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x
        attn = _base.Attention(configs=self.configs, name="attn")

        pre_attn = []
        gates = []
        for i, value in enumerate(xs):
            normalized = value
            if value is not None:
                normalized, gate = RMSNorm(name=_base._name("pre_attention_norm", i))(  # noqa: SLF001
                    value, adarms_cond[i]
                )
            pre_attn.append(normalized)
            gates.append(gate if value is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [
            _base._gated_residual(x, y, gate)  # noqa: SLF001
            for x, y, gate in zip(xs, post_attn, gates, strict=True)
        ]
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        gates = []
        for i, (value, config) in enumerate(zip(xs, self.configs, strict=True)):
            transformed = value
            if value is not None:
                transformed, gate = RMSNorm(name=_base._name("pre_ffw_norm", i))(  # noqa: SLF001
                    value, adarms_cond[i]
                )
                transformed = lora.FeedForward(
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_base._name("mlp", i),  # noqa: SLF001
                    lora_config=config.lora_configs.get("ffn"),
                )(transformed)
            out.append(transformed)
            gates.append(gate if value is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [
            _base._gated_residual(x, y, gate)  # noqa: SLF001
            for x, y, gate in zip(xs, out, gates, strict=True)
        ]
        return sharding.activation_sharding_constraint(xs), kv_cache


@at.typecheck
class Module(nn.Module):
    configs: Sequence[Config]
    embed_dtype: str
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    adarms: bool = False

    def setup(self):
        assert all(config.depth == self.configs[0].depth for config in self.configs)
        self.embedder = _base.Embedder(
            vocab_size=_base.PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,
            name="embedder",
        )
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(5,),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        self.final_norms = [RMSNorm(name=_base._name("final_norm", i)) for i in range(len(self.configs))]  # noqa: SLF001

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b *t _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda value: value.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)
        embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic)
        assert all(value.dtype == jnp.dtype(self.embed_dtype) for value in embedded if value is not None)
        return [
            norm(value, cond)[0] if value is not None else value
            for norm, value, cond in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache

    def init(self, use_adarms: Sequence[bool]):
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, config.width)) for config in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[
                jnp.zeros((1, config.width)) if use else None
                for use, config in zip(use_adarms, self.configs, strict=True)
            ],
        )
