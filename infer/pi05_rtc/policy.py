"""Dynamic RTC prefix conditioning, using the exact trained normalization."""

# Protocol numbers must exclude bool, which is an int subclass.
# pylint: disable=unidiomatic-typecheck


# Optional backend imports keep the robot/CLI dependency boundary lightweight.
# pylint: disable=import-outside-toplevel


from pathlib import Path
import numpy as np

from infer.common.contract import Capabilities
from training.common.identity import digest, file_digest
from training.pi05_rtc import backend, transforms


class Policy:
    """Load a self-describing checkpoint; input prefixes use physical units."""

    # Keep checkpoint/model setup together; imports are intentionally lazy.
    # pylint: disable-next=too-many-locals
    def __init__(self, checkpoint):
        tokenizer_path = backend.local_assets()
        backend.activate()
        import jax
        import jax.numpy as jnp
        from openpi.models import (
            model,
        )
        from openpi.training.config import (
            ModelTransformFactory,
        )
        from openpi.shared import (
            normalize,
            nnx_utils,
        )
        from openpi import (
            transforms as ops,
        )
        from openpi.experiments.arx5_joint_rtc import (
            checkpoints,
        )
        from openpi.experiments.arx5_joint_rtc.model_config import (
            Pi05RtcConfig,
        )

        if jax.default_backend() != "gpu":
            raise RuntimeError(
                "pi05_rtc requires a GPU; check --gpus all and NVIDIA runtime"
            )
        checkpoint = Path(checkpoint)
        metadata = checkpoints.validate_policy_type(checkpoint / "assets")
        if metadata.get("algorithm_sha256") != backend.algorithm_identity():
            raise ValueError("checkpoint numerical implementation mismatch")
        if metadata.get("contract") != transforms.CONTRACT:
            raise ValueError(
                "checkpoint UR12e training/serving contract mismatch"
            )
        norm_path = checkpoint / "assets/ur12e"
        if metadata.get("norm_sha256") != file_digest(
            norm_path / "norm_stats.json"
        ):
            raise ValueError("checkpoint normalization content mismatch")
        if metadata.get("tokenizer_sha256") != file_digest(tokenizer_path):
            raise ValueError("checkpoint tokenizer identity mismatch")
        delay = metadata["max_delay"]
        config = Pi05RtcConfig(
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            discrete_state_input=True,
            max_delay=delay,
        )
        network = config.load(
            model.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
        )
        network.eval()
        self.sample = nnx_utils.module_jit(network.sample_actions)
        stats = normalize.load(norm_path)
        self.preprocess = ops.compose(
            [
                ops.Normalize(stats, use_quantiles=True),
                *ModelTransformFactory()(config).inputs,
            ]
        )
        self.unscale = ops.Unnormalize(stats, use_quantiles=True)
        self.rng = jax.random.key(42)
        self.checkpoint_sha256 = checkpoints.checkpoint_params_sha256(
            checkpoint
        )
        self.capabilities = Capabilities(
            "pi05_rtc",
            50,
            True,
            delay,
            digest(metadata | {"checkpoint_sha256": self.checkpoint_sha256}),
        )

    # Explicit named intermediate tensors make the prefix transform auditable.
    # pylint: disable-next=too-many-locals
    def infer(self, request):
        """Rebase physical prefix against this request's current joint state."""
        import jax
        import jax.numpy as jnp
        from openpi.models.model import (
            Observation,
        )

        observation = request["observation"]
        prefix = np.asarray(request["prefix"], dtype=np.float32)
        delay = request["delay"]
        if (
            type(delay) is not int
            or not 0 <= delay < self.capabilities.max_delay
            or prefix.shape != (delay, 7)
            or not np.isfinite(prefix).all()
        ):
            raise ValueError("invalid RTC prefix/delay")
        state = np.asarray(observation["state"], dtype=np.float32)
        absolute_prefix = np.tile(state, (50, 1))
        absolute_prefix[:delay] = prefix
        inputs = transforms.inputs(
            state, observation["images"], observation["prompt"], absolute_prefix
        )
        inputs = self.preprocess(inputs)
        batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        self.rng, key = jax.random.split(self.rng)
        result = self.sample(
            key,
            Observation.from_dict(batch),
            action_prefix=batch["actions"],
            delay=jnp.asarray([delay], dtype=jnp.int32),
        )
        normalized = np.asarray(result[0])
        if not np.array_equal(normalized[:delay], inputs["actions"][:delay]):
            raise ValueError("sampler did not preserve hard prefix")
        physical = self.unscale({"actions": normalized[:, :7]})["actions"]
        physical = transforms.absolute(physical, state)
        physical[:delay] = prefix
        return physical
