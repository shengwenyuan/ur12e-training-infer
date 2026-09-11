"""1011 RTC training lifecycle, with an explicit UR12e v3 loader."""

import functools
import logging
import os
import platform
from etils import epath
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import wandb
from openpi.training import sharding
from openpi.training import utils as training_utils
from openpi.experiments.arx5_joint_rtc import checkpoints


# Preserve the inspected upstream loop structure for numerical review.
# pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
def run_training(
    config, dataset, *, save_checkpoints: bool, run_identity: dict
) -> None:
    """Run the shared training primitives under an RTC-owned lifecycle."""
    from training.pi05_rtc import loader
    from scripts import train as base_train

    base_train.init_logging()
    logging.info("Running on: %s", platform.node())
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must divide "
            f"device count {jax.device_count()}."
        )
    if config.resume and not save_checkpoints:
        raise ValueError("cannot resume when checkpoint saving is disabled")

    jax.config.update(
        "jax_compilation_cache_dir",
        os.environ.get(
            "JAX_COMPILATION_CACHE_DIR",
            str(epath.Path("~/.cache/jax").expanduser()),
        ),
    )
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    checkpoint_manager = None
    resuming = False
    if save_checkpoints:
        checkpoint_manager, resuming = checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir,
            keep_period=config.keep_period,
            overwrite=config.overwrite,
            resume=config.resume,
        )
    if save_checkpoints:
        from training.common.identity import write_json

        write_json(config.checkpoint_dir / "run.json", run_identity)
    base_train.init_wandb(
        config, resuming=resuming, enabled=config.wandb_enabled
    )
    data_loader = loader.create(config, dataset)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(
        "Initialized data loader:\n%s", training_utils.array_tree_to_info(batch)
    )

    images_to_log = [
        wandb.Image(
            np.concatenate(
                [np.array(image[i]) for image in batch[0].images.values()],
                axis=1,
            )
        )
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)
    train_state, train_state_sharding = base_train.init_train_state(
        config, init_rng, mesh, resume=resuming
    )
    jax.block_until_ready(train_state)
    logging.info(
        "Initialized train state:\n%s",
        training_utils.array_tree_to_info(train_state.params),
    )
    if resuming:
        train_state = checkpoints.restore_state(
            checkpoint_manager, train_state, data_loader
        )

    ptrain_step = jax.jit(
        functools.partial(base_train.train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )
    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(  # pylint: disable=not-callable
                train_rng, train_state, batch
            )
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            if not all(
                np.isfinite(value).all()
                for value in jax.tree.leaves(reduced_info)
            ):
                raise FloatingPointError("non-finite training metrics")
            pbar.write(
                f"Step {step}: "
                + ", ".join(
                    f"{key}={value:.4f}" for key, value in reduced_info.items()
                )
            )
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)
        should_save = (
            step % config.save_interval == 0 and step > start_step
        ) or step == config.num_train_steps - 1
        if save_checkpoints and should_save:
            checkpoints.save_state(
                checkpoint_manager,
                train_state,
                data_loader,
                step,
                policy_metadata=config.policy_metadata or {},
            )
    if checkpoint_manager is not None:
        logging.info("Waiting for checkpoint manager to finish")
        checkpoint_manager.wait_until_finished()
