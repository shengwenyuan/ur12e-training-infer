"""Offline held-out action error and latency measurements without a robot."""

# Dataset/model imports remain optional for the normal robot client.
# pylint: disable=import-outside-toplevel
import argparse
from dataclasses import asdict
import importlib.metadata
import os
from pathlib import Path
import platform
import time
import uuid
import numpy as np

from infer.common.contract import Capabilities, PROTOCOL, ENVELOPE_FIELDS, chunk
from infer.common import wire
from training.common.identity import write_json


def observation(dataset, index):
    """Read an exported observation using its physical camera-role names."""
    row = dataset.official()[index]
    from training.common.dataset import rgb_images

    images = rgb_images(row)
    return {
        "state": dataset.states[index],
        "images": images,
        "prompt": row["task"],
        "simulated": dataset.manifest["simulated"],
    }


def _infer_window(dataset, index, capabilities, infer):
    request = {
        "protocol": PROTOCOL,
        "request_id": uuid.uuid4().hex,
        "epoch": 0,
        "anchor": 0,
        "contract_id": capabilities.contract_id,
        "model": capabilities.model,
        "observation": observation(dataset, index),
        "prefix": np.empty((0, 7), dtype=np.float32),
        "delay": 0,
    }
    start = time.perf_counter()
    response = infer(request)
    elapsed = (time.perf_counter() - start) * 1000
    for key in ENVELOPE_FIELDS:
        if response.get(key) != request[key]:
            raise ValueError(f"benchmark response {key} mismatch")
    actions = chunk(response["actions"], capabilities.horizon)
    return elapsed, actions


def measure(dataset, capabilities, infer, count=50, warmup=5):
    """Measure request latency and non-padded future action error."""
    if count < 1 or warmup < 0:
        raise ValueError("positive samples and nonnegative warmup required")
    indices = np.linspace(
        0, len(dataset) - 1, min(count, len(dataset)), dtype=int
    )
    latency = []
    warmup_latency = []
    errors = []
    rows = []
    for iteration in range(warmup + len(indices)):
        index = int(indices[max(0, iteration - warmup)])
        elapsed, actions = _infer_window(dataset, index, capabilities, infer)
        if iteration < warmup:
            warmup_latency.append(elapsed)
            continue
        latency.append(elapsed)
        # Current-row target and repeated tail would artificially improve error.
        valid = min(capabilities.horizon, int(dataset.ends[index]) - index + 1)
        if valid > 1:
            errors.append(
                np.abs(
                    actions[1:valid]
                    - dataset.window(index, capabilities.horizon)[1:valid]
                )
            )
        rows.append(
            {
                "row": index,
                "latency_ms": elapsed,
                "valid_future_steps": max(0, valid - 1),
                "predicted_actions": actions.tolist(),
            }
        )
    return _summary(
        dataset, capabilities, latency, warmup_latency, errors, rows
    )


def _summary(dataset, capabilities, latency, warmup_latency, errors, rows):
    percentiles = np.percentile(latency, [50, 95, 99]).tolist()
    mean_error = np.concatenate(errors).mean(axis=0) if errors else None
    return {
        "schema_version": 1,
        "dataset_sha256": dataset.identity,
        "simulated_data": dataset.manifest["simulated"],
        "capabilities": asdict(capabilities),
        "conditioning": (
            "independent zero-prefix windows; " "not closed-loop task success"
        ),
        "warmup_ms": warmup_latency,
        "latency_ms": dict(zip(("p50", "p95", "p99"), percentiles))
        | {"mean": float(np.mean(latency))},
        "requests_per_second": 1000 / float(np.mean(latency)),
        "joint_mae_rad": (
            float(np.mean(mean_error[:6])) if mean_error is not None else None
        ),
        "gripper_mae_raw": (
            float(mean_error[6]) if mean_error is not None else None
        ),
        "rows": rows,
    }


def environment():
    """Record package and available GPU identity for reproducible comparison."""
    versions = {}
    for package in ("jax", "jaxlib", "torch", "lerobot", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": versions,
        "image": os.environ.get("UR12E_IMAGE_ID", "unspecified"),
    }


def main(argv=None):
    """Benchmark a local GPU checkpoint or a remote server."""
    args = _arguments(argv)
    from training.common.dataset import ExportedDataset

    dataset = ExportedDataset(
        args.dataset, allow_simulated=args.allow_simulated
    )
    runtime = environment()
    if args.checkpoint:
        from training.pi05_rtc.backend import local_assets

        local_assets(args.tokenizer)
        from infer.pi05_rtc.policy import Policy
        from infer.pi05_rtc.server import respond

        policy = Policy(args.checkpoint)
        import jax

        if jax.default_backend() != "gpu":
            raise RuntimeError(
                "GPU checkpoint benchmark requires a GPU backend"
            )
        runtime["devices"] = [
            {
                "id": device.id,
                "kind": device.device_kind,
                "platform": device.platform,
            }
            for device in jax.devices()
        ]
        runtime["checkpoint_sha256"] = policy.checkpoint_sha256
        result = measure(
            dataset,
            policy.capabilities,
            lambda request: respond(policy, request),
            args.samples,
            args.warmup,
        )
        runtime["transport"] = "in_process"
    else:
        from websockets.sync.client import connect

        with connect(
            args.server,
            open_timeout=10,
            close_timeout=1,
            max_size=wire.MAX_BYTES,
            compression=None,
        ) as connection:
            caps = Capabilities(**wire.unpack(connection.recv(timeout=10)))

            def infer(request):
                connection.send(wire.pack(request))
                return wire.unpack(connection.recv(timeout=30))

            result = measure(dataset, caps, infer, args.samples, args.warmup)
        runtime["transport"] = "websocket_round_trip"
    write_json(args.output, result | {"environment": runtime})
    print(f"Wrote {args.output}")


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    source.add_argument("--server")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--allow-simulated", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; choose a new report path")
    return args


if __name__ == "__main__":
    main()
