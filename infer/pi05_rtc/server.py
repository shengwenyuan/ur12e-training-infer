"""Serve model-selected inference; the fake backend has no GPU dependencies."""

# Protocol numbers must exclude bool, which is an int subclass.
# pylint: disable=unidiomatic-typecheck


# Optional backend imports keep the robot/CLI dependency boundary lightweight.
# pylint: disable=import-outside-toplevel


import argparse
import asyncio
from dataclasses import asdict
import numpy as np

from infer.common import wire
from infer.common.contract import (
    Capabilities,
    HOME,
    PROTOCOL,
    ENVELOPE_FIELDS,
    vector,
)


def warmup(policy):
    """Compile the fixed model input shape before accepting robot clients."""
    policy.infer(
        {
            "observation": {
                "state": np.r_[HOME, 0.0],
                "images": {
                    role: np.zeros((224, 224, 3), dtype=np.uint8)
                    for role in ("wrist", "third_left", "third_right")
                },
                "prompt": "model warmup",
            },
            "prefix": np.empty((0, 7), dtype=np.float32),
            "delay": 0,
        }
    )


class FakePolicy:
    """Non-RTC backend proving the client is independent of pi05 tensors."""

    capabilities = Capabilities("fake", 8, False, 0, "fake-hold-v1")

    def infer(self, request):
        """Return stationary synthetic targets, never a meaningful robot
        policy.
        """
        return np.tile(vector(request["observation"]["state"]), (8, 1))


def respond(policy, request):
    """Validate the request envelope before model computation."""
    caps = policy.capabilities
    if (
        request.get("protocol") != PROTOCOL
        or request.get("contract_id") != caps.contract_id
        or request.get("model") != caps.model
    ):
        raise ValueError("protocol or model contract mismatch")
    for key in ("epoch", "anchor"):
        if type(request.get(key)) is not int or request[key] < 0:
            raise ValueError(f"invalid {key}")
    if (
        not isinstance(request.get("request_id"), str)
        or not request["request_id"]
    ):
        raise ValueError("request ID required")
    return {key: request[key] for key in ENVELOPE_FIELDS} | {
        "actions": policy.infer(request)
    }


async def serve(policy, host, port):
    """One serialized policy instance owns the model RNG and inference state."""
    from websockets.asyncio.server import (
        serve as websocket_serve,
    )

    lock = asyncio.Lock()

    async def handler(connection):
        await connection.send(wire.pack(asdict(policy.capabilities)))
        async for message in connection:
            async with lock:
                response = respond(policy, wire.unpack(message))
            await connection.send(wire.pack(response))

    async with websocket_serve(
        handler, host, port, max_size=wire.MAX_BYTES, compression=None
    ):
        await asyncio.Future()


def main(argv=None):
    """Load a fake backend or an explicitly selected trained RTC checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pi05_rtc", "fake"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--tokenizer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.model == "pi05_rtc":
        if not args.checkpoint:
            parser.error("--checkpoint is required")
        from infer.pi05_rtc.policy import (
            Policy,
        )

        from training.pi05_rtc.backend import local_assets

        local_assets(args.tokenizer)
        policy = Policy(args.checkpoint)
        print("Warming model before opening the server socket", flush=True)
        warmup(policy)
    else:
        policy = FakePolicy()
    asyncio.run(serve(policy, args.host, args.port))


if __name__ == "__main__":
    main()
