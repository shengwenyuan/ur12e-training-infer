#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import PurePosixPath
import re
import shlex


def _absolute_path(value: str) -> str:
    if not re.fullmatch(r"/[A-Za-z0-9._/-]+", value):
        raise argparse.ArgumentTypeError("must be a shell-safe absolute path")
    return str(PurePosixPath(value))


def _job_name(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", value):
        raise argparse.ArgumentTypeError("must match [a-z0-9][a-z0-9-]{0,62}")
    return value


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _environment(value: str) -> tuple[str, str]:
    name, separator, environment_value = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise argparse.ArgumentTypeError("must be NAME=VALUE")
    return name, environment_value


def build_job(args: argparse.Namespace) -> dict[str, object]:
    envs = [
        {"name": "LOG_COLLECTION", "value": "true"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "AIHC_JOB_NAME", "value": args.name},
    ]
    envs.extend({"name": name, "value": value} for name, value in args.env)
    return {
        "name": args.name,
        "queue": args.queue,
        "jobFramework": "pytorch",
        "jobSpec": {
            "command": shlex.join(args.command_arg),
            "image": args.image,
            "imageConfig": {"username": "", "password": ""},
            "replicas": 1,
            "resources": [
                {"name": args.gpu_resource, "quantity": args.gpus},
                {"name": "cpu", "quantity": args.cpu},
                {"name": "memory", "quantity": args.memory},
                {"name": "sharedMemory", "quantity": args.shared_memory},
            ],
            "envs": envs,
            "enableRDMA": args.enable_rdma,
            "hostNetwork": False,
        },
        "faultTolerance": False,
        "priority": args.priority,
        "datasources": [
            {
                "id": "",
                "type": "cfs",
                "sourcePath": "/",
                "mountPath": args.cfs_mount_path,
                "name": "",
                "options": {
                    "sizeLimit": 0,
                    "medium": "",
                    "readOnly": False,
                    "cfsInstanceId": args.cfs_instance_id,
                    "cfsMountPoint": args.cfs_endpoint,
                    "datasetVersion": "",
                },
            }
        ],
        "codeSource": {
            "filePath": "",
            "mountPath": "",
            "id": "",
            "bosObjectName": "",
            "bosTemporaryToken": "",
            "bosEndpoint": "",
            "bosBucket": "",
        },
        "enableBccl": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an AIHC job specification")
    parser.add_argument("--command-arg", action="append", required=True)
    parser.add_argument("--env", action="append", type=_environment, default=[])
    parser.add_argument("--gpus", type=_positive_int, required=True)
    parser.add_argument("--cpu", type=_positive_int, required=True)
    parser.add_argument("--memory", type=_positive_int, required=True)
    parser.add_argument("--shared-memory", type=_positive_int, required=True)
    parser.add_argument("--name", type=_job_name)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu-resource", required=True)
    parser.add_argument("--cfs-instance-id", required=True)
    parser.add_argument("--cfs-endpoint", required=True)
    parser.add_argument("--cfs-mount-path", type=_absolute_path, required=True)
    parser.add_argument("--priority", choices=("low", "normal", "high"), default="normal")
    parser.add_argument("--enable-rdma", action="store_true")
    args = parser.parse_args()
    if args.name is None:
        date = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d-%H%M%S")
        args.name = f"robot-policy-training-{args.gpus}gpu-{date}"
    return args


def main() -> None:
    print(json.dumps(build_job(parse_args()), indent=2))


if __name__ == "__main__":
    main()
