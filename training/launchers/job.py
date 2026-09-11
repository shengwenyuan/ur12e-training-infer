"""Render an AIHC job using the inherited schema; never submit jobs."""

# Protocol numbers must exclude bool, which is an int subclass.
# pylint: disable=unidiomatic-typecheck


import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from training.launchers.vendor.aihc import build_job


def render(settings, command):
    """Keep platform resources and mounts separate from model profiles."""
    if not command or not all(isinstance(arg, str) and arg for arg in command):
        raise ValueError("nonempty structured command required")
    for key in ("gpus", "cpu", "memory", "shared_memory"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"invalid {key}")
    for key in ("name", "queue", "image", "gpu_resource"):
        if not isinstance(settings.get(key), str) or not settings[key].strip():
            raise ValueError(f"missing {key}")
    mounts = settings["mounts"]
    if not mounts:
        raise ValueError("explicit storage mounts required")
    args = SimpleNamespace(
        **{
            key: settings[key]
            for key in (
                "name",
                "queue",
                "image",
                "gpu_resource",
                "gpus",
                "cpu",
                "memory",
                "shared_memory",
            )
        },
        command_arg=command,
        env=list(settings.get("env", {}).items()),
        enable_rdma=settings.get("enable_rdma", False),
        priority=settings.get("priority", "normal"),
        cfs_mount_path="/unused",
        cfs_instance_id="",
        cfs_endpoint="",
    )
    job = build_job(args)
    job["datasources"] = []
    for mount in mounts:
        if (
            mount["type"] not in ("cfs", "pfsl2")
            or not mount["mount_path"].startswith("/")
            or not mount["source_path"].startswith("/")
        ):
            raise ValueError("invalid storage mount")
        job["datasources"].append(
            {
                "id": "",
                "type": mount["type"],
                "sourcePath": mount["source_path"],
                "mountPath": mount["mount_path"],
                "name": mount["id"] if mount["type"] == "pfsl2" else "",
                "options": {
                    "sizeLimit": 0,
                    "medium": "",
                    "readOnly": mount["read_only"],
                    "cfsInstanceId": (
                        mount["id"] if mount["type"] == "cfs" else ""
                    ),
                    "cfsMountPoint": mount.get("endpoint", ""),
                    "datasetVersion": "",
                },
            }
        )
    return job


def main(argv=None):
    """Print a reviewable manifest with shell-quoted command arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    print(
        json.dumps(
            render(
                json.loads(args.deployment.read_text(encoding="utf-8")), command
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
