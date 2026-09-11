"""Run pinned checks over owned code, excluding preserved vendor source."""

import os
import subprocess
import sys
from pathlib import Path

from training.pi05_rtc import backend


def main():
    """Use the active development environment and explicit source paths."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "black",
            "--check",
            "training",
            "infer",
            "tests",
            "--exclude",
            "/vendor/",
        ],
        check=True,
    )
    os.environ.setdefault("PYLINTHOME", "/tmp/ur12e-pylint")
    backend.activate()
    from pylint.lint import Run  # pylint: disable=import-outside-toplevel

    files = [
        str(path)
        for base in ("training", "infer")
        for path in Path(base).rglob("*.py")
        if "vendor" not in path.parts
    ]
    files.extend(str(path) for path in Path("tests").glob("test_*.py"))
    result = Run(files, exit=False)
    raise SystemExit(result.linter.msg_status)


if __name__ == "__main__":
    main()
