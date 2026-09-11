"""Read-only admission and numeric windows for official UR12e LeRobot v3."""

# Optional backend imports keep the robot/CLI dependency boundary lightweight.
# pylint: disable=import-outside-toplevel


import json
from pathlib import Path
import numpy as np

from training.common.identity import digest, file_digest

ROLES = ("wrist", "third_left", "third_right")
ACTION = "same_row_absolute_measured_joints_rad_and_gripper_raw"


class ExportedDataset:
    """Keep physical source time separate from the retained-row sequence."""

    def __init__(self, root: str | Path, *, allow_simulated: bool = False):
        import pyarrow.parquet as pq

        self.root = Path(root).resolve()
        if (
            self.root.name.endswith(".partial")
            or (self.root / "failure.json").exists()
        ):
            raise ValueError("incomplete dataset")
        self.manifest = json.loads(
            (self.root / "export.json").read_text(encoding="utf-8")
        )
        self.info = json.loads(
            (self.root / "meta/info.json").read_text(encoding="utf-8")
        )
        self._validate_contract(allow_simulated)
        paths = sorted((self.root / "data").rglob("*.parquet"))
        if not paths:
            raise ValueError("empty dataset")
        columns = (
            "index",
            "episode_index",
            "frame_index",
            "observation.state",
            "action",
            "source.state_receipt_ns",
        )
        rows = [
            row
            for path in paths
            for row in pq.read_table(path, columns=list(columns)).to_pylist()
        ]
        rows.sort(key=lambda row: row["index"])
        self.states = np.asarray(
            [r["observation.state"] for r in rows], dtype=np.float32
        )
        self.actions = np.asarray([r["action"] for r in rows], dtype=np.float32)
        self.episodes = np.asarray(
            [r["episode_index"] for r in rows], dtype=np.int64
        )
        self._validate_rows(rows)
        paths = sorted(
            p
            for p in self.root.rglob("*")
            if p.is_file() and p.suffix in (".json", ".parquet", ".mp4")
        )
        if any(p.is_symlink() for p in self.root.rglob("*")):
            raise ValueError("dataset must not contain symlinks")
        self.identity = digest(
            {str(p.relative_to(self.root)): file_digest(p) for p in paths}
        )
        self._reader = None

    def _validate_contract(self, allow_simulated):
        verification = self.manifest["verification"]
        if (
            self.manifest.get("schema_version") != 1
            or self.manifest.get("format") != "lerobot-v3.0"
            or self.info.get("codebase_version") != "v3.0"
        ):
            raise ValueError("unsupported dataset version")
        if self.manifest.get("action") != ACTION:
            raise ValueError("unexpected action representation")
        if self.manifest.get("simulated") is not False and not allow_simulated:
            raise ValueError("fixture dataset requires --allow-simulated")
        if any(
            verification.get(key) is not True
            for key in (
                "all_numeric_rows_exact",
                "all_output_rgb_decoded",
                "all_admitted_source_rgb_depth_verified",
            )
        ):
            raise ValueError("export verification incomplete")
        for key, shape in [
            ("observation.state", [7]),
            ("action", [7]),
            ("observation.flange", [16]),
        ]:
            feature = self.info["features"][key]
            if feature["shape"] != shape or feature["dtype"] != "float32":
                raise ValueError(f"invalid feature {key}")
        for role in ROLES:
            feature = self.info["features"][f"observation.images.{role}"]
            if feature["dtype"] != "video" or feature["shape"] != [3, 480, 640]:
                raise ValueError(f"invalid RGB feature {role}")

    def _validate_rows(self, rows):
        verification = self.manifest["verification"]
        if (
            [r["index"] for r in rows] != list(range(len(rows)))
            or len(rows) != self.info["total_frames"]
            or len(rows) != verification["frames"]
        ):
            raise ValueError("frame identity/count mismatch")
        if (
            not np.array_equal(self.states, self.actions)
            or not np.isfinite(self.states).all()
            or np.any((self.states[:, 6] < 0) | (self.states[:, 6] > 255))
        ):
            raise ValueError("invalid measured state/action or gripper")
        self.ends = np.empty(len(rows), dtype=np.int64)
        unique = list(dict.fromkeys(self.episodes))
        if (
            unique != list(range(self.info["total_episodes"]))
            or len(unique) != verification["episodes"]
        ):
            raise ValueError("episode identity/count mismatch")
        for episode in unique:
            indices = np.flatnonzero(self.episodes == episode)
            if not np.array_equal(
                indices, np.arange(indices[0], indices[-1] + 1)
            ):
                raise ValueError("interleaved episodes")
            selected = [rows[i] for i in indices]
            if [r["frame_index"] for r in selected] != list(
                range(len(indices))
            ):
                raise ValueError("invalid frame index")
            times = np.array(
                [
                    int(np.asarray(r["source.state_receipt_ns"]).reshape(-1)[0])
                    for r in selected
                ],
                dtype=np.int64,
            )
            if np.any(np.diff(times) <= 0):
                raise ValueError("physical source time is not increasing")
            self.ends[indices] = indices[-1]

    def __len__(self):
        return len(self.states)

    def window(self, index: int, horizon: int) -> np.ndarray:
        """Use current-row first target and repeated terminal targets."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        indices = np.minimum(
            np.arange(index, index + horizon), self.ends[index]
        )
        return self.actions[indices].copy()

    def official(self):
        """Open the pinned official loader locally, without Hub downloads."""
        if self._reader is None:
            from lerobot.datasets.lerobot_dataset import (
                LeRobotDataset,
            )

            import torch

            torch.set_num_threads(1)
            self._reader = LeRobotDataset(
                "local/ur12e",
                root=self.root,
                video_backend="pyav",
                download_videos=False,
            )
        return self._reader


def rgb_images(row: dict) -> dict:
    """Convert official CHW float RGB into physical-role HWC uint8 inputs."""
    return {
        role: np.rint(np.asarray(row[f"observation.images.{role}"]) * 255)
        .clip(0, 255)
        .astype(np.uint8)
        .transpose(1, 2, 0)
        for role in ROLES
    }
