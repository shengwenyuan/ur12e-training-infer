"""Explicit robot boundaries and a motion-free deterministic simulator."""

# Optional backend imports keep the robot/CLI dependency boundary lightweight.
# pylint: disable=import-outside-toplevel


import fcntl
from pathlib import Path
import time
import numpy as np

from infer.common.contract import HOME, ROLES


class Ownership:
    """Share this lock path with every collector and inference actuator
    owner.
    """

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The ownership resource spans the entire runtime, ending in close().
        # pylint: disable-next=consider-using-with
        self.stream = path.open("a", encoding="utf-8")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.stream.close()
            raise

    def close(self):
        """Release only after the robot is stopped."""
        self.stream.close()


class FakeRobot:
    """Test-only robot: synthetic feedback/images are labeled at the
    boundary.
    """

    simulated = True

    def __init__(self):
        self.state = np.r_[HOME, 0.0]
        self.writes = []

    def go_home(self, q):
        """Expose deterministic arrival for software tests, not hardware
        claims.
        """
        self.state[:6] = q
        self.writes.append("HOME")

    def stop_hold(self):
        """Record that no HOME/open command follows a stop."""
        self.writes.append("HOLD")

    def command(self, action):
        """Apply a software-only target."""
        self.state = action.copy()
        self.writes.append("ACTION")

    def feedback(self):
        """Return fresh simulated state."""
        return {
            "state": self.state.copy(),
            "velocity": np.zeros(6),
            "received": time.monotonic(),
            "healthy": True,
        }

    def observation(self, prompt):
        """Return explicit synthetic RGB inputs."""
        return {
            "state": self.state.copy(),
            "images": {
                role: np.zeros((224, 224, 3), dtype=np.uint8) for role in ROLES
            },
            "prompt": prompt,
            "simulated": True,
        }

    def heartbeat(self):
        """Fake robot has no controller watchdog."""

    def close(self):
        """No physical resources are owned."""


def open_robot(kind, station=None):
    """Reject unaccepted physical devices before creating any SDK connection."""
    if kind == "fake":
        return FakeRobot()
    from infer.client.hardware import (
        connect_station,
    )

    return connect_station(station)
