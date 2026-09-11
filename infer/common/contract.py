"""Model-neutral physical observations and bounded inference messages."""

# Protocol numbers must exclude bool, which is an int subclass.
# pylint: disable=unidiomatic-typecheck


from dataclasses import dataclass
from typing import Protocol
import numpy as np

HOME = np.array([0, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
ROLES = ("wrist", "third_left", "third_right")
PROTOCOL = 1
ENVELOPE_FIELDS = (
    "protocol",
    "request_id",
    "epoch",
    "anchor",
    "contract_id",
    "model",
)


@dataclass(frozen=True)
class Capabilities:
    """Describe the selected server without importing a model implementation."""

    model: str
    horizon: int
    rtc: bool
    max_delay: int
    contract_id: str

    def __post_init__(self):
        if not self.model or not self.contract_id:
            raise ValueError("backend identity is required")
        if type(self.rtc) is not bool or (self.rtc and self.max_delay < 1):
            raise ValueError("invalid RTC capability")
        if (
            type(self.horizon) is not int
            or self.horizon < 1
            or type(self.max_delay) is not int
            or not 0 <= self.max_delay <= self.horizon
        ):
            raise ValueError("invalid backend capabilities")


def vector(value, size: int = 7) -> np.ndarray:
    """Reject malformed/nonfinite values before motion or normalization."""
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"expected {size} finite values")
    return result


def chunk(value, horizon: int) -> np.ndarray:
    """Model output is always absolute radians and raw gripper position."""
    result = np.asarray(value, dtype=np.float64)
    if (
        result.shape != (horizon, 7)
        or not np.isfinite(result).all()
        or np.any((result[:, 6] < 0) | (result[:, 6] > 255))
    ):
        raise ValueError("invalid physical action chunk")
    return result


class RobotAdapter(Protocol):
    """Exclusive device boundary; methods must fail rather than reconnect."""

    def go_home(self, q: np.ndarray) -> None:
        """Begin a bounded asynchronous HOME move."""

    def stop_hold(self) -> None:
        """Request deceleration and preserve the current grasp."""

    def command(self, action: np.ndarray) -> None:
        """Accept one checked absolute physical action."""

    def feedback(self) -> dict:
        """Return measured state, velocity, freshness and health."""
