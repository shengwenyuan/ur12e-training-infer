"""HOME-before-policy and independent supervisor lease enforcement."""

from dataclasses import dataclass
import numpy as np

from infer.common.contract import HOME, RobotAdapter, vector


@dataclass(frozen=True)
class Limits:
    """Simulation defaults; physical deployment requires measured acceptance."""

    lease_seconds: float = 0.3
    feedback_seconds: float = 0.25
    home_timeout: float = 45.0
    home_tolerance: float = 0.01
    settle_seconds: float = 0.3
    stationary_speed: float = 0.01
    max_joint_step: float = 0.02
    max_joint_speed: float = 0.3
    max_gripper_step: float = 15.0

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in self.__dict__.values()):
            raise ValueError("limits must be finite and positive")


class Session:
    """One control-loop owner serializes stop, HOME and policy writes."""

    def __init__(self, robot: RobotAdapter, limits: Limits = Limits()):
        self.robot = robot
        self.limits = limits
        self.phase = "DISARMED"
        self.reason = ""
        self.epoch = 0
        self.lease_until = -1.0
        self.home_started = None
        self.settled_since = None
        self.last_action = None

    def renew(self, now: float) -> None:
        """Only an admitted, timely supervisor heartbeat renews authority."""
        self.lease_until = now + self.limits.lease_seconds

    def start(self, now: float, feedback: dict | None = None) -> None:
        """Every start requires HOME, including starts following a HOLD."""
        if self.phase not in ("DISARMED", "HOLDING") or now >= self.lease_until:
            raise RuntimeError(
                "start requires a live supervisor and idle session"
            )
        if feedback is None:
            feedback = self.robot.feedback()
        if (
            not feedback["healthy"]
            or not 0
            <= now - feedback["received"]
            <= self.limits.feedback_seconds
            or np.max(np.abs(vector(feedback["velocity"], 6)))
            >= self.limits.stationary_speed
        ):
            self.stop("HOME preflight failed", fault=True)
            raise RuntimeError(
                "HOME requires fresh healthy stationary feedback"
            )
        vector(feedback["state"])
        self.epoch += 1
        self.phase = "HOMING"
        self.home_started = now
        self.settled_since = None
        self.last_action = None
        try:
            self.robot.go_home(HOME.copy())
        except Exception:
            self.stop("HOME command failed", fault=True)
            raise

    def stop(self, reason: str = "Space", *, fault: bool = False) -> None:
        """Revoke first; report HOLD only after actual settling on later
        ticks.
        """
        self.epoch += 1
        self.reason = reason
        self.phase = "FAULT" if fault or self.phase == "FAULT" else "STOPPING"
        self.settled_since = None
        self.last_action = None
        try:
            self.robot.stop_hold()
        except Exception:
            self.phase = "FAULT"
            self.reason = "stop/hold unconfirmed"
            raise

    def tick(self, now: float, feedback: dict) -> None:
        """Use measured arrival; never grant policy authority from a timer
        alone.
        """
        try:
            q = vector(feedback["state"])
            qd = vector(feedback["velocity"], 6)
        except ValueError:
            self.stop("invalid feedback", fault=True)
            raise
        active = self.phase in ("HOMING", "READY", "RUNNING", "STOPPING")
        if active and (
            (self.phase != "STOPPING" and now >= self.lease_until)
            or not 0
            <= now - feedback["received"]
            <= self.limits.feedback_seconds
            or not feedback["healthy"]
        ):
            self.stop("supervisor/feedback health lost", fault=True)
            return
        stationary = np.max(np.abs(qd)) < self.limits.stationary_speed
        if self.phase == "HOMING":
            if now - self.home_started > self.limits.home_timeout:
                self.stop("HOME timeout", fault=True)
                return
            arrived = np.max(np.abs(q[:6] - HOME)) < self.limits.home_tolerance
            self.settled_since = (
                (now if self.settled_since is None else self.settled_since)
                if arrived and stationary
                else None
            )
            if (
                self.settled_since is not None
                and now - self.settled_since >= self.limits.settle_seconds
            ):
                self.phase = "READY"
        elif self.phase == "STOPPING":
            self.settled_since = (
                (now if self.settled_since is None else self.settled_since)
                if stationary
                else None
            )
            if (
                self.settled_since is not None
                and now - self.settled_since >= self.limits.settle_seconds
            ):
                self.phase = "HOLDING"

    def execute(
        self, action: np.ndarray, now: float, feedback: dict, period: float
    ) -> None:
        """Check authority again immediately before the sole actuator write."""
        self.tick(now, feedback)
        if self.phase not in ("READY", "RUNNING"):
            raise RuntimeError("policy control requires confirmed HOME")
        try:
            action = vector(action)
        except ValueError:
            self.stop("invalid action", fault=True)
            raise
        actual = vector(feedback["state"])
        anchor = actual if self.last_action is None else self.last_action
        bound = min(
            self.limits.max_joint_step, self.limits.max_joint_speed * period
        )
        if (
            np.any(np.abs(action[:6] - anchor[:6]) > bound)
            or np.any(
                np.abs(action[:6] - actual[:6]) > self.limits.max_joint_step
            )
            or abs(action[6] - anchor[6]) > self.limits.max_gripper_step
            or not 0 <= action[6] <= 255
            or np.any(np.abs(action[:6]) > 2 * np.pi)
        ):
            self.stop("invalid or discontinuous action", fault=True)
            raise ValueError("action exceeds physical limits")
        self.phase = "RUNNING"
        self.robot.command(action)
        self.last_action = action.copy()
