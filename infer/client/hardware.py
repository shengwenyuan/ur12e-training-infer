"""UR12e/Hand-E integration behind explicit, station-specific acceptance."""

# Protocol numbers must exclude bool, which is an int subclass.
# pylint: disable=unidiomatic-typecheck


# Optional backend imports keep the robot/CLI dependency boundary lightweight.
# pylint: disable=import-outside-toplevel


import json
from pathlib import Path
import socket
import time
import numpy as np

from infer.common.contract import HOME
from infer.client.cameras import Cameras
from infer.client.gripper import GripperWorker
from infer.client.lifecycle import Limits


def _line(connection):
    result = bytearray()
    while len(result) < 256:
        part = connection.recv(1)
        if not part:
            raise ConnectionError("device closed connection")
        result.extend(part)
        if part == b"\n":
            return result.decode("ascii").strip()
    raise ValueError("oversized device reply")


class HandE:
    """Bounded raw-register commands; startup never activates or opens grasp."""

    def __init__(self, connection, speed, force):
        self.connection = connection
        self.speed = speed
        self.force = force
        self.target = None

    def get(self, name):
        """Read measured raw state with exact response-name validation."""
        self.connection.sendall(f"GET {name}\n".encode())
        parts = _line(self.connection).split()
        if (
            len(parts) != 2
            or parts[0] != name
            or not parts[1].isdigit()
            or not 0 <= int(parts[1]) <= 255
        ):
            raise ValueError("invalid Hand-E readback")
        return int(parts[1])

    def position(self):
        """Require already activated and fault-free gripper feedback."""
        if self.get("STA") != 3 or self.get("FLT") != 0:
            raise RuntimeError("Hand-E not ready; activate manually")
        return self.get("POS")

    def move(self, value):
        """Preserve the last grasp request during arm holds."""
        target = int(round(value))
        if not 0 <= target <= 255:
            raise ValueError("invalid gripper command")
        if self.target == target:
            return
        self.connection.sendall(
            (
                f"SET POS {target} SPE {self.speed} "
                f"FOR {self.force} GTO 1\n"
            ).encode()
        )
        reply = bytearray()
        while len(reply) < 3:
            part = self.connection.recv(3 - len(reply))
            if not part:
                raise ConnectionError("Hand-E command connection closed")
            reply.extend(part)
        if bytes(reply) != b"ack":
            raise RuntimeError("Hand-E command not acknowledged")
        self.target = target

    def close(self):
        """Close the socket without release/reset/activation commands."""
        self.connection.close()


class Robot:
    """Sole control owner; stalled main loop also stops watchdog updates."""

    simulated = False

    def __init__(self, control, receiver, gripper, cameras, station):
        self.control = control
        self.receiver = receiver
        self.gripper = GripperWorker(gripper)
        self.cameras = cameras
        self.station = station
        self.target = None
        self.mode = "hold"
        self.last_stamp = None
        self.last_receipt = 0.0

    def feedback(self):
        """Timestamp-bracket UR getters; retain receipt time on repeated
        data.
        """
        for _ in range(5):
            stamp = self.receiver.getTimestamp()
            q = self.receiver.getActualQ()
            qd = self.receiver.getActualQd()
            healthy = (
                self.receiver.isConnected()
                and self.control.isConnected()
                and self.receiver.getRobotMode() == 7
                and self.receiver.getSafetyMode() == 1
            )
            if stamp == self.receiver.getTimestamp():
                break
        else:
            raise RuntimeError("no stable UR feedback view")
        if self.last_stamp is not None and stamp < self.last_stamp:
            raise RuntimeError("UR timestamp regressed")
        if stamp != self.last_stamp:
            self.last_stamp = stamp
            self.last_receipt = time.monotonic()
        gripper_state = self.gripper.snapshot()
        return {
            "state": np.r_[q, gripper_state],
            "velocity": np.asarray(qd),
            "received": self.last_receipt,
            "healthy": healthy,
        }

    def go_home(self, q):
        """Start the adopted branch-preserving HOME move asynchronously."""
        self.target = None
        self.mode = "move"
        if not self.control.moveJ(
            q.tolist(),
            self.station["home_speed"],
            self.station["home_acceleration"],
            True,
        ):
            raise RuntimeError("HOME rejected")

    def command(self, action):
        """Set a checked target; heartbeat streams it at the configured rate."""
        self.target = action[:6].copy()
        self.mode = "servo"
        self.gripper.move(action[6])

    def heartbeat(self):
        """Never kick watchdog from a separate thread that could mask a
        stall.
        """
        if self.mode == "servo":
            if not self.control.servoJ(
                self.target.tolist(),
                0,
                0,
                1 / self.station["servo_hz"],
                0.1,
                300,
            ):
                raise RuntimeError("servo target rejected")
        if not self.control.kickWatchdog():
            raise RuntimeError("controller watchdog update failed")

    def stop_hold(self):
        """Stop current motion without power-off, HOME or gripper release."""
        mode = self.mode
        self.mode = "hold"
        self.target = None
        self.gripper.cancel_pending()
        if mode == "servo":
            if not self.control.servoStop(self.station["stop_deceleration"]):
                raise RuntimeError("servo stop unconfirmed")
        elif mode == "move":
            self.control.stopJ(self.station["stop_deceleration"], True)

    def observation(self, prompt):
        """Use fresh measured state and the single owned camera source."""
        feedback = self.feedback()
        return {
            "state": feedback["state"],
            "images": self.cameras.snapshot(),
            "prompt": prompt,
            "simulated": False,
        }

    def close(self):
        """Release all resources; never unlock faults or restart programs."""
        try:
            self.stop_hold()
        finally:
            try:
                self.control.stopScript()
            finally:
                self.control.disconnect()
                self.receiver.disconnect()
                self.gripper.close()
                self.cameras.close()


def validate_station(path):
    """Complete the configuration gate before any physical connection."""
    if not path:
        raise ValueError("hardware requires an accepted station file")
    station = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in (
        "motion_accepted",
        "home_route_accepted",
        "stop_hold_accepted",
        "gripper_accepted",
        "camera_accepted",
    ):
        if station.get(key) is not True:
            raise ValueError(f"station acceptance missing: {key}")
    q = np.asarray(station["ready_q_rad"])
    if q.shape != (6,) or not np.allclose(q, HOME, atol=1e-8, rtol=0):
        raise ValueError("station HOME differs from collection HOME")
    if not station.get("serial") or not station.get("host"):
        raise ValueError("station controller identity is required")
    for key in (
        "servo_hz",
        "home_speed",
        "home_acceleration",
        "stop_deceleration",
        "watchdog_hz",
    ):
        value = station[key]
        if (
            type(value) not in (int, float)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"invalid station {key}")
    if station["servo_hz"] < 2 * station["watchdog_hz"]:
        raise ValueError("watchdog frequency does not fit servo cadence")
    for key in ("gripper_speed", "gripper_force"):
        if type(station[key]) is not int or not 0 <= station[key] <= 255:
            raise ValueError(f"invalid {key}")
    if (
        set(station["cameras"]) != {"wrist", "third_left", "third_right"}
        or len(set(station["cameras"].values())) != 3
        or not all(station["cameras"].values())
    ):
        raise ValueError("three distinct registered cameras required")
    Limits(**station["policy_limits"])
    return station


def connect_station(path):
    """Verify serial and normal state before RTDEControl can upload a script."""
    station = validate_station(path)
    host = station["host"]
    with socket.create_connection((host, 29999), timeout=1) as dashboard:
        _line(dashboard)
        dashboard.sendall(b"get serial number\n")
        if _line(dashboard) != station["serial"]:
            raise ValueError("controller serial mismatch")
        dashboard.sendall(b"is in remote control\n")
        if _line(dashboard).lower() != "true":
            raise RuntimeError("controller must already be in remote mode")
    import rtde_receive  # pylint: disable=import-error
    import rtde_control  # pylint: disable=import-error

    receiver = rtde_receive.RTDEReceiveInterface(host)
    control = gripper = cameras = None
    try:
        if (
            receiver.getRobotMode() != 7
            or receiver.getSafetyMode() != 1
            or receiver.getRuntimeState() != 1
            or np.max(np.abs(receiver.getActualQd())) > 0.01
        ):
            raise RuntimeError(
                "controller must already be normal and stationary"
            )
        connection = socket.create_connection(
            (host, station["gripper_port"]), timeout=0.05
        )
        gripper = HandE(
            connection, station["gripper_speed"], station["gripper_force"]
        )
        gripper.position()
        cameras = Cameras(station["cameras"])
        control = rtde_control.RTDEControlInterface(host, station["servo_hz"])
        if not control.setWatchdog(station["watchdog_hz"]):
            raise RuntimeError("controller watchdog installation failed")
        return Robot(control, receiver, gripper, cameras, station)
    except BaseException:
        receiver.disconnect()
        if control is not None:
            control.stopScript()
            control.disconnect()
        if gripper is not None:
            gripper.close()
        if cameras is not None:
            cameras.close()
        raise
