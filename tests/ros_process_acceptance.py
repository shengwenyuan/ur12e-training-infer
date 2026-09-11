"""Run the real ROS/keyboard entrypoints with an explicitly fake robot."""

import os
from pathlib import Path
import pty
import socket
import subprocess
import sys
import tempfile
import time

import rclpy
from std_msgs.msg import String
import json


def main():
    """Verify HOME, RUNNING, Space HOLD, restart HOME and supervisor loss."""
    rclpy.init()
    node = rclpy.create_node("acceptance_observer")
    states = []
    node.create_subscription(
        String,
        "/ur12e/infer/status",
        lambda message: states.append(json.loads(message.data)),
        10,
    )

    def wait_for(predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if predicate():
                return
        raise AssertionError(f"timeout; recent states: {states[-5:]}")

    processes = []
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "processes.log"
        with output.open("w", encoding="utf-8") as log:
            try:
                server = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "infer.pi05_rtc.server",
                        "--model",
                        "fake",
                        "--port",
                        "8123",
                    ],
                    stdout=log,
                    stderr=log,
                )
                processes.append(server)
                deadline = time.monotonic() + 10
                while True:
                    try:
                        with socket.create_connection(
                            ("127.0.0.1", 8123), timeout=0.1
                        ):
                            break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise
                        time.sleep(0.05)
                client = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "infer.client.main",
                        "--server",
                        "ws://127.0.0.1:8123",
                        "--robot",
                        "fake",
                        "--prompt",
                        "integration fixture",
                        "--lock",
                        directory + "/owner.lock",
                    ],
                    stdout=log,
                    stderr=log,
                )
                processes.append(client)
                master, slave = pty.openpty()
                supervisor = subprocess.Popen(
                    [sys.executable, "-m", "infer.supervisor.main"],
                    stdin=slave,
                    stdout=log,
                    stderr=log,
                )
                processes.append(supervisor)
                os.close(slave)
                wait_for(
                    lambda: bool(states) and states[-1]["phase"] == "DISARMED"
                )
                time.sleep(1)
                os.write(master, b"r")
                wait_for(lambda: states[-1]["phase"] == "RUNNING")
                assert any(s["phase"] == "HOMING" for s in states)
                os.write(master, b" ")
                wait_for(lambda: states[-1]["phase"] == "HOLDING")
                boundary = len(states)
                wait_for(lambda: len(states) > boundary + 5)
                assert all(s["phase"] == "HOLDING" for s in states[boundary:])
                os.write(master, b"r")
                wait_for(lambda: states[-1]["phase"] == "RUNNING")
                assert any(s["phase"] == "HOMING" for s in states[boundary:])
                supervisor.kill()
                supervisor.wait(timeout=3)
                wait_for(lambda: states[-1]["phase"] == "FAULT", timeout=3)
                print(
                    json.dumps(
                        {
                            "result": "PASS",
                            "phases": list(
                                dict.fromkeys(s["phase"] for s in states)
                            ),
                            "checks": [
                                "HOME before policy",
                                "Space hold",
                                "HOME on restart",
                                "supervisor kill revokes client",
                            ],
                        },
                        indent=2,
                    )
                )
                os.close(master)
            except BaseException:
                log.flush()
                print(output.read_text(encoding="utf-8"))
                raise
            finally:
                for process in reversed(processes):
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                node.destroy_node()
                rclpy.shutdown()


if __name__ == "__main__":
    main()
