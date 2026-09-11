"""Robot-side executor; network requests never own the stop/control loop."""

import argparse
from contextlib import ExitStack
import time
from websockets.sync.client import connect

from infer.client.robot import Ownership, open_robot
from infer.client.runtime import Runtime
from infer.common.contract import Capabilities
from infer.common.ros import Link
from infer.common import wire


def main(argv=None):
    """Require supervisor start and measured HOME before policy actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="ws://127.0.0.1:8000")
    parser.add_argument("--robot", choices=("fake", "hardware"), default="fake")
    parser.add_argument("--station")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--action-hz", type=float, default=10.0)
    parser.add_argument("--namespace", default="/ur12e/infer")
    parser.add_argument("--lock", default="/tmp/ur12e-control.lock")
    args = parser.parse_args(argv)
    if not 0 < args.action_hz <= 50:
        parser.error("action-hz must be in (0,50]")
    with ExitStack() as cleanup:
        owner = Ownership(args.lock)
        cleanup.callback(owner.close)
        connection = connect(
            args.server,
            open_timeout=5,
            close_timeout=1,
            max_size=wire.MAX_BYTES,
            compression=None,
        )
        cleanup.callback(connection.close)
        caps = Capabilities(**wire.unpack(connection.recv(timeout=5)))
        if caps.model == "fake" and args.robot != "fake":
            raise ValueError("fake policy requires a fake robot")
        robot = open_robot(args.robot, args.station)
        cleanup.callback(robot.close)
        link = Link("client", args.namespace)
        cleanup.callback(link.close)
        runtime = Runtime(
            robot, link, connection, caps, args.prompt, args.action_hz
        )
        # Stop motion before releasing ROS and the ownership lock.
        cleanup.callback(runtime.close)
        while True:
            now = time.monotonic()
            if not runtime.step(now):
                break
            period = 1 / getattr(robot, "station", {}).get("servo_hz", 120)
            time.sleep(max(0, period - (time.monotonic() - now)))


if __name__ == "__main__":
    main()
