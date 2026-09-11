"""Independent keyboard supervisor: r starts via HOME, Space holds, q quits."""

import argparse
import select
import sys
import termios
import time
import tty
import uuid

from infer.common.ros import Link


def main(argv=None):
    """Heartbeat only while the client reports a fresh matching session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="/ur12e/infer")
    args = parser.parse_args(argv)
    if not sys.stdin.isatty():
        parser.error(
            "interactive TTY required; attach a terminal to supervisor"
        )
    link = Link("supervisor", args.namespace)
    boot = uuid.uuid4().hex
    sequence = 0
    status = {}
    original = termios.tcgetattr(sys.stdin)
    print("r: GO HOME then policy | Space: stop and hold | q: quit", flush=True)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            now = time.monotonic()
            for update in link.poll():
                if not status or update.get("session") == status["session"]:
                    status = update
            ready = select.select([sys.stdin], [], [], 0.02)[0]
            key = sys.stdin.read(1) if ready else ""
            operation = {"r": "start", " ": "stop", "q": "quit"}.get(
                key, "heartbeat"
            )
            if status and (
                operation in ("stop", "quit")
                or 0 <= now - status.get("sent", 0) <= 0.3
            ):
                sequence += 1
                link.send(
                    {
                        "boot": boot,
                        "session": status["session"],
                        "sequence": sequence,
                        "sent": time.monotonic(),
                        "operation": operation,
                    }
                )
            if key == "q":
                break
    except KeyboardInterrupt:
        pass
    finally:
        # Loss of these heartbeats also revokes P1's bounded lease.
        if status:
            link.send(
                {
                    "boot": boot,
                    "session": status["session"],
                    "sequence": sequence + 1,
                    "sent": time.monotonic(),
                    "operation": "stop",
                }
            )
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original)
        link.close()


if __name__ == "__main__":
    main()
