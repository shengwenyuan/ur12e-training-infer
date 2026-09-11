"""Keep bounded gripper socket traffic off the arm control loop."""

import threading
import time


class GripperWorker:
    """One socket owner, latest target and a measured feedback deadline."""

    def __init__(self, device):
        self.device = device
        self.state = (device.position(), time.monotonic())
        self.error = None
        self.pending = None
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.closed.is_set():
                with self.lock:
                    target, self.pending = self.pending, None
                # Once claimed, a grasp request may finish during an arm stop.
                # No socket I/O ever holds the control-loop snapshot lock.
                if target is not None:
                    self.device.move(target)
                position = self.device.position()
                with self.lock:
                    self.state = (position, time.monotonic())
                self.closed.wait(0.05)
        except Exception as error:  # pylint: disable=broad-exception-caught
            with self.lock:
                self.error = error

    def snapshot(self):
        """Fail closed on missing gripper feedback without waiting for I/O."""
        with self.lock:
            position, received = self.state
            error = self.error
        if error is not None or time.monotonic() - received > 0.25:
            raise RuntimeError("gripper feedback unavailable") from error
        return position

    def move(self, value):
        """Replace an unsent request; do not enqueue an unbounded history."""
        with self.lock:
            self.pending = value

    def cancel_pending(self):
        """Discard unsent targets while preserving the issued grasp."""
        with self.lock:
            self.pending = None

    def close(self):
        """Close only after the caller has stopped the arm."""
        self.closed.set()
        self.device.close()
        self.thread.join(timeout=1)
