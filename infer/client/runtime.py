"""One nonblocking control-loop iteration, isolated from CLI and ROS setup."""

from concurrent.futures import ThreadPoolExecutor
import time

from infer.client.lifecycle import Limits, Session
from infer.client.rtc import Scheduler
from infer.common.supervision import Receiver
from infer.common import wire


class Runtime:
    """Own the session, scheduler and a single bounded network worker."""

    # This owner keeps its asynchronous request and control state together.
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, robot, link, connection, capabilities, prompt, action_hz
    ):
        self.robot = robot
        self.link = link
        self.connection = connection
        self.prompt = prompt
        self.action_period = 1 / action_hz
        self.session = Session(
            robot,
            Limits(**getattr(robot, "station", {}).get("policy_limits", {})),
        )
        self.receiver = Receiver()
        self.scheduler = Scheduler(
            capabilities, min(20, max(1, capabilities.horizon // 2))
        )
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.future_epoch = None
        self.request_started = 0.0
        self.next_action = 0.0
        self.next_status = 0.0

    def _query(self, request):
        self.connection.send(wire.pack(request))
        return wire.unpack(self.connection.recv(timeout=3))

    def _stop(self, reason, fault=False):
        self.scheduler.reset()
        self.session.stop(reason, fault=fault)

    def _control(self, now):
        for message in self.link.poll():
            now = time.monotonic()
            if not self.receiver.accept(message, now):
                continue
            self.session.renew(now)
            operation = message["operation"]
            if operation == "start" and self.session.phase in (
                "DISARMED",
                "HOLDING",
            ):
                self.scheduler.reset()
                feedback = self.robot.feedback()
                self.session.start(time.monotonic(), feedback)
            elif operation in ("stop", "quit"):
                self._stop("Space")
                if operation == "quit":
                    return False
        return True

    def _result(self):
        if self.future is None or not self.future.done():
            return
        current = (
            self.future_epoch == self.scheduler.epoch
            and self.session.phase in ("READY", "RUNNING")
        )
        try:
            response = self.future.result()
            if current:
                self.scheduler.accept(response)
        except Exception as error:  # pylint: disable=broad-exception-caught
            # Any transport/model exception revokes authority.
            if current:
                self._stop(str(error), fault=True)
        self.future = None

    def _infer(self, now):
        if self.future is None:
            request = self.scheduler.request(
                self.robot.observation(self.prompt)
            )
            if request is not None:
                self.future_epoch = self.scheduler.epoch
                self.request_started = now
                self.future = self.pool.submit(self._query, request)
        if self.future is not None and now - self.request_started > 3:
            self._stop("model server timeout", fault=True)
        elif self.scheduler.accepted and now >= self.next_action:
            try:
                action = self.scheduler.pop()
                feedback = self.robot.feedback()
                self.session.execute(
                    action,
                    time.monotonic(),
                    feedback,
                    self.action_period,
                )
                self.next_action = now + self.action_period
            except Exception as error:  # pylint: disable=broad-exception-caught
                # SDK exceptions must never leave queued actions executable.
                self.scheduler.reset()
                if self.session.phase != "FAULT":
                    self.session.stop(str(error), fault=True)

    def step(self, now: float) -> bool:
        """Process stop before responses/targets, and recheck lease before
        I/O.
        """
        if not self._control(now):
            return False
        phase = self.session.phase
        feedback = self.robot.feedback()
        self.session.tick(time.monotonic(), feedback)
        if self.session.phase == "FAULT" and phase != "FAULT":
            self.scheduler.reset()
        self._result()
        if self.session.phase in ("READY", "RUNNING"):
            self._infer(now)
        if now >= self.next_status:
            self.link.send(
                {
                    "session": self.receiver.session_id,
                    "phase": self.session.phase,
                    "reason": self.session.reason,
                    "epoch": self.session.epoch,
                    "sent": now,
                    "simulated": self.robot.simulated,
                }
            )
            self.next_status = now + 0.05
        self.robot.heartbeat()
        return True

    def close(self):
        """Revoke and stop before waiting for any bounded network cleanup."""
        try:
            self._stop("client shutdown")
        finally:
            try:
                self.connection.close()
            finally:
                self.pool.shutdown(wait=True, cancel_futures=True)
