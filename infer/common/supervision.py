"""Admit fresh, ordered supervisor commands; stale ROS queues cannot arm."""

# Protocol numbers must exclude bool, which is an int subclass.
# pylint: disable=unidiomatic-typecheck


import math
import uuid


class Receiver:
    """Bind a client session to one supervisor boot identity and sequence."""

    def __init__(self, session_id=None, max_age=0.3):
        self.session_id = session_id or uuid.uuid4().hex
        self.boot = None
        self.sequence = -1
        self.max_age = max_age

    def accept(self, message, now):
        """Reject replay, future timestamps and unexpected supervisor
        takeover.
        """
        if message.get("session") != self.session_id:
            return False
        sent = message.get("sent")
        sequence = message.get("sequence")
        if (
            type(sent) not in (int, float)
            or not math.isfinite(sent)
            or not 0 <= now - sent <= self.max_age
        ):
            return False
        if (
            type(sequence) is not int
            or sequence <= self.sequence
            or not isinstance(message.get("boot"), str)
        ):
            return False
        if self.boot is not None and message["boot"] != self.boot:
            return False
        if message.get("operation") not in (
            "heartbeat",
            "start",
            "stop",
            "quit",
        ):
            return False
        self.boot = message["boot"]
        self.sequence = sequence
        return True
