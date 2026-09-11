"""Single-inflight chunk scheduling using consumed action steps, not wall
time.
"""

from collections import deque
import uuid
import numpy as np

from infer.common.contract import PROTOCOL, chunk


class Scheduler:
    """An epoch owns one queue; late responses never revive a stopped run."""

    def __init__(self, capabilities, execution_steps=20):
        self.capabilities = capabilities
        maximum = (
            capabilities.horizon // 2
            if capabilities.rtc
            else capabilities.horizon
        )
        if not 1 <= execution_steps <= maximum:
            raise ValueError("execution steps must fit two windows")
        self.execution_steps = execution_steps
        self.epoch = 0
        self.consumed = 0
        self.queue = deque()
        self.pending = None
        self.next_request = 0
        self.delay = 0
        self.accepted = False

    def reset(self):
        """Invalidate all old responses and committed future targets."""
        self.epoch += 1
        self.consumed = 0
        self.queue.clear()
        self.pending = None
        self.accepted = False
        self.next_request = 0
        self.delay = 0

    def request(self, observation):
        """Snapshot the previous absolute tail; server rebases it to this
        state.
        """
        if self.pending is not None or self.consumed < self.next_request:
            return None
        caps = self.capabilities
        delay = (
            min(self.delay + 1, caps.max_delay - 1, len(self.queue))
            if caps.rtc and self.accepted
            else 0
        )
        delay = max(delay, 0)
        prefix = np.asarray(list(self.queue)[:delay], dtype=np.float64).reshape(
            -1, 7
        )
        request = {
            "protocol": PROTOCOL,
            "request_id": uuid.uuid4().hex,
            "epoch": self.epoch,
            "anchor": self.consumed,
            "contract_id": caps.contract_id,
            "model": caps.model,
            "observation": observation,
            "prefix": prefix,
            "delay": delay,
        }
        self.pending = request
        return request

    def accept(self, response):
        """Install only still-future actions and enforce protected prefix
        parity.
        """
        request = self.pending
        if request is None or response.get("epoch") != self.epoch:
            return False
        for key in ("protocol", "request_id", "anchor", "contract_id", "model"):
            if response.get(key) != request[key]:
                raise ValueError(f"response {key} mismatch")
        values = chunk(response["actions"], self.capabilities.horizon)
        delay = self.consumed - request["anchor"]
        if delay > self.execution_steps or delay + self.execution_steps > len(
            values
        ):
            raise ValueError("RTC response exceeded execution horizon")
        if self.capabilities.rtc and delay >= self.capabilities.max_delay:
            raise ValueError("RTC delay outside training support")
        protected = request["delay"]
        if protected and not np.allclose(
            values[:protected], request["prefix"], atol=1e-5, rtol=0
        ):
            raise ValueError("hard prefix changed")
        self.queue = deque(values[delay:].copy())
        self.delay = delay
        self.pending = None
        self.accepted = True
        self.next_request = request["anchor"] + self.execution_steps
        return True

    def pop(self):
        """Consume exactly one logical target; starvation is a stop
        condition.
        """
        if not self.queue:
            raise RuntimeError("action queue underrun")
        self.consumed += 1
        return self.queue.popleft()
