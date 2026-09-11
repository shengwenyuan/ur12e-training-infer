"""Behavior tests for HOME, revocation and asynchronous chunk races."""

# Pytest fixture names and explicit case names supply the test interface.
# pylint: disable=missing-function-docstring,redefined-outer-name,import-outside-toplevel,duplicate-code


import numpy as np
import pytest
from infer.common.contract import HOME, Capabilities
from infer.common.supervision import Receiver
from infer.common import wire
from infer.client.lifecycle import Session
from infer.client.robot import FakeRobot, Ownership
from infer.client.rtc import Scheduler
from infer.pi05_rtc.server import respond, FakePolicy
from infer.client.hardware import validate_station


def feedback(now=1, q=None, qd=None):
    return {
        "state": np.r_[HOME, 0.0] if q is None else q,
        "velocity": np.zeros(6) if qd is None else qd,
        "received": now,
        "healthy": True,
    }


def ready():
    robot = FakeRobot()
    session = Session(robot)
    session.renew(1)
    session.start(1, feedback())
    session.tick(1, feedback())
    session.renew(1.31)
    session.tick(1.31, feedback(1.31))
    assert session.phase == "READY"
    return session, robot


def test_policy_blocked_until_measured_home_and_settling():
    robot = FakeRobot()
    session = Session(robot)
    session.renew(1)
    session.start(1, feedback())
    wrong = np.r_[HOME + 0.5, 0.0]
    session.tick(1.1, feedback(1.1, wrong))
    with pytest.raises(RuntimeError):
        session.execute(np.r_[HOME, 0.0], 1.1, feedback(1.1, wrong), 0.1)
    assert robot.writes == ["HOME"]


@pytest.mark.parametrize("phase", ["HOMING", "READY", "RUNNING"])
def test_space_revokes_every_phase_and_holds_without_home(phase):
    session, robot = ready()
    session.phase = phase
    session.stop()
    assert session.phase == "STOPPING" and robot.writes[-1] == "HOLD"
    with pytest.raises(RuntimeError):
        session.execute(np.r_[HOME, 0.0], 1.32, feedback(1.32), 0.1)
    session.tick(1.4, feedback(1.4))
    session.tick(1.71, feedback(1.71))
    assert session.phase == "HOLDING"
    session.renew(1.72)
    session.start(1.72, feedback(1.72))
    assert session.phase == "HOMING" and robot.writes == [
        "HOME",
        "HOLD",
        "HOME",
    ]


def test_stationary_off_home_never_grants_policy():
    session = Session(FakeRobot())
    session.renew(1)
    session.start(1, feedback())
    for now in (1.1, 1.5, 2.0):
        session.renew(now)
        session.tick(now, feedback(now, np.zeros(7)))
    assert session.phase == "HOMING"


@pytest.mark.parametrize("fault", ["lease", "stale", "health", "nan"])
def test_faults_never_write_actions(fault):
    session, robot = ready()
    state = feedback(1.32)
    now = 1.32
    if fault == "lease":
        now = 2
    if fault == "stale":
        state["received"] = 0
    if fault == "health":
        state["healthy"] = False
    if fault == "nan":
        state["state"][0] = np.nan
    if fault == "nan":
        with pytest.raises(ValueError):
            session.execute(np.r_[HOME, 0.0], now, state, 0.1)
    else:
        with pytest.raises(RuntimeError):
            session.execute(np.r_[HOME, 0.0], now, state, 0.1)
        assert session.phase == "FAULT"
    assert "ACTION" not in robot.writes


def test_nonfinite_or_discontinuous_action_stops():
    session, robot = ready()
    target = np.r_[HOME, 0.0]
    target[0] = 1
    with pytest.raises(ValueError):
        session.execute(target, 1.32, feedback(1.32), 0.1)
    assert session.phase == "FAULT" and robot.writes[-1] == "HOLD"


def test_supervisor_lease_admission_rejects_replay_and_boot_takeover():
    receiver = Receiver("test")
    msg = {
        "session": "test",
        "boot": "a",
        "sequence": 1,
        "sent": 1.0,
        "operation": "heartbeat",
    }
    assert receiver.accept(msg, 1.1)
    assert not receiver.accept(msg, 1.1)
    assert not receiver.accept(msg | {"sequence": 2, "sent": 2.0}, 1.2)
    assert not receiver.accept(msg | {"sequence": 2}, 1.4)
    assert not receiver.accept(msg | {"sequence": 2, "boot": "b"}, 1.2)
    assert receiver.accept(msg | {"sequence": 2, "operation": "stop"}, 1.2)


def test_exclusive_device_owner(tmp_path):
    first = Ownership(tmp_path / "lock")
    with pytest.raises(BlockingIOError):
        Ownership(tmp_path / "lock")
    first.close()
    Ownership(tmp_path / "lock").close()


def response(request, values):
    return {
        k: request[k]
        for k in (
            "protocol",
            "request_id",
            "epoch",
            "anchor",
            "contract_id",
            "model",
        )
    } | {"actions": values}


def test_rtc_consumed_delay_and_late_epoch_are_independent_of_wall_time():
    scheduler = Scheduler(Capabilities("pi05_rtc", 50, True, 10, "test"))
    actions = np.tile(np.r_[HOME, 0.0], (50, 1))
    first = scheduler.request({})
    assert scheduler.accept(response(first, actions))
    for _ in range(20):
        scheduler.pop()
    request = scheduler.request({})
    assert request["delay"] == 1
    for _ in range(3):
        scheduler.pop()
    assert scheduler.accept(response(request, actions))
    assert len(scheduler.queue) == 47 and scheduler.delay == 3
    scheduler.reset()
    assert (
        not scheduler.accept(response(request, actions)) and not scheduler.queue
    )


def test_rtc_protected_prefix_mismatch_and_unsupported_delay():
    scheduler = Scheduler(Capabilities("pi05_rtc", 50, True, 10, "test"))
    actions = np.tile(np.r_[HOME, 0.0], (50, 1))
    first = scheduler.request({})
    scheduler.accept(response(first, actions))
    for _ in range(20):
        scheduler.pop()
    request = scheduler.request({})
    changed = actions.copy()
    changed[0, 0] = 1
    with pytest.raises(ValueError, match="prefix"):
        scheduler.accept(response(request, changed))
    for _ in range(10):
        scheduler.pop()
    with pytest.raises(ValueError, match="support"):
        scheduler.accept(response(request, actions))


def test_fake_backend_has_different_horizon_and_no_rtc():
    policy = FakePolicy()
    scheduler = Scheduler(policy.capabilities, 4)
    request = scheduler.request({"state": np.r_[HOME, 0.0]})
    result = respond(policy, wire.unpack(wire.pack(request)))
    assert scheduler.accept(wire.unpack(wire.pack(result)))
    assert scheduler.pop().shape == (7,) and len(scheduler.queue) == 7


def test_wire_rejects_object_arrays_and_bad_shapes():
    with pytest.raises(ValueError):
        wire.pack(np.array([object()]))
    import msgpack

    bad = msgpack.packb(
        {
            "__array__": True,
            "dtype": "f8",
            "shape": [1000000, 1000000],
            "data": b"1",
        },
        use_bin_type=True,
    )
    with pytest.raises(ValueError):
        wire.unpack(bad)


def test_hardware_example_cannot_connect():
    with pytest.raises(ValueError, match="acceptance"):
        validate_station("infer/config/station.example.json")


def test_home_preflight_rejects_stale_feedback_before_motion():
    robot = FakeRobot()
    session = Session(robot)
    session.renew(1)
    with pytest.raises(RuntimeError):
        session.start(1, feedback(0))
    assert robot.writes == ["HOLD"] and session.phase == "FAULT"


def test_home_timeout_and_stop_failure_cannot_report_ready():
    robot = FakeRobot()
    session = Session(robot)
    session.renew(1)
    session.start(1, feedback())
    session.renew(47)
    session.tick(47, feedback(47, np.zeros(7)))
    assert session.phase == "FAULT"

    def failed_stop():
        raise OSError("disconnected")

    robot.stop_hold = failed_stop
    with pytest.raises(OSError):
        session.stop()
    assert (
        session.phase == "FAULT" and session.reason == "stop/hold unconfirmed"
    )


def test_benchmark_excludes_current_row_and_repeated_episode_tail():
    from infer.benchmark import measure

    class Dataset:
        """Three rows with analytically known future errors."""

        states = np.array(
            [[i] * 6 + [i * 10] for i in range(3)], dtype=np.float32
        )
        ends = np.array([2, 2, 2])
        identity = "fixture"
        manifest = {"simulated": True}

        def __len__(self):
            return 3

        def official(self):
            return [
                {
                    "task": "fixture",
                    **{
                        f"observation.images.{role}": np.zeros((3, 4, 5))
                        for role in ("wrist", "third_left", "third_right")
                    },
                }
            ] * 3

        def window(self, index, horizon):
            return self.states[np.minimum(np.arange(index, index + horizon), 2)]

    policy = FakePolicy()
    result = measure(
        Dataset(),
        policy.capabilities,
        lambda request: respond(policy, request),
        3,
        1,
    )
    assert result["joint_mae_rad"] == pytest.approx(4 / 3)
    assert result["gripper_mae_raw"] == pytest.approx(40 / 3)
    assert [row["valid_future_steps"] for row in result["rows"]] == [2, 1, 0]
    assert len(result["warmup_ms"]) == 1 and result["latency_ms"]["p95"] >= 0


def test_single_step_non_rtc_backend_is_supported():
    scheduler = Scheduler(Capabilities("single", 1, False, 0, "single-v1"), 1)
    request = scheduler.request({})
    scheduler.accept(response(request, np.tile(np.r_[HOME, 0.0], (1, 1))))
    assert scheduler.pop().shape == (7,)


def test_model_assets_require_a_local_mount(monkeypatch, tmp_path):
    from training.pi05_rtc.backend import local_assets

    monkeypatch.delenv("UR12E_TOKENIZER_PATH", raising=False)
    with pytest.raises(ValueError, match="mounted"):
        local_assets()
    path = tmp_path / "tokenizer.model"
    path.write_bytes(b"fixture")
    assert local_assets(path) == path
    assert __import__("os").environ["HF_HUB_OFFLINE"] == "1"


def test_space_cannot_clear_latched_fault():
    session, robot = ready()
    session.stop("lease lost", fault=True)
    session.stop()
    session.tick(2, feedback(2))
    session.tick(3, feedback(3))
    assert session.phase == "FAULT" and robot.writes[-1] == "HOLD"
    session.renew(3)
    with pytest.raises(RuntimeError):
        session.start(3, feedback(3))


def test_gripper_socket_stall_does_not_block_stop_or_snapshot():
    import threading
    from infer.client.gripper import GripperWorker

    entered = threading.Event()
    release = threading.Event()

    class Device:
        """Hold a polling read until the control-thread assertions finish."""

        calls = 0

        def position(self):
            self.calls += 1
            if self.calls > 1:
                entered.set()
                assert release.wait(2)
            return 12

        def move(self, value):
            raise AssertionError(f"canceled target sent: {value}")

        def close(self):
            release.set()

    worker = GripperWorker(Device())
    try:
        assert entered.wait(1)
        worker.move(42)
        worker.cancel_pending()
        assert worker.snapshot() == 12
        assert worker.pending is None
    finally:
        worker.close()


def test_server_warmup_has_no_robot_or_rtc_history():
    from infer.pi05_rtc.server import warmup

    class Model:
        """Capture only the model request; no actuator interface exists."""

        def infer(self, request):
            assert request["delay"] == 0
            assert request["prefix"].shape == (0, 7)
            np.testing.assert_array_equal(
                request["observation"]["state"][:6], HOME
            )
            assert set(request["observation"]["images"]) == {
                "wrist",
                "third_left",
                "third_right",
            }
            assert all(
                image.shape == (224, 224, 3)
                for image in request["observation"]["images"].values()
            )

    warmup(Model())
