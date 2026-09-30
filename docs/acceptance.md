# Pipeline landing acceptance — 2026-09-12

Original software acceptance is recorded below. Update 2026-09-15: the selected
Docker Hub dependency image plus frozen source completed five real pi05-base
training steps on two H200s; see the [Explorer acceptance](explorer-two-gpu-plan.md)
and [compact report](checks/explorer-smoke-10351656.json). Checkpoint roundtrip
job 10351772 also passed with real model/AdamW/EMA restore and online W&B
resume; see its [report](checks/explorer-roundtrip-10351772.json). Full training
completed all five segments and 5,000 optimizer steps; final checkpoint 4999
was verified through its saved scalar step, commit marker and assets. See the
[completion report](checks/explorer-full-completion.json). Station
acceptance remains open. No physical device was contacted and no
Docker Hub image was pushed by this work. The tables below retain the original
2026-09-12 evidence and its scope.

## Executed checks

| Scope | Result | What this establishes |
| --- | --- | --- |
| Owned software tests | 35 passed | HOME and hold lifecycle, lease/replay/epoch rejection, RTC queues, non-RTC backend, gripper worker isolation, export admission, transforms, normalization, launcher and checkpoint asset callbacks |
| Preserved RTC numerical tests | 8 passed on Mac and Linux amd64 image CPU | Tiny JAX/Flax configurations verify conditioning, prefix timesteps, loss masking and hard prefix sampling; copied algorithm source hashes agree |
| Official LeRobot v3 fixture | Passed | 224 rows / 3 episodes; three actual RGB video streams; episode-clamped action windows and normalization |
| Full training input batch | Passed | Local tokenizer; measured state `(8, 32)` and action tensor `(8, 50, 32)` reach inherited loader/model preprocessing |
| Black / Pylint | Passed | Owned code formatting and static checks; preserved vendor source is excluded |
| ROS process integration | Passed | Real ROS Jazzy, separate client and keyboard supervisor, fake server/robot; HOME, Space HOLD, HOME on restart, supervisor kill → FAULT |
| Linux amd64 robot image | Built and imported | ROS, RTDE and RealSense dependencies import; CLI works; physical SDK behavior remains untested |
| Linux amd64 GPU image | Built, imports and `pip check` passed | Installed package tested away from source directory with CPU-only JAX; no NVIDIA execution |
| Dataset → WebSocket → report | Passed | Five actual exported observations and one warmup through a fake model; JSON actions/error/latency output |

The ROS process test records `DISARMED → HOMING → READY → RUNNING → STOPPING →
HOLDING`, then requires HOME again and faults after supervisor termination.
This is application process testing, **not URSim or hardware stop acceptance**.
The unit suite includes Space during HOMING/READY/RUNNING, stale feedback,
missing lease, nonfinite/discontinuous actions, exclusive process ownership,
late RTC responses, changed prefixes and an unavailable physical station gate.

The checkpoint test executes the asset callback and validates preserved metadata
and normalization. It is **not** an optimizer/parameter save-and-resume training
run. The numerical tests use dummy model dimensions, not pretrained pi05 weights.
Offline action-error accounting is checked with an analytic synthetic fixture;
no learned-model accuracy or GPU speed is reported.

Raw compact outputs: [software](checks/software.txt),
[RTC numerical](checks/rtc-numerical.txt), [style](checks/style.txt),
[ROS processes](checks/ros-processes.json), [GPU imports](checks/gpu-import.txt),
[GPU dependencies](checks/gpu-dependencies.txt),
[image CPU numerical tests](checks/gpu-image-cpu-numerical.txt),
[fake-model benchmark](checks/fake-benchmark.json).

## Reproduction

From the repository root with its Python 3.12 development dependencies:

```sh
UR12E_TEST_DATASET=/absolute/exported/lerobot python -m pytest -q tests
python -m training.check
PYTHONPATH=training/pi05_rtc/vendor/openpi/src:training/pi05_rtc/vendor/openpi \
  python -m pytest -q \
  training/pi05_rtc/vendor/openpi/src/openpi/experiments/arx5_joint_rtc/model_test.py

docker build --platform linux/amd64 -f docker/Dockerfile.robot -t ur12e-infer:development .
docker run --rm --platform linux/amd64 -e ROS_DOMAIN_ID=174 \
  -v "$PWD:/app:ro" ur12e-infer:development \
  python /app/tests/ros_process_acceptance.py
```

The fixture path used here was the existing simulated exporter output at
`/Users/shengwenyuan/neu/ur12e_data_exporter/artifacts/lerobot`. It remains
read-only. The unit run produced one inherited `ml_collections` SyntaxWarning;
the numerical suite produced JAX/Flax deprecation warnings (348 in the image). Neither is suppressed
as a passing hardware or production inference result.

## Acceptance IDs

- TI01-A01/A02: software boundaries and dependency/source provenance implemented;
  CPU checks pass. Both image recipes are hash-pinned.
- TI02-A01/A02/A03: passed on the existing official simulated export.
- TI03-A01: preserved-source numerical suite passed on CPU. Full-scale GPU
  equivalence is not run. TI03-A02: dry-run and launcher rendering passed.
- TI03-A03/A04: **NOT RUN** — NVIDIA GPU, selected mounted pretrained parameters
  and a training dataset/run budget are required for smoke/save/resume.
- TI04-A01/A02: fake-protocol and source sampler components passed separately;
  a real trained-checkpoint WebSocket round trip is **NOT RUN**.
- TI04-A03, TI05-A01, TI05-A05: software/ROS fake-process checks passed.
  HOME route and measured physical settling remain unverified.
- TI05-A02/A03/A04: **NOT RUN** — URSim, real-camera shadow and physical arm/gripper
  tests. Station example acceptance flags intentionally remain false.
- TI06-A01: local image/build/import checks only. Docker Hub publishing and
  colleague GPU-host checks are not run.
- TI06-A02: report schema and synthetic error accounting passed; learned policy
  effect, RTC closed-loop behavior and GPU latency are not measured.

## Resolved integration failures and limitations

Early fixture checks found PyArrow tuple-column selection and scalar source
receipt handling errors. Both were fixed, then the complete fixture tests passed.
The first ROS process run exposed a monotonic timestamp captured before feedback
receipt; moving the check timestamp after the read resolved it. The process
acceptance was rerun successfully. These were actual failures, not skipped tests.

Hardware uses receipt-time camera freshness/skew, not the collector's accepted
capture synchronization. Station limits, HOME route clearance, servo cadence,
stop deceleration/latency/drift, controller watchdog and gripper behavior require
station-specific validation. Space clears unsent gripper targets; a request
already claimed by the I/O worker may complete and is not converted to release.
Faults are latched until a new client process. Hardware startup checks stationary,
stopped controller state, remote mode and serial before RTDEControl construction.
No automatic unlock, power-on, gripper activation or reconnect is implemented.

## Image build record

Two local development images were built before the last serving/report refinements:

- GPU: `ur12e-training:development`, local image ID
  `sha256:282abedd409281cb322d761c5b0c307c5f4662ca0605d61a03f28b18b47e90aa`.
  Package imports, `pip check` and the 8 RTC numerical tests passed on this
  image with CPU-only JAX.
- Robot: `ur12e-infer:development`, local image ID
  `sha256:e3c6321778e693de4c19b5bb4e8b1288ede50c31baa8f44668f3abe18fe77143`.
  Installed-package ROS/RTDE/RealSense imports and the actual two-process
  fake-robot test passed.

These are development artifacts, not a release of the final source snapshot.
The GPU image predates startup warmup, the explicit GPU guard and checkpoint
identity in reports. The robot control implementation tested in its image is
unchanged; later edits concern serving/reporting. Build from the delivered source
before future distribution.

On 2026-09-12 the user limited current acceptance to a simple working pipeline
and deferred resource-heavy work. The additional image refresh was explicitly
terminated at image export (exit 143); its queued robot rebuild was also stopped.
This was user-requested cancellation, not a dependency/build failure. The final
source and Docker recipes are delivered, while final-image refresh, NVIDIA
smoke/resume/inference/latency, URSim/camera and physical acceptance remain deferred.
No Docker Hub image was published. No further computational checks are pending
in the background from this task.
