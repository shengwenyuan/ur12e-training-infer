# UR12e training and inference

Multi-model training and inference with pi05 training-time RTC as the first
backend. Training inherits the pinned 1011 implementation; inference is local
code with a model server and two robot-side processes. No safeinfer dependency.

- `training/`: exported dataset admission, model-specific training and launchers.
- `infer/`: model serving, action client, independent ROS supervisor and evaluation.
- `docs/`: [aligned plan](docs/pipeline-landing.md), source provenance and results.

HOME is `[0, -90, -90, -90, 90, 0]` degrees, in the collector's joint order.
The supervisor's **r** starts GO HOME, verifies arrival/settling, then permits
policy actions. **Space** interrupts HOME or policy execution and holds the
attained pose. A later **r** requires HOME again. **q** stops and exits.

## Development

Python 3.12 is the selected baseline. GPU and ROS dependencies are separate.
The complete GPU environment is hash-locked for Linux x86_64. CPU development
uses the same numerical JAX/Flax versions without CUDA libraries.

```sh
uv venv --python 3.12 .venv-py312
uv pip install --python .venv-py312/bin/python -r training/pi05_rtc/requirements-cpu.in
uv pip install --python .venv-py312/bin/python --no-deps -e .
.venv-py312/bin/python -m training.check
.venv-py312/bin/python -m pytest -q
```

The pre-existing `.venv` was preserved and is not the validated environment;
use the fresh Python 3.12 environment above or the supplied images.

Dataset tests require `UR12E_TEST_DATASET=/absolute/exported/lerobot`. They read
an existing exporter fixture and never invoke the exporter or hardware.
Preserved numerical model tests are also available in the vendor RTC directory;
see [acceptance](docs/acceptance.md) for the exact command and tested scope.

## Training

Pretrained/checkpoint downloading is outside this repository. Supply mounted
local parameter/checkpoint directories and a tokenizer file. Runtime does not
download model resources; tokenizer identity is checked against each checkpoint.

Use a completed official LeRobot v3 export. State/action are measured six joints
in radians plus raw Hand-E position. Training forms current-row future windows,
uses relative joint targets and absolute gripper values, and never crosses an
episode boundary. Every retained row is one action step, independent of original
physical timestamps or the later execution cadence. Three RGB views are required.

```sh
ur12e-train validate --dataset /datasets/task
ur12e-train norm-stats --dataset /datasets/task --norm-root /output/norm/task
ur12e-train smoke --dataset /datasets/task --norm-root /output/norm/task \
  --initialization /weights/pi05_base/params --tokenizer /weights/paligemma_tokenizer.model \
  --checkpoint-root /output/checkpoints \
  --config training/pi05_rtc/config/profile.toml --dry-run
```

Remove `--dry-run` on a configured GPU host to execute the five-step smoke.
Use `train` for a checkpoint-saving run. Choose a new profile run name for a new
run; `--resume` explicitly resumes a compatible saved run. The default profile
uses 1 GPU, batch 8, 10,000 steps, inherited optimizer/schedule and RTC delay
range [0,10). It also supports an explicitly configured 8-GPU FSDP run. W&B is
disabled by default; enabling it uses the host's ordinary W&B environment.
Fresh norm outputs and existing run directories are never silently replaced.
Fixtures require `--allow-simulated`; their training is not task-performance data.

AIHC manifests preserve the 1011 job schema and independently configured
read-only dataset/weight mounts and writable output/cache mounts. They are
rendered, not submitted:

```sh
python -m training.launchers.job --deployment /config/deployment.json -- \
  python -m training.cli smoke --dataset /datasets/task \
  --norm-root /output/norm/task --initialization /weights/pi05_base/params \
  --tokenizer /weights/paligemma_tokenizer.model \
  --checkpoint-root /output/checkpoints
```

Use `training/launchers/deployment.example.json` as a template. Replace queue,
image and storage placeholders with actual cloud resources before submission.

## Inference

Start the model server in the GPU environment, using a numbered checkpoint
folder containing `params/` and `assets/`:

```sh
ur12e-model-server --model pi05_rtc --checkpoint /checkpoints/pi05_rtc/your-run/9999 \
  --tokenizer /weights/paligemma_tokenizer.model
```

The pi05 server requires a visible GPU and completes model warmup before
opening its socket. Wait for server startup before launching the client.

In two ROS Jazzy terminals on the robot computer:

```sh
ur12e-infer --robot fake --server ws://GPU_HOST:8000 --prompt 'pick up the object'
ur12e-supervisor
```

The server binds loopback by default. Use `--host 0.0.0.0` on a deliberately
exposed server and configure network access. For a software-only end-to-end
check, use `ur12e-model-server --model fake`; this backend uses 8-step chunks
without RTC and has no model dependencies.

Hardware mode requires an explicitly accepted station JSON based on
`infer/config/station.example.json`. The example cannot connect to hardware.
Its motion settings are candidates, not physical acceptance. Match collection's
HOME and shared control-lock path, register serials/cameras, and complete
route/hold/gripper/camera acceptance before enabling flags. The constructor
checks the controller serial and normal stationary readback before opening a
motion interface. The single control loop owns RTDE/watchdog. Gripper socket polling uses a
separate thread; Space discards unsent grasp targets and preserves an already
issued grasp request. Camera threads
never command actuators. No gripper activation, fault unlocking or reconnect is
automatic. Physical cadence, route clearance and hold performance remain to be
measured on the actual station.

Camera snapshot admission currently uses host receipt freshness/skew. It does
not claim the collector's capture-time synchronization acceptance. Source depth
is not a pi05 input. Robot policy action frequency is independently configured
with `--action-hz`; it is not derived from exporter fps.

## Docker and colleague evaluation

See [Docker Hub colleague guide](docs/docker-colleague-guide.md) for versioned
builds, external mounts, GPU model serving and offline error/latency reports.
Both images target Linux amd64. Runtime users can override UID/GID to match
mounted directories; examples never require a privileged container by default.

The current status and unrun GPU/physical cases are recorded in
[acceptance](docs/acceptance.md). No Docker Hub image has been published by this
implementation task.
