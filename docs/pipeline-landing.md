# First Pipeline Landing: pi05 Training-Time RTC

> **Code style requirement: Economical code, exceptional readability, and excellent abstraction design.**

- Primary module: TI03; related modules: TI01, TI02, TI04, TI05, TI06.
- Parent: [module registry](../meta_plan.md).
- Status: implemented / GPU and station acceptance pending; authorized on 2026-09-12.
- Updated: 2026-09-12.

## Confirmed requirements

The user requests a multi-model repository at
`/Users/shengwenyuan/neu/ur12e-training-infer`. Its first training pipeline and
first inference example use pi05 training-time RTC and substantially inherit
the 1011 training implementation. Model-specific code must remain subordinate
to the training/inference architecture.

There are two local inference application processes: a client that connects
to a model server and executes actions, and an independent ROS supervisor that
interrupts the run on Space. A stopped run must leave the arm stationary via
an explicitly defined stop/hold behavior. The model server is an additional
process, usually on a GPU host; it is not counted as one of these two local
application processes. ROS infrastructure processes are also outside this count.

The user explicitly confirmed on 2026-09-12 that every retained exporter row
is one training action step, with the execution cadence configured separately.
Do not resample the dataset to physical time for this baseline. Its nominal
fps is a sequence/video indexing convention, not measured robot frequency.

Implementation is authorized on 2026-09-12. Read the two supplied reference repositories
without modifying them. The user explicitly excluded safeinfer on 2026-09-12: do not inspect, import,
vendor or depend on it. Training and inference are implemented in this repository
under the two main paths `training/` and `infer/`.

## Source audit and inheritance boundary

Inspected source: `robot-policy-training` HEAD
`f7a169c7f98e40c73afe831985bc55e0af519fb6`, with local modifications to RTC
profiles, contract, training, norm computation and experiment registration.
The untracked `kai0_arx5_shake_fold` directory includes an external RTC runtime
adapter. HEAD alone does not identify the inspected working-tree contents.
Before copying, capture a source inventory, relevant file hashes and the
working-tree patch. Do not change or commit the reference repository.

Inspected exporter: clean HEAD
`1484b15079803c159b71b4810b1443a2a3176f91`.
The target currently contains residual dotfiles, caches and a Python 3.11
virtual environment, but no visible application source or `.git` directory.
Bootstrap must inspect and preserve existing files; no bulk cleanup is implied.

| Area | Inherit | Required adaptation |
| --- | --- | --- |
| RTC algorithm | JAX/Flax pi05, clean prefix/noisy postfix, per-action timestep, postfix-only loss, hard prefix throughout sampling | Preserve numerical semantics and compare fixed inputs with the captured source |
| Training lifecycle | TOML validation, norm-stats, dry-run, smoke/train, FSDP, optimizer/schedule, EMA, checkpoint save/resume, run identity | UR names, dataset provenance, independent paths; backend-specific dependencies |
| Robot data | Relative joint action transform and absolute gripper semantics | ARX5 14D to UR12e 7D; three actual camera roles; no Aloha kinematic/gripper conversion |
| Dataset loader | Future-window behavior and numeric-only norm projection | Old `lerobot.common.datasets` APIs to official v3; exporter `export.json` instead of ARX5 `snapshot.json` |
| Deployment | Launcher structure, separate weight/data/norm/checkpoint roots; existing AIHC job rendering | Target resource configuration and model-neutral outer commands; AIHC and Docker scope aligned |
| Inference | Model RTC sampler, policy metadata, WebSocket transport concepts | Explicit per-request prefix handling and UR robot runtime integration |

The training repository's generic `Policy.infer` forwards static sample kwargs
and optional noise, but does not itself define a per-request RTC prefix protocol.
Its experiment adapter imports `pi05_arx5_inference` from external
`pi05_jax_safeinfer` (documented adapter reference: `d749a99`). Thus the complete
safe robot runtime and supervisor are not verified by reading this training
repository. It is intentionally excluded from this project. Implement the missing runtime
and supervisor locally; do not describe ordinary `serve_policy.py` as an
already complete RTC deployment.

## Proposed repository layout and boundaries

```text
training/
  common/                 # run identity, dataset admission, launcher contracts
  pi05_rtc/               # first model: transforms, model, norm, smoke/train
    vendor/openpi/        # pinned inherited implementation and narrow patches
    config/               # backend-specific TOML profiles
  launchers/              # local/remote and optional AIHC wrappers
infer/
  common/                 # observation/action contracts and backend selection
  client/                 # observation assembly, execution, session lifecycle
  supervisor/             # ROS supervision, keyboard, revocation, status
  pi05_rtc/               # model server and RTC adapter/protocol
  config/                 # identity-free station and deployment examples
docs/
```

These are proposed responsibility boundaries, not a commitment to scaffold
empty packages. Keep shared pi05 model/transform code in the training backend's
installable package; its inference server imports that pinned package rather
than maintaining another model implementation. The robot-side client never
imports the training dependency stack. Preserve upstream package names inside
the isolated vendor/backend environment to reduce migration churn. Keep source
notices and licenses with copied files.

Use an explicit small backend registry and typed contracts. Shared robot code
must not import OpenPI/JAX. Backend capabilities describe chunking/RTC support;
future models may use different horizons, tensor shapes and dependencies.
Use a fake backend to verify this boundary without implementing a second model.

Keep GPU training/serving dependencies separate from the ROS 2 Jazzy robot
runtime. Inherited Python 3.11 and exporter Python 3.12 requirements require an
explicit compatibility check; copying the existing target `.venv` is not an
environment specification. Pin compatible versions after validation, and keep
any v3 compatibility changes isolated from algorithm changes.

## TI02: Data and time contract

Read only completed official LeRobot v3 exports with successful verification
in `export.json`. Validate schema, feature names, episode boundaries, tasks,
source/fixture identity, counts and provenance. Reject incomplete output and
unexpected data contracts before norm computation or training. Do not require
or fabricate the old ARX5 snapshot structure inside an exporter dataset.

- `observation.state` and same-row `action`: float32 `[q1..q6, gripper]`.
  Joints are absolute measured radians; Hand-E is measured raw position 0–255.
- Three RGB inputs: `wrist`, `third_left`, `third_right`; preserve role identity.
  Proposed pi05 slots: third_left -> base_0_rgb, wrist -> left_wrist_0_rgb,
  third_right -> right_wrist_0_rgb. The last name is a model slot, not a second
  physical wrist. Record this mapping identically in training and serving.
- `observation.flange`: 16 row-major values representing base-from-flange.
  Preserve it for auditing; the initial joint policy need not consume it.
- Depth stays in source MCAP; first pi05 input is RGB, state and task prompt.
- Preserve integer source clocks and sample intervals. Do not use `delta_ns`
  or nominal dataset fps as a hidden execution command.

Proposed parity baseline: horizon H=50, with target row `min(i+k, episode_end)`
for k=0..49. This preserves the source's current-row first action and
end-of-episode clamping. Do not add a next-row shift or cross episode boundaries.
Training and norm statistics must use the same window/clamping rules, including
short exporter episodes. Document tail repetition in the training contract.

Encode the six target joints relative to the observation joint vector at the
window anchor; do not accumulate adjacent deltas. Keep the gripper absolute.
Pad the model representation to 32 dimensions inside pi05 only, and return
exactly seven physical dimensions after output conversion. Compute fresh
UR12e normalization assets from this projection. Never reuse ARX5 statistics.

Bind normalization/checkpoint identity to dataset provenance, window rule,
camera mapping, action encoding, gripper units and RTC profile. Verify transform
round trips and numeric norm-path equivalence to training samples.

Training sample spacing, policy action execution rate and low-level robot
servo rate are separate settings. Changing execution cadence changes physical
behavior and delay measured in seconds, so record it in each inference run.
Do not claim reconstruction of the original demonstration timing. Robot-side
interpolation or rate limiting is execution behavior, not dataset resampling;
it must preserve the scheduler's logical action-step accounting.

The exporter currently documents a flange-bearing stub input contract and
synthetic public fixtures, not completed physical collector compatibility.
Offline landing may use clearly labeled fixtures. Meaningful policy evaluation
requires compatible real data and tuned exporter sampling parameters.

## TI03: Training parity and first deliverable

Preserve the source baseline: pi05 action_dim=32, horizon=50, token length=200,
discrete state input, independent uniform delays in [0,10), prefix t=0,
postfix-only loss and exact hard-prefix sampling. Reuse the source TOML's
optimizer, schedule, seed, EMA and lifecycle defaults as a named parity profile;
actual task, dataset counts, paths and resources must come from UR inputs.

Provide model-selected commands for validate, norm-stats, smoke, train and
resume, plus a dry-run summary. Exact CLI spelling is an implementation choice.
Expose all independent path roots and capture config/source/dependency hashes.
Keep secrets outside profiles and command output. Preserve explicit full-run
intent and overwrite/resume checks; this plan does not submit a training job.

Preserve the 5-step smoke path without W&B or checkpoint saving. Add a separate
short checkpoint round-trip test: a passing no-save smoke cannot prove save,
reload, resume or inference compatibility. Support the inherited 1/8 GPU
configurations first; no unverified multi-node or arbitrary-GPU promise.

First landing is complete when an exporter fixture passes admission and norm
computation, launch commands are concrete, GPU smoke passes in the selected
environment, and a checkpoint is saved/reloaded under the same contract.
Full task training and physical inference are later acceptance stages.

## TI04/TI05: Inference and interruption

```text
GPU process: model server <---- requests/actions ----> Local P1: client/executor
                                                        | ROS status/lease
Local P2: keyboard + ROS supervisor ---------------------+
                                                        |
                                               sole robot control adapter
```

P1 owns camera/robot access for this application (or uses a separately agreed
existing observation service), receives model chunks and executes authorized
actions. Its network request path must not block the servo or stop path.
P2 independently reads keyboard input, supervises health and revokes permission.
P2 must not open a second competing motion writer. Collection and inference
must also acquire mutually exclusive station control ownership.

Aligned lifecycle: DISARMED -> HOMING -> READY -> RUNNING -> STOPPING -> HOLDING;
faults latch FAULT. Arming/start/resume are explicit operator operations.
Space revokes authority, advances the control epoch, clears queued actions and
invalidates in-flight responses, then requests bounded deceleration and hold.
No late result, server reconnect or repeated key event may restart motion.
Hand-E stopping/holding is an explicit device contract, not inferred from arm
hold. Default proposal is to preserve the gripper's held target without an
automatic open command.

Aligned Space behavior is a controlled stop and position hold. Manual
freedrive/gravity compensation is a separate deliberate mode; it is not a
substitute for the fixed-pose requirement. Exact actuator behavior and numeric
limits require the UR/Hand-E integration and physical validation.

A ROS stop message or process kill alone does not establish a hold. P1 must
check a bounded supervisor lease; robot-side watchdog behavior must stop motion
if P1 or the host stalls. Verify P1 loss, P2 loss, model server loss, ROS loss,
stale observations, empty action queues and malformed actions. Reuse collector
adapter/ownership contracts where suitable, while distinguishing its URSim
checks from unrun physical acceptance. Do not promise powered position holding
after power loss or override controller safety stops.

RTC requests carry protocol/model contract identity, request ID, control epoch,
anchor action sequence, observation, execution horizon, committed prefix and
estimated delay in logical action steps. Serialize/rebase the previous absolute
joint targets against the new observation, apply the same normalization as
training, and invert the transform on output. A normalized prefix from an old
observation must not be reused against a new anchor unchanged.

Start with no previous prefix. For steady-state RTC, protect committed actions,
compute actual delay from consumed steps, reject stale/mismatched responses,
atomically replace only valid future queue entries and stop on underflow or
unsupported delay/horizon. Implement and validate this scheduling inside `infer/`; do not depend on an
external safeinfer package. Generic websocket chunking is not evidence
that training-time RTC conditioning is active. Chunk smoothing, Kai0 variants,
Point2, online training and DAgger are outside this first baseline.

## Ordered work and acceptance

1. Align this plan and resolve the training deployment choice.
2. Capture source provenance; establish minimal backend boundaries/environments.
3. Implement and test the v3 adapter and UR data transforms.
4. Migrate RTC primitives and training lifecycle; validate parity and launchers.
5. Run separately authorized GPU smoke and checkpoint round trip.
6. Integrate server/client RTC protocol and independent ROS supervision.
7. Run software faults, URSim and camera shadow, then separately agreed physical
   tests with numerical stop/hold limits and operator procedure.

| Case | Pass criterion | Environment |
| --- | --- | --- |
| TI01-A01/A02 | Shared code imports without GPU backend; exact copied source/dependency identities recorded | Local software |
| TI02-A01 | Actual official v3 fixture loads all three videos, numeric state and task; incomplete exports rejected | Offline software |
| TI02-A02 | Independent expected windows/deltas/camera mapping agree; no cross-episode samples; 7D round trip passes | Offline software |
| TI02-A03 | Norm path agrees with training projection; dataset/encoding/profile mutations invalidate assets | Offline software |
| TI03-A01 | Fixed-input conditioning, timestep, loss and hard-prefix results match the captured source | CPU/GPU numerical tests as supported |
| TI03-A02 | Validate/dry-run and manifests expose exact config/resources/roots; no hidden task or personal path defaults | Software |
| TI03-A03 | Selected 1/8 GPU 5-step smoke completes with finite loss and expected device count | GPU, no robot |
| TI03-A04 | Short run saves/reloads correct metadata and norm assets; resume advances steps; fresh initialization stays distinct | GPU, no robot |
| TI04-A01/A02 | Prefix really reaches sampler; protected actions stay unchanged; consumed-step delay and stale response tests pass | Fake transport + GPU server |
| TI04-A03 | Fake non-RTC backend runs through client/supervisor without OpenPI dependency | Software |
| TI05-A01 | Space/fault race tests revoke epoch, clear actions, reject late responses and prevent automatic restart | Software fault injection |
| TI05-A02 | Client/supervisor/transport stalls trigger bounded stop; sole ownership and held drift pass configured limits | URSim |
| TI05-A03 | Three camera roles, state freshness, server connection and Space lifecycle operate with actuator writes disabled | Real cameras, no robot motion |
| TI05-A04 | Operator-agreed stop latency, settling displacement and held drift limits pass; gripper and each fault path recorded | Physical UR12e/Hand-E, separately aligned |
| TI05-A05 | Every start requires measured HOME and settling; Space interrupts HOME; a fault cannot grant authority | Software passed; physical route/settling NOT RUN |

## Alignment record and open decisions

Confirmed: multi-model architecture; pi05-RTC first training and inference;
two local inference processes; Space interruption; reference repositories
read-only; retained-row training with separately configured execution cadence;
no safeinfer inspection/dependency; local training and inference implementation
under the two main paths `training/` and `infer/`.

On 2026-09-12 the user approved the concrete plan, default position hold,
AIHC/Docker inheritance and implementation. Versions are an engineering choice.
The user additionally requires GO HOME before every model-control handoff.
HOME is [0, -pi/2, -pi/2, -pi/2, pi/2, 0] rad, as recorded in the collector's
`docs/m06-control-motion/home-proposal.md`; never wrap its joint branches.
Measure arrival and settling before READY. Every new start after HOLD requires
HOME again. Space interrupts HOMING as well as RUNNING; it never returns HOME.
A HOME timeout/fault cannot grant policy authority. Preserve a single HOME
configuration and reject a station value that differs from this adopted pose.
Add TI05-A05 for this HOME-before-policy gate and interruption at every phase.

Docker delivery includes a GPU training/serving image and a separate ROS Jazzy
robot image, explicit mounts and a portable local/AIHC launch interface.
No cloud job submission or physical motion follows from implementation approval.

Before GPU execution: select dataset/task, hardware/image, storage roots,
training budget and numerical run parameters. Before physical inference: agree
control integration, gripper hold behavior, policy/servo rates, freshness/lease
limits, stop latency and held drift thresholds. Unknown numbers are not acceptance
criteria that have already passed.

## Implementation and validation results

Implementation is delivered under `training/` and `infer/`. See
[acceptance.md](acceptance.md) for the actual checks and per-ID conclusions:
35 owned tests, 8 preserved numerical tests, official v3 video/batch validation
and real ROS process tests with a fake robot. Docker environment checks are
recorded there separately. GPU training/checkpoint resume, learned-policy GPU
serving, URSim, real cameras and physical motion remain NOT RUN.
No cloud job or Docker Hub publishing has been performed.

## TI06: Colleague image evaluation (aligned 2026-09-12)

The user requests future Docker Hub image sharing for GPU inference effect and
speed tests. Deliver versioned GPU/robot image builds and explicit publish
commands. No image push is performed without a publishing request. Colleagues
mount checkpoints and held-out datasets, run the model server or an in-process
benchmark, and receive JSON with environment identity, warmup, P50/P95/P99
latency and per-window action predictions. Offline joint/gripper errors exclude
the current-row target and padded tail; they do not establish closed-loop robot
task success. Physical evaluation remains a separate acceptance stage.

## Mounted model assets (aligned 2026-09-12)

The user explicitly excludes pretrained-checkpoint downloading from repository
responsibility. Training initialization and inference checkpoints must be local
mounted directories; the repository supplies path arguments only. Tokenizer
resources also use `--tokenizer` / `UR12E_TOKENIZER_PATH`, with no implicit
download. Checkpoint metadata binds the tokenizer content hash. Docker images
contain code/dependencies only, never pretrained or fine-tuned checkpoints.

## Resource-limited delivery decision (2026-09-12)

The user requests only simple pipeline validation on the Mac. Stop extra image
rebuilds and compute-heavy checks; defer GPU training, learned-model performance
and physical acceptance until resources are available. Existing software, fake
ROS process, exported-data and environment results are sufficient for this
first code delivery. The additional final-image refresh was explicitly canceled;
see acceptance.md for the exact tested image versions and remaining work.
