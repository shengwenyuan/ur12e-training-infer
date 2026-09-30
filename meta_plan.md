# UR12e Training and Inference: Module Plan

> **Code style requirement: Economical code, exceptional readability, and excellent abstraction design.**

Status: GPU smoke, checkpoint resume and full 5,000-step training accepted; rollout and station acceptance pending. Updated: 2026-09-15.

Build a multi-model training and inference repository. The first backend is
pi05 training-time RTC, inherited from the local 1011 implementation. Shared
interfaces must describe UR12e observations, actions, runs and control authority
without assuming OpenPI, JAX, a 32-dimensional tensor or RTC. The two main
implementation paths are `training/` and `infer/`. Do not inspect or depend on
the external safeinfer project; inference is implemented here.

The collector owns acquisition; ur12e_data_exporter owns offline MCAP-to-LeRobot
v3 conversion; this repository owns model training, serving and supervised
inference. Reference repositories remain unchanged.

## Stable module registry

These TI IDs belong to this repository. They do not replace the collection
repository's M IDs or the exporter's E IDs. Do not renumber or reuse them.

| ID | Responsibility | First acceptance cases |
| --- | --- | --- |
| TI01 | Repository boundaries, backend selection and reproducible environments | TI01-A01: shared interfaces load without OpenPI; TI01-A02: source and dependency provenance |
| TI02 | Exported dataset admission and training projection | TI02-A01: official v3 loading; TI02-A02: windows, cameras and action transforms; TI02-A03: normalization identity |
| TI03 | pi05-RTC training and launchers | TI03-A01: source algorithm parity; TI03-A02: dry-run and launch manifests; TI03-A03: GPU smoke; TI03-A04: checkpoint round trip and resume |
| TI04 | Model serving and robot-side inference client | TI04-A01: train/serve contract; TI04-A02: RTC queue behavior; TI04-A03: model-independent fake backend |
| TI05 | ROS supervision, exclusive robot control and interruption | TI05-A01: software fault injection; TI05-A02: URSim stop/hold; TI05-A03: camera shadow; TI05-A04: physical stop/hold; TI05-A05: HOME gate |
| TI06 | Docker Hub distribution and portable colleague evaluation | TI06-A01: image/entrypoint validation; TI06-A02: offline error and latency reports |

The shared first-landing plan is [docs/pipeline-landing.md](docs/pipeline-landing.md).
Its primary module is TI03; it also specifies the TI01/TI02 prerequisites and
TI04/TI05 follow-on work. More models are future work, not empty implementations
or a requirement to inherit from a pi05 base class.

## Delivery sequence

1. Align the concrete plan and pin the exact reference snapshot.
2. Establish model-neutral boundaries and the LeRobot v3 adapter.
3. Land pi05-RTC norm statistics, smoke/train/resume and deployment scripts.
4. Land its model server and RTC client under the common inference interfaces.
5. Validate the two local processes through faults, URSim, camera shadow and
   separately agreed physical tests.

Implementation is authorized. Explorer source/data staging and CPU preparation
have passed; the approved two-H200 smoke job 10351656 completed all five steps successfully
with finite metrics and exit 0. No physical hardware acceptance has been performed. GO HOME before model handoff and Space
position hold are confirmed; TI05-A05 covers this gate. Software acceptance has passed; see [acceptance](docs/acceptance.md) for the
GPU and physical cases that remain NOT RUN.

The user additionally confirmed future Docker Hub image sharing for colleagues
on GPU devices. Prepare release commands and offline effect/latency evaluation;
do not publish until requested. Version/dependency selection is delegated.

Current delivery gate: keep Mac checks lightweight. Explorer image/data/weight
preparation, GPU smoke job 10351656 and checkpoint roundtrip job 10351772
have passed. The user authorized online W&B and full 5,000-step training;
all five dependent segments completed successfully, starting with job 10351884.
Final checkpoint 4999 contains optimizer step 5000; W&B run is jenvhjmn. Physical acceptance and image publication
remain separate. See the [Explorer record](docs/explorer-two-gpu-plan.md).


RTC follow-up (2026-09-30): the user approved a fresh pi05-base run on the same
dataset with uniform action-prefix lengths 0..9. CPU preparation 10697077 passed; GPU smoke/restore gate 10697078 is queued.
The gate will submit the authorized 5,000-step full run automatically after
successful RTC smoke and checkpoint recovery checks. See [RTC training v1](docs/explorer-rtc-training-v1.md).
Inference/runtime changes and hardware evaluation are outside this launch.
