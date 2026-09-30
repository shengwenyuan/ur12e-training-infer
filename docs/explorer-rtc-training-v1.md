# Explorer RTC training v1

> **Code style requirement: Economical code, exceptional readability, and excellent abstraction design.**

Date: 2026-09-30 (Asia/Shanghai). Status: CPU preparation PASS; GPU gates queued; full training authorized after gates. Primary module
TI03; related TI01/TI02/TI06. The user approved the proposed RTC configuration
and explicitly requested a new training launch on explorer-jona. Only training
is in scope. Inference-device changes, benchmarks, Isaac and hardware motion
are excluded from this run. Use the repository ur12e-train skill.

## Aligned experiment

Use all 87 cleaned intervals / 8,800 rows from 30 original demonstrations,
dataset identity `77c98a1e61be400c73572ff2e68ed69e85cd8c620f3c540a3ee08ab8fbf9f6e7`.
Initialize a fresh experiment from mounted pi05-base; do not resume the previous
zero-prefix checkpoint. Model: horizon 50, action dimension 32, three RGB views,
200 text tokens, retained-row action steps. Training-time RTC samples a uniform
prefix length from 0 through 9 (`maximum_delay_exclusive=10`), clean prefix at
OpenPI timestep zero, postfix-only loss. All other training settings match the
verified baseline: global batch 16, FSDP 2, workers 0, 5,000 total updates, seed
42, warmup 250, peak LR 2.5e-5, cosine decay to 2.5e-6 at step 5,000, inherited
AdamW and EMA .999, checkpoint interval 250 / keep period 1,000.

Profile: `training/pi05_rtc/config/ur12e_pick_place_rtc_2gpu.toml`.
Run name: `ur12e_pick_place_pi05_rtc_2gpu_v1`. The proposed future inference
settings (10 flow steps, 10/15 Hz, initial protected prefixes 5/7 and replanning
every 10 actions) are not training settings and are not implemented here.

## Deployment and isolation

Verified login explorer-jona resolves to Unix user li.xinle on explorer-01;
Slurm billing account x.li1. Use the proven gpu-short allocation: two H200s,
one node/process, 16 CPUs, 256 GiB RAM, at most two hours per allocation. The
multigpu Unix group now exists, but a partition named multigpu is unavailable;
do not infer longer-job eligibility from that group. Keep the proven bounded
1,000 -> 2,000 -> 3,000 -> 4,000 -> 5,000 cumulative-budget chain. The learning
rate schedule remains 5,000 steps in every segment. The inherited loader resets
its seeded iterator on resume, so this is not bitwise equivalent to one process.

Reuse image `salswy/ur12e-training-infer:0.1.0-gpu`, SIF SHA256
`1396c34cdc60adad1fc8473102ebb5d1690df791ac2ad6f9a6d59ae7a308a22b`, with frozen source
`/home/li.xinle/ur12e-training-infer/releases/a8acb317cfed` and a separately hashed
external TOML. No dependency rebuild or numerical-code changes are needed.
Online W&B: entity sals-northeastern-university, project ur12e_training. Recheck
authentication using the explicitly provided Mac credential file without printing
it, and reuse/refresh only this login's private credential reference.

Persistent root: `/projects/vla_training/li.xinle/ur12e-training-infer`.
New outputs: `runs/pick-place-rtc-2gpu-v1`; preparation scripts/profiles:
`/home/li.xinle/ur12e-training-infer/preparation/20260930-rtc-v1`.
The historical launcher internally names its norm subdirectory
`assets/pick-place-zero-prefix-v1`; this is a legacy path label only. It resolves
inside the NEW isolated output root and must contain a fresh RTC=10 marker.
Never reuse or relabel the previous experiment's marker.

## Gates and exact-once launch

1. Validate the complete TOML with the real loader and CLI dry-run; check existing
   jobs/records and writable storage. Freeze profile and operational-script hashes.
2. In a scheduled CPU job, verify image/source/data/base-weight inputs, run pip
   check, create/verify matching normalization, and load a real RGB/action batch.
3. In one two-H200 gate job, run five-step nonzero-prefix-capable smoke with W&B
   disabled. Run a separate two-step online save and restore to four total steps.
   Verify saved scalar step, committed checkpoint/assets and W&B ID continuity.
4. After gate success, automatically submit the five full segments once. The first
   full job also depends on successful termination of the GPU gate; each successor
   requires its predecessor to succeed. Persist every submission ID immediately.
   Before each resume, verify the exact predecessor's committed optimizer step
   and matching RTC metadata. Never fall back to base initialization on a missing
   predecessor. Preserve logs and all historical output roots.
5. Verify each final segment checkpoint and record terminal global step 5,000,
   W&B URL and completion limits. Retention may leave only the latest checkpoint
   because the inherited zero-based indices do not align with keep_period=1000.

Acceptance: TI03-A02 local parser/launch checks; TI02-A01/A03 runtime data/norm;
TI03-A03 real RTC GPU smoke; TI03-A04 RTC save/restore; full training completion
requires successful Slurm jobs plus verified saved optimizer step 5,000. These
are NOT RUN until actual evidence is recorded. No rollout quality is claimed.

## Launch progress

The complete RTC TOML parses with the actual backend and CLI dry-run. Its SHA256
is `9cc5422ede140914d35343c4e59ad369d27d15a9ae23cec5add459108db4fc04`. All five
segment profiles differ from it only in their cumulative step budget; the two
roundtrip profiles differ from each other only in that budget. Bash/Python
operational-script syntax, whitespace and 17 lightweight tests PASS (6 dataset
or backend cases deselected). Local Black passes. Local full Pylint/three backend
tests could not run successfully because the lightweight Mac environment lacks
optional numerical dependencies; the complete checks are required inside the
selected image during scheduled CPU preparation instead.

W&B authentication was revalidated for sals / sals-northeastern-university. Frozen
source and operational-file hashes were verified remotely before submission.
The current Slurm QoS confirms gpu-short allows two GPUs per job/user; gpu permits
only one per job. The exact two-H200 / 256-GiB request passes scheduler test-only.

Submitted once:
- CPU preparation `10697077`, short, 8 CPUs / 32 GiB / 30 minutes.
- RTC GPU gates `10697078`, gpu-short, 2 H200 / 16 CPUs / 256 GiB / 45 minutes,
  afterok:10697077. A preparation failure cancels this dependent job.

The GPU gate only submits the five authorized full segments after smoke, full
2-to-4-step checkpoint restore, saved scalar-step checks and W&B ID continuity
PASS. Every full segment verifies its required predecessor before resume and
verifies its new committed checkpoint after training. The first full job also
depends on successful termination of the gate job.

No inference files were changed and no inference device was contacted.

The selected image subsequently passed the full required style checks (Black
and Pylint 10.00/10) and all 20 selected tests, with 3 dataset cases deselected.
This resolves the Mac optional-dependency limitation without installing the
training stack on the Mac. Image/base/data SHA256 checks and pip check PASS.
[Initial exact launch record](checks/explorer-rtc-launch-v1.json).

CPU preparation job `10697077` completed with exit 0:0 in 4m03s. TI02-A01/A03
PASS for all 87 intervals / 8,800 rows; action batch [16,50,32], three image
batches [16,224,224,3]. Matching RTC=10 normalization identity:
`c656ed140b89bdfef0cd673ee834fd74ec3c6ca96411999acdec903163f074d4`.
[Preparation report](checks/explorer-rtc-prepare-10697077.json). TI03-A02 PASS
for parser, image checks, operational syntax and the exact scheduler request.

At handoff, GPU gate `10697078` is PENDING with its CPU dependency satisfied.
TI03-A03/A04 for this RTC configuration remain NOT RUN until that job executes.
Full segments have not yet been submitted: the frozen gate script automatically
submits them only after all GPU checks succeed, recording their IDs in the remote
`full-launch.json`. No second user confirmation is needed. The full W&B run ID
is created only when its first training segment begins. Normal queue waiting
is not a failure. The previous experiment's monitor remains paused; no new
recurring monitor was requested or created for this launch.
