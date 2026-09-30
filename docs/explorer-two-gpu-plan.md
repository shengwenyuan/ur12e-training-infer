# Explorer two-GPU pi05 first run

> **Code style requirement: Economical code, exceptional readability, and excellent abstraction design.**

Date: 2026-09-15. Primary module TI03; related TI01/TI02/TI06.
Status: GPU smoke, checkpoint roundtrip and full 5,000-step training PASS; rollout acceptance pending. The user selected two GPUs,
all cleaned data, and pi05-base initialization. On 2026-09-15 the user clarified
initially requested alignment only, then explicitly approved the proposed
zero-prefix profile, image preparation and an automatic smoke launch once
checks pass. The user subsequently supplied a W&B credential and explicitly
authorized full 5,000-step training after successful smoke. Complete the
checkpoint roundtrip gate, then launch automatically with online W&B. Smoke
must disable W&B. Image and asset locations
have been resolved through read-only SSH and registry metadata inspection.

## Experiment

Use all 30 recorded UR12e pick-and-place demonstrations, exported as 87 intervals
and 8,800 retained rows. The complete dataset is already SHA256-verified at
`/projects/vla_training/li.xinle/ur12e-training-infer/datasets/cleaned-20260915`.
No interval concatenation, new cleaning, random row validation split or
checkpoint download. This is supervised adaptation to UR12e; task simplicity
is motivation to try it, not proof of cross-embodiment generalization.

Aligned active profile: [ur12e_pick_place_2gpu.toml](../training/pi05_rtc/config/ur12e_pick_place_2gpu.toml). The earlier draft is retained as review history.
Use the current RTC backend with maximum_delay_exclusive=1 (delay always zero),
not a claim that an independent vanilla backend already exists. This isolates
learning for the planned synchronous Isaac evaluation. If RTC training is
preferred now, select delay 0..9 instead and freeze a distinct run/norm identity.

Full fine-tuning, global batch 16 (8 samples per GPU), bf16 model compute,
horizon 50, 200 text tokens, 5,000 steps, peak LR 2.5e-5, 250 warmup steps,
cosine decay to 2.5e-6 at step 5,000, inherited AdamW, EMA .999, seed 42.
This is about 9.09 dataset-equivalent passes over anchors, not a promise of
convergence. Log every 10 steps, save every 250, retain every 1,000.
Keep workers=0 for the first reproducible chain; measure loading and compute
before a separate throughput-tuned run. W&B runs online in project `ur12e_training`, team
`sals-northeastern-university`, using the user-supplied private credential.

Use all data for training as requested. Evaluate separately in Isaac; report
success rate across object placements and seeds. Offline action metrics exclude
current-row trivial targets and repeated terminal padding (21.73% at horizon
50). Export reports also identify an object-settling pause cut, so learning
quality remains empirical rather than established by numeric round-trip tests.

## Resource and environment contract

Account x.li1, partition gpu-short, one node, two H200s, one JAX process,
FSDP=2, 16 CPUs, maximum two hours per allocation. Smoke/roundtrip requested
128 GiB RAM; full training requests 256 GiB after the roundtrip accounting
reported approximately 172 GiB peak RSS. The numerical
sharding code supports divisors of the device count; only our profile guard
currently restricts selection to 1 or 8. Add 2 to the supported topology and
keep exact allocated GPU count and batch divisibility checks. Do not launch
one full training process per GPU. Leave inherited numerical sources unchanged.

Selected image: `salswy/ur12e-training-infer:0.1.0-gpu`.
Registry index digest:
`sha256:6e7a5d4efb48f0bb3781c04da9b40c0090ce7fbaea701aa342f775268228d6e7`.
Linux amd64 image manifest digest:
`sha256:c77b06ca1de82247a62d6d0bac1a530cbb38c970a0029dc7470594c24228dfe6`.
The image was converted to SIF, verified and executed on two H200s; the
acceptance timeline below records the exact hashes and job results.

Assets on PC `ur12e-collection`:
- Initialization: `/home/robot2026fall/ur12e-models/openpi/pi05_base/params`.
- Tokenizer: `/home/robot2026fall/ur12e-models/openpi/big_vision/paligemma_tokenizer.model`.
- Tokenizer SHA256: `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`.
- Base directory occupies approximately 12 GiB. Orbax PyTree handler metadata,
  OCDBT manifest and commit marker are present. This is structural inspection,
  not successful full model restore or upstream checksum verification.

These assets have since been copied to Explorer and fully checksum-verified. Existing PC environments inspected include Isaac,
exporter and lightweight inference dependencies, but no complete training
runtime was established. Host Python 3.9 is not our runtime.
Use Apptainer with frozen code/data/assets read-only and persistent output plus
scratch cache read/write. Do not rebuild/download environments on every run.

## Implementation after alignment

1. Accept FSDP=2 in the owned profile validator and test supported counts,
   unsupported counts, batch divisibility, and inherited two-device CPU mesh
   sharding using a tiny synthetic array (not model training on the Mac).
2. Promote the selected TOML into training/pi05_rtc/config. Keep model/action
   semantics unchanged except the explicitly selected RTC delay configuration.
3. Add an Explorer launcher with CPU preparation and GPU smoke/roundtrip/train
   modes, explicit mounted paths and image identity, offline W&B, persistent
   logs, fail-fast errors and no implicit downloads or overwrite. Create Slurm
   output directories before submission. Freeze source/config/asset identities.
4. CPU preparation: quota, repository dataset admission, RGB batch decoding,
   normalization generation or identity-checked reuse. Do not normalize against
   unrelated Berkeley UR5 data or reuse the exporter's generic stats.json.
5. Two-GPU smoke: same batch/profile, five steps, check GPU count, weight loading,
   image/action tensors, finite loss/gradients, memory and compile/step timing.
6. Checkpoint roundtrip: separate run with two train steps and save_interval=1;
   resume to four total steps with all other identity fields unchanged. Verify
   saved serving metadata and checkpoint restore before full training.
7. Full-data run: 5,000-step profile with a distinct name/output path. Inspect
   the first checkpoint and actual timing. If the two-hour limit is insufficient,
   resume from the latest completed checkpoint with the same profile; do not
   promise completion in one allocation or enqueue an unbounded job chain.

## Acceptance

- TI03-A02: profile validation, launcher dry-run, mount/command/resource inspection,
  shell syntax and failure behavior before submission.
- TI03-A03: real two-H200 smoke with the selected image and base weights.
- TI03-A04: separate two-to-four-step checkpoint save/restore/resume.
- TI02-A01/A03: actual selected runtime dataset/batch loading and bound norm marker.
- Full run: record Slurm IDs, model/image/data/config identity, W&B directory,
  completed steps and final checkpoint. A successful train run does not establish
  Isaac task success; rollout acceptance remains separate.

Historical pre-implementation results: original code and complete data uploaded; all dataset file
checksums match. Two H200s in gpu-short pass scheduler test-only and fit its
reported two-GPU QoS limit. The exact 16-CPU / 128-GiB / two-hour resource request also passed test-only.
Inherited FSDP was tested read-only on two virtual CPU devices with an 8x8
array: mesh creation and sharding PASS. This is not a GPU/model smoke.
Draft TOML syntax and selected values PASS; the current profile guard still
rejects FSDP=2 pending the planned change. New implementation and GPU
acceptance remain NOT RUN. No task submitted by this plan.

## Approved execution scope

User confirmation on 2026-09-15: proceed with the aligned two-GPU zero-prefix
profile, verify/pull the selected image, stage the existing PC assets, prepare
full training and automatically launch smoke if checks pass. Do not ask for
smoke authorization again. No new pretrained downloader or image rebuild is
requested. Mount a frozen source snapshot over /app to use the owned FSDP=2
validator with the existing dependency image; verify dependency-lock and vendor
numerical identities before use. Image revision label is an uncommitted build,
so the OCI digest alone does not establish source parity.

## Execution record (in progress)

- Implemented owned profile support for 1/2/8 FSDP devices; inherited numerical
  code unchanged. Promoted profile: `training/pi05_rtc/config/ur12e_pick_place_2gpu.toml`.
- Added `training/launchers/explorer.sh` and CPU-only `explorer_prepare.py`.
  Preparation verifies the image dependency lock, dataset, norm identity and a
  real RGB/action batch. It does not restore model weights or claim GPU readiness.
- CPU preparation can overlap checkpoint transfer. Standalone tokenizer was
  copied and SHA256 verified. Runtime asset paths are now
  `/pretrained/openpi/pi05_base/params` and `/pretrained/paligemma_tokenizer.model`;
  the host mount root is the project's `pretrained/` directory.
- TI03-A02 software PASS: 18 related tests, shell syntax, Black and Pylint 10/10.
  Dataset integration is deferred to the actual selected image on the cluster.
- Frozen source: `/home/li.xinle/ur12e-training-infer/releases/f6d7c2183240`.
  All 155 files verified against source_snapshot.json. Archive SHA256:
  `b074bdffd5d9dac782143d26bb68eed25de8b1ba9918121e6ec11b9a9d92bf0d`.
- Runtime dependency lock SHA256:
  `ce635024c72b332e8c095ed3945e0345f87f5c366dfaa5cb82321bc6452011ca`.
- Image preparation job: 10351278 (CPU short). Group project quota query reports
  1 TiB soft / 1.5 TiB hard limit. Image conversion is in progress.
- Data/runtime preparation job: 10351337 (CPU short), dependent on image task
  termination and independently gated by image existence, pip check and lock
  identity. Missing or incompatible image fails preparation; it cannot launch GPU work.
- Operations files: `/home/li.xinle/ur12e-training-infer/preparation/20260915/`.
- Persistent output: `/projects/vla_training/li.xinle/ur12e-training-infer/runs/pick-place-zero-prefix-2gpu-v1`.
- No GPU smoke, checkpoint roundtrip or full training has run yet.

### Image and CPU preparation results

Image task 10351278 completed pull/conversion and pip check, then exited 1
because the default host /tmp bind hid /tmp/requirements.lock in its final
probe. This was an inspection-path failure, not a failed SIF conversion. The
launcher already disables that bind with --no-mount tmp. Independent job
10351337 completed in 31 seconds (exit 0), including pip check, exact image
dependency-lock equality, dataset admission, fresh normalization and a real
batch. No image rebuild or second image download was needed.

SIF SHA256: `1396c34cdc60adad1fc8473102ebb5d1690df791ac2ad6f9a6d59ae7a308a22b`.
Python 3.12.3, JAX 0.5.3, Flax 0.10.2, Torch 2.7.1, LeRobot 0.6.1,
Transformers 5.4.0. Dataset identity:
`77c98a1e61be400c73572ff2e68ed69e85cd8c620f3c540a3ee08ab8fbf9f6e7`.
Normalization marker identity:
`0fec1496e939d65976a855ff76345e73a2acc98efeab3439ada2bb4d0849ee0d`.
Profile SHA256: `8ce6f65c57ba70fca5a680d3e0fa9c9017ebfadf71caefad7ba840261effdb80`.
Algorithm SHA256: `0f584cd3430b7066f06498fc5b6c4a16a4555178b3c46651c9d3043ed4c92980`.
Batch: actions [16,50,32]; each of three RGB views [16,224,224,3].
Report: `runs/pick-place-zero-prefix-2gpu-v1/reports/prepare-10351337.json`.
TI02-A01/A03 CPU runtime preflight PASS; GPU model restoration remains untested
until smoke. TI06-A01 selected image CPU/dependency checks PASS, GPU pending.

### Weight verification and smoke submission

Weight transfer completed. CPU verification job 10351508 completed in 40 seconds,
exit 0: all 30 files and the exact file set match the PC SHA256 manifest. Published
asset directory: `/projects/vla_training/li.xinle/ur12e-training-infer/pretrained/openpi`.
The standalone tokenizer independently matches the previously recorded hash.

Smoke job **10351656**, `ur12e-smoke-2gpu-v1`, submitted once after all preparation
gates passed and source/profile identities were rechecked. Account x.li1,
partition gpu-short, one node, two H200s, one process, 16 CPUs, 128 GiB RAM,
two-hour limit, global batch 16, five optimizer steps. W&B is disabled.
The initial scheduler observation was PENDING (Priority). The job subsequently
ran on d4055 and completed successfully; see the final acceptance below.

Launch record: `/home/li.xinle/ur12e-training-infer/preparation/20260915/smoke-launch.json`.
Log: `/projects/vla_training/li.xinle/ur12e-training-infer/runs/pick-place-zero-prefix-2gpu-v1/logs/smoke-10351656.log`.
Monitoring: `squeue -j 10351656` and
`sacct -j 10351656 --format=JobID,State,Elapsed,ExitCode`.
TI03-A03 was pending while queued and is now PASS. TI03-A04 checkpoint
roundtrip and full training remain NOT RUN. Smoke does not save checkpoints. The 5,000-step
configuration is prepared, but no full training job has been submitted.

### Final GPU smoke acceptance: PASS

Job 10351656 completed on d4055 in 00:02:38, exit 0:0. Both allocated GPUs
were verified as NVIDIA H200. Actual pi05-base weights and training state
initialized, and all five forward/backward/update steps completed:

| Step | Loss | Gradient norm |
| --- | --- | --- |
| 0 | 0.0335 | 0.3382 |
| 1 | 0.0406 | 0.2981 |
| 2 | 0.0280 | 0.2046 |
| 3 | 0.0326 | 0.3366 |
| 4 | 0.0354 | 0.4178 |

All recorded metrics are finite. W&B was disabled and no checkpoint was saved,
as required by smoke mode. An observed GPU-memory snapshot during initialization
was approximately 65 GiB per device; this is not a measured peak. XLA emitted
a slow constant-folding diagnostic during first-step compilation, then completed
normally. First-step compilation dominates this short run; do not treat total
time divided by five as steady-state throughput or as inference latency.

TI03-A03 PASS (two-GPU zero-prefix configuration), TI06-A01 selected image +
mounted-source GPU execution PASS, TI02-A01/A03 actual data and normalization
PASS. TI03-A04 save/restore/resume is still NOT RUN. No full training job,
learned-policy Isaac success evaluation, real-time RTC delay test or physical
robot acceptance was performed. Five finite-loss steps establish the pipeline,
not learned UR12e task success.

Compact evidence: [smoke report](checks/explorer-smoke-10351656.json).
All tasks started by this preparation/smoke chain are terminal; no GPU task
from this chain remains active. Full-training artifacts are ready for the next
checkpoint roundtrip gate and a separately authorized full launch.


## Online launch authorization and implementation (2026-09-15)

The user explicitly authorized online W&B and the full run after smoke. Extend
the owned launcher with an optional externally mounted TOML and explicit
online/offline W&B selection. Read the credential from a private host file,
pass it only through the container environment, and never include it in source,
TOML, command arguments, reports or logs. W&B API lookup resolved the actual
team to `sals-northeastern-university` (the organization slug differs).

The checkpoint acceptance uses a distinct run name and 2 then 4 total steps,
with save/log intervals of 1, the same initialization, dataset, normalization,
optimizer and 5,000-step LR schedule. Online W&B also exercises strict run-ID
resume. Full training retains the approved TOML and a separate fresh run. The
2-hour Slurm limit may require resuming the same experiment from its last
completed checkpoint; changing only the total step budget is already supported.
No image rebuild or vendor numerical changes are required.

### Launch state and bounded continuation

Source release: `/home/li.xinle/ur12e-training-infer/releases/a8acb317cfed`
(156 files, archive SHA256
`5d219dc9f37681fbfddc8fd7f29f1dca02ef2f429158a58d1ce61b7282cabffe`).
Owned launcher tests: 20 PASS, 3 dataset cases deselected; training.check PASS
(Black / Pylint 10.00), bash syntax PASS. Dependency image and numerical
algorithm identity remain unchanged. Roundtrip job `10351772` submitted,
with online W&B and external 2/4-step TOMLs; acceptance pending execution.

After that gate succeeds, submit five bounded segments with cumulative budgets
1,000/2,000/3,000/4,000/5,000. Each successor requires the previous Slurm job
to succeed; dependency failure cancels the successor rather than allocating
GPUs. Every segment uses the entire dataset and the original 5,000-step LR
schedule. Only the permitted total-step budget differs. Resumed model, AdamW,
EMA and global step share a single experiment and W&B run ID. The inherited
loader restarts its seeded iterator on process resume; this is not bitwise
equivalent to one uninterrupted run, though each 1,000-step segment traverses
all 8,800 anchors. Record every job ID immediately after submission.


## Current acceptance and submitted full run

- TI03-A04 PASS: job `10351772`, node d4055, COMPLETED / exit 0:0, elapsed
  5m51s. The 2-step run saved checkpoint 1; a new process restored all training
  state and executed steps 2 and 3, then saved checkpoint 3. Metadata and saved
  normalization match. W&B online resume retained run ID `oeeoisav`.
  [Compact acceptance](checks/explorer-roundtrip-10351772.json).
- Final Slurm accounting reports MaxRSS 180025704 KiB (about 172 GiB). Full
  jobs therefore request 256 GiB, rather than relying on the earlier 128 GiB
  request. GPU count, batch size and training hyperparameters are unchanged.
- Full training is authorized and submitted. It is not yet complete. Initial
  scheduler status: first segment PENDING (Priority); successors PENDING
  (Dependency). Full-run GPU steps and first checkpoint remain unverified
  until the scheduler allocates resources.

| Cumulative target | Slurm job | Dependency |
| --- | --- | --- |
| 1,000 | 10351884 | None |
| 2,000 | 10351885 | afterok:10351884 |
| 3,000 | 10351886 | afterok:10351885 |
| 4,000 | 10351888 | afterok:10351886 |
| 5,000 | 10351889 | afterok:10351888 |

[Exact launch record](checks/explorer-full-launch-20260915.json).
The full experiment name is `ur12e_pick_place_pi05_zero_prefix_2gpu_v1`; it
uses all 87 intervals / 8,800 rows from 30 source demonstrations. Online W&B
project: https://wandb.ai/sals-northeastern-university/ur12e_training .
The full-run W&B ID will be created when the first GPU process starts.

Persistent logs, normalization and checkpoints:
`/projects/vla_training/li.xinle/ur12e-training-infer/runs/pick-place-zero-prefix-2gpu-v1`.
Checkpoint subdirectory: `checkpoints/pi05_rtc/ur12e_pick_place_pi05_zero_prefix_2gpu_v1`.
Frozen operational scripts and segment TOMLs:
`/home/li.xinle/ur12e-training-infer/preparation/20260915-online-v1`.
Secrets remain in a private home-directory credential file, outside source,
images, profiles, manifests and logs. No additional image publication occurred.

Monitor without resubmitting:

```sh
ssh explorer-jona 'squeue -j 10351884,10351885,10351886,10351888,10351889'
```

Only successful predecessor jobs release their successors. A failed or timed-out
segment stops the bounded chain; inspect its log and last committed checkpoint
before scheduling any repair. No Isaac rollout or hardware acceptance has run.


## Full training completion

The scheduled supervision verified all five original jobs COMPLETED with exit
0:0. No repair or replacement jobs were needed. Total allocation runtime was
3h19m38s; queue time is excluded. The final TensorStore scalar `train_state.step`
was read directly and equals 5,000. Checkpoint index 4,999 uses the inherited
zero-based naming convention. Its commit timestamp is present; parameter and
training-state metadata/OCDBT manifests exist, and saved normalization matches
the run's normalization. Only final checkpoint directory `4999` is currently
retained. The last logged metrics at step 4,990 were loss 0.0020 and gradient
norm 0.0621; these are training metrics, not rollout success measurements.

Final checkpoint on Explorer:
`/projects/vla_training/li.xinle/ur12e-training-infer/runs/pick-place-zero-prefix-2gpu-v1/checkpoints/pi05_rtc/ur12e_pick_place_pi05_zero_prefix_2gpu_v1/4999`

[W&B run](https://wandb.ai/sals-northeastern-university/ur12e_training/runs/jenvhjmn)
was retained across all five segments.
[Completion evidence](checks/explorer-full-completion.json) records actual job
status, scalar-step verification and remaining limits. The remote launch record
now includes this completion result. Earlier queued states above are historical.

No full reload of the final model or Isaac/hardware rollout was performed in
this check; the earlier real checkpoint roundtrip remains the restore evidence.
Supervision is complete and its two-hour automation has been paused.
