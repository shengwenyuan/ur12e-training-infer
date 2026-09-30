# Explorer first pi05 training run

The initial eight-GPU proposal below is superseded by the approved
[two-GPU plan and execution record](explorer-two-gpu-plan.md).

> **Code style requirement: Economical code, exceptional readability, and excellent abstraction design.**

- Primary module: TI03; related: TI01, TI02, TI04, TI06.
- Date: 2026-09-15. Status: code/data staging complete; training configuration proposed.
- Authorized: SSH inspection, code staging, and preparation for the complete cleaned dataset. No GPU job or image build is submitted. Vanilla model/simulator implementation is not part of this preparation.
- Source: `d0b71bcc977fa08fa84b887e308865fa7beb73e3`.

## Verified environment

`explorer-jona` connects as `li.xinle` to `login.explorer.northeastern.edu`
(observed node `explorer-01`). This is a login node; run training, compilation,
normalization and substantial validation inside scheduled compute jobs.
The host Python is 3.9.25, while this repository requires Python 3.12.
Use the repository GPU environment in Apptainer 1.4.5, not the host Python or
the existing Cosmos virtual environment. The host successfully displayed the
lightweight CLI help; that does not validate its full runtime.

Existing colleague work:

- `~/pi05/slurm/pi05_multigpu_benchmark.slurm`: eight-H200 resource and Apptainer skeleton; training command is still a placeholder. Its referenced `/scratch/li.xinle/pi05/containers/openpi.sif` does not exist.
- `/projects/vla_training/li.xinle/cosmos_robomind/`: persistent code, data, container, cache, checkpoint, log and output directory layout; inspected top levels contain mostly scaffolding.
- `/scratch/li.xinle/cosmos/`: Cosmos framework and virtual environment, Berkeley UR5 adapter/transform tests and patches, checkpoint preparation/conversion, GPU configuration validation and SFT smoke scripts, evaluation outputs, and Isaac input clips. These artifacts show prior work; their current correctness was not revalidated.
- `/scratch/li.xinle/datasets/berkeley_autolab_ur5`, `/scratch/li.xinle/robomind/`: previous datasets and processing layout.

Scheduler account: `x.li1`. The `multigpu` partition allows groups `multigpu,rc`;
this login belongs to `users,vla_training`. A read-only `sbatch --test-only`
probe for one node / eight H200s / 32 CPUs / 256 GiB / two hours failed with
`User's group not permitted to use this partition`. No job was created.
The partition exists but is absent from this user's default visible list.
Access approval must name this exact login. There is no current allocation or
reservation visible for the account. Do not substitute the ordinary `gpu`
partition for eight GPUs: its QoS limits a job to one GPU.

## Storage and source staging

Code snapshot: `/home/li.xinle/ur12e-training-infer/releases/d0b71bc`.
It is an extracted Git archive, not a working Git clone. The retained
`source.tar` SHA256 is
`489316e844f1daf2e042d5bd345db0b52e27b13b1ce53d13860158b4a7de54ea`.
It matches the local archive. Existing colleague files were not replaced.

Persistent run root: `/projects/vla_training/li.xinle/ur12e-training-infer`.
Created subdirectories: `datasets/`, `pretrained/`, `runs/`, `containers/`.
This is a personal namespace inside the group's shared project, not a separate
personal storage allocation. Compute jobs can bind these shared paths.
Scratch root: `/scratch/li.xinle/ur12e-training-infer`, with `cache/`, `tmp/`,
`logs/`. Prefer persistent run logs/checkpoints under `runs/`; scratch is for
rebuildable caches and temporary work. Scratch is not backed up and is subject
to monthly purge. Home has a small quota and should hold code/configuration.
Actual project quota and compute-node read throughput remain unmeasured.
Run `check-quota /projects/vla_training` in a CPU `short` allocation before
placing large checkpoints. Do not confuse filesystem-wide free space with
this project's quota.

Mount contract for Apptainer:

| Host | Container | Mode |
| --- | --- | --- |
| Code snapshot | `/app` | read-only |
| Selected TOML directory | `/config` | read-only |
| Dataset snapshot | `/data` | read-only |
| Existing pi05 JAX params and tokenizer | `/pretrained` | read-only |
| Persistent run output | `/runs` | read/write |
| Scratch cache | `/cache` | read/write |

No checkpoint or tokenizer downloader is added. Supply the OpenPI JAX/Orbax
`params` directory and the tokenizer file explicitly; a PyTorch safetensors
checkpoint is not interchangeable with this loader. Supply an exact GPU
image tag/digest or an existing SIF path and SHA256. Convert/pull a chosen image
in an appropriate compute/build job, not as login-node training work.

## Dataset decision

Use the entire PC dataset, confirmed by the user on 2026-09-15:
`/tmp/ur12e-cleaning-validation-20260915/lerobot/cleaned` on
`ur12e-collection` (`ur12e-flexlab`). It contains 30 original demonstrations,
87 exported intervals, 8,800 rows, three RGB streams, and 391,116,689 bytes.
Task: `Pick up the red block and place it inside the small box.`
It is recorded data (`simulated=false`), with `validation_only=true` and an
unsettled temporary input bridge label. Its numeric/media round trip passed
the exporter validation. The accompanying report explicitly records an
object-settling pause cut (candidate 53); first-run conclusions must therefore
remain exploratory rather than treating cleaning semantics as fully accepted.

Episode lengths: min 10, median 81, max 295; 31 intervals have fewer than 50
rows. With current-row windows and terminal repetition, 21.73% of the targets
at horizon 50 are repeated terminal padding. Exclude padding and the trivial
current-row target from offline action-error metrics. Never join across cuts.
Nominal 10 fps is a row/video indexing convention, not physical control timing.

Transfer complete files without re-encoding into
`datasets/cleaned-20260915.partial`, verify every file against source SHA256,
then rename to `datasets/cleaned-20260915`. Preserve `export.json`, `meta`,
`data` and `videos`. Transfer PASS: all 10 files and the exact file set matched source SHA256; the final directory has been published. Checksums and verification output are adjacent to it. The original data remain on the PC. An SSH stream can
relay through the Mac without storing the dataset or using UI automation.
For future large transfers, the official endpoint is `xfer.discovery.neu.edu`;
its host identity must be verified before first use. No SSH host-key checks
were disabled for this preparation.

Current loader trains on the complete supplied snapshot: it has no configurable
held-out episode selector. This first run may use all data, with separately
specified Isaac trials for evaluation. If adding an offline validation split,
group by the original 30 demonstrations and create reproducible views; splitting
the 87 intervals at random leaks source demonstrations. This requires a small
separate adapter/export change, not a nonexistent TOML option.

## Proposed first-run hyperparameters

These are proposals, not submitted or silently installed overrides.

| Setting | Proposal / implementation status |
| --- | --- |
| Backend | `pi05_rtc`; public CLI currently has no `pi05` backend |
| Fine-tuning | Existing full fine-tuning; LoRA/freeze selection is not exposed |
| Precision | Inherited bf16 model compute; do not describe every parameter/optimizer buffer as bf16 |
| State/action | 6 measured joint radians + raw gripper; relative joint targets, absolute gripper; tensor padded to 32 |
| Horizon / text tokens | 50 / 200, currently fixed by schema |
| RTC delay | Proposed Isaac baseline 1: delay=0; current RTC default 10 samples delay 0..9 retained rows |
| Resources | One node, one JAX process, eight H200s, FSDP=8; not eight independent Python processes |
| Global batch | 8 for smoke; proposed pilot 32 (4 per GPU), subject to measured memory/throughput; do not multiply by eight again |
| Budget | Pilot 1,000 optimizer steps; at batch 32 this is 3.64 dataset-equivalent passes over 8,800 anchors; extend toward 5,000 only after rollout review |
| LR | Peak 2.5e-5; warmup 100 steps; cosine decay 1,000 steps to 2.5e-6 |
| Optimizer | Inherit AdamW betas .9/.95, eps 1e-8, decay 1e-10, gradient clip 1.0 |
| EMA / seed | .999 / 42 |
| Data workers | 0 for initial smoke; benchmark 4 for full run, then freeze for that run |
| Logs / saves | Log every 10; save every 250; keep period 500; check checkpoint disk use |
| W&B | Disabled automatically in smoke; enable for train, initially offline if authentication/network is pending |
| Smoke | 5 optimizer steps, real data and real initialized model; no checkpoints are saved by this mode |

Changing the run budget alone is allowed on resume, but changing the LR schedule,
RTC semantics, worker setting or other recorded profile fields fails the current
resume identity check. Either retain the original schedule when extending, or
start a distinct experiment. Normalization is tied to dataset/model/RTC identity;
RTC delay changes require a separately validated norm marker even if the numeric
statistics would coincide.

## Vanilla pi05 and Isaac

If Isaac pauses physics during model inference and advances explicitly between
observations/actions, inference wall time no longer creates simulation-time
action delay. Synchronous pi05 is then a useful first policy-quality baseline;
RTC is not necessary solely to compensate for the slow GPU. This does not make
model inference itself faster, and physics/rendering still need resources.
Continue to simulate action execution and choose replanning intervals in
simulation steps. Real-time execution or intentionally injected inference delay
still provides a reason to test RTC.

The vendored standard Pi0Config(pi05=True) exists, but the public trainer and
server currently construct the RTC implementation. Setting
`maximum_delay_exclusive=1` gives delay=0 training (no conditioned action prefix,
all action positions contribute to loss). This is a zero-prefix RTC baseline,
not a separately tested vanilla backend. Genuine vanilla selection and a
synchronous Isaac client need explicit implementation and acceptance. Do not
pause the existing hardware client/ROS heartbeat and expect its wall-clock
watchdogs to become simulation-clock aware automatically.

Future configuration should separate model training settings from simulation
settings: backend/RTC mode, horizon, denoising steps, execution/replanning steps,
physics dt, rendering cadence, simulation clock mode and evaluation seeds.
These simulator and vanilla options are not accepted by today's strict TOML.

## Launch sequence and CLI contract

1. Resolve multigpu group access; rerun the same `sbatch --test-only` resource probe.
2. Freeze source revision, complete dataset checksum manifest, TOML, exact image/SIF, JAX params and tokenizer paths.
3. Use a CPU `short` job for quota/admission, all three RGB decode checks and `norm-stats`. Reuse stats only after verifying `validation.json` against the exact dataset/profile.
4. Use a scheduled GPU smoke job. Check NVIDIA driver, JAX GPU backend/count (eight for FSDP=8), model restore, actual image/action batch, forward/backward, finite loss/gradients, memory and first-compile timing. Do not install/download on each job start.
5. Separately run a short `train` experiment to save, reload and resume a checkpoint: `smoke` deliberately cannot establish this acceptance case.
6. Run the frozen pilot using `sbatch`, with distinct output/run names. W&B project is `ur12e_training`; entity is supplied by the user. Persist W&B directory, run ID and Slurm logs under the run root. For online logging, arrange credentials outside Git and do not print them. Offline runs can be synced later.
7. Monitor with `squeue -u li.xinle`, `sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode`, and the persisted logs. Evaluate saved checkpoints in Isaac; record success rate and failures, not training loss alone.

Draft Slurm resource header (requires access; no task has been submitted):

```bash
#!/bin/bash
#SBATCH --account=x.li1
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
set -euo pipefail
```

Create log directories before `sbatch` and pass absolute `--output`/`--error`
paths at submission; creating them inside the script is too late for Slurm.
Use `apptainer exec --nv` with the mount contract above and `/opt/venv/bin/python`.
Launch exactly one process. Bind the frozen code to `/app` and set the working
directory to `/app`; preserve Slurm's GPU visibility. Pass cache/W&B/offline
environment settings explicitly through Apptainer. Do not rely on image
ENTRYPOINT behavior; use `exec` and the explicit Python module:

```bash
/opt/venv/bin/python -m training.cli smoke \
  --model pi05_rtc --config /config/pilot.toml \
  --dataset /data --norm-root /runs/assets/pilot \
  --checkpoint-root /runs/checkpoints \
  --initialization /pretrained/pi05_base/params \
  --tokenizer /pretrained/paligemma_tokenizer.model
```

Use `validate` with `--dataset`, `norm-stats` with `--dataset --config --norm-root`,
then replace `smoke` with `train` for the actual run. `--dry-run` validates only
argument/profile structure; it does not load data or demonstrate GPU readiness.

## Acceptance and missing inputs

- TI03-A02 PASS: matching source archive uploaded; remote CLI help works; scheduler preflight conclusively reports missing group access.
- TI03-A03 BLOCKED: multigpu access, selected GPU runtime and mounted initialization assets are not ready. No real GPU smoke performed.
- TI03-A04 NOT RUN: checkpoint save/load/resume on GPU.
- TI02-A01 transfer PASS: all 10 files match source SHA256 and exact file set. The exporter reports a numeric/media PASS; this repository's full reader/batch admission in the selected training runtime remains NOT RUN.
- TI06-A01 NOT RUN: selected Docker-to-Apptainer image execution on H200.
- Requested inputs: access approval for `li.xinle`; exact existing image/SIF identity; local JAX checkpoint/tokenizer locations; W&B entity and preferred online/offline mode. No password, SSH private key or W&B API key should be pasted into the conversation.

References: [H200 quick start](https://rc-docs.northeastern.edu/en/latest/gpus/quickstart-h200.html),
[GPU access](https://rc-docs.northeastern.edu/en/latest/gpus/accessinggpus.html),
[storage](https://rc.northeastern.edu/data-storage-options/),
[scratch policy](https://rc.northeastern.edu/scratch-space-policy/),
[file transfer](https://rc-docs.northeastern.edu/en/latest/datamanagement/transferringdata.html).
The supplied service portal redirected to institutional sign-in; its authenticated
contents were not read, no credentials were entered, and further UI/login use is excluded.
