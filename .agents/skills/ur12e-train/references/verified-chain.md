# Verified Explorer chain: historical reference

This is evidence from September 2026, not a required account, resource preset or
new launch approval. The reference is self-contained so a fresh checkout can use
the skill even if the original operational records are not versioned alongside it.

## What actually succeeded

All cleaned pick-and-place data was used: 30 source demonstrations became 87
exported intervals and 8,800 retained rows. Dataset SHA256 identity:
`77c98a1e61be400c73572ff2e68ed69e85cd8c620f3c540a3ee08ab8fbf9f6e7`.
Action time means retained row steps, not uniformly sampled physical time.
Initialization was mounted pi05-base JAX params with a mounted tokenizer.

The model used horizon 50, action_dim 32, 200 text tokens, discrete state,
relative six-joint actions plus absolute gripper, and three RGB cameras.
RTC `maximum_delay_exclusive=1`: zero-prefix only. This run proves the training
pipeline; it does not prove nonzero-prefix RTC behavior or task success.

Global batch 16, FSDP 2, workers 0, 5,000 steps, seed 42, EMA .999; warmup 250,
peak LR 2.5e-5, decay_steps 5000, final LR 2.5e-6. AdamW: b1 .9, b2 .95,
eps 1e-8, weight_decay 1e-10, clip_gradient_norm 1.0. Log interval 10,
save interval 250, keep period 1000; norm batch 64, smoke steps 5.

Historical login: explorer-jona / li.xinle; billing account x.li1. Slurm used
partition gpu-short, two H200s on one node, one task, 16 CPUs, 256 GiB RAM and
a two-hour limit per full segment. Discover new users/accounts and quotas anew.
The 128 GiB roundtrip request reported about 172 GiB MaxRSS, motivating the
larger full-run request. GPU-only smoke did not reveal checkpoint RAM demand.

| Stage | Job | Result |
| --- | --- | --- |
| CPU dataset/normalization preparation | 10351337 | PASS |
| Five-step GPU smoke, W&B disabled | 10351656 | PASS |
| Separate 2 -> 4-step save/resume, online W&B | 10351772 | PASS |
| Cumulative steps 1,000 | 10351884 | COMPLETED, 0:0 |
| Cumulative steps 2,000 | 10351885 | COMPLETED, 0:0 |
| Cumulative steps 3,000 | 10351886 | COMPLETED, 0:0 |
| Cumulative steps 4,000 | 10351888 | COMPLETED, 0:0 |
| Cumulative steps 5,000 | 10351889 | COMPLETED, 0:0 |

Full segments used success dependencies and retained one W&B run:
https://wandb.ai/sals-northeastern-university/ur12e_training/runs/jenvhjmn .
That entity/project is historical; resolve and align W&B for each new identity.
Total full-job runtime: 3h19m38s, excluding queue waits. No repairs were required.

## Frozen environment and paths

Image: `salswy/ur12e-training-infer:0.1.0-gpu`.
Linux amd64 manifest:
`sha256:c77b06ca1de82247a62d6d0bac1a530cbb38c970a0029dc7470594c24228dfe6`.
Converted SIF SHA256:
`1396c34cdc60adad1fc8473102ebb5d1690df791ac2ad6f9a6d59ae7a308a22b`.
The image revision label was dirty; compatibility was established with an
independently frozen read-only source mount, not inferred from that label.

Project root was `/projects/vla_training/li.xinle/ur12e-training-infer`:
- Image: `containers/ur12e-0.1.0-gpu-c77b06ca.sif`.
- Dataset: `datasets/cleaned-20260915`.
- Initialization: `pretrained/openpi/pi05_base/params`.
- Tokenizer: `pretrained/paligemma_tokenizer.model`.
- Output: `runs/pick-place-zero-prefix-2gpu-v1`.
- Final checkpoint below output:
  `checkpoints/pi05_rtc/ur12e_pick_place_pi05_zero_prefix_2gpu_v1/4999`.

Frozen source: `/home/li.xinle/ur12e-training-infer/releases/a8acb317cfed`;
156-file archive SHA256:
`5d219dc9f37681fbfddc8fd7f29f1dca02ef2f429158a58d1ce61b7282cabffe`.
Operational scripts and segment TOMLs:
`/home/li.xinle/ur12e-training-infer/preparation/20260915-online-v1`.
Source identity is separate from the source's base Git commit. Some successful
launch helpers were uncommitted at skill creation; their presence in another
checkout must be checked. This skill does not install or commit those helpers.

The frozen `training/launchers/explorer.sh` accepts prepare, smoke and train
(with --resume for train). Its external inputs are UR12E_IMAGE, UR12E_DATASET,
UR12E_PRETRAINED, UR12E_OUTPUT, UR12E_CACHE, optional UR12E_PROFILE, and for online
W&B: UR12E_WANDB_MODE=online, UR12E_WANDB_ENTITY, UR12E_WANDB_KEY_FILE.
The credential file belongs to the selected user; do not copy another login's key.
The wrapper's hardcoded normalization subpath contains `zero-prefix`; when
adapting to a new RTC run, use an isolated output/asset set and matching marker,
not the historical zero-prefix assets. Its fixed initialization/tokenizer layout
also needs to match the mounted directory or be explicitly adapted.

Apptainer used --cleanenv, --no-mount tmp, --pwd /app, --nv for GPU mode,
PYTHONPATH=/app and read-only source/data/pretrained/profile binds. The image's
Python is /opt/venv/bin/python. --no-mount tmp prevents the host /tmp bind from
hiding the image dependency-lock file. Keep writable TMPDIR and compilation
cache under the assigned scratch mount; preserve Slurm CUDA_VISIBLE_DEVICES.
Image conversion, normalization and large checksum operations used CPU jobs.

## Completion semantics and known limits

Checkpoint index 4999 was committed and its TensorStore train_state.step scalar
was read as 5000. Parameter/state metadata and OCDBT manifests existed; policy
metadata and saved normalization matched. Normalization file SHA256:
`046695fc3cca5528f2cfd768125bf16a116b7296a8c92b7c7e7703231648556b`.
Only final directory 4999 was retained. Do not assume keep_period=1000 preserves
all segment ends: directory indices are zero-based and the loop skips periodic
saving at a segment's start, so the retention rule may miss those boundaries.

For the saved scalar, the read-only TensorStore specification used driver zarr,
an ocdbt kvstore with base `file:///.../4999/train_state/`, and path `step`.
A bare filesystem string without file:// is not a valid kvstore URL. Use the
actual checkpoint metadata to select the format in future versions.

The final model was not fully reloaded during completion verification. Earlier
2 -> 4-step acceptance did exercise model/AdamW/EMA restore. There was no Isaac
or physical rollout; loss at the last logged step 4990 was .0020, not a task
success rate. The requested two-hour monitor was paused after verification.
