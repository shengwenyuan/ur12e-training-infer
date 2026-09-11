# Docker distribution and GPU evaluation

The repository builds two Linux amd64 images: `VERSION-gpu` for pi05-RTC
training/model serving/evaluation, and `VERSION-robot` for ROS Jazzy client and
supervisor. The GPU image uses Python 3.12, CUDA 12.6.3, JAX 0.5.3 and Flax
0.10.2. The robot image includes ROS Jazzy, ur_rtde 1.6.5 and RealSense 2.56.5.
Resolved Python dependencies include hashes; Docker base images have digests.
Apt packages follow the selected Ubuntu repositories at build time; distribute
the tested image digest for exact reuse rather than assuming byte-identical rebuilds.

Image creation and successful imports do not establish GPU numerical execution,
physical timing, task success or cross-host compatibility. See acceptance.md.

## Build and publish when requested

```sh
# Build both versioned images locally.
bash docker/release.sh YOUR_DOCKERHUB_NAMESPACE 0.1.0

# Publishing is a separate explicit operation, after testing that exact version.
docker login
bash docker/release.sh YOUR_DOCKERHUB_NAMESPACE 0.1.0 --push
```

No Hub namespace is hardcoded and no credentials enter the build context.
Record the pushed digest and give colleagues `namespace/repository@sha256:...`
for a repeatable comparison. OCI labels record the requested version and source
revision; an uncommitted build is labeled as such rather than inventing a revision.
The script builds only; it does not download weights, launch training or submit
cloud jobs. `--push` is never the default.

## Required external mounts

| Container path | Contents | Mode |
| --- | --- | --- |
| `/weights` | User-supplied pretrained params and `paligemma_tokenizer.model` | Read-only |
| `/checkpoint` | One selected trained checkpoint with `params/` and `assets/` | Read-only |
| `/datasets` | Completed exported LeRobot v3 datasets, preferably held out for evaluation | Read-only |
| `/output` | Fresh training/norm/evaluation output destinations | Writable |
| `/cache` | Optional dataset/library caches; never the source of required weights | Writable |

There is no pretrained-checkpoint downloader. The owner supplies files from an
existing trusted source. `--initialization`, `--checkpoint` and `--tokenizer`
are local-path interfaces. No parameter or tokenizer download occurs at runtime.
Missing resources fail with a path error. Checkpoints bind camera/action/model,
normalization, numerical implementation and tokenizer identities.

Use a GPU host with a working Docker NVIDIA runtime. Match the runtime UID/GID
to host mount ownership (replace the example `1000:1000`). The GPU workload is
JAX; robot-side ROS is not required for offline evaluation.

## One-command offline effect and speed check

```sh
docker run --rm --gpus all --user 1000:1000 \
  -e UR12E_IMAGE_ID='YOUR_NAMESPACE/ur12e-training-infer@sha256:YOUR_DIGEST' \
  -v /host/weights:/weights:ro \
  -v /host/checkpoint:/checkpoint:ro \
  -v /host/datasets:/datasets:ro \
  -v /host/reports:/output \
  -v /host/cache:/cache \
  --entrypoint python YOUR_NAMESPACE/ur12e-training-infer:0.1.0-gpu \
  -m infer.benchmark --checkpoint /checkpoint \
  --tokenizer /weights/paligemma_tokenizer.model \
  --dataset /datasets/held-out-task --samples 50 --warmup 5 \
  --output /output/evaluation-new.json
```

The report includes separate warmup latency, P50/P95/P99 and mean request
latency, sequential requests/second, checkpoint content identity, GPU model,
each predicted action chunk, mean absolute
joint error in radians and gripper error in raw units. Current-row targets and
repeated episode tails are excluded from action errors. Image decode happens
before timing; local timing includes preprocessing, model computation, device
synchronization and output conversion. Reported requests/second is sequential
service throughput, not control-loop frequency or parallel serving capacity.
For the in-process benchmark, the first warmup includes cold JAX compilation;
zero warmup intentionally leaves
cold-start cost in the measured distribution.

This is independent-window, zero-prefix, open-loop evaluation. It does not
measure closed-loop success or RTC queue continuity. Use the dedicated runtime
acceptance and separately authorized robot trials for those conclusions. Public
synthetic exporter fixtures require `--allow-simulated` and prove pipeline
operation only. Compare models on the same held-out dataset/task identities;
training-data error is not generalization performance.

## Model server and network latency

```sh
docker run --rm --gpus all --user 1000:1000 \
  -p 127.0.0.1:8000:8000 \
  -v /host/weights:/weights:ro -v /host/checkpoint:/checkpoint:ro \
  -v /host/cache:/cache \
  --entrypoint python YOUR_NAMESPACE/ur12e-training-infer:0.1.0-gpu \
  -m infer.pi05_rtc.server --model pi05_rtc --checkpoint /checkpoint \
  --tokenizer /weights/paligemma_tokenizer.model --host 0.0.0.0
```

The model server completes a dummy-observation warmup before listening, so
remote request timing excludes its startup compilation.

Use `infer.benchmark --server ws://HOST:8000 --dataset ... --output ...` from
an environment with the dataset dependencies to measure round-trip latency.
The client report identifies its own package/platform environment; record the
server image digest and GPU separately when comparing remote hosts. The server
binds loopback by default; choose explicit reachability or an SSH tunnel for
other devices. No network location or cloud credential is embedded in images.

For real execution, the colleague also starts client and supervisor in the
robot image and supplies the accepted station file and shared lock mount. See
README.md. The Compose example defaults to a fake robot and grants no USB or
physical control privileges. Add the specific host's device access only during
its integration; Docker cannot validate robot route clearance or payload.

## Reproducibility and platform limits

The copied algorithm files and their source hashes are recorded in
source-manifest.json. GPU and robot locks are generated for Linux x86_64/Python
3.12. `requirements-cpu.in` is the portable development input; resolve it for
the development host rather than installing CUDA wheels on a Mac. Container
builds and import checks on Mac's amd64 emulation do not measure GPU speed.
No checkpoint, dataset, lab serial, credential or run output belongs in an image.
