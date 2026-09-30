#!/usr/bin/env bash
# Run inside a Slurm allocation; submission is a separate, explicit action.
set -euo pipefail
MODE=${1:?Usage: explorer.sh prepare|smoke|train [--resume]}
shift
case "$MODE" in
  prepare|smoke) (( $# == 0 )) || exit 2 ;;
  train) (( $# == 0 )) || [[ $# == 1 && $1 == --resume ]] || exit 2 ;;
  *) exit 2 ;;
esac
: "${SLURM_JOB_ID:?Run this launcher in a Slurm allocation}"
: "${UR12E_IMAGE:?Set the verified SIF path}"
: "${UR12E_DATASET:?Set the immutable dataset directory}"
: "${UR12E_PRETRAINED:?Set the existing OpenPI asset directory}"
: "${UR12E_OUTPUT:?Set the persistent output directory}"
: "${UR12E_CACHE:?Set the scratch cache directory}"
SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PROFILE=${UR12E_PROFILE:-$SOURCE/training/pi05_rtc/config/ur12e_pick_place_2gpu.toml}
[[ -f $PROFILE ]] || { echo "Missing training profile" >&2; exit 1; }
for path in "$UR12E_IMAGE" "$UR12E_DATASET" "$UR12E_PRETRAINED"; do
  [[ -e $path ]] || { echo "Missing mounted input: $path" >&2; exit 1; }
done
mkdir -p "$UR12E_OUTPUT/reports" "$UR12E_CACHE/tmp" "$UR12E_CACHE/wandb"
export APPTAINERENV_PYTHONPATH=/app
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_HF_DATASETS_OFFLINE=1
export APPTAINERENV_TMPDIR=/cache/tmp
export APPTAINERENV_JAX_COMPILATION_CACHE_DIR=/cache/jax
export APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
export APPTAINERENV_WANDB_MODE=disabled
export APPTAINERENV_OMP_NUM_THREADS=4
export APPTAINERENV_OPENBLAS_NUM_THREADS=4
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
OPTIONS=(--cleanenv --no-mount tmp --pwd /app
  --bind "$SOURCE:/app:ro" --bind "$UR12E_DATASET:/data:ro"
  --bind "$PROFILE:/config/profile.toml:ro"
  --bind "$UR12E_PRETRAINED:/pretrained:ro"
  --bind "$UR12E_OUTPUT:/runs" --bind "$UR12E_CACHE:/cache")
ARGS=(--config /config/profile.toml --dataset /data
  --norm-root /runs/assets/pick-place-zero-prefix-v1
  --checkpoint-root /runs/checkpoints
  --initialization /pretrained/openpi/pi05_base/params
  --tokenizer /pretrained/paligemma_tokenizer.model)
if [[ $MODE == prepare ]]; then
  export APPTAINERENV_JAX_PLATFORMS=cpu
  MODULE=training.launchers.explorer_prepare
  ARGS+=(--report "/runs/reports/prepare-${SLURM_JOB_ID}.json")
else
  : "${CUDA_VISIBLE_DEVICES:?Slurm must set allocated GPU visibility}"
  export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
  export APPTAINERENV_JAX_PLATFORMS=cuda
  OPTIONS+=(--nv)
  MODULE=training.cli
  ARGS=("$MODE" "${ARGS[@]}" "$@")
  if [[ $MODE == train ]]; then
    mkdir -p "$UR12E_OUTPUT/wandb"
    export APPTAINERENV_WANDB_MODE=${UR12E_WANDB_MODE:-offline}
    case "$APPTAINERENV_WANDB_MODE" in
      online)
        : "${UR12E_WANDB_ENTITY:?Set the W&B team}"
        : "${UR12E_WANDB_KEY_FILE:?Set a private credential file}"
        IFS= read -r APPTAINERENV_WANDB_API_KEY < "$UR12E_WANDB_KEY_FILE"
        [[ -n $APPTAINERENV_WANDB_API_KEY ]] || exit 2
        export APPTAINERENV_WANDB_API_KEY
        export APPTAINERENV_WANDB_ENTITY="$UR12E_WANDB_ENTITY"
        ;;
      offline|disabled) unset APPTAINERENV_WANDB_API_KEY ;;
      *) echo "Unsupported W&B mode" >&2; exit 2 ;;
    esac
    export APPTAINERENV_WANDB_DIR=/runs/wandb
  fi
fi
printf 'Slurm job %s: %s, source=%s, image=%s\n' \
  "$SLURM_JOB_ID" "$MODE" "$SOURCE" "$UR12E_IMAGE"
exec apptainer exec "${OPTIONS[@]}" "$UR12E_IMAGE" \
  /opt/venv/bin/python -m "$MODULE" "${ARGS[@]}"
