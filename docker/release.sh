#!/usr/bin/env bash
# Build an immutable version tag locally; pushing requires the explicit flag.
set -euo pipefail
if (( $# < 2 )); then
  echo 'Usage: docker/release.sh DOCKERHUB_NAMESPACE VERSION [--push]' >&2
  exit 2
fi
NAMESPACE=$1
VERSION=$2
MODE=${3:---load}
[[ $NAMESPACE =~ ^[a-z0-9][a-z0-9_-]*$ ]] || exit 2
[[ $VERSION =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || exit 2
[[ $MODE == --load || $MODE == --push ]] || exit 2
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
REVISION=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo uncommitted)
if [[ $REVISION != uncommitted ]] && [[ -n $(git -C "$ROOT" status --porcelain) ]]; then
  REVISION="$REVISION-dirty"
fi
for ROLE in gpu robot; do
  docker buildx build --platform linux/amd64 "$MODE" \
    --build-arg "VERSION=$VERSION" --build-arg "SOURCE_REVISION=$REVISION" \
    -f "$ROOT/docker/Dockerfile.$ROLE" \
    -t "$NAMESPACE/ur12e-training-infer:$VERSION-$ROLE" "$ROOT"
done
