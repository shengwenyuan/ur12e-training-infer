#!/usr/bin/env bash
set -euo pipefail
exec "${UR12E_PYTHON:-python3}" -m training.cli "$@"
