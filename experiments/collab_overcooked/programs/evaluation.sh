#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"

"$PY" -m multimodal_comms.apps.collab_overcooked.evaluation --test_mode build_in
"$PY" -m multimodal_comms.apps.collab_overcooked.organize_result
"$PY" -m multimodal_comms.apps.collab_overcooked.convert_result
