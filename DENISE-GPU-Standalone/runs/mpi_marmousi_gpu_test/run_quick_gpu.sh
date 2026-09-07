#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec /home/ywb/DENISE-GPU-Standalone/bin/denise_standalone_gpu \
  DENISE_marm_OBC_gpu_quick.inp FWI_workflow_quick.inp
