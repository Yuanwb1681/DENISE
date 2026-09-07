#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec mpirun -np 1 /home/ywb/DENISE-GPU/bin/denise_gpu \
  DENISE_marm_OBC_gpu.inp FWI_workflow_marmousi.inp
