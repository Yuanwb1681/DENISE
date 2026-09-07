#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
nohup mpirun -np 1 /home/ywb/DENISE-GPU/bin/denise_gpu \
  DENISE_marm_OBC_gpu.inp FWI_workflow_marmousi.inp \
  > gpu_console.log 2>&1 &
echo $! > gpu_mpirun.pid
echo "GPU run started: PID $(cat gpu_mpirun.pid), log $PWD/gpu_console.log"
