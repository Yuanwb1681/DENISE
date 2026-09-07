# DENISE standalone single-GPU branch

This tree is independent from `/home/ywb/DENISE-GPU`.  It targets one CUDA GPU
and does not link an MPI implementation.  The local `include/mpi.h` implements
the legacy DENISE single-process control calls only; there is no communication
runtime and the executable must be launched directly.

## Build

```bash
make -C libcseife -j4
make -C src -j4 denise
```

The executable is `bin/denise_standalone_gpu`.

## Marmousi FWI

```bash
cd /home/ywb/DENISE-GPU-Standalone/runs/marmousi_fwi
/home/ywb/DENISE-GPU-Standalone/bin/denise_standalone_gpu \
  DENISE_marm_OBC_gpu.inp FWI_workflow_marmousi.inp
```

Both `NPROCX` and `NPROCY` must remain 1. Do not use `mpirun`.

## Current migration state

- No MPI library or `mpirun` dependency.
- Single-rank halo exchanges are absent from P-SV propagation.
- P-SV propagation, sources, receivers, wavefield storage, correlation and
  energy accumulation execute on CUDA.
- Managed propagation allocations are prefetched to the GPU before propagation.
- The five large forward-wavefield history buffers use explicit `cudaMalloc`;
  they remain device-only from forward storage through adjoint correlation.
- All 15 live elastic P-SV fields and nine CPML state matrices use explicit
  device-only allocations and GPU clearing. The optimized path requires `L=0`
  and `SNAP=0`; unsupported modes stop explicitly instead of accessing device
  pointers from host code.
- P-SV gradient parameter conversion, Plessix-Mulder weighting, regularization,
  scaling and per-shot stacking are fused into GPU kernels for the Marmousi
  `EPRECOND=3` workflow.
- Butterworth filtering is parallelized across traces, and the production L2
  residual, reverse-time adjoint source and objective reduction run on CUDA.
- The shared line-search/model update used by every P-SV optimizer performs its
  six maximum reductions and bounded Vp/Vs/rho trial/commit update on CUDA.
- L-BFGS gradient scaling/history copies run on CUDA. Its full two-loop CUDA
  recursion is available with `DENISE_GPU_ENABLE_LBFGS_RECURSION=1`, but is not
  the default because many small synchronized reductions are slower on the
  tested GTX 1650. PCG also performs its
  three-class packing/unpacking, all four supported beta formulas, direction
  update, and previous-gradient update on CUDA.
- Default L-BFGS recursion and model/gradient checkpoint file I/O still require
  host-visible storage at iteration boundaries.

CPU fallbacks can be selected with `DENISE_GPU_DISABLE_MODEL_UPDATE=1` and
`DENISE_GPU_DISABLE_ITERATION_OPS=1`. Multi-iteration regressions are provided
in `tests/psv_fwi_lbfgs_iter2` and `tests/psv_fwi_pcg_iter2`; both force two
iterations so the non-initial optimizer paths are exercised.

The 500x174, 250-step P-SV forward regression completes in 0.19-0.21 seconds of
solver-reported time, down from 0.46 seconds. The four-stage FWI regression
completes in 3.82 seconds, down from 8.99 seconds before device-only live fields.
Both regressions and
the final Vp/Vs/rho models are bitwise identical to the retained MPI+CUDA branch.
