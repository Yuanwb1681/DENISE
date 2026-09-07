# MPI Marmousi input copied for standalone GPU testing

This directory contains the required inputs copied from
`/home/ywb/DENISE-Black-Edition/par`: acquisition geometry, the three start
models, the four-stage workflow, and all 100 shots (x/y components) under
`su/MARMOUSI_spike`. Historical MPI output files were not copied.

Run the validated one-shot, one-stage test directly (no `mpirun`):

```bash
./run_quick_gpu.sh
```

Run the original 100-shot, four-stage FWI configuration directly:

```bash
./run_full_gpu.sh
```

Both GPU inputs use `NPROCX=NPROCY=1`. The quick case preserves the original
6-second record length so that the copied SU observations remain compatible.
