# GPU port work

Code for the `nnp-train` GPU port described in `../GPU_PORTING_PLAN.md`. Kept
separate from `src/` so it doesn't touch the shared, upstream-tracked build
tree until it's ready to be integrated (see the plan's Phase 6).

## `smoke/`

Standalone CUDA smoke tests: from-scratch reimplementations of individual
n2p2 compute kernels (not linked against `libnnp`), each validated against
the exact same math read out of the corresponding CPU source file. Used to
prove out the CUDA toolchain and one kernel's numerics in isolation before
any integration work.

- `symfnc_exprad_test.cu` — radial symmetry function (type 2, `SymFncExpRad`)
  with the TANHU cutoff (`cutoff_type 2`), one CUDA thread per atom. Validates
  against `src/libnnp/SymFncExpRad.cpp`/`CutoffFunction.cpp` on two systems:
  a synthetic 630-atom system with realistic (81-132) neighbor counts
  (matching `temp/H2O_2G`'s measured neighbor statistics), and real neighbor
  geometry loaded from `dump_real_neighbors.cpp`'s output (pass its output
  file as argv[1]).
- `dump_real_neighbors.cpp` — one-off diagnostic linked against the already
  -built `lib/libnnp.a`, using n2p2's own `ElementMap`/`Structure` classes
  (not a reimplementation) to load `temp/H2O_2G/input.data` and dump real
  H-H neighbor pairs from the real `Structure::calculateNeighborList()`, so
  the CUDA test can validate against physically real geometry, not just
  synthetic random data.
- `run.slurm` — builds and runs both (dumper first, then the CUDA test
  against its output) on a Booster A100 (`sbatch gpu/smoke/run.slurm`). No
  GSL/BLAS linking needed for the dumper (confirmed `libnnp.a` has no
  undefined `gsl_*` symbols for this usage) -- convenient, since the system
  linker can't parse `libgsl.so`'s compressed debug sections anyway.
