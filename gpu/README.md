# GPU port work

Code for the `nnp-train` GPU port described in `../GPU_PORTING_PLAN.md`. Kept
separate from `src/` so it doesn't touch the shared, upstream-tracked build
tree until it's ready to be integrated (see the plan's Phase 6).

## `smoke/`

Standalone CUDA smoke tests: from-scratch reimplementations of individual
n2p2 compute kernels (not linked against `libnnp`), each validated against
the exact same math read out of the corresponding CPU source file. Used to
prove out the CUDA toolchain and each kernel's numerics in isolation before
any integration work.

Symmetry function coverage (7 of n2p2's 11 leaf types in `src/libnnp/SymFnc*.cpp`):

| Type | File | Status |
|---|---|---|
| `SymFncExpRad` (2) | `symfnc_exprad_test.cu`, `symfnc_family_test.cu` | done |
| `SymFncExpRadWeighted` (12) | `symfnc_family_test.cu` | done |
| `SymFncCompRad` (20) | `symfnc_family_test.cu` | done |
| `SymFncCompRadWeighted` (21) | `symfnc_family_test.cu` | done |
| `SymFncExpAngn` (3) | `symfnc_family_test.cu` | done |
| `SymFncExpAngnWeighted` (13) | `symfnc_family_test.cu` | done |
| `SymFncExpAngw` (9) | `symfnc_family_test.cu` | done |
| `SymFncCompAngn` / `CompAngnWeighted` / `CompAngw` / `CompAngwWeighted` | — | **not done** — different angle-space (acos) parameterization, see below |

- `symfnc_exprad_test.cu` — the original ExpRad-only smoke test (one CUDA
  thread per atom), kept as a synthetic-system regression check (`./
  symfnc_exprad_test`, no arguments). Superseded for real-data validation by
  `symfnc_family_test.cu`.
- `symfnc_family_test.cu` — covers 7 types across the radial and Exp-angular
  families, one CUDA thread per selected central atom, all validated against
  real `H2O_2G` neighbor geometry (`./symfnc_family_test
  real_neighbors_full.txt`). `SymFncExpRad`/`SymFncExpAngn` use real
  production parameters straight out of `temp/H2O_2G/input.nn`; the other 5
  types use representative parameters on the same real geometry, since that
  input.nn doesn't define instances of them. Validates the unscaled energy
  accumulator and the central atom's own derivative only (see the file's
  header comment for the exact scope, consistent with `symfnc_exprad_test.cu`).
- **Not yet covered**: the 4 "Compact angular" types use a fundamentally
  different parameterization — a `CompactFunction` of `acos(cos theta)`
  directly (compact support in angle-space), not the classic
  `(1+lambda*cos)^zeta` form the Exp-angular family uses. Needs its own
  read-through of `SymFncBaseCompAng.{h,cpp}` before porting; left for a
  follow-up rather than rushed.
- `dump_real_neighbors.cpp` — one-off diagnostic linked against the already
  -built `lib/libnnp.a`, using n2p2's own `ElementMap`/`Structure` classes
  (not a reimplementation) to load `temp/H2O_2G/input.data` and dump every
  atom's complete, unfiltered neighbor list (element, distance, displacement
  vector) from the real `Structure::calculateNeighborList()`. One dump feeds
  every symmetry function type's test — each applies its own element-
  selection rule (or none, for the `*Weighted` variants) exactly like the
  corresponding `SymFnc::calculate()` does.
- `run.slurm` — builds the dumper and both CUDA tests and runs them in
  sequence on a Booster A100 (`sbatch gpu/smoke/run.slurm`). No GSL/BLAS
  linking needed for the dumper (confirmed `libnnp.a` has no undefined
  `gsl_*` symbols for this usage) — convenient, since the system linker
  can't parse `libgsl.so`'s compressed debug sections anyway.
