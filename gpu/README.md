# GPU port work

Code for the `nnp-train` GPU port described in `../GPU_PORTING_PLAN.md`. Kept
separate from `src/` so it doesn't touch the shared, upstream-tracked build
tree until it's ready to be integrated (see the plan's Phase 6) -- as of
Phase 6's first pass, one real, narrow integration point now exists in
`src/` too: `src/libnnpgpu/` and a `Mode::calculateAtomicNeuralNetworks()`
call site gated behind `N2P2_GPU`/`make GPU=1`, described at this file's end.

## Final status summary

This file is a chronological development log -- the rest of it reads top to
bottom as the actual investigation happened, including dead ends. This
section is the opposite: a single, current-state reference for whoever
picks this up next, so they don't have to reconstruct it from ~40
`### Follow-up:` entries. Everything below is on `temp/H2O_4G` (the
electrostatics/charge-transfer dataset, `HDNNP_4G`), full 1254-structure
production scale, `memorize_symfunc_results` on, GPU side = 4xA100 on one
Booster node, 32 MPI ranks under a single shared NVIDIA MPS daemon, CPU
side = one full `dcgp_usr_prod` node (112 cores). Numbers below are a
fresh, clean rerun at the current commit specifically for this summary,
not copy-pasted from earlier entries (though they land close to the
numbers those entries already reported, which is itself a useful
cross-check).

### Where things stand, end to end

| | Stage 1 (charge/electronegativity) | Stage 2 (energy + force) |
|---|---|---|
| CPU, 112 cores (dcgp, full node) | `~7.5s`/epoch | `~695.9s`/epoch |
| GPU, 4xA100, 32 ranks (MPS) | `~4.4-4.5s`/epoch | `~84.5-87.4s`/epoch |
| **GPU vs CPU-112** | **`~1.7x` faster** | **`~8.0x` faster** |

Both stages now clearly favor GPU on this single-node-multi-GPU deployment
shape, on this problem size. That wasn't true for most of this file's
history -- stage 2 spent a long stretch **slower** than CPU-112 (as bad as
`2.36x` slower) before the fixes below flipped it, and stage 1 was `~1.5x`
slower right up until the Ewald-matrix-assembly investigation. The
single-node/4-GPU/shared-MPS deployment shape itself was validated correct
very early and never needed to change -- every win below came from fixing
what ran *on* that deployment, not the deployment itself.

### Successfully shipped (in the production `src/` tree today, behind `N2P2_GPU`/`make GPU=1`)

**New GPU device code:**
- `GpuNeuralNetwork.h/.cu` -- `gpuNnForwardDEdG` (batched NN forward +
  `dEdG`, `HDNNP_2G` and `HDNNP_4G`'s `"short"`/`"elec"` branches),
  `gpuNnEnergyDEdcSum` (stage-2 energy weight-Jacobian, summed),
  `gpuNnForceDFdcSum` (stage-2 force weight-Jacobian, summed, widened to
  4G's extra charge-input column). `25-28x` standalone on the forward
  pass; end-to-end effect folded into the totals below.
- `GpuKalmanFilter.h/.cu` -- the two `O(N^2 m)` steps of the Kalman weight
  update (`X=P.H`, the `K.X^T` covariance downdate term), `P` persistent
  on GPU for the filter object's lifetime. Shared by both stages. Two real
  correctness bugs found and fixed here via real end-to-end data (a
  hand-rolled GPU inverse that was numerically wrong for ill-conditioned
  production Jacobians, and a missing `selfadjointView`-equivalent
  symmetrization) -- see List 2 below for the inverse, which was removed
  entirely rather than kept broken.
- `GpuForces.h/.cu` -- owner-centric edge-list, `atomicAdd`-scatter port
  of `Mode::calculateForces()`'s short-range term, persistent per-structure
  topology cache. `HDNNP_2G` and (widened, no new device code)
  `HDNNP_4G`'s shared short-range loop. Single biggest win in this whole
  file: `HDNNP_4G` stage 2 `1311s -> 149.3s` (`~8.8x`) from this port alone.
- `GpuElecForces.h/.cu` -- 4G-specific electrostatic-force port
  (pre-multiplied `lambdaTotal`/`lambdaElec` into `dChidG`, owner-centric
  self+edge structure, dense `dAdrQ` reduction). Stage 2, 4G only, capped
  at a modest `~12%` win by `dAdrQ`'s per-call re-upload cost (`~9.5MB`
  /structure, genuinely changes every call -- see "what's left" below).
- `GpuQeqSolver.h/.cu` -- cuSOLVER LU-based batched replacement for the
  charge-equilibration `ColPivHouseholderQR` solve. Shared by both stages.
  Numerical safety pre-validated against real production matrices
  (condition number, LU-vs-QR agreement) before writing any CUDA -- exactly
  the check that would have caught the Kalman inverse bug earlier.
- `GpuEwaldReal.h/.cu` -- owner-centric edge-list, `atomicAdd`-scatter port
  of the Ewald real-space erfc double loop, persistent per-structure
  topology cache -- the `GpuForces.cu` template applied to a second,
  unrelated part of the code. Stage 1 primarily (also stage 2 via the
  shared `chargeEquilibration()` machinery). `~4.3x` further stage-1 win on
  its own.

**CPU-only algorithmic fixes (no device code, same effort, same file):**
- `AConstrainedQr` factorize-once fix (was refactorizing per solve across
  5 call sites) -- `~10x`/`~126x` on the pieces it touched, shared by both
  stages, bit-identical.
- Stage-1 Jacobian-assembly loop reassociation
  (`O(numAtoms^2 numWeights) -> O(numAtoms^2)+O(numAtoms numWeights)`) --
  `~3.1x`, bit-identical.
- `memorize_symfunc_results` enabled -- pure config change, `~1.76x` at
  full scale for `~1.2%` extra memory.
- Reciprocal-space Ewald sum reformulated as two GEMMs instead of an
  `O(numAtoms^2 x numK)` `cos()` loop -- caught and fixed a real
  triangle-placement bug along the way; a small win for *this* dataset
  specifically (`numK=3`, real-space-dominated), likely a bigger one for a
  dataset with a larger k-space grid.
- Two redundant-recomputation fixes, same bug shape found twice
  independently: stage 2's `calculateForces()` called twice per force
  candidate (`~40-50%` stage-2 win) and stage 1's `chargeEquilibration()`
  called twice per accepted candidate (part of the `~16-20%` stage-1 win).
- `collectDGdxiaAllAtoms()` -- precomputed CSR reverse-neighbor index
  replacing an `O(numAtoms x numNeighbors)` per-atom scan, computed once
  per structure. Stage 2, 4G only, `~27%` further win, bit-identical.

**Infrastructure (enables everything above):** `src/libnnpgpu/` build
integration, `NeuralNetwork::hasGpuCompatibleArchitecture()` gating,
NVIDIA MPS (single shared daemon, all 4 GPUs) fixing the 32-rank/1-GPU
contention that had made an early GPU pass *slower* than CPU net, spreading
ranks across all 4 GPUs (8/GPU), persistent per-architecture/per-structure
device-state caching everywhere instead of `cudaMalloc`/`cudaFree` per
call, plus several correctness fixes surfaced along the way
(`PM_TRAIN_RK0` silently running on every rank, a build-race heisenbug,
`nnp-dataset` never having been run).

### Tried, built, validated or debugged, and explicitly reverted

1. **Hand-rolled GPU Gauss-Jordan matrix inverse** (inside the Kalman
   port) -- passed its synthetic test, diverged **~10 orders of
   magnitude** on real end-to-end data (ill-conditioned real Jacobians
   vs. well-conditioned synthetic ones). Removed entirely, moved back to
   host Eigen.
2. **`GpuElecForces`'s `dChidG`/`dAdrQ`/`pEelecpr` caching** -- built to
   avoid re-uploading these every call; a live cross-check caught a real
   staleness bug (a code path that flips `hasAMatrix` true without
   refreshing the cache). Reverted entirely rather than ship an uncertain
   partial fix.
3. **Batched force-Jacobian dispatch** (`gpuNnForceDFdcSumBatch`) --
   validated bit-identical against 7232 live samples, **not a correctness
   bug** -- reverted because `task_batch_size_force 1` means every real
   candidate has exactly one batch member, so the batching path is pure
   overhead for zero benefit under this training config. Would need a
   training-methodology change (raising that hyperparameter) to ever
   engage, out of scope for a performance-only pass.
4. **`gpuNnChargeDEdc()`, the stage-1 `"fwd"` loop port** -- validated
   bit-exact on the first try, then caused a reproducible **`6-9%`
   regression** at full 32-rank MPS scale. Every component of the branch
   was bracketed and summed to less than half of measured `Q_err` -- the
   gap is inter-rank desynchronization/stalling from added GPU-queue
   contention, invisible to any single rank's own timer, the same failure
   class as the `GpuForces.cu` first-pass regression but with no
   topology-cache-style fix available here (the inputs genuinely change
   every call).

### What's left, and my honest opinion on it

Every remaining cost either bracketed or reasoned about in this file falls
into one of three buckets, and none of them look like quick wins:

1. **Already GPU-accelerated, capped by per-call dispatch/transfer
   overhead under MPS, not by unaccelerated compute.** Stage 1's
   `chargeEquilibration()` trial-loop calls (`~26%` of `Q_err`, all
   running through `GpuEwaldReal`/`GpuQeqSolver` already) and
   `GpuElecForces`'s `dAdrQ` re-upload are the two clearest examples. The
   `gpuNnChargeDEdc()` revert just demonstrated, concretely and
   reproducibly, that adding *more* small per-call GPU dispatch under this
   MPS setup is as likely to make things worse as better -- this isn't a
   theoretical risk anymore, it's a measured one. Chasing this further
   needs batching *across* candidates/structures into fewer, larger calls
   (the same idea the reverted batched-`calculateDFdc` attempt used), not
   another single-call port.
2. **Structurally blocked by training hyperparameters, not code.** The
   batched-force-Jacobian dispatch is *built and validated*, sitting idle
   behind `task_batch_size_force 1`. Turning it on is a one-line
   `input.nn` change with an already-working GPU path behind it -- but it
   changes training *convergence behavior* (more candidates per weight
   update), which is a modeling decision, not a performance one, and isn't
   this kind of engineering pass's call to make unilaterally.
3. **Cheap enough that porting isn't worth the risk.** Stage 1's `dq`/
   `jac` brackets are `~0.24s`/`~0.12s` per epoch -- genuinely small in
   absolute terms now that the two big stage-1 levers (Ewald real-space,
   the redundant-call fixes) are shipped. The CPU-only reciprocal-sum GEMM
   is similarly small for *this* dataset's Ewald parameters specifically.

**My honest read: there's no non-negligible low-hanging fruit left in the
"port more code to GPU" direction.** The two genuinely large,
well-scoped opportunities this file identified (stage 2's force/NN-forward
path, stage 1's Ewald matrix assembly) are both done, and both flips
(`2.36x` slower `-> 8.0x` faster; `1.5x` slower `-> 1.7x` faster) are
banked. What remains is either (a) a real but substantial architectural
change -- batching across candidates to cut per-call MPS-contention
overhead, which is new design work with the same numerical-safety burden
every port in this file has needed, not a quick follow-up -- or (b) a
training-methodology lever (`task_batch_size_force`) that isn't a
performance-engineering decision at all. If asked to prioritize, I'd stop
here rather than chase (a) speculatively: the last two things tried in
this exact spirit (the batched-Jacobian attempt, the `fwd` port) were both
correct and both didn't pay off, for two different structural reasons
that a third attempt in the same style would very plausibly hit again.

## `smoke/`

Standalone CUDA smoke tests: from-scratch reimplementations of individual
n2p2 compute kernels (not linked against `libnnp`), each validated against
the exact same math read out of the corresponding CPU source file. Used to
prove out the CUDA toolchain and each kernel's numerics in isolation before
any integration work.

Symmetry function coverage: **all 11 of n2p2's leaf types** in
`src/libnnp/SymFnc*.cpp` are ported and validated:

| Type | Status |
|---|---|
| `SymFncExpRad` (2) | done |
| `SymFncExpRadWeighted` (12) | done |
| `SymFncCompRad` (20) | done |
| `SymFncCompRadWeighted` (23) | done |
| `SymFncExpAngn` (3) | done |
| `SymFncExpAngnWeighted` (13) | done |
| `SymFncExpAngw` (9) | done |
| `SymFncCompAngn` (21) | done |
| `SymFncCompAngnWeighted` (24) | done |
| `SymFncCompAngw` (22) | done |
| `SymFncCompAngwWeighted` (25) | done |

- `symfnc_exprad_test.cu` — the original ExpRad-only smoke test (one CUDA
  thread per atom), kept as a synthetic-system regression check (`./
  symfnc_exprad_test`, no arguments). Superseded for real-data validation by
  `symfnc_family_test.cu`.
- `symfnc_family_test.cu` — all 11 types, one CUDA thread per selected
  central atom, all validated against real `H2O_2G` neighbor geometry
  (`./symfnc_family_test real_neighbors_full.txt`). `SymFncExpRad`/
  `SymFncExpAngn` use real production parameters straight out of
  `temp/H2O_2G/input.nn`; the other 9 types use representative parameters on
  the same real geometry, since that input.nn doesn't define instances of
  them. Validates the unscaled energy accumulator and the central atom's own
  derivative only (see the file's header comment for the exact scope,
  consistent with `symfnc_exprad_test.cu`). Two mathematically distinct
  angular parameterizations are both covered: the classic
  `(1+lambda*cos theta)^zeta` form (Exp-angular family) and a `CompactFunction`
  applied directly to `acos(cos theta)` — compact support in angle-space
  (Compact-angular family). For the latter's `*Weighted` variants, the
  atomic-number weight is folded into the angular part (`ang`/`dang`) rather
  than the radial part, unlike the Exp-angular weighted variants (which fold
  it into the exponential) — worth double-checking against
  `SymFncCompAngnWeighted.cpp` if this code is ever touched, since it's easy
  to misplace which factor the weight multiplies.
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

## `soa/`

Phase 1 (`../GPU_PORTING_PLAN.md` §5) — data layout redesign, done step by
step starting with the layout itself, validated in isolation before any
kernel is rewritten to consume it.

**Step 1 (done): `AtomBatch`**, a struct-of-arrays / CSR flattening of a real
n2p2 `Structure`'s atoms and neighbor lists, replacing the current
array-of-structs `std::vector<Atom>` (each owning its own
`std::vector<Atom::Neighbor>`, each with its own `std::vector<double>`
cache/`std::vector<Vec3D>` dGdr — pointer-chasing, non-coalesced on GPU):

- `AtomBatch.h`/`.cpp` — atoms reordered element-major (all element-0 atoms
  first, then element-1, ...; `elementOffset` gives the block boundaries —
  this is the layout Phase 3's per-element batched NN GEMMs need), positions
  as separate contiguous `x`/`y`/`z` arrays, and neighbor lists flattened
  into CSR (`neighborOffset` prefix-sum + flat `neighborElement`/`neighborD`/
  `neighborDx,Dy,Dz`/`neighborAtomSorted` arrays). Same shape as
  `smoke/`'s `real_neighbors_full.txt` dump, generalized: built directly
  from `Structure` in memory (no text round-trip), element-sorted, with
  explicit offsets instead of a linear scan.
- `build_batch_test.cpp` — host-side-only validation (no CUDA yet): builds
  an `AtomBatch` from the real H2O_2G structure and checks, atom by atom and
  neighbor by neighbor, that it's a lossless re-encoding of
  `Structure`/`Atom::neighbors` (permutation bijectivity, element grouping,
  exact position/neighbor values). 630 atoms, 66868 neighbor entries,
  405621 checks, all passing — confirms the new layout before any kernel
  depends on it.

**Step 2 (done): `symfnc_exprad_soa_test.cu`**, wiring `SymFncExpRad`'s CUDA
kernel (math copied verbatim from `smoke/symfnc_family_test.cu`, already
validated there) directly to a real `AtomBatch`, replacing `smoke/`'s
`loadRealSystem()` text-file parsing entirely — this binary loads
`temp/H2O_2G/input.data` via `ElementMap`/`Structure` itself, builds one
`AtomBatch`, copies its CSR arrays to the device, and runs the kernel. Two
things the layout buys for free, demonstrated here:

- Central-atom selection for a given element is the contiguous slice
  `[elementOffset[ec], elementOffset[ec+1])` of already-sorted atoms — no
  scan, no `selectedAtoms` gather array, unlike `smoke/`'s `runCase()`.
- The kernel launch operates directly on that sorted range; thread `t` maps
  to sorted atom index `begin + t`.

Both real-parameter `ExpRad` cases (H central, e1=H, the two `eta`/`rs`
pairs `input.nn` actually uses) pass to ~1e-15 (420 selected H atoms each).
Together with step 1's build_batch_test.cpp, this chains
GPU-kernel-vs-`AtomBatch`-vs-`Structure` correctness end to end.

`run.slurm` builds `AtomBatch.o` once (g++, `-std=c++14` to link cleanly
against nvcc's host objects) and reuses it for every step's binary; needs
`--gres=gpu:1` from step 2 onward.

**Step 3 (done): `symfnc_exprad_group_test.cu`**, persistent per-atom SF
storage (`AtomBatch::G`) written by a "grouped" kernel — the first kernel to
*write into* `AtomBatch` rather than only read it. Adds:

- `AtomBatch::sfCountPerElement`/`gBlockOffset`/`G` and `allocateSfStorage()`
  — one flat array with one contiguous block per element (block size =
  atoms-of-that-element x sfCountPerElement[e], row-major so one atom's SF
  values are contiguous), since each central element has its own
  independently-sized symmetry function list in `input.nn`. `gIndex(s, k)`
  gives the flat offset for sorted atom `s`'s `k`-th symmetry function.
- The kernel mirrors `src/libnnp/SymGrpExpRad.cpp::calculate()`'s actual
  optimization: multiple symmetry functions sharing the same element
  filter/cutoff radius evaluate the cutoff function **once per neighbor**
  and reuse it across all group members, instead of each one re-walking the
  neighbor list independently (what step 2's single-SF kernel did). Two
  ExpRad instances (H central/e1=H, the same two real `eta`/`rs` pairs used
  in step 2) are evaluated together per H atom and written straight into
  `AtomBatch::G`'s device mirror.
- Validated against a CPU reference built with the same grouped structure
  (840 stored values, H block = 420 atoms x 2 SFs), to ~1e-15, and
  cross-checked against `gIndex()`'s own index arithmetic inline.

**Step 4 (done, folded into `symfnc_exprad_group_test.cu`)**: both
derivative storages `SymFncExpRad.cpp` actually produces per matching
neighbor (`Vec3D dij = p1 * n.dr; atom.dGdr[index] += dij; n.dGdr[...] -=
dij;`) — closing the gap `smoke/`'s original per-type tests explicitly left
out of scope ("Neighbor-side derivative bookkeeping is out of scope"):

- `AtomBatch::dGdx`/`dGdy`/`dGdz` — the owner atom's own derivative, summed
  over its matching neighbors. Same per-element block layout/indexing as
  `G` (`gIndex()`).
- `AtomBatch::neighborSfOffset` + `neighborDGdx`/`Dy`/`Dz` — one entry **per
  neighbor slot**, not summed, addressed via `neighborSfIndex(s,
  neighborSlot, sfIndex)`: a CSR-of-blocks mirroring `neighborOffset`, but
  each atom's block width is `sfCountPerElement[that atom's element]`
  instead of 1. This is the exact quantity Phase 3's force assembly will
  scatter-add onto each neighbor atom's own force
  (`Training::collectDGdxia`/`calculatePairForceShort`) — the ~46.5%-of-
  wall-time cost center §3a's profiling identified.
- Both allocated together by `allocateSfStorage()`, zero-initialized (a
  non-matching neighbor slot's derivative is exactly zero — the CPU source
  agrees, since that neighbor doesn't contribute to the sum at all).
- Validated: 840 owner-derivative values and 89346 individual neighbor-slot
  derivative values, all GPU vs. CPU to ~1e-16, plus `neighborSfIndex()`'s
  index arithmetic cross-checked inline (mirroring step 3's `gIndex()`
  check).

**Step 5 (done): `symfnc_expangn_group_test.cu`**, neighbor-side derivatives
for the narrow angular (`SymFncExpAngn`, type 3) family — closes the last
gap left by `smoke/`'s original per-type tests ("Neighbor-side derivative
bookkeeping is out of scope" applied there to *all* angular types; step 4
already closed it for the radial family). Needed to cover the real
35/42-wide H2O_2G production network end to end (`e2e/`'s first pass only
covered the 16-wide radial subset, since this piece didn't exist yet).

- Re-derived line by line from `src/libnnp/SymFncExpAngn.cpp:79-225`: an
  angular triple (central atom, neighbors `j` and `k`) produces THREE force
  vectors (`drij`, `drik`, `drjk`), split as `atom.dGdr[index] += drij +
  drik` (central atom, already validated in `smoke/`), `nj.dGdr[...] -=
  drij + drjk`, `nk.dGdr[...] -= drik - drjk` (both new). Grouped over all
  of one element's real instances at once (19 for H, 26 for O), sharing the
  `(j,k)` pair enumeration and cutoff geometry across members, generalized
  to a per-member `(e1,e2)` filter like `e2e/`'s radial generalization.
- Since a neighbor slot can appear as `j` in one pair and `k` in another,
  its neighbor-derivative entry must **accumulate** (`+=`) across the whole
  double loop — unlike the radial family's neighborDGdx (step 4), where
  each slot got exactly one direct write. Still no atomics needed: a given
  central atom's neighbor-derivative block is only ever touched by the one
  thread computing that atom's own symmetry functions.
- Validated against an independent CPU pass of the same `__host__
  __device__` function: both elements pass on the first run (`G`, the
  central atom's own derivative, and every individual neighbor-slot
  derivative all matching to ~1e-16).

## `nn/`

Phase 3 (`../GPU_PORTING_PLAN.md` §5) — NN forward/backward + force
assembly, the phase Phase 0's profiling identified as the actual dominant
cost (~78% of wall time: force assembly ~46.5%, NN backward/Jacobian
~30.8%), unlike symmetry functions (~4%). Picked over further broadening
`soa/`'s symmetry-function coverage because `AtomBatch`'s element-grouped
`G` storage is exactly the layout this phase needs (weights are per-element,
atoms are already sorted by element) — the Phase 1 groundwork was written
with this phase in mind, not just for symmetry functions.

**Step 1 (done): `nn_forward_test.cu`**, the NN forward pass on GPU, one
thread per atom (same pattern as `smoke/`'s symmetry-function kernels),
reading straight from `AtomBatch::G` and writing into `AtomBatch::energy`
(new atom-level output slot, indexed directly by sorted atom position, no
block layout needed since it's one scalar per atom unlike `G`). Mirrors
`Mode::calculateAtomicNeuralNetworks()` (`src/libnnp/Mode.cpp:1666`): one
shared set of weights per element, propagated independently per atom.

- H2O_2G's real architecture from `temp/H2O_2G/input.nn`
  (`global_hidden_layers_short 2`, `global_nodes_short 25 25`,
  `global_activation_short t t l`): input (35 for H, 42 for O — real
  `symfunction_short` counts in that file) → 25 (tanh) → 25 (tanh) → 1
  (identity).
- Ground truth is a **real** `nnp::NeuralNetwork` (linked from
  `lib/libnnp.a`, not a reimplementation), same architecture, random
  weights via its own `initializeConnectionsRandomUniform()`, propagated
  the standard way (`setInput`/`propagate`/`getOutput`) — same "reuse real
  n2p2 classes as the CPU reference" approach as `smoke/`'s
  `dump_real_neighbors.cpp`. `G` values are synthetic (this step validates
  the forward pass in isolation, decoupled from `soa/`'s symmetry-function
  kernels); atom counts per element (420 H, 210 O) come from the real
  H2O_2G structure via `AtomBatch`.
- Weight layout matches `NeuralNetwork::getConnections()`'s documented
  per-layer order exactly (`W[j*numCur+k]` = weight from previous-layer
  neuron `j` to current-layer neuron `k`) — already the "X (atoms x
  numPrev) times W (numPrev x numCur)" shape a later batched-GEMM version
  would want.
- Validated both element architectures (H: 420 atoms, O: 210 atoms) to
  ~1e-15 against `NeuralNetwork::propagate()`.

**Step 2 (done, folded into `nn_forward_test.cu`)**: the NN backward pass,
`dEdG` — the derivative of an atom's NN energy output with respect to each
of its own symmetry-function inputs (`Atom::dEdG` in
`Mode::calculateAtomicNeuralNetworks()`; not to be confused with Phase 1's
`dGdx`/`neighborDGdx`, the derivative of a symmetry function with respect
to atom *coordinates* — `dEdG` is the other half of the force chain rule,
`dE/dx = dE/dG * dG/dx`).

- `AtomBatch::dEdG` — new storage, same per-element block layout/indexing
  as `G` (`gIndex()`), added alongside the existing arrays in
  `allocateSfStorage()`.
- The kernel ports `NeuralNetwork::calculateDEdG()`'s **exact algorithm**
  (`src/libnnp/NeuralNetwork.cpp:396`): for each input `k`, forward-propagate
  a unit sensitivity through the layers (multiply by weights and `dfdx` at
  each stage) — not a from-scratch reverse-mode backprop, which would reach
  the same numbers through a differently-ordered computation; this ports
  the CPU's specific algorithm, the same "exact port" approach used for the
  symmetry functions. Since both hidden layers are tanh, `dfdx = 1 -
  value^2` is recovered directly from the forward pass's already-computed
  activations (no separate pre-activation storage needed); the output
  layer is identity, so its `dfdx` is exactly 1.
- Computed in the same kernel launch as the forward pass (one thread per
  atom does both), mirroring how `Mode.cpp` calls `propagate()` then
  `calculateDEdG()` using the neuron state `propagate()` just left behind.
- Validated against the real `NeuralNetwork::calculateDEdG()`: both element
  architectures match to ~5e-15.

**Step 3 (done): `nn_backward_dfdc_test.cu`**, the weight-Jacobian pass,
`calculateDFdc`/`calculateD2EdGdc` — the other half of the ~30.8% NN
backward cost (`dEdG`, step 2, is the first half). This is what
`Training::update()`'s Jacobian assembly actually feeds to the weight
updater (Kalman filter / gradient descent both fit *forces*, so they need
`dF/dc`, not just `dE/dc`).

- `AtomBatch::connCountPerElement`/`dFdcBlockOffset`/`dFdc` +
  `allocateWeightJacobianStorage()`/`dFdcIndex()` — a new, separate
  per-element block store (can't reuse `gIndex()`/`sfCountPerElement`
  since a network's connection count and its symmetry-function count are
  different numbers).
- `calculateDFdc(dFdc, dGdxyz)` computes `dFdc[c] = -sum_k (d²E/dc dG_k) *
  dGdxyz[k]`, where `dGdxyz[k]` is `dG_k/dx` for some atom/coordinate —
  `AtomBatch::dGdx`/`neighborDGdx` (Phase 1, step 4) would supply this in a
  full integration; here it's synthetic, validating the NN math in
  isolation first, same incremental approach as steps 1-2's synthetic `G`.
- The kernel is a hand-specialized, unrolled port of
  `calculateD2EdGdc()`/`calculateDEdb()`/`calculateDxdG()`'s exact
  algorithm for this fixed architecture (2 hidden tanh(25) + linear
  output(1)) — re-derived directly from `NeuralNetwork.cpp:444-719` rather
  than translated as a generic per-layer loop, since porting the *exact*
  CPU algorithm (not a from-scratch re-derivation of the same math) is this
  project's established approach throughout.
- **One real bug, caught and fixed by the CPU comparison, not by
  inspection**: the first pass dropped a `dfdx2[j]` factor when computing
  `dEdb_hidden2[j]` (`calculateDEdb`'s recursion multiplies by the
  *destination* layer's own `dfdx`, easy to lose track of when hand-porting
  nested index arithmetic) — every `dFdc` entry touching `W2`/`b1`/`W1` came
  out wrong by an O(1) amount (max error ~12, obviously not roundoff) until
  fixed. Worth remembering: this class of bug is exactly why every step in
  this port validates against the real CPU class numerically rather than
  trusting a hand derivation on its own.
- Validated: 1,029,630 total `dFdc` values (H: 420 atoms × 1576
  connections, O: 210 atoms × 1751 connections) against the real
  `NeuralNetwork::calculateDFdc()`, to ~2.5e-14.

Batching the forward/dEdG pass with cuBLAS is now done — see `gemm/` below.
Force assembly (the single biggest cost center) is covered next, in
`force/`.

## `force/`

Force assembly — `Mode::calculateForces()`/`Atom::calculatePairForceShort()`,
the single biggest cost center Phase 0's profiling identified (~46.5% of
wall time), untouched until now. Combines Phase 3's `dEdG` (`nn/`) with
Phase 1's per-neighbor derivative storage (`soa/AtomBatch.h` step 4:
`dGdx,Dy,Dz` + `neighborDGdx,Dy,Dz`) into actual per-atom forces.

**Done: `force_assembly_test.cu`**. The exact formula
(`src/libnnp/Atom.cpp:372-402`, the `N2P2_FULL_SFD_MEMORY` branch — the one
matching `AtomBatch`'s storage shape, one derivative entry per (neighbor
slot, symmetry function), no per-element filter table needed):
`F_self_i = -sum_k dEdG_i[k]*dGdr_i[k]`, and atom `j`'s contribution to
neighbor `i`'s force is `-sum_k dEdG_j[k] * (j's neighbor-entry-for-i).dGdr[k]`.

- `Mode::calculateForces()` computes this from atom `i`'s perspective: loop
  over `i`'s unique neighbors `j`, then **re-scan** all of `j`'s neighbors
  looking for `i` (an O(k²) search per atom, mitigated on CPU by a compact
  per-element symmetry-function table this port doesn't use). `AtomBatch`'s
  layout makes that re-scan unnecessary: `neighborDGdx,Dy,Dz` is already
  addressed by (owner atom `j`, its neighbor slot, symmetry function), so
  the natural GPU formulation is a **scatter**: one thread per atom `j`
  adds its own self-force directly, then walks its *own* neighbor list
  once, atomically adding each pair contribution straight onto that
  neighbor's force accumulator (`atomicAdd` on `double`, native since
  compute capability 6.0, no custom implementation needed on an A100).
- `AtomBatch::forceX,Y,Z` — new atom-level output slot (same shape as
  `energy`: one 3-vector per atom, no per-element block layout).
- Ground truth is **not** the real `Atom::calculateSelfForceShort()`/
  `calculatePairForceShort()` here (unlike `NeuralNetwork` in `nn/`): using
  them would need either the compact per-element table (full
  `Mode`/`Element`/`Settings` setup, far heavier than this project's other
  smoke tests) or rebuilding `libnnp.a` with `-DN2P2_FULL_SFD_MEMORY`, which
  silently changes `Atom`'s struct layout (`Atom.h` conditionally adds a
  member under that macro) relative to the rest of the already-built
  library — an ABI mismatch across translation units, not a safe option
  for a validation test. Instead this validates the scatter-add
  **algorithm** independently: an explicit, differently-ordered "gather" CPU
  reference (for each target atom `i`, self term plus a scan over every
  *other* atom `j`'s neighbor slots looking for `i`) computes the same sums
  via a genuinely different traversal than the GPU's scatter, so an
  addressing bug in either one would very likely disagree with the other.
- `dEdG`/`dGdx,Dy,Dz`/`neighborDGdx,Dy,Dz` values are synthetic random
  numbers on the real H2O_2G `AtomBatch` — this step validates the assembly
  formula/addressing in isolation, same incremental philosophy as every
  step before it. Real values are *not* expected to sum to zero net force
  here (that Newton's-third-law property depends on the `dGdr`/`-dGdr` sign
  relationship real `SymFnc` code enforces, which synthetic data doesn't
  reproduce) — only GPU-vs-CPU agreement is checked.
- Validated on the first run: 630 atoms, 66868 neighbor entries, GPU vs.
  independent CPU gather to ~2.4e-13.

Next steps (not yet done): determinism (`atomicAdd` on doubles is not
run-to-run bit-reproducible — the plan flags this explicitly, an
alternative would be a neighbor-major segmented reduction). Wiring real
values through instead of synthetic ones is covered next, in `e2e/`.

## `e2e/`

End-to-end integration: chains every piece validated so far — Phase 2's
radial *and* narrow-angular symmetry function math, Phase 3's NN
forward/backward, and force assembly — into one real GPU pipeline for the
real H2O_2G structure, using **real** geometry and **real**
symmetry-function parameters parsed straight out of `temp/H2O_2G/input.nn`,
not synthetic data at each stage like every step before this one existed.
This is what was recommended after Phase 3 closed out: every piece so far
had been validated in isolation (`dEdG` used random `G`; force assembly
used random `dEdG`/derivatives), so nothing had exercised the *wiring*
between phases with real, physically-connected numbers.

**Done: `e2e_single_structure_test.cu`**, now covering the **full real
35/42-wide production network** (16 radial + 19 angular = 35 for H; 16
radial + 26 angular = 42 for O) — the first version of this file only
covered the 16-wide radial subset, since the angular family's neighbor-side
derivatives didn't exist yet; those were ported and validated separately
first (`soa/symfnc_expangn_group_test.cu`), then wired in here.

- **Symmetry-function index order is not arbitrary**: `input.nn` defines,
  for a given central element, all of that element's type-2 (radial)
  instances contiguously, followed by all of its type-3 (angular) instances
  (verified directly against the file — no interleaving), so concatenating
  `[radial members][angular members]` in parse order reproduces the real
  production `G`-vector layout exactly, not just "a" 35/42-wide network.
  Both families write into the *same* per-atom `G`/`dGdx`/`neighborDGdx`
  block via an `sfIndexOffset` (radial at `[0, 16)`, angular at
  `[16, 35)` for H, etc.) — a new parameter both grouped kernels needed,
  since each family's own member count no longer equals the atom's full
  width once two families share one block.
- **GPU pipeline**: all data stays resident on device across the four
  kernel launches (radial symmetry functions → angular symmetry functions
  → NN forward+`dEdG` → force assembly) — only the initial geometry/weights
  upload and the final energy/force download cross the host/device
  boundary, mirroring what a real batched training step would do.
- **CPU reference**: fully independent — the same real `nnp::NeuralNetwork`
  class, and the same `symFncExpRadGroupReal()`/`symFncExpAngnGroupReal()`
  `__host__ __device__` functions called directly on the host (specifically
  re-checking the *wiring*/indexing, including the two families sharing one
  `G` vector, not the core per-neighbor math again, which earlier phases
  already validated bit-exact). Force assembly's CPU side reuses `force/`'s
  independent "gather" traversal.
- **One latent bug caught and fixed before the final run, by inspection
  this time, not by a failing comparison**: the first draft of the combined
  kernels staged each atom's neighbor-derivative contributions in a
  fixed-size local array (`double neighTmpX[512]`) sized by guesswork, then
  copied it out to the real strided location afterward — with real
  per-atom neighbor counts and up to 26 members, this could silently
  overflow for any atom with more neighbors than `512/26`, corrupting stack
  memory. Fixed by having the grouped device functions write directly into
  the global `neighborDGdx,Dy,Dz` array at its real `(neighborSlot *
  sfStride + sfIndexOffset + member)` address — no local staging buffer at
  all, and no fixed size to get wrong. Worth remembering as its own
  category of bug alongside the earlier dropped-`dfdx`-factor one: not
  every mistake in this port shows up as a wrong number against the CPU
  reference — some are memory-safety bugs that need to be caught by
  re-reading the code, not just by comparing outputs.
- Validated: 630 atoms, 66868 neighbor entries, 23520 `G` values (the real
  35/42-wide count), all matching the independent CPU reference to ~5e-15
  (energy, forces, and `G` all individually checked).

**Done: Stage 5**, wiring `kalman/`'s Kalman filter into this pipeline with a
**real** per-structure energy Jacobian in place of `kalman_test.cu`'s
synthetic `H`/`xi` — closing the loop from "symmetry functions → NN →
forces" all the way to "→ weight update". This is a genuine energy update,
the dominant update type in this project's config (`short_energy_fraction
1.0` vs. `short_force_fraction 0.0041`) — exactly what `Training::update`
does for the `"energy"` property once per structure.

- **The missing piece was `NeuralNetwork::calculateDEdc()`**
  (`src/libnnp/NeuralNetwork.cpp:444`) — the derivative of the atomic energy
  output w.r.t. every connection (weight+bias), *not* `calculateDFdc` (the
  force-Jacobian already ported in `../nn/nn_backward_dfdc_test.cu`, which
  isn't what an energy-fit update needs). Re-derived by hand and confirmed
  against the real source to reduce to exactly the standard backprop deltas
  `calculateDFdc` already computes as intermediates (`dEdbHidden2`,
  `dEdbHidden1`): `dE/db3=1`, `dE/dW3[j]=h2[j]`, `dE/db2[j]=W3[j]·dfdx2[j]`,
  `dE/dW2[i][j]=dEdbHidden2[j]·h1[i]`, `dE/db1[i]=dfdx1[i]·Σ_j
  W2[i][j]·dEdbHidden2[j]`, `dE/dW1[jin][i]=dEdbHidden1[i]·G[jin]` — same
  `[W1,b1,W2,b2,W3,b3]` flat order as `calculateDFdc`/`getConnections()`.
- **Structure-level Jacobian = per-atom `dEdc` summed over that element's
  atoms**, since every atom of an element shares that element's weights —
  the same reasoning as the total energy itself being a per-atom sum. The
  combined `N=3327` state vector is built from the network's *actual current*
  weights (`connH`/`connO`, already in hand from earlier stages), not zero
  like `kalman_test.cu`'s proof-of-concept — a real update moves existing
  weights, it doesn't start a delta from scratch. Observation is `m=1`: this
  structure's energy residual (`structure.energyRef` — parsed by the real
  `Structure` class from `input.data` — minus the already-computed total
  energy).
- **The Kalman recursion kernels are copied verbatim from `kalman/
  kalman_test.cu`** (already validated there at this exact `N=3327` size);
  only the Jacobian/observation feeding them is new. Validated the same way:
  GPU vs. an independent CPU implementation vs. the real `nnp::KalmanFilter`
  class (same debug-info-stripping link workaround as `kalman/`, added to
  this directory's `run.slurm` too).
- Validated: `max|dEdc_gpu-dEdc_cpu| = 2.3E-13` (3327 combined connections),
  `max|w_gpu-w_cpu| = 1.1E-16`, `max|w_gpu-w_real| = 2.2E-16`. Applying the
  updated weights and re-running the forward pass moves the total energy
  from `1702.3` to `487.1` for this random, untrained network — reported
  informationally only, since a single Kalman step from random initial
  weights/`P`/`eta` isn't expected to land anywhere near the real reference
  energy (`-4.15`), and "did the loss get better" was deliberately not used
  as a pass/fail criterion (unlike the bit-exact comparisons above).

**Done: Stage 6**, a force-fit Kalman update — same wiring as Stage 5, but
using `calculateDFdc` (the force-weight Jacobian, `../nn/
nn_backward_dfdc_test.cu`) instead of `calculateDEdc`, since this project's
config also does force updates (`short_force_fraction 0.0041`, far less
often than energy's `1.0` but still part of a real training run).

- **A single force component depends on more than one atom's network**,
  unlike energy: atom T's force is `-Σ_k dEdG_T[k]·dGdx_T[k]` (T's own
  network) *minus* a sum over every atom J that lists T as a neighbor of
  `-Σ_k dEdG_J[k]·neighborDGdx[J,slot(T),k]` (J's own network) — exactly the
  self+neighbor-scatter structure `forceAssemblyKernel` (Stage 4) already
  uses for the scalar `dEdG·dGdx` product. The insight this reuses: `d(dEdG_
  atom[k])/dc` is exactly what `calculateDFdc`'s internal `calculateD2EdGdc`
  already computes per input `k`, before contracting with whatever
  `dGdxyz` array is handed to it — so calling the *same* `nnCalculateDFdc`
  on a contributing atom's own G/weights with `dGdxyz = neighborDGdx[J,
  slot(T),:]` (instead of J's own `dGdx`) gives exactly J's contribution to
  T's force Jacobian, accumulating a full per-weight vector via `atomicAdd`
  instead of a scalar.
- **`forceJacobianKernel`**: one thread per atom J (same shape as
  `forceAssemblyKernel`), fixed to a single target atom/component (sorted
  atom 0, x) — if `J == target`, contract its own `dGdx` (self term); scan
  J's own neighbor list for any slot pointing at the target and contract
  `neighborDGdx` for each match (neighbor term). Both cases call the same
  `nnCalculateDFdc` (copied verbatim from `../nn/nn_backward_dfdc_test.cu`)
  into a fixed `double[2048]` local buffer — safe here, unlike the earlier
  buffer-overflow bug, since this file's architecture caps real connection
  counts at a compile-time-known 1751 (O), not an unbounded runtime neighbor
  count.
- **CPU reference** calls the real `nnH`/`nnO.calculateDFdc()` directly, one
  call per contributing atom, using the same self+neighbor-scan gather
  `forceXCpu` above already does. Stage 6 runs independently of Stage 5 (not
  chained after it) — it first resets `nnH`/`nnO`'s connections and the GPU
  weight buffers back to the original random weights Stage 5 mutated for its
  informational re-run, so both stages start from the same baseline.
- Validated: `max|dFdc_gpu-dFdc_cpu| = 6.7E-16` (3327 combined connections),
  `max|w_gpu-w_cpu| = 1.1E-16`, `max|w_gpu-w_real| = 3.9E-16`.

## `kalman/`

Phase 4 — the weight-update algorithm (`src/libnnptrain/KalmanFilter.cpp`),
i.e. the piece that turns Jacobians (Phase 3's `calculateDFdc`/
`calculateD2EdGdc` output) and errors into an updated weight vector. This is
what nnp-train actually calls once per scheduled structure/force-component
during training (not once per epoch) — `Training::update()`
(`src/libnnptrain/Training.cpp:2975-3010`) gathers one Jacobian column and
one error value per MPI rank (`MPI_Gather`, `parallel_mode 0` =
`PM_TRAIN_RK0`, so the `KalmanFilter` object only ever exists on rank 0),
then calls `KalmanFilter::update()`.

**Done: `kalman_test.cu`** — only `KT_STANDARD` is ported (`kalman_type 0` in
`temp/H2O_2G/input.nn:79`, the only mode this project's config uses;
`KT_FADINGMEMORY` exists in the CPU code but isn't exercised here, same
"port what's actually used" choice already made for symmetry functions).
The algorithm, replicated kernel-by-kernel from `KalmanFilter.cpp:126-197`:
`X = P.H` → `A = HᵀX + R` (`R = I/eta`) → `K = X.A⁻¹` → `P -= K.Xᵀ`, `P += Q`
(`Q = I*q`) → `w += K.xi`, with the `eta`/`q` exponential schedules applied
in the exact same order as the real code (`eta` grows *before* being used to
build `R` that step; `q` decays *after* being used to build `Q` that step —
getting this backwards would silently use the wrong step's value).

- **State size**: `update_strategy 0` (`US_COMBINED`) in `input.nn:42` means
  *one* filter covers both elements' weights combined — `N = 3327` for
  H2O_2G (1576 H-connections + 1751 O-connections), `P` dense `3327×3327`
  (~89 MB, not block-diagonal — H/O cross-covariance is tracked). `m`
  (observation count) equals the MPI rank count doing the gather — 32 in
  this project's job scripts. Both a small smoke-test size (`N=50, m=5`)
  and this real production size (`N=3327, m=32`) are exercised.
- **Three independent implementations validated against each other**: the
  CUDA kernels; a from-scratch nested-loop CPU reference with its own
  hand-rolled Gauss-Jordan inversion (catches CUDA indexing bugs); and the
  **real** `nnp::KalmanFilter` class itself, linked from `lib/libnnptrain.a`
  and run in lockstep on identical `H`/`xi` sequences. Its `P`/`K` are
  private with no getters, so only its externally-visible weight vector `w`
  is checked — but since every step's `w` update depends on the full `P`/`K`
  chain, a bug anywhere in the recursion would surface in `w` within a step
  or two, making this a strong ground-truth check despite the limited
  visibility.
- **The `m×m` inverse uses a single-thread (`<<<1,1>>>`) Gauss-Jordan
  kernel**, deliberately not parallel — `m` is small (32 in production) and
  this step is about proving the recursion's numerics correct, not
  performance. The augmented matrix is a caller-sized device buffer, not a
  fixed-size local array — the earlier `e2e/` buffer-overflow bug made that
  mistake once already; not repeating it here even though `m` is small
  enough that a guessed size would probably have been fine in practice.
  `updateP` (`O(N²·m)` per call) is the dominant cost and the real target
  for a later cuBLAS `dsyrk`/`dgemm`-based rewrite.
- **Linking `lib/libnnptrain.a` hit a new variant of the GSL debug-section
  problem** noted elsewhere in this README: `KalmanFilter.o`'s own
  `.debug_info` section is compressed in a way this cluster's `ld` (binutils
  2.30) can't decompress ("unable to initialize decompress status for
  section .debug_info"), which makes `ld` reject the *entire* archive
  ("File format not recognized") rather than just that one section. Fix:
  extract just the four objects actually needed (`KalmanFilter.o`/
  `Updater.o` from `libnnptrain.a`, `Stopwatch.o`/`utility.o` from
  `libnnp.a` — confirmed via `nm -u` to have zero undefined GSL/MPI runtime
  symbols) and `strip --strip-debug` them before linking, instead of linking
  the full archives.
- Validated: both sizes PASS, `max|P_gpu-P_cpu| = 2.8E-14`,
  `max|w_gpu-w_cpu| = 9.5E-18`, `max|w_gpu-w_real| = 3.3E-17` (production
  size, 5 sequential updates).

Wiring this into `e2e/` with a real per-structure Jacobian (not synthetic
`H`/`xi`) is now done — see `e2e/`'s Stage 5 (energy) and Stage 6 (force,
`calculateDFdc`'s output instead of `calculateDEdc`'s) below.

All of `gemm/`'s cuBLAS batching (NN forward/`dEdG`, `calculateDFdc`/
`calculateDEdc`, and this filter's own `updateP`) is now done — see `gemm/`
below.

Phase 6 (build system integration, `N2P2_GPU` flag, first real call site
into `nnp-predict`/`nnp-train`) is now done for the NN forward+`dEdG`
piece — see `src/libnnpgpu/` below.

## `gemm/`

Performance work: replaces Phase 3's one-thread-per-atom NN forward+`dEdG`
kernel (`nn/nn_forward_test.cu`) with cuBLAS GEMMs batched across every atom
of one element at once — the "many tiny per-atom MLPs → a handful of
per-element batched evaluations" opportunity `GPU_PORTING_PLAN.md`'s Phase 3
section names, not yet exploited by any prior step (every kernel so far has
been one CUDA thread doing one atom's tiny sequential MLP, correctness-first
by design).

**Done: `nn_forward_gemm_test.cu`.** Since every atom of one element shares
the same weights, this isn't "batched GEMM" in the cuBLAS
`gemmStridedBatched` sense (many independent small matrix pairs) — it's a
single ordinary GEMM per layer per element, with only the data (`G`)
operand varying row-by-row:

- Forward: `G (numAtoms×numIn) . W1 (numIn×25) → H1pre`, bias+`tanh`
  (elementwise) → `H1 . W2 (25×25) → H2pre`, bias+`tanh` → `H2 . W3 (25×1)`
  + bias → energy.
- `dEdG` (the per-input forward-sensitivity sweep) re-expressed as two more
  GEMMs plus cheap elementwise ops: `v2 = dfdx2 .* W3` (broadcast) →
  `v1 = v2 . W2ᵀ` → `v1s = dfdx1 .* v1` → `dEdG = v1s . W1ᵀ`. `W1ᵀ`/`W2ᵀ`
  are precomputed once on the host (tiny matrices, ≤42×25) so every cuBLAS
  call stays a plain `CUBLAS_OP_N` via the standard row-major-via-
  column-major trick (`gemmRowMajor()`), rather than mixing that trick with
  cuBLAS transpose flags — a well-known source of sign/axis mistakes best
  avoided by construction rather than by careful bookkeeping.
- Validated three ways: the GEMM path, the already-validated one-thread-
  per-atom kernel (copied verbatim from `nn/nn_forward_test.cu`), and the
  real `nnp::NeuralNetwork` class — all agreeing to ~6e-15 on energy and
  `dEdG`, for both elements.
- **Timed, not just validated** (`cudaEvent`-based, 500 repeated launches
  for a steady-state average): **25.6× for H (420 atoms), 27.9× for O (210
  atoms)** — a real measured win even at this project's modest real batch
  sizes, better than expected going in (small batches can leave GEMM/cuBLAS
  call overhead dominant; it didn't here).

**Done: `nn_dedc_gemm_test.cu`** — `calculateDEdc` (the energy-weight
Jacobian, `../e2e/`'s Stage 5), batched with genuine
`cublasDgemmStridedBatched` calls, the first real use of "batched" GEMM in
the cuBLAS sense in this port (the forward pass above used ordinary,
non-batched `dgemm` throughout, since every atom shared the same weight
matrix). Turns out almost the entire computation is either already
produced by the forward/`dEdG` pipeline above, or reduces to two per-atom
**rank-1 outer products**:

- `dE/dW3 = H2` and `dE/db2 = v2`, `dE/db1 = v1s` are direct aliases of
  arrays the forward/`dEdG` GEMM pipeline already computed — no new work.
  `dE/db3 = 1` is a constant.
- `dE/dW2[atom] = outer(h1[atom,:], v2[atom,:])` and `dE/dW1[atom] =
  outer(G[atom,:], v1s[atom,:])` are batched rank-1 (`k=1`)
  `cublasDgemmStridedBatched` calls (`outerProductBatched()`), writing
  directly into the right offset of each atom's packed `[W1,b1,W2,b2,W3,b3]`
  block (matching `AtomBatch::dFdcIndex`'s layout, so this could feed
  `../e2e/`'s Stage 5 directly) via the stride parameter, no separate
  packing/copy for those two blocks.
- Validated three ways (GEMM path, the existing per-atom kernel copied from
  `../e2e/`'s Stage 5, the real `NeuralNetwork::calculateDEdc()`) to ~3e-15.
  **8.5× (H) / 8.8× (O)** speedup.

**Done: `nn_dfdc_gemm_test.cu`** — the harder of the two Jacobian passes:
`calculateDFdc` (the force-weight Jacobian, `../nn/
nn_backward_dfdc_test.cu`) loops over every input `k0`, and each iteration
touches every connection, so this keeps a host-side loop over `k0` (≤42
iterations, cheap) and batches the atom dimension inside each iteration via
a mix of ordinary shared-weight GEMMs (`u . W2`, `jacBiasHidden2 . W2ᵀ` —
same trick as the forward pass, since `W1`/`W2`/`W3` don't depend on the
atom) and batched rank-1 outer products (the same `cublasDgemmStridedBatched`
tool `nn_dedc_gemm_test.cu` introduced, now with explicit `alpha`/`beta` so
contributions **accumulate** across all `k0` iterations into `dFdc_W2`/
`dFdc_W1` rather than overwriting). The trickiest piece ported faithfully:
`calculateDFdc`'s `deltaTerm` (the real algorithm's `dE/db_hidden1`
contribution to `dFdc`'s `W1` block, which lands *only* at the row matching
the input actually being differentiated) becomes a small elementwise kernel
touching just that one row per `k0`, not a GEMM. Validated three ways (GEMM
path, the existing per-atom kernel, the real
`NeuralNetwork::calculateDFdc()`) to ~2-3e-14. **11.3× (H) / 10.4× (O)**
speedup — despite roughly 14 kernel/GEMM launches per `k0` iteration (≤42
of them), launch overhead didn't erase the win.

**Done: `kalman_gemm_test.cu`** — the two dominant `O(N²·m)` steps of
`KalmanFilter::update()` (`X = P.H`, and the `K.Xᵀ` term of `P -= K.Xᵀ`),
`~354M` multiply-adds each at the real `N=3327, m=32` production size; every
other step (`A = Hᵀ.X`, `K = X.Ainv`, the `m×m` inverse, `w += K.xi`) is
`O(N·m²)` or smaller (`<1%` of the total flops) and is deliberately left as
`kalman/`'s existing naive kernel. Unlike every other file in this
directory, `P`/`H`/`X`/`K` here are **column-major**, not row-major —
`kalman/kalman_test.cu` deliberately stores them that way to match
`Eigen::Map<MatrixXd>`'s layout, so the same buffers can be handed to the
real `KalmanFilter` class unchanged. That means cuBLAS's native
column-major form applies directly here: `X = P.H` and `K = X.Ainv` are
plain `cublasDgemm(OP_N, OP_N)` calls, and `P -= K.Xᵀ` just needs
`CUBLAS_OP_T` on `X` — no row-major-via-column-major trick, no materialized
transpose needed.

- **A first draft got this wrong**: it reused this directory's row-major
  `gemmRowMajor()` helper out of habit, without checking that Kalman's data
  layout is the opposite convention — it silently computed the wrong matrix
  product. Caught immediately (not subtly) by the three-way validation:
  `P`/`w` diverging from both the naive kernel and the two independent CPU
  references, worsening with every update. Fixed by using cuBLAS's native
  column-major calls directly instead of forcing a trick built for a
  different file's layout. Worth remembering: a helper correct in one file
  isn't automatically correct in another with a different convention, even
  within the same directory.
- Validated three ways (GEMM path vs. the existing naive-kernel path vs. two
  independent CPU references — an own-loop implementation and the real
  `nnp::KalmanFilter` class) at both sizes, to ~1e-15.
- **Modest speedup, reported honestly**: **1.22×** (small) / **1.39×**
  (production size) — much smaller than the NN pieces above. The naive
  kernel here was already a reasonably efficient memory-access pattern
  (each thread's dot product reads contiguous rows), unlike the NN forward
  kernel's redundant serial per-atom work; cuBLAS still wins, just not
  dramatically.

Not pursued: `cublasDsymm` to exploit `P`'s symmetry for `X = P.H` (would
roughly halve that GEMM's cost) — judged not worth the added
side/uplo-semantics risk for a further ~2× on top of an already-modest win.

All three of Phase 4's GPU-side cost centers this port has profiled or
ported so far are now batched with cuBLAS in some form.

## `src/libnnpgpu/` (Phase 6: build system integration)

Everything above lives entirely under `gpu/` — standalone test harnesses,
each with its own duplicated copy of the math and its own `main()`, never
linked into the real `nnp-train`/`nnp-predict` binaries. Phase 6
(`GPU_PORTING_PLAN.md`) is the first step that actually changes the shared,
upstream-tracked `src/` tree: a real, minimal-footprint call site, scoped
deliberately narrow (see the three options this was scoped from — build
skeleton only / one real call site / full pipeline — "one real call site"
was chosen as the smallest slice that proves the whole pattern end-to-end,
matching `GPU_PORTING_PLAN.md` §7's own advice).

**What's wired in**: `Mode::calculateAtomicNeuralNetworks()`'s `HDNNP_2G`
branch (`src/libnnp/Mode.cpp:1672`) — the NN forward+`dEdG` pass, batched
via cuBLAS exactly as `gpu/gemm/nn_forward_gemm_test.cu` already validated
(25.6×/27.9× there). Deliberately **not** wired in this pass: symmetry
function computation (stays on CPU — `Atom::G` is already populated by the
time this function runs, so there was nothing to gain by touching it here)
and forces/Kalman (out of this pass's chosen scope).

- **`src/libnnpgpu/`**: a new, separate library (`GpuNeuralNetwork.h/.cu`),
  built only when `GPU=1` is passed to `make` (otherwise a clean no-op —
  see its own `makefile`). Its one function, `gpuNnForwardDEdG()`, is
  `nn_forward_gemm_test.cu`'s forward+`dEdG` pipeline generalized from that
  file's hardcoded 25/25 hidden-layer sizes to **run-time** sizes — because
  architecture (hidden layer count/sizes/activations) is a property n2p2
  reads from `input.nn`'s `global_hidden_layers_short`/`global_nodes_short`/
  `global_activation_short` keywords (with optional per-element overrides),
  not a fixed constant; several of this repo's other bundled example
  datasets use different sizes (e.g. `examples/nnp-predict/Anisole_SCAN`
  uses 30/20, asymmetric). A hardcoded-25/25 GPU path would have been a
  real, silent correctness bug for any dataset that didn't happen to match
  H2O_2G's numbers.
- **`NeuralNetwork::hasGpuCompatibleArchitecture()`** (new, purely additive
  public method, `src/libnnp/NeuralNetwork.h/.cpp`): checks a network has
  exactly two `AF_TANH` hidden layers, a single-neuron `AF_IDENTITY` output
  layer, neuron normalization disabled, and hidden sizes within a bound —
  the shape `src/libnnpgpu` supports. `Mode::calculateAtomicNeuralNetworks()`
  checks this for every element before taking the GPU path and falls back
  to the exact original per-atom CPU loop, atom by atom, if any element's
  network doesn't match (a real run-time possibility, not a formality —
  proven by `gpu/gemm/libnnpgpu_test.cu`, which deliberately exercises a
  30/20 architecture alongside H2O_2G's 25/25 to confirm the generalization
  is genuinely correct, not just re-lucky on one dataset's numbers, ~5e-15
  agreement in both cases).
- **Build system**: a new `GPU` make variable (default off, mirroring how
  `COMP`/`MODE` already work), threaded from the master `src/makefile`
  through `src/libnnp/makefile`/`src/libnnpgpu/makefile`/`src/application/
  makefile`. `GPU=1` adds `-DN2P2_GPU` and an extra include path where
  needed, and links `libnnpgpu.a` plus `-lcudart -lcublas` into any
  `APP_CORE` binary (`nnp-predict` and friends — the mechanism applies to
  all of them since they share `Mode::calculateAtomicNeuralNetworks()`,
  though only `nnp-predict` has actually been run and validated with
  `GPU=1` so far). A plain `make` (no `GPU=1`) is completely unaffected —
  confirmed by rebuilding `nnp-train` afterward with the same makefiles
  and no changes to its build. **g++ links the nvcc-compiled
  `libnnpgpu.a` into the final binary with no special handling** — no need
  to swap the final linker to `nvcc`, unlike every `gpu/*/run.slurm` script
  so far (those always used `nvcc` as the final linker; this is the first
  place that didn't need to).
- **Validated two ways**: `gpu/e2e_predict_check/mode_gpu_call_site_test.cpp`
  calls the actual public `Mode::calculateAtomicNeuralNetworks()` directly
  (its class-level doc comment literally shows this exact call as intended
  usage) on a real `Prediction` object built from real H2O_2G
  `input.nn`/trained weights, with synthetic `Atom::G` (valid since this
  function only reads `Atom::G` and writes `Atom::energy`/`dEdG` — confirmed
  by reading its source) — `max|E_cpu-E_gpu| = 3.9E-14`,
  `max|dEdG_cpu-dEdG_gpu| = 2.3E-13`. **And, the real end-to-end
  deliverable**: `gpu/e2e_predict_check/run_nnp_predict_gpu_vs_cpu.slurm`
  builds and runs the actual `nnp-predict` binary itself, twice (once
  CPU-only, once `GPU=1`), on the real H2O_2G structure, and diffs
  `energy.out`/`nnatoms.out`/`nnforces.out` — `max abs diff` ~1e-16/1e-17
  across all 630 atoms' energies and forces. Both agree with each other and
  with the sensible physical result (`Ediff` ~4e-5 eV/atom against the
  reference for this trained model).
- **A red herring along the way, now resolved**: `nnp-predict` initially
  segfaulted reliably on this branch (confirmed via a core dump and `gdb`
  backtrace, inside `SymGrpExpRad::calculate()`/`SymGrpExpAngn::calculate()`
  — code this Phase 6 pass never touched, called well before
  `calculateAtomicNeuralNetworks()`). Root-caused by systematically bisecting
  optimization flags and rebuild state (a `-O0` build never crashed; ASan/
  UBSan at the exact production flags never crashed either; the crash
  location shifted between separate crashing runs; and, critically, the
  *identical* `-O3 -march=native` build stopped crashing entirely, 0/20
  repeated runs, once every object file was rebuilt from a `make clean`
  state on the actual execution node). Conclusion: it was a **stale,
  non-portable prebuilt `lib/libnnp.a`** left over from before this
  investigation, compiled with `-march=native` on a different machine than
  where it was later executed — not a source bug, and not anything this
  port introduced. A full `make clean && make` on the actual target node
  resolved it completely; `mode_gpu_call_site_test.cpp` is kept anyway since
  it's a useful, lighter-weight regression check of just this call site.
- **A real gotcha this Phase 6 pass *did* introduce**, discovered while
  restoring a clean state afterward: `lib/libnnp.a` is one shared archive
  across every application, and whether `Mode.o` inside it references
  `gpuNnForwardDEdG()` is baked in by whichever `GPU=` value `libnnp` was
  *last* built with — not tracked per-application. Building `GPU=1
  nnp-predict` and then plain `nnp-train` (no `GPU=1`) without rebuilding
  `libnnp` first fails at `nnp-train`'s link step with an undefined
  reference (`APP_TRAINING`/`APP_DATASET`'s link lines aren't GPU-aware).
  Documented directly in `src/application/makefile` next to the `GPU=1`
  block; the fix when switching `GPU=` values is to rebuild `libnnp` first
  (`cd src/libnnp && make clean && make ... GPU=<value>`).

### Follow-up: `nnp-train`'s energy-Jacobian call site

The natural second call site, chosen because it's the one place in
`nnp-train` where the already-measured 8.5×/8.8× `calculateDEdc` speedup
(`gemm/nn_dedc_gemm_test.cu`, above) would actually show up as real
wall-clock time, unlike `nnp-predict`'s single-shot usage pattern.
**What's wired in**: `Training::update()`'s `"energy"` Jacobian branch
(`src/libnnptrain/Training.cpp`, the `if (k == "energy")` block), `HDNNP_2G`
only — this loop runs once per selected update candidate (i.e. potentially
many times per epoch, unlike `nnp-predict`'s one-shot call), accumulating
each atom's `calculateDEdc()` output straight into the Kalman filter's
Jacobian row for that structure's total energy.

- **New library function, `gpuNnEnergyDEdcSum()`** (`src/libnnpgpu/
  GpuNeuralNetwork.h/.cu`): batches the forward pass + `calculateDEdc()` for
  every atom of one element, but returns the **sum over atoms**, not a
  per-atom array — the only thing `Training::update()` ever does with
  `calculateDEdc()`'s output is add it into one shared Jacobian row, so
  there is no reason to materialize (or transfer off the GPU) a per-atom
  result at all. This makes the atom-axis reduction cheaper here than in
  `nn_dedc_gemm_test.cu`'s original per-atom design: instead of batched
  rank-1 (`k=1`) `cublasDgemmStridedBatched` outer products followed by a
  separate sum, the atom axis is directly the **contracted** dimension of
  two GEMMs (`dE/dW1_sum = Gᵀ.v1s`, `dE/dW2_sum = H1ᵀ.v2`), computed via a
  new `gemmRowMajorATransB()` helper (row-major-via-column-major trick,
  `CUBLAS_OP_T` on the second cuBLAS operand instead of a materialized host
  transpose). The four bias/output-layer sums (`dE/db1`, `dE/db2`, `dE/dW3`)
  reuse the same helper with a device "ones" vector as the contracted
  operand (i.e. the reduction is *also* just a GEMM, not a separate kernel);
  `dE/db3`'s sum is the constant `numAtoms`, set directly on the host.
- **Validated standalone**: `gpu/gemm/libnnpgpu_dedc_test.cu` (`run_libnnpgpu_
  dedc.slurm`) checks `gpuNnEnergyDEdcSum()`'s energy and summed-`dEdc`
  output against the real `NeuralNetwork::calculateDEdc()` summed by hand
  over every atom, for both H2O_2G's real architecture (25/25 hidden,
  H and O) and, again, the deliberately-different Anisole_SCAN-like 30/20
  case — `max|E_gpu-E_cpu| ≤ 4.9E-15`, `max|dEdcSum_gpu-dEdcSum_cpu| ≤
  2.4E-13` across all three, ALL PASS.
- **Dispatch in `Training.cpp`**: mirrors `Mode.cpp`'s pattern exactly —
  checks `hasGpuCompatibleArchitecture()` for every element, and only when
  all pass, gathers `G` per element across every atom of the current
  structure, calls `gpuNnEnergyDEdcSum()` once per element, scatters
  `it->energy` back, and adds the returned sum directly into
  `pu.jacobian`. Falls back to the untouched original per-atom
  `calculateDEdc()` loop (now wrapped in `if (!doneOnGpu)`) for `HDNNP_4G`
  (extra charge input neuron, electrostatics coupling — out of scope here)
  or any GPU-incompatible architecture. `HDNNP_4G`/`HDNNP_Q`'s other
  branches (`"force"`, `"charge"`) are untouched.
- **A second, genuinely new build-system gotcha found and fixed** (distinct
  from the `libnnp.a`-GPU-state-stickiness one above, which was only
  *documented*, not fixed, until now): the master `src/makefile`'s
  `$(APP_LIBNNPTRAIN)` rule (covers `nnp-train`, `nnp-checkdw`, and all of
  `APP_DATASET`) depended only on `libnnp`/`libnnptrain` and never forwarded
  `GPU=$(GPU)` or depended on `libnnpgpu` — unlike `$(APP_LIBNNP)`, which
  already had both. Building `make GPU=1 nnp-train` from a clean tree
  therefore compiled and linked `nnp-train.o` with `-DN2P2_GPU` (that part
  *is* inherited automatically — GNU Make re-exports command-line variables
  to sub-`make`s) but never actually built `lib/libnnpgpu.a` first, so the
  link failed: `cannot find ../../lib/libnnpgpu.a`. Fixed by adding the same
  conditional `libnnpgpu` prerequisite (and explicit `GPU=$(GPU)` forwarding)
  to `$(APP_LIBNNPTRAIN)` that `$(APP_LIBNNP)` already had. Also closed the
  previously-just-documented half of this gap: `src/application/makefile`'s
  `$(APP_TRAINING)`/`$(APP_DATASET)` link rules now add `-DN2P2_GPU`, the
  `libnnpgpu` include path, and `libnnpgpu.a`/`$(PROJECT_LDFLAGS_GPU)` when
  `GPU=1`, exactly like `$(APP_CORE)` already did.
- **The real end-to-end deliverable**: `gpu/e2e_train_check/run_nnp_train_
  gpu_vs_cpu.slurm` builds the actual `nnp-train` binary twice (CPU-only,
  then `GPU=1`, each via a full clean rebuild) and runs **one real training
  epoch** (32 MPI ranks, the real, full H2O_2G production dataset and
  `input.nn`, fixed `random_seed` so both runs start from identical
  randomized initial weights and an identical train/test split) end to end
  through the Kalman-filter updater. Result: `learning-curve.out` and the
  post-epoch-1 `weights.001.000001.out`/`weights.008.000001.out` (H's 1576
  and O's 1751 connections) match **exactly** — `0.000E+00` max abs
  difference, bit-identical to all 17 printed significant digits — while
  clearly differing from the epoch-0 (pre-update) weights, confirming a real
  Kalman update happened and both code paths agree on it precisely, not just
  approximately.
- **Honest performance note**: this call site touches only the energy
  Jacobian, which `temp/H2O_2G/timing.out` shows is a small fraction of
  per-epoch time (`Etrain` ~1.8s vs `Ftrain` ~150s per epoch in that log) —
  the force-Jacobian update (`calculateDFdc`, still CPU-only) dominates.
  Wall-clock speedup for `nnp-train` as a whole is expected to be small
  until forces/Kalman are also ported (see next steps); this pass's value
  is the same as `nnp-predict`'s: a second real, validated call site
  proving the pattern generalizes, not a training-speed claim.

### Follow-up: `nnp-train`'s force-Jacobian call site

The harder of the two Jacobian passes, and the one that actually matters
for `nnp-train`'s wall-clock time: `timing.out` shows `Ftrain` (the
force-Jacobian update) dominates the epoch (~150s vs ~1.8s for `Etrain`
above). **What's wired in**: `Training::update()`'s `"force"` Jacobian
branch (`HDNNP_2G` only, same scope restriction as the energy branch) —
for one fixed `(atom, coordinate)` update candidate, every atom in the
structure contributes `calculateDFdc()`'s per-connection output (using
*its own* `dGdxia` from `collectDGdxia()`) to the same Jacobian row.

- **New library function, `gpuNnForceDFdcSum()`** (`src/libnnpgpu/
  GpuNeuralNetwork.h/.cu`): ports `gemm/nn_dfdc_gemm_test.cu`'s algorithm —
  a host-side loop over each of the `numIn` inputs (cheap, at most a few
  dozen iterations), each touching every connection — but, exactly like
  `gpuNnEnergyDEdcSum()`, only ever needs the atom-axis **sum**, not a
  per-atom array. That turns every one of the standalone benchmark's
  batched RANK-1 `cublasDgemmStridedBatched` outer products into a single
  ordinary `cublasDgemm` contracting the atom dimension directly (e.g.
  `dFdc_W2`'s per-atom `outer(P[atom,:],jacBH2[atom,:])` summed over atoms
  is just `Pᵀ.jacBH2`, a plain GEMM) — simpler *and* cheaper than the
  per-atom-preserving version, not just cheaper to transfer off the device.
  The one new wrinkle: `dGdxyz`'s per-input "column" needs to be a
  *contiguous* device array for this trick to apply (the atom-summed GEMMs
  contract over the atom axis, which requires that axis's operand
  contiguous) — but `dGdxyz` arrives atom-major, so a one-time
  `transposeKernel()` produces `dGdxyzᵀ` before the input loop, making each
  iteration's column access contiguous. Also computes energy and `dEdG`
  in the same pass (reusing the same `v1s`/`W1ᵀ` intermediates
  `gpuNnForwardDEdG()` uses), since `Mode::calculateForces()` still needs
  correct `it->energy`/`it->dEdG` afterward regardless of the Jacobian path.
- **Validated standalone, first try**: `gpu/gemm/libnnpgpu_dfdc_test.cu`
  (`run_libnnpgpu_dfdc.slurm`) checks energy, `dEdG`, and the atom-summed
  `dFdc` against the real `NeuralNetwork` class (`calculateDFdc()` summed by
  hand over every atom), for H2O_2G's real 25/25 architecture (H and O) and
  the Anisole_SCAN-like 30/20 case — `max|E| ≤ 4.9E-15`, `max|dEdG| ≤
  6.4E-15`, `max|dFdcSum| ≤ 1.4E-13` across all three, ALL PASS with no
  fixes needed (the derivation above was worked out on paper before coding,
  gemm-call-by-gemm-call, against `nn_dfdc_gemm_test.cu`'s reference
  algorithm).
- **Dispatch in `Training.cpp`**: same pattern as the energy branch —
  `collectDGdxia()` writes into a single shared scratch member
  (overwritten every call), so every atom's result is first collected into
  a per-atom `vector` before any batching can happen; then, per element,
  gathers `G` and `dGdxyz` across the structure's atoms, calls
  `gpuNnForceDFdcSum()` once, scatters `energy`/`dEdG` back, and adds the
  returned sum into `pu.jacobian`. Falls back to the untouched original
  per-atom loop (wrapped in `if (!doneOnGpu)`) for `HDNNP_4G`,
  `N2P2_FULL_SFD_MEMORY` builds, or any GPU-incompatible architecture.
- **The real end-to-end deliverable**: rerunning the same
  `gpu/e2e_train_check/run_nnp_train_gpu_vs_cpu.slurm` used for the energy
  branch (unchanged) now exercises *both* GPU Jacobian paths together,
  since forces are trained by default in `H2O_2G`'s `input.nn`. Result:
  identical to the energy-only run — `learning-curve.out` and both
  elements' post-epoch-1 weights (3327 connections total) match **exactly**,
  `0.000E+00` max abs difference, bit-identical to all 17 printed
  significant digits, while the logged force RMSE columns confirm a real,
  substantial force-driven update happened (not a degenerate/no-op
  comparison).

### Follow-up: `KalmanFilter::update()` — the third and last call site

The last of Phase 4's three profiled GPU cost centers to reach a real
binary. Unlike the two Jacobian branches above (called from
`Training.cpp`, stateless per call), `KalmanFilter::update()` needed its
own design: `P` is a persistent `~89 MB` (`N=3327`) matrix carried across
*every* weight update for the life of a training run, so the natural
"upload state, compute, download result" pattern used everywhere else in
this port doesn't apply here without erasing the entire win.

**What's wired in**: only the two genuinely dominant `O(N²·m)` steps —
`X = P.H` and the `K.Xᵀ` term of the covariance downdate `P -= K.Xᵀ` — via
two new library functions, `gpuKalmanComputeX()`/`gpuKalmanUpdateP()`
(`src/libnnpgpu/GpuKalmanFilter.h/.cu`), `KT_STANDARD` only (the only mode
`input.nn` actually configures, same scoping `gpu/kalman/kalman_test.cu`
already chose). `P` lives on the GPU for the lifetime of one
`KalmanFilter` object (created lazily on first use); only `H` (uploaded),
`X` (downloaded once), and `K` (uploaded once) cross the PCIe bus per
call — all small relative to `P`. **Deliberately not ported**: `A = HᵀX +
R`, the `m×m` inverse, `K = X.Ainv`, and `w += K.ξ` all stay on the host,
using the exact same Eigen expressions `KalmanFilter::update()` always
used — this is `O(N·m²)`, under 1% of the flops, and (see below) porting
it turned out to be a real correctness trap, not just an unnecessary one.

**Two real bugs found and fixed here, both only visible on real
production data** — the most important lesson from this whole call site:

1. **Hand-rolled GPU matrix inverse vs. Eigen's LU inverse.** The first
   version of this integration also ported the `m×m` inverse to a
   hand-rolled GPU Gauss-Jordan kernel (mirroring `gpu/kalman/
   kalman_test.cu`'s already-validated standalone benchmark). A standalone
   test with synthetic random `H`/`ξ` data passed cleanly. But a real
   end-to-end `nnp-train` run (one full epoch, the real `H2O_2G` dataset,
   32 MPI ranks) diverged by **~10 orders of magnitude** after that one
   epoch. Root cause: Gauss-Jordan with partial pivoting and Eigen's
   `PartialPivLU`-based `.inverse()` are *different algorithms* — both
   correct for a well-conditioned matrix, but not numerically
   interchangeable for a near-singular one, and real, correlated
   production Jacobian data can produce a far more ill-conditioned `A`
   than uncorrelated random test data ever would by construction. Fixed
   by removing the GPU inverse entirely and moving `A`/inverse/`K`/`w`
   back to the host with Eigen, as described above — this also simplified
   the library (no Gauss-Jordan kernel, no `A`/`Ainv` device buffers at
   all).
2. **`P`'s implicit symmetry.** The real code computes
   `X = P.selfadjointView<Lower>() * H` — Eigen deliberately reads *only*
   `P`'s lower triangle and mirrors it, rather than trusting the full
   stored matrix. This matters because the *other* update,
   `P.noalias() -= K * X.transpose()`, is a plain dense operation with no
   re-symmetrization, so `P` can and does accumulate genuine
   floating-point asymmetry over many `update()` calls (the class's own
   commented-out diagnostic, `"Max. deviation of symmetric form of P"`,
   is a hint the original authors were aware of this). The CPU path
   "self-heals" every single call by only ever reading the lower
   triangle; a first fix of bug 1 left `gpuKalmanComputeX()`'s GEMM
   reading the *full* stored `P` (both triangles) via a plain
   `cublasDgemm`, which doesn't self-heal and instead reads back whatever
   asymmetry has built up. This fix alone took the real end-to-end
   divergence from ~10 orders of magnitude down to ~4-5 orders — much
   better, but still a real, visible divergence, not noise. Fixed by
   switching to `cublasDsymm` (`CUBLAS_SIDE_LEFT`/`CUBLAS_FILL_MODE_LOWER`)
   — the direct GPU equivalent of `selfadjointView<Lower>()`.
   - Neither bug was, or could have been, caught by the standalone
     synthetic-data test alone: bug 1 needs a real ill-conditioned matrix
     (random test data is never ill-conditioned by construction) and bug 2
     needs *many* real update() calls compounding real asymmetry (the
     standalone test's own reference path doesn't use `selfadjointView`
     either, so it can't see this class of bug regardless of iteration
     count). Both were only caught by a real, full end-to-end `nnp-train`
     run — a second, independent confirmation (after the earlier
     stale-`libnnp.a` crash) that this project's "validate the real
     binary, not just the synthetic microbenchmark" discipline matters.
- **Validated standalone** (`gpu/gemm/libnnpgpu_kalman_test.cu`,
  `run_libnnpgpu_kalman.slurm`): drives the actual persistent-state API
  across many iterations (unlike `gemm/kalman_gemm_test.cu`, which called
  the GEMM pipeline directly with everything freshly allocated each time),
  cross-checked against both an independent from-scratch CPU reference and
  the real `nnp::KalmanFilter` class. Four cases: small, production-sized
  (`N=3327, m=32`, 20 updates), the *real* `H2O_2G` production schedule
  (real `kalman_epsilon`/`q0`/`qtau`/`qmin`/`eta`/`etatau`/`etamax` values
  from `input.nn`, normalized by the real ~310 updates/epoch this dataset
  produces — see `temp/H2O_2G/updater.000.out`), and a varying-`m` case
  exercising the growable scratch-buffer logic. All four: `max|P_gpu-P_cpu|
  ≤ 1.1E-13`, `max|w_gpu-w_real| ≤ 3.4E-14`, PASS.
- **The real end-to-end deliverable**: `gpu/e2e_train_check/run_nnp_train_
  gpu_vs_cpu.slurm` (unchanged from the Jacobian rounds) now exercises all
  three GPU call sites together. Result, after both fixes above:
  `learning-curve.out`'s energy/force RMSE values agree to **4-5
  significant figures** between CPU and GPU (e.g. `6.98988453E-06` vs
  `6.98888908E-06`), and the post-epoch-1 weights (3327 connections total)
  differ by at most `~1.5E-3` absolute — small, physically reasonable
  numbers, not a divergence, but *not* bit-identical either, unlike the two
  Jacobian call sites. That distinction is expected and not a red flag:
  the Jacobian branches are single-pass, embarrassingly-parallel
  reductions (bit-identical results whenever the same numbers are summed
  in the same order); the Kalman recursion is an iterative feedback loop
  where `P` from one call feeds the next, so small floating-point
  differences from a different GEMM implementation's summation order
  compound over the ~310 real updates in a way that's intrinsic to the
  algorithm, not a sign of a remaining bug.
- **Honest performance finding — a real win that doesn't show up net,
  and why** (resolved two follow-ups down, via NVIDIA MPS — read on):
  the `update()` step itself measurably sped up — `timing.out`'s
  `F_upd` column (the force-branch Kalman update time, which dominates
  since force candidates vastly outnumber energy candidates) dropped from
  `12.14s` (CPU) to `0.39s` (GPU), a genuine `~31×`. But the *energy*-branch
  update column, `E_upd`, got **slower** (`1.58s → 5.37s`), and — more
  importantly — `F_com` (MPI communication time for the force branch)
  ballooned from `2.96s` to `50.45s`, and the **overall epoch got slower**
  (`156.0s → 211.2s`, about `1.35×` slower), even though the piece of code
  this pass actually touched got dramatically faster in isolation.

### Follow-up: `PM_TRAIN_RK0` was silently running the update on every rank

Investigating the finding above turned up a real, pre-existing bug (not
something this port introduced, but one it made *costly* for the first
time): `temp/H2O_2G/input.nn` sets `parallel_mode 0` (`PM_TRAIN_RK0`),
whose class-level documentation
(`Training.h`'s `ParallelMode` enum) reads *"Weight update is carried out
on rank 0 and new weights are redistributed to all tasks."* But
`Training::update()`'s actual weight-update loop called
`updaters.at(i)->update()` **unconditionally on every rank**, with no
`myRank == 0` guard at all. Under `PM_TRAIN_RK0`, the preceding
`MPI_Gather`/`MPI_Reduce` calls only deliver the fully-assembled
error/Jacobian to rank 0 — every other rank's `update()` call runs on its
own incomplete, stale local data — and the resulting weights are
unconditionally overwritten by the `MPI_Bcast` immediately after, while
`status()`/log output is rank-0-only too. So every non-zero rank's
`update()` call was **pure wasted work, computed and immediately
discarded, every single time**, on both `GradientDescent` and
`KalmanFilter`. Cheap enough on CPU (a small dense linear-algebra call)
not to matter or be noticed; once `KalmanFilter::update()` started
dispatching to the GPU, "cheap and wasted" became "32 concurrent,
wasted CUDA calls against one shared device," which is exactly the kind
of contention the finding above was pointing at.

**Fixed** by guarding the whole `updaters.at(i)->update()` loop (and its
`setError`/`setJacobian`/`setSizeObservation` calls) with
`if (myRank == 0 || parallelMode == PM_TRAIN_ALL)` in `Training.cpp` —
making the code actually implement what `PM_TRAIN_RK0` already claimed.
Verified safe by confirming neither updater's `update()` contains any
MPI collective (so skipping it on non-zero ranks can't deadlock) and that
no other code path reads a non-rank-0 updater's internal state (`P`,
`eta`, `q`, Adam's `m`/`v`) — only rank 0's `status()`/log output is ever
written. Re-running the full CPU-vs-GPU comparison after this fix
reproduced the **exact same** `1.253E-03` learning-curve/weights diff as
before (confirming the eliminated work was truly inconsequential to the
result), and measurably helped the two things it targeted: `F_com`
dropped from `50.45s` to `19.44s` (`2.6×` better, though still `5.4×`
worse than CPU's `3.58s`) and `E_upd` flipped from *slower-than-CPU*
(`5.37s`) to faster (`0.05s`, since the few energy-branch calls no longer
pay 32-way redundant overhead).

**But the overall epoch barely moved** (`211.2s → 212.8s` — within
run-to-run noise). Breaking down every `timing.out` column's GPU-minus-CPU
delta for this run ranks the actual contributors by absolute impact:

| column | CPU | GPU | Δ |
|---|---|---|---|
| `F_err` (the force-Jacobian dispatch itself) | `137.4s` | `183.5s` | **`+46.1s`** |
| `F_com` | `3.58s` | `19.44s` | `+15.9s` |
| `E_err` (energy-Jacobian dispatch) | `0.24s` | `5.41s` | `+5.2s` |
| `F_upd` (this fix's target) | `12.31s` | `0.39s` | `-11.9s` |
| `E_upd` (this fix's target) | `1.60s` | `0.05s` | `-1.6s` |

`F_err`/`E_err` are where `gpuNnForceDFdcSum()`/`gpuNnEnergyDEdcSum()`
actually run — the two Jacobian call sites validated for *correctness*
earlier in this port, but never benchmarked at full 32-rank concurrency
until now. They're the **dominant** remaining cost, and unlike the Kalman
update, this work is **not redundant** — every rank genuinely needs its
own local structure's Jacobian, so there's no rank-0-and-broadcast trick
available here. The most likely explanation is the same underlying
mechanism as the Kalman finding (32 MPI ranks issuing frequent CUDA calls
against one physical GPU with no MPS configured contend for the device
regardless of whether the work is redundant or not), just without an
easy fix, since the parallel work itself is real and necessary.

**Bottom line at this point**: the `PM_TRAIN_RK0` fix is a genuine,
low-risk correctness improvement (matches documented behavior, verified
safe, measurably helps what it targets) and is kept regardless of the net
epoch-time outcome. But it revealed that this cluster's "32 ranks, 1
shared GPU, no MPS" resource allocation was the more fundamental
bottleneck for this workload's calling pattern (many small, frequent GPU
calls from many concurrent processes).

### Follow-up: NVIDIA MPS resolves the remaining contention — a real net win

`nvidia-smi` on a compute node confirms `nvidia-cuda-mps-control`/
`nvidia-cuda-mps-server` are installed and the GPU (A100-SXM-64GB, compute
capability 8.0) is in `Default` compute mode — MPS works without needing
`EXCLUSIVE_PROCESS` mode on Volta+, and needs no elevated privileges to
start as a regular user. **One real gotcha**: MPS's control channel is a
UNIX domain socket, so `CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY`
must point at a node-local filesystem -- a first attempt pointing them at
the NFS/parallel-filesystem-backed project directory failed silently
(`nvidia-cuda-mps-control -d` exited 1, both log files empty); pointing
them at `/tmp` on the compute node instead worked immediately (confirmed
via `control.log`).

`gpu/e2e_train_check/run_nnp_train_gpu_mps.slurm` builds the real GPU
`nnp-train` binary once and runs the exact same one-epoch, 32-rank,
real-`H2O_2G` job twice — without MPS, then with the control daemon
started first — and compares. Correctness first: `learning-curve.out`
matches **exactly**, `0.000E+00` max abs diff, between the MPS and
no-MPS runs (MPS only changes scheduling, never results, as expected).
Performance:

| column | CPU | GPU, no MPS | GPU + MPS | MPS vs no-MPS | **MPS vs CPU** |
|---|---|---|---|---|---|
| epoch (total) | `158.8s` | `209.7s` | `109.4s` | `1.92×` | **`1.45×`** |
| `Ftrain` | `153.3s` | `200.4s` | `106.0s` | `1.89×` | `1.45×` |
| `F_err` (force-Jacobian dispatch) | `137.4s` | `185.5s` | `104.4s` | `1.78×` | `1.32×` |
| `F_com` (MPI wait) | `3.58s` | `14.50s` | `1.24s` | `11.7×` | `2.88×` |
| `Etrain` | `1.85s` | `5.56s` | `0.22s` | `24.9×` | `8.31×` |
| `E_err` (energy-Jacobian dispatch) | `0.24s` | `5.43s` | `0.17s` | `31.5×` | `1.41×` |
| `F_upd`/`E_upd` (Kalman update) | `12.31s`/`1.60s` | `0.39s`/`0.05s` | `0.34s`/`0.04s` | ~flat | `36.6×`/`43.2×` |

With MPS enabled, **every single column beats the CPU baseline** —
including `F_err`/`E_err`, the two Jacobian dispatches that looked like an
unfixable, fundamental bottleneck in the finding above. That confirms the
diagnosis was right: the underlying GPU work was always fast (as the
standalone, single-process benchmarks throughout this port always
showed); it was specifically 32 concurrent CUDA contexts serializing
against each other with no MPS that made it look slow in the full
MPI job. `nnp-train`'s real epoch time drops from `158.8s` to `109.4s`,
a genuine **`1.45×` end-to-end speedup** — the first time in this whole
port that a full `nnp-train` run, not just an isolated kernel or a single
call site, is honestly faster on the GPU than on the CPU.

This is still a single epoch on a shared, unreserved cluster node, not a
controlled multi-run statistical study — but going from "consistently
~1.3-1.9× slower without MPS" to "consistently faster with it, across
every single timing column" is a large, consistent enough swing not to
be sampling noise. MPS is not yet wired into the routine validation
scripts (`run_nnp_train_gpu_vs_cpu.slurm` doesn't start it); a real
production job would need to start `nvidia-cuda-mps-control -d` (pointed
at a node-local, e.g. `/tmp`, pipe/log directory) before `mpirun`, and
stop it afterward.

Next steps (not yet done): fold MPS startup/teardown into
`run_nnp_train_gpu_vs_cpu.slurm` (and any real production job script) so
it's the default rather than a separate one-off test; run a multi-epoch
benchmark under MPS to confirm the `1.45×` holds up over a full training
run, not just one epoch; extend all three dispatches above from
`HDNNP_2G` to `HDNNP_4G` (needs the extra charge input neuron and
electrostatics coupling handled).

### Follow-up: made the three dispatch functions reuse device state -- didn't help `F_err`, and that's a useful negative result

With `F_err` (the force-Jacobian dispatch) now `95%` of the GPU epoch even
with MPS enabled, the obvious hypothesis was the same one that motivated
`GpuKalmanFilter`'s whole design: all three functions in
`GpuNeuralNetwork.cu` were stateless, `cudaMalloc`-everything/
`cudaFree`-everything-every-call (`gpuNnForceDFdcSum()` alone allocated
and freed roughly three dozen buffers per call), at a calling frequency
(once per update candidate, ~300+ times/epoch) similar to Kalman's.
`cudaMalloc`/`cudaFree` carry real fixed driver overhead independent of
buffer size, so this looked like the same problem with the same fix.

**Implemented**: all three functions (`gpuNnForwardDEdG`,
`gpuNnEnergyDEdcSum`, `gpuNnForceDFdcSum`) now cache a persistent,
per-architecture device state, keyed by `(numIn, numHidden1, numHidden2)`
-- in practice one entry per element (H, O, ...), created lazily on first
use. Weight-dependent buffers are allocated once and only re-uploaded
(`cudaMemcpy`, not `cudaMalloc`) every call, since their VALUES change
every Kalman update even though their SIZE never does; atom-dependent
buffers grow on demand (never shrink), the same `ensureMCapacity()`
pattern `GpuKalmanFilter.cu` already used. The public API is completely
unchanged -- `Mode.cpp`/`Training.cpp` needed no edits at all.

**Validated correct**: `gpu/gemm/libnnpgpu_test.cu`/`libnnpgpu_dedc_test.cu`/
`libnnpgpu_dfdc_test.cu` were all rewritten to call their function
*repeatedly*, with fresh random weights and a varying (growing/shrinking)
atom count every call, instead of once -- a single-shot test can't catch a
stale-state-reuse or capacity-growth bug. All three: `ALL PASS`, `~1e-13`
to `~1e-15` against the real `NeuralNetwork` class, across every
architecture and call in the sequence.

**But the real end-to-end measurement told a different story than
expected**: rerunning the same MPS comparison with this change in place,
`F_err` did not improve -- `104.4s → 108.9s` (within run-to-run noise on a
shared cluster, arguably slightly worse), and the overall epoch was
likewise flat (`109.4s → 113.2s`). **The hypothesis was wrong**: allocator
churn was not the dominant remaining cost after all (it may have
contributed a little, given `F_com` did drop further, `1.24s → 0.44s`,
but that's a small piece of a `~110s` total). The persistent-state design
is still kept -- it's strictly better practice regardless (no more
allocator churn, matches the pattern already validated for Kalman, zero
downside once validated correct) -- but it does not explain `F_err`'s
cost, and this section is left in as an honest record of a fix that
didn't work, not removed to make the narrative cleaner.

**Best remaining guess, not yet confirmed**: `gpuNnForceDFdcSum()`'s
inner loop runs once per input (`numIn`, up to ~40 for this project's
real elements) and issues on the order of 15-20 kernel/cuBLAS launches
per iteration -- roughly 600-800 total GPU launches in a single call, all
serialized on one stream. Kernel/cuBLAS launch dispatch carries its own
fixed host-side latency, separate from allocation; at that call count, an
otherwise-small per-launch overhead could plausibly dominate the wall
time in a way persistent buffers can't fix, since every one of those
launches still has to happen. This is a guess, not a measurement --
confirming it needs actual GPU profiling (`nsys`/`ncu`) to see where the
time in one real call actually goes, which hasn't been done yet.

Next steps (not yet done): profile a single real `gpuNnForceDFdcSum()`
call with `nsys`/`ncu` to find out where the `~185ms`/call is actually
going, rather than guessing a third time; if launch count is confirmed as
the driver, consider restructuring the `k0` loop to batch multiple
inputs' worth of work into fewer, larger launches instead of one
GEMM/kernel set per input.

### Follow-up: profiled a single `gpuNnForceDFdcSum()` call -- launch count in isolation is not the bottleneck; multi-rank GPU sharing is the prime suspect now

Built a minimal, targeted harness (`gpu/gemm/profile_dfdc.cu`,
`gpu/gemm/run_profile_dfdc.slurm`) instead of guessing a third time:
construct H2O_2G's real "H" architecture (35 inputs, 25/25 hidden),
call `gpuNnForceDFdcSum()` with a representative 420-atom input five
times to warm up the persistent per-architecture state (weight buffers,
capacity growth, cuBLAS handle, driver JIT), then bracket exactly ONE
further steady-state call with `cudaProfilerStart()`/`cudaProfilerStop()`
so `nsys profile --capture-range=cudaProfilerApi` and
`ncu --profile-from-start off` capture only that one call, on an
otherwise idle GPU (single process, `--ntasks-per-node=1`, no MPS, no
other rank sharing the device).

**Measured, in isolation:**

| Metric | Value |
| --- | --- |
| Kernel launches in the one call | `817` |
| Total GPU kernel execution time (`nsys cuda_gpu_kern_sum`, all 817 launches summed) | `~3.12ms` |
| Wall-clock span, first kernel start to last kernel end (queried directly from the `.sqlite` export's `CUPTI_ACTIVITY_KIND_KERNEL` table) | `~4.59ms` |
| Host-side `cudaLaunchKernel` dispatch overhead (`nsys cuda_api_sum`, 817 calls) | `~2.41ms` total, `~2.9us`/call avg |
| Individual kernel durations (`ncu --print-summary per-kernel`) | `~2.1-2.2us` (elementwise/scale kernels) up to `~6.4-11.1us` (cutlass GEMM kernels); nothing anomalous for their problem sizes |
| One large `cudaMemset` (zeroing an accumulator/scratch buffer) | `~355us` |
| `cudaMemcpy` (12 calls, H2D+D2H) | `~210us` total |

Every one of these numbers is small and clean. The kernels are legitimately
tiny, dispatch overhead is normal, and the whole call -- ~800 launches and
all -- completes end-to-end in **about 4.6ms** when nothing else is
competing for the GPU.

**This changes the diagnosis.** `~4.6ms`/call in isolation is roughly
20-40x smaller than the `~150-185ms`/call that `F_err`'s real, measured
total (`104.4s`-`108.9s` over an epoch's worth of update candidates)
implies in the actual 32-rank MPS run. The earlier "launch-count/cuBLAS
overhead" hypothesis predicted that ~800 launches would be
*intrinsically* costly -- that's now directly measured and ruled out: in
isolation they are cheap. What's left is the one variable this harness
deliberately removed: **32 concurrent MPI ranks sharing one physical GPU
through MPS**, each independently issuing its own ~800 tiny back-to-back
launches at the same time, every training step. MPS lets those ranks'
kernels execute concurrently instead of time-slicing whole contexts (that
was the earlier, already-confirmed win -- see the MPS section above), but
each rank's hundreds of launches still has to pass through the MPS
server's shared submission path; with 32 ranks doing that simultaneously,
aggregate launch volume (up to ~26,000 near-simultaneous tiny launches
per training step, cluster-wide) plausibly creates queueing/dispatch
contention at the GPU that a single isolated process structurally cannot
reproduce.

**What's confirmed vs. still a hypothesis, explicitly:**
- CONFIRMED (directly measured): one isolated call's launch count (817),
  its total kernel time (~3.1ms), its wall-clock span (~4.6ms), and that
  none of its individual kernels or launches are anomalously slow.
- NOT YET CONFIRMED: that 32-way MPS contention specifically (rather than
  something else) is what inflates this to `~150-185ms`/call in the real
  run. Confirming that requires profiling the actual multi-rank job under
  load -- e.g. `nsys`/`ncu` attached to one representative rank while the
  other 31 are simultaneously running against the same MPS-shared GPU --
  which is a meaningfully harder profiling setup than this isolated
  harness and was not attempted here.

Next steps (not yet done): profile one rank of the real 32-rank MPS job
(not an isolated single-process harness) to directly confirm or rule out
multi-rank contention as the cause of the `~150-185ms`/call figure. If
confirmed, the fix is a different shape of problem than "make the kernels
faster" -- it becomes about *how many concurrent launch-heavy processes
share one GPU* (e.g. fewer MPI ranks per GPU, or consolidating multiple
ranks' Jacobian work behind one process/queue), not about the kernels
themselves, which this profiling run shows are already fast.

### Follow-up: profiled rank 0 of the REAL 32-rank MPS run -- contention confirmed, and it's blocking/sync stalls, not slower kernels

Ran the actual training job (32 MPI ranks, MPS enabled, real `H2O_2G`
data, 1 epoch) exactly as in the MPS section above, but with `nsys`
tracing rank 0 only (`gpu/e2e_train_check/profile_rank0_wrapper.sh` +
`run_nnp_train_gpu_mps_profile.slurm`). `nsys` does a single-pass trace,
not `ncu`'s multi-pass kernel replay, so it doesn't re-execute or reorder
kernels and can't desync rank 0 from the other 31 at an MPI collective --
it only adds recording overhead to rank 0's own calls. The run completed
normally (energy/force RMSEs matched the expected order of magnitude,
no hang, no crash), confirming this is safe to do.

This epoch's `timing.out`: `F_err = 107.9s` over `F_count = 275` force
updates on rank 0 -- an average of **`~392ms` per update**, close to
(actually somewhat worse than) the earlier `~150-185ms` back-of-envelope
guess.

**Individual kernels and launch dispatch are essentially unchanged from
the isolated baseline** (`nsys cuda_gpu_kern_sum` on rank 0's whole-epoch
trace):

| Kernel | Isolated (1 call, no contention) | Real run (whole epoch, rank 0, med.) |
| --- | --- | --- |
| cutlass GEMM (`nt`) | `~9.5us` | `~9.0us` |
| cutlass GEMM (`nn`) | `~6.4us` | `~6.8us` |
| `gemvNSP_kernel` | `~4.8us` | `~4.0us` |
| `splitKreduce_kernel` | `~3.7us` | `~3.2us` |
| elementwise/scale kernels | `~2.1-2.2us` | `~2.6-2.8us` |
| `cudaLaunchKernel` dispatch | `~2.9us` avg | `~3.7us` avg / `~3.2us` median |

Same kernels, same problem sizes, same durations, give or take noise.
**This directly rules out "kernels execute slower under contention" and
"launch dispatch is slower under contention"** -- neither is true. Total
GPU kernel execution time summed over rank 0's *entire* `131.5s` epoch
(every kernel, every dispatch function, every Kalman update, all `514555`
kernel launches) is only **`~2.26s`** -- `~1.7%` of the epoch. The GPU
itself is doing almost no work, from rank 0's point of view, essentially
the whole time.

**What's actually eating the time is host-side blocking on the shared
GPU** (`nsys cuda_api_sum`, rank 0, whole epoch):

| Blocking API | Calls | Total time | Notable outliers |
| --- | --- | --- | --- |
| `cudaDeviceSynchronize` | `2498` | `17.45s` | median `30us`, max `44.6ms` |
| `cudaFree` | `122` | `1.39s` | **one single call: `1.376s`** |
| `cudaMemcpy` | `26902` | `1.08s` | max `9.4ms` |

`cudaFree` and (for host-visible pointers) `cudaMemcpy` both carry an
implicit device synchronization -- they block until all outstanding work
on the device has drained. A single `cudaFree` call blocking for
`1.376s`, or `cudaDeviceSynchronize` occasionally taking `44.6ms` when its
own rank's queued work is `~4.6ms` worth of kernels in total, is the
signature of a GPU whose queue is backed up by the other 31 ranks'
concurrent submissions, not of this rank's own work being slow. In the
isolated single-process baseline, `cudaDeviceSynchronize` was called
exactly twice for `~293us` total; here it's called `2498` times for
`17.45s` on a training run that never asks for that many syncs when run
alone.

**What's confirmed vs. still open, explicitly:**
- CONFIRMED: `F_err`'s real per-update average (`~392ms`) is roughly
  `85x` the isolated per-call cost (`~4.6ms`).
- CONFIRMED: neither kernel execution time nor launch dispatch overhead
  is elevated under real 32-rank contention -- both match the isolated
  baseline closely.
- CONFIRMED: rank 0 spends `~2.26s` of a `131.5s` epoch actually running
  GPU kernels, and at least `~20s` blocked inside synchronizing CUDA API
  calls (`cudaDeviceSynchronize`/`cudaFree`/`cudaMemcpy`) whose durations
  have no counterpart in the isolated baseline -- direct evidence of
  32-way GPU-sharing contention via MPS, not of anything being
  computationally slower.
- NOT YET EXPLAINED: the confirmed blocking-wait time (`~20s`) is a
  meaningful chunk of `F_err`'s `107.9s` but doesn't account for all of
  it. The remainder is most likely legitimate CPU-side work (symmetry
  function / data-marshaling code around each GPU dispatch call, which
  predates the GPU port and runs regardless) whose share simply looks
  much larger now that the GPU part of each call is contention-dominated
  rather than compute-dominated -- but this is not directly measured yet,
  only inferred by elimination, and is flagged as such rather than
  presented as confirmed.

Next steps (not yet done): the earlier "batch launches into fewer, larger
kernels" idea is now much less promising -- kernels aren't the problem,
contention for the shared queue is. A more promising direction is
reducing how many independent processes hammer one GPU's queue
simultaneously (e.g. fewer MPI ranks per GPU with each rank handling a
larger batch, or consolidating multiple ranks' Jacobian work behind one
process). Finer-grained NVTX instrumentation of the CPU-only sections of
`Training::update()`'s PART 2 loop would help confirm or rule out the
"remaining time is legitimate CPU-side work" inference above.

### Follow-up: spread the 32 ranks across all 4 GPUs -- confirms GPU contention is real but a MINORITY of F_err's cost

Direct test of the "fewer ranks per GPU" idea proposed above: this node
has 4 A100s (`sinfo` reports `gpu:a100:4`), so
`run_nnp_train_gpu_mps_4gpu.slurm` + `rank_gpu_wrapper.sh` reran the same
32-rank, 1-epoch, real `H2O_2G` job with the ranks spread 8-per-GPU
across all 4, instead of all 32 on one. A single MPS daemon (no
`CUDA_VISIBLE_DEVICES` restriction on the daemon itself) serves all 4
GPUs; each client rank sets its own `CUDA_VISIBLE_DEVICES` to pick a GPU
-- this is the pattern NVIDIA's own MPS docs describe for multi-GPU
nodes. (An earlier attempt ran 4 *separate* per-GPU daemons and hit
intermittent `cublasCreate()` failures on some ranks, almost certainly a
startup race between the 4 simultaneous daemon spawns; switching to one
shared daemon fixed it outright, and is simpler besides.)

| | 32 ranks / 1 GPU (previous section) | 32 ranks / 4 GPUs, 8/GPU |
| --- | --- | --- |
| `F_err` | `107.9s` | `87.78s` |
| `F_err` avg/update (`275` updates) | `~392ms` | `~319ms` |
| epoch total | `112.2s` | `93.2s` |

**The fix helped, but nowhere near as much as a naive "contention scales
linearly with ranks-per-GPU" prediction would suggest.** Quartering the
ranks sharing each GPU (32 -> 8) should have quartered any purely
GPU-queue-contention-driven cost, but `F_err` only dropped `~19%`
(`20.12s`), not anywhere close to `~75%`.

**This number is not a coincidence, though -- it lines up almost exactly
with what was directly measured before.** The previous section's rank-0
trace of the 32-ranks/1-GPU run found `~19.9s` of confirmed host-side
blocking-wait time (`cudaDeviceSynchronize` `17.45s` +
`cudaFree` `1.39s` + `cudaMemcpy` `1.08s`) that had no counterpart in the
isolated single-process baseline. The `F_err` reduction measured here,
`20.12s`, matches that `~19.9s` almost to the second. Put together, this
is a coherent, cross-validated picture:
- The GPU-sharing-contention component of `F_err` is real, is now
  quantified independently two different ways (direct trace measurement,
  and an end-to-end fix that removes almost exactly that much time), and
  is worth roughly **`~20s`** in this one-epoch benchmark.
- But that `~20s` is only `~18%` of `F_err`'s total (`107.9s`). The
  remaining `~82%` (`~88s`) is **not** explained by GPU-sharing
  contention and was **not** helped by spreading ranks across more GPUs
  -- consistent with the earlier section's "not yet explained" inference
  that most of the remaining time is legitimate CPU-side work (symmetry
  function / data-marshaling code around each GPU dispatch call), since
  all 32 ranks still run on the *same* 32 CPU cores in both
  configurations here -- changing which GPU a rank talks to cannot touch
  a CPU-bound cost.

**Practical takeaway, and its limit:** requesting all available GPUs on
a node and spreading ranks across them (rather than defaulting to one
GPU) is a legitimate, free `~15-20%` win whenever multiple GPUs are
available, and should be the default for future GPU training jobs on
this cluster. But it is not close to a full fix for GPU training being
slower than CPU overall -- the dominant remaining cost is now
strongly implicated as CPU-side, not GPU-side at all.

**What's confirmed vs. still open, explicitly:**
- CONFIRMED: spreading 32 ranks across 4 GPUs (8/GPU) cuts `F_err` by
  `~20s` (`~19%`), and this reduction closely matches the independently
  measured GPU-blocking-wait time from the single-GPU trace.
- CONFIRMED (by elimination): the majority of `F_err`'s cost is
  unaffected by GPU-sharing pattern, meaning it is not a GPU-contention
  problem at all.
- NOT YET CONFIRMED: that the remaining `~88s` is specifically CPU-side
  symmetry-function/data-marshaling work, as opposed to some other
  as-yet-unmeasured cost. This is inferred by elimination (it isn't
  kernel time, isn't launch overhead, isn't the GPU-contention component
  quantified above), not directly measured.

Next steps (not yet done): profile the CPU side of `Training::update()`'s
PART 2 loop directly (e.g. `perf`, or NVTX ranges around the CPU-only
symmetry-function code bracketing each `gpuNn*` dispatch call) to confirm
or rule out the "remaining `~88s` is CPU-side work" inference, rather
than leaving it as elimination-by-exclusion.

### Follow-up: instrumented `Training::update()`'s PART 1/PART 2 directly -- found the real answer, and it's not the GPU dispatch at all

Rather than guess a third time, added *temporary* `Stopwatch` timers
directly in `src/libnnptrain/Training.cpp` (reverted afterward -- not
part of any commit) bracketing every substantial call inside the
`k == "force"` code path, and printed their per-epoch totals as a
`DEBUG` line from `printEpoch()`. Ran the same 32-ranks/1-GPU/MPS,
1-epoch, real `H2O_2G` job (`run_nnp_train_cpu_breakdown.slurm`).

The first pass only instrumented PART 2 (`collectDGdxia` +
`gpuNnForceDFdcSum` + PART 2's own `calculateForces()` call) and left
`~70s` of `F_err` still unexplained. Reading through PART 1 ("Find
update candidate") explained why: `H2O_2G`'s `input.nn` sets
`selection_mode 2` (`SM_THRESHOLD`) with `rmse_threshold_trials 3` --
meaning **PART 1 itself, before PART 2's GPU-accelerated Jacobian step
even runs, loops up to 3 times per selected update**, and each trial
calls `calculateSymmetryFunctionGroups()`, `calculateAtomicNeuralNetworks()`,
and **`calculateForces()`** (the same whole-structure, O(atoms x
neighbors), OpenMP-only, entirely CPU force loop in `Mode.cpp` --
`calculateSelfForceShort()`/`calculatePairForceShort()` over every atom
and its unique neighbors) to decide whether that candidate's RMSE
exceeds the threshold. Added three more timers there
(`trial_symfunc`/`trial_nn`/`trial_forces`) and reran.

**Full breakdown of `F_err` (`82.87s` this run), accounting for
essentially all of it (`82.57s` measured, `~0.3s` unattributed):**

| Component | Time | Share of `F_err` |
| --- | --- | --- |
| `trial_forces` -- PART 1's `SM_THRESHOLD` loop's `calculateForces()`, up to 3x/update | **`61.16s`** | **`~74%`** |
| `force_calcforces` -- PART 2's own (single) `calculateForces()` call | `8.46s` | `~10%` |
| `force_gpu` -- the actual GPU Jacobian dispatch (`gpuNnForceDFdcSum`), this whole investigation's original focus | `11.67s` | `~14%` |
| `trial_nn` -- PART 1's `calculateAtomicNeuralNetworks()` | `1.04s` | `~1%` |
| `force_collect` / `trial_symfunc` | `~0.25s` / `~0.00s` | `~0%` |

`trial_symfunc` being essentially free confirms `memorize_symfunc_results`
(set in `H2O_2G`'s `input.nn`) is doing its job -- symmetry functions are
cached, not recomputed. But **`calculateForces()`, called up to 4 times
per force-update (3 `SM_THRESHOLD` trials + 1 final PART 2 call), is
entirely CPU-only, was never GPU-ported at any point in this project, and
accounts for `~84%` of `F_err`'s total cost.** The GPU Jacobian dispatch
this entire investigation (rank-0 profiling, 4-GPU spread) focused on is
real, measurable, and was optimized correctly -- but it was only ever
`~14%` of the problem.

**This reframes the whole investigation.** Every prior follow-up in this
file (MPS, persistent GPU state, launch-count profiling, rank-0 tracing,
multi-GPU spreading) was legitimate, correctly executed, and honestly
reported -- but all of it targeted the smaller of two costs.
`calculateForces()` was never a suspect until it was directly measured,
because nothing about GPU contention or launch counts pointed at it --
it doesn't call into `src/libnnpgpu` at all, in either PART 1's trial
loop or PART 2.

**What's confirmed vs. still open, explicitly:**
- CONFIRMED (directly measured, accounts for ~100% of `F_err`, not
  elimination-by-exclusion this time): `calculateForces()` calls,
  driven by `SM_THRESHOLD`'s trial mechanism, are `~84%` of `F_err`'s
  cost; the GPU dispatch is `~14%`.
- NOT YET DONE: `calculateForces()` has no GPU implementation at all --
  `src/libnnpgpu` doesn't touch it. Whether it's a good GPU-porting
  target (it's an O(atoms x neighbors) pairwise sum, structurally similar
  in shape to symmetry function evaluation, which was out of scope for
  this whole `HDNNP_2G`-dispatch-focused port) is an open question, not
  yet assessed.
- WORTH CHECKING SEPARATELY: `rmse_threshold_trials 3` is a training
  hyperparameter, not a performance one -- reducing it would cut PART 1's
  trial cost roughly proportionally but changes training dynamics
  (fewer candidates considered before accepting one), so it is a
  trade-off to discuss with whoever owns the training recipe, not a free
  performance win to just apply.

Next steps (not yet done, and now the clearly higher-value ones): decide
whether `calculateForces()` is worth GPU-porting given it is `~84%` of
`F_err` versus the already-GPU-accelerated dispatch's `~14%`; separately,
raise the `rmse_threshold_trials` trade-off with whoever owns the
training recipe, since it directly multiplies PART 1's cost independent
of any GPU work.

### Follow-up: GPU-ported `calculateForces()` -- two failed attempts, then a real ~12x win

Went after the higher-value target identified above. First confirmed
`calculateForces()` really had never been touched by this port: only one
commit on this branch (`1fe973a`) edits `Mode.cpp` for GPU work, and its
diff is entirely inside `calculateAtomicNeuralNetworks()`, nowhere near
`calculateForces()` (verified via `git show`/`git log` on `Mode.cpp`, not
just a assumption).

**The math**: `Mode::calculateForces()` sums, for every atom `i`, a self
term (`-sum_k dEdG_i[k] * dGdr_i[k]`, no cross-atom dependency) plus a
pair term contributed by every neighbor `j` whose own symmetry functions
depend on `i`'s position (`-dEdG_j[table] * dGdr_j-wrt-i`). The CPU code
computes the pair term **atom-`i`-centric with an inner search**: for
each `i`, loop its unique neighbors `j`, then search `j`'s *own* neighbor
list for the entry pointing back to `i`. Reformulating this
**owner-`j`-centric** instead -- one direct pass over each atom's own
neighbor list, scatter-adding each contribution straight into the
*target* atom's accumulator -- computes the exact same sum without the
inner search, and is naturally a GPU-friendly edge list: one thread per
edge, `atomicAdd` into the target atom's force. Two kernels
(`src/libnnpgpu/GpuForces.cu`): `selfForceKernel` (one thread/atom, no
atomics needed) and `pairForceKernel` (one thread/edge, atomic scatter).

**Validated correct** against the real `Mode::calculateForces()` (not a
reimplementation) via `gpu/gemm/libnnpgpu_forces_test.cu`, using real
`H2O_2G` structures read through the actual `Prediction` class. Building
this test surfaced a real, previously-undetected, unrelated bug: `Atom::
toPhysicalUnits()`/`toNormalizedUnits()` (`src/libnnp/Atom.cpp`) has a
loop, over each atom's neighbor list, that's supposed to convert *the
neighbor's* `dGdr` but instead re-multiplies *the atom's own* `dGdr` by
`convLength` once per neighbor entry -- with ~100+ neighbors this
produces `convLength^100`-scale garbage (`~1e180`+ in an early test run).
It's dormant in ordinary `nnp-predict` usage since nothing reads `dGdr`
after that conversion runs; this test is a new consumer that does, so it
hit the bug immediately. Worked around it by calling `evaluateNNP()`
directly instead of `predict()` (skips the buggy conversion entirely,
compares everything in the NNP's native normalized units, which is all
this test needs) rather than touching `Atom.cpp` -- flagging the bug
here as a real, separate finding, not fixing it as part of this
GPU-porting work.

**Attempt 1 (stateless, rebuild-and-reupload-every-call) measured
WORSE, not better.** This mirrors the exact phased approach that worked
fine for `GpuNeuralNetwork.cu`'s three dispatch functions -- but real
end-to-end measurement (same 32-ranks/1-GPU/MPS/1-epoch/real-`H2O_2G`
scenario used throughout this file) showed `F_err` going from
`104-110s` to **`136.3s`**, and the whole training loop `130s -> 221.7s`.
Root cause: `calculateForces()`'s edge list is one to two orders of
magnitude bigger than anything the NN dispatch functions ever moved (up
to `~1.68M` entries for one real structure, tens of MB per call), called
up to 4x per force update (`SM_THRESHOLD`'s 3 trials + 1 final PART 2
call), under the same 32-way MPS contention that the rank-0 profiling
section above already showed makes `cudaMalloc`/`cudaMemcpy`/`cudaFree`
themselves become blocking, multi-millisecond-to-second stalls.
Correctness was unaffected (`learning-curve.out` matched prior runs) --
only speed regressed.

**Attempt 2: persistent per-structure topology cache -- correct, but ran
out of GPU memory.** Unlike the NN dispatch functions' weight/`G`/`dGdxyz`
inputs (which genuinely change every call), `calculateForces()`'s
topology (`dEdGOffset`, `dGdrSelf`, the whole edge list) is **purely
geometric** and never changes during training -- only `dEdG` does (it
depends on the NN's current weights). Split the API into
`gpuForcesUploadTopology()` (called once per structure, keyed by
`Structure::index`) and `gpuForcesCompute()` (called every time,
re-uploads only the small `dEdG` array). Validated correct --
`libnnpgpu_forces_test.cu` was extended to upload topology once per
structure then call `gpuForcesCompute()` repeatedly with *manually
perturbed* `dEdG` (simulating repeated weight updates against fixed
geometry), each compared against a fresh real `Mode::calculateForces()`
call for that exact perturbed `dEdG` -- `ALL PASS`, `~2e-14`. But the
real 32-ranks/1-GPU/MPS run crashed: **`out of memory`** on multiple
ranks. Each structure's cached edge list is `~54MB` (`H2O_2G`,
`~1.68M` edges x 32 bytes/edge); each rank caches roughly `35-40`
distinct structures over an epoch (`~1129` train structures / 32 ranks);
`32` ranks x `~2GB`/rank far exceeds what one A100 can hold once
everything else sharing that GPU is accounted for.

**Fix: combine with the earlier 4-GPU-spread finding.** Rather than add
a memory-bounded eviction policy (real added complexity, and its benefit
shrinks whenever eviction churn is high -- a live option if ever
revisited), spread the same 32 ranks 8-per-GPU across this node's 4
A100s (reusing `rank_gpu_wrapper.sh` and the single shared MPS daemon
pattern from the earlier MPS-contention follow-up section) --
`run_nnp_train_gpu_forces_4gpu.slurm`. This cuts the aggregate cached
memory per GPU to `~8` ranks x `~2GB` = `~16GB`, comfortably within one
A100's capacity.

**Result: a real, clean, apples-to-apples ~12x win.**

| | CPU `calculateForces()`, 4-GPU spread (earlier section) | GPU `calculateForces()` (cached), 4-GPU spread (this run) |
| --- | --- | --- |
| `F_err` | `87.78s` | **`7.12s`** |
| epoch total | `93.19s` | **`8.11s`** |
| training loop total | (not recorded) | `29.09s` |

`F_err`: **`12.3x`** faster. Epoch: **`11.5x`** faster. Against the very
first CPU-forces/1-GPU/32-rank baseline this whole investigation started
from (`F_err = 107.9s`), that's **`15.2x`**. `learning-curve.out`'s
energy/force RMSEs match every prior run almost exactly (e.g.
`E_train = 6.98885538E-06` here vs. `6.98889E-06`/`6.98908E-06` in
earlier runs) -- correctness held throughout every attempt, including
the two that regressed on speed.

**What's confirmed vs. still open, explicitly:**
- CONFIRMED: `gpuForcesUploadTopology()`/`gpuForcesCompute()` compute the
  exact same result as the real CPU `Mode::calculateForces()`, including
  across repeated calls with varying `dEdG` against fixed, cached
  topology.
- CONFIRMED: the real end-to-end `~12x` `F_err` speedup, measured
  against the closest available clean baseline (same 4-GPU-spread
  infrastructure, only `calculateForces()` differs).
- CONFIRMED (separate, incidental finding): `Atom::toPhysicalUnits()`/
  `toNormalizedUnits()` has a real bug (converts the wrong `dGdr` inside
  its neighbor loop) -- dormant today, but worth fixing or at least
  tracking separately from this GPU-porting work.
- NOT YET DONE: this measurement combines two changes (the
  `calculateForces()` GPU port itself, and the 4-GPU spread it now
  requires for memory reasons) -- both were independently validated
  before combining (the 4-GPU spread's own effect was already measured
  in isolation in the earlier section, `~19%` on the old CPU-forces
  code), but a fully isolated "GPU forces on exactly 1 GPU" number isn't
  available since that configuration doesn't fit in memory.
- NOT YET DONE: no memory-bounded eviction policy exists for the
  topology cache -- it currently grows unboundedly per rank for as many
  distinct structures as that rank visits over training. This was
  sufficient here (with the 4-GPU spread), but would need revisiting for
  larger datasets, more ranks per GPU, or bigger structures.
- NOT YET DONE: multi-epoch validation (this is still a 1-epoch
  benchmark, matching every other measurement in this file) and
  extending beyond `HDNNP_2G`.

Next steps (not yet done): run a multi-epoch benchmark to confirm the
`~12x` holds over a full training run, not just one epoch; consider
whether the `Atom::toPhysicalUnits()`/`toNormalizedUnits()` bug found
along the way should be fixed given it's now known to be real, even
though dormant; decide whether a memory-bounded eviction policy is
worth the complexity if future datasets/configurations need more ranks
per GPU than this one comfortably supports.

### Follow-up: a real 100-epoch run -- 17.8x, and the persistent cache doesn't leak

Ran the "not yet done" item above for real: `run_nnp_train_100ep_{cpu,gpu}.slurm`,
100 epochs, real `H2O_2G`, `normal` QOS (`1-00:00:00` limit -- a 100-epoch
CPU-only run doesn't fit in the debug queue's 30 minutes). Each build now
happens in its own isolated `git worktree`
(`git worktree add --detach n2p2_devel_100ep_{cpu,gpu} gpu-portability`),
**not** the shared main checkout -- running two concurrent `make clean &&
make ...` cycles against the same `src/lib/bin` tree is exactly the
build-race failure mode documented in the `nnp-dataset` fix below, and
this was the first time in the whole project two long training builds
were deliberately run at the same time.

**Result:**

| | CPU | GPU (4-GPU spread, cached topology) |
| --- | --- | --- |
| 100-epoch wall time | `3.97h` (`14300s`) | **`13.4min`** (`802.5s`) |
| steady-state epoch time | `~142s` | **`~7.8s`**, flat for all 100 epochs |

**`17.8x`** end to end -- better than the `~12.3x`/`~11.5x` single-epoch
numbers above, and, importantly, the GPU epoch time does not drift
upward over 100 epochs. That was a real risk worth checking: the
persistent per-structure topology cache (`GpuForces.cu`) grows as new
structures are visited across an epoch, and this is the first time it
was exercised over many repeated epochs rather than one -- a slow
memory/time leak would have shown up as the per-epoch time creeping up
over the run. It didn't; `~7.6-7.8s` the entire way.

**Correctness across 100 epochs, checked two ways, and one of them was
initially misleading in a way worth documenting.** The raw per-epoch
*training*-candidate RMSE in `learning-curve.out` diverges hugely
between the two runs by epoch 10 (500%+ relative difference at some
epochs) despite an identical fixed random seed (`random_seed 12345`).
That number is a red herring, not a correctness signal: it reflects
whichever specific candidates got sampled that epoch, and Kalman-filter
training is a recursive, chaotically-sensitive process where tiny
floating-point order-of-operation differences between CPU and GPU math
compound fast over many sequential updates -- expected, and it would
happen between any two nominally-identical runs whose arithmetic order
differs even slightly (different rank count, different BLAS, etc.), not
something specific to this GPU port.

The metric that actually answers "did both models learn the same
thing" is **test-set RMSE**, evaluated fresh on the same 118 held-out
structures every epoch -- and there the two runs track each other
closely the entire way (epoch 0 and 1 match to near machine precision;
by epoch 100, energy test RMSEpa `2.77e-6` (CPU) vs `2.32e-6` (GPU),
force test RMSE `3.865e-4` (CPU) vs `3.862e-4` (GPU)). Cross-checked
independently with `nnp-dataset` (see below) on the final epoch-100
weights against the same `test.data`: energy RMSE `1.742e-3` (CPU) vs
`1.462e-3` (GPU), force RMSE `3.865e-4` (CPU) vs `3.862e-4` (GPU),
matching the training loop's own periodic test evaluation almost
exactly -- both a correctness confirmation and a sanity check that the
two independent measurement paths (training-loop test eval vs. a
separate `nnp-dataset` run reading the saved weights) agree.

### Follow-up: nnp-dataset had never been run in this whole project -- fixed a real usability gap, and hit a build-race heisenbug along the way

Wanted to use `nnp-dataset` for the energy/force parity comparison
above, but it had literally never been compiled or run in this project
before. `Dataset::distributeStructures()` already took an optional
`fileName` parameter (default `"input.data"`), but `nnp-dataset.cpp`'s
`main()` never exposed it -- no way to point the tool at a held-out
`test.data` split directly, only ever a file literally named
`input.data`. Added an optional second CLI argument,
`nnp-dataset <shuffle> [<data_file>]`, fully backward compatible
(verified: default 2-arg output is byte-identical to explicit
3-arg output pointed at a same-content, differently-named file).

**This surfaced a real, very confusing bug during validation that had
nothing to do with the patch.** An early test run segfaulted inside
`SymGrpExpRad::calculate()` -- non-reproducibly: it crashed on some
runs and not others with the *exact same* command, the *exact same*
weights, the *exact same* input, sometimes fixed by adding unrelated
debug print statements. That pattern -- fixed by changes that shouldn't
matter -- is the signature of memory-layout-sensitive undefined
behavior, not a logic bug, and the actual root cause turned out to be
exactly that: two of this session's jobs were running `make clean &&
make ...` in the *same shared* `src/lib/bin` tree at overlapping times,
and one job's `make clean` deleting `.o` files while another job's
`make` was mid-compile/link produced a silently corrupted binary. A
clean, non-concurrent rebuild made the "bug" vanish completely, with
zero source changes. Confirmed via a careful decision tree (isolated
`-np 1` vs `-np 2`, growing dataset sizes, `gdb` -- which couldn't even
parse this binary's debug sections -- then `-fsanitize=address`, which
also couldn't reach the fault before a debug-queue timeout, before
simply re-running the *exact* prior command and having it pass clean).
Fixed going forward the same way the 100-epoch runs above do: isolate
concurrent builds into separate git worktrees rather than trust the
shared tree to serialize itself.

**What's confirmed vs. still open, explicitly:**
- CONFIRMED: the `17.8x` 100-epoch speedup, with no per-epoch drift
  over the full run.
- CONFIRMED: CPU- and GPU-trained models are statistically
  indistinguishable in test-set quality throughout training, cross-checked
  two independent ways (the training loop's own periodic test
  evaluation, and a separate `nnp-dataset` run against the saved
  weights).
- CONFIRMED (documented as a real, separate finding): concurrent
  `make clean && make ...` against a shared `src/lib/bin` tree from
  multiple jobs is a genuine build-race hazard in this project's build
  system, not hypothetical -- it was hit for real, and cost real
  debugging time before being correctly attributed.
- NOT YET DONE: `nnp-dataset`'s new argument and the git-worktree
  isolation pattern are both new; neither has been exercised outside
  this one investigation yet.

Next steps (not yet done): apply the same git-worktree isolation
pattern by default any time two builds might run concurrently in this
repo, not just for the 100-epoch scripts; consider whether
`nnp-dataset`'s test-set comparison is worth adding as a standard
post-training step in future benchmark scripts, given how directly
useful it was here.

### Follow-up: the honest GPU speedup, against a real CPU-class node, is ~5.6x -- not 17.8x

Every CPU baseline in this file so far ran on `boost_usr_prod` (Leonardo's
GPU partition) -- 32-core Intel Xeon Platinum 8358 (Ice Lake) nodes,
chosen only because that's where the GPU comparison also had to run
(same node, for a fair same-hardware A/B). Leonardo has a separate,
dedicated CPU partition, `dcgp_usr_prod` -- 112-core Intel Xeon Platinum
8480+ (Sapphire Rapids) nodes, two full CPU generations newer, 2 sockets
x 56 cores, `AMX`/`avx512_bf16`/`avx_vnni` Ice Lake doesn't have, 105MB
L3 vs. 48MB. Raised as a direct concern: is the `17.8x` headline number
overstating the real GPU win because Booster's CPUs are the weak side of
the comparison, not a representative CPU baseline?

**Yes, partially.** Ran the same 100-epoch CPU-only `H2O_2G` benchmark
on `dcgp_usr_prod` at 32, 64, and 112 (full node) cores, each in its own
isolated git worktree (the three ran concurrently, alongside the
build-race lesson from the section above):

| Config | 100-epoch total | Epoch time | vs. `802.5s` GPU |
| --- | --- | --- | --- |
| Booster CPU, 32 cores (this file's baseline so far) | `3.97h` | `143.0s` | `17.8x` |
| DCGP CPU, 32 cores (same core count, newer silicon) | `3.14h` | `113.2s` | `14.1x` |
| DCGP CPU, 64 cores | `1.87h` | `67.2s` | `8.4x` |
| **DCGP CPU, 112 cores (full node)** | **`1.25h`** | **`44.9s`** | **`5.6x`** |

Two separate effects, both real:
1. **Same core count, different CPU generation**: 32 DCGP cores beat 32
   Booster cores by `1.26x` on identical settings -- Sapphire Rapids is
   genuinely faster than Ice Lake for this workload, not just a core-count
   story.
2. **DCGP has 3.5x the cores per node** (112 vs. 32) -- scaling
   32->64->112 cores gave `84%`/`85%` parallel efficiency per doubling-ish
   step, `72%` overall 32->112, consistent with this project's very
   first Phase 0 profiling flagging MPI communication/rank-count
   overhead as a real, separate cost.

Combined, the **honest GPU speedup against the strongest CPU
configuration actually tested (a full 112-core DCGP node) is `~5.6x`,
not `17.8x`**. `17.8x` was a correct, honestly-measured number for what
it actually compared (GPU vs. one specific 32-core Ice Lake
configuration) -- but quoting it as "the" GPU speedup without the
CPU-partition caveat overstates the real win by roughly `3x`. This
correction was requested and run *after* the `17.8x` number had already
been shared externally; this section exists so the record is accurate
going forward, not to retract what was already said in good faith with
the data available at the time.

**What's confirmed vs. still open, explicitly:**
- CONFIRMED: `1.26x` same-core-count CPU generation gap (Sapphire
  Rapids vs. Ice Lake) for this exact workload.
- CONFIRMED: real GPU speedup vs. the best CPU baseline tested so far
  (112-core DCGP) is `~5.6x`.
- NOT YET DONE: DCGP nodes allow up to 16 nodes/job
  (`MaxNodes=16` on `dcgp_usr_prod`) -- multi-node CPU scaling (beyond
  one 112-core node) hasn't been tested, so `~5.6x` is the speedup
  against the best *single-node* CPU config, not necessarily the best
  possible CPU config on this cluster.
- NOT YET DONE: the GPU side of this comparison is still the same
  4-GPU/32-rank Booster configuration used throughout this file --
  worth asking whether a like-for-like "1 node's worth of GPU vs. 1
  node's worth of CPU" framing changes anything, though 4 GPUs on 1
  Booster node already *is* that framing on the GPU side.

Next steps (not yet done): consider whether multi-node DCGP CPU scaling
is worth measuring before treating `~5.6x` as the final word; when
reporting GPU speedups from this port going forward, quote the CPU
baseline's hardware explicitly (partition, CPU model, core count)
rather than a bare multiplier.

### Follow-up: 4G-HDNNP -- two CPU bugs (~10x, ~24%) before any GPU work, then a GPU port of what was actually slow

Everything above is `HDNNP_2G` (short-range only). Porting `HDNNP_4G`
(electrostatics + non-local charge transfer) started with real
profiling on a real periodic dataset (`temp/H2O_4G`, 1254 structures,
630 atoms/structure) rather than assuming the electrostatics/charge
equilibration math was the natural GPU target -- a good thing, since
the two dominant costs found were both plain CPU algorithmic bugs, not
GPU-shaped problems at all.

**Bug 1, stage 1 (charge NN training):** `Structure::calculateDQdChi()`
(and four sibling functions -- `calculateDQdJ`, `calculateDQdr`,
`calculateForceLambdaTotal`, `calculateForceLambdaElec`) each called
`A.colPivHouseholderQr()` fresh for every solve against the same
`(numAtoms+1)x(numAtoms+1)` charge-equilibration matrix, instead of
factorizing once and reusing it -- turning an `O(N^3)` solve into an
`O(N^4)` one (`calculateDQdChi` alone loops over every atom in the
structure). Fixed by adding a persistent `Structure::AConstrainedQr`
member, factorized once in `calculateElectrostaticEnergy()` and reused
via `.solve()` everywhere else. Measured on the real dataset: stage-1
epoch time `270s -> 26.6s` (`~10x`), `calculateDQdChi` itself
`250s -> 2.0s` (`~126x`), `learning-curve.out.stage-1` bit-identical
before/after.

**Bug 2, stage 2 (force training):** `Mode::calculateForces()` was
being recomputed from scratch on every `SM_THRESHOLD` trial of the same
update candidate (up to `rmse_threshold_trials` times), even though
trials share the same structure and the same not-yet-updated weights --
the electrostatics half of this exact redundancy already had a cache
(`Structure::hasAMatrix`); the short-NN-forward + `calculateForces()`
pair had no equivalent guard. Fixed with a per-candidate `forcesValid`
flag, declared fresh inside the batch loop (never a persistent
`Structure`-level flag, since forces genuinely go stale between
different candidates -- each one applies a real Kalman filter update).
`~24%` faster, bit-identical results, smaller win than bug 1 because
most `SM_THRESHOLD` candidates apparently resolve in one trial already.

**Then, and only then, GPU work.** Fine-grained profiling of what was
left (temporary `Stopwatch` brackets around `Mode.cpp`'s `HDNNP_4G`
force block and `Training.cpp`'s `calculateDQdr`/`calculateForces` call
sites, same throwaway-instrumentation methodology as the `F_err`
investigation above) found: `calculateForceLambdaTotal()`/`Elec()` --
two more solves against the now-cached factorization -- cost under 1ms
per call combined; `calculateDQdr()` cost `0.05s` per epoch on rank 0.
Meanwhile the whole `calculateForces()` call cost `~17.3s` per epoch on
that same rank -- a `346x` gap. The difference: an `O(numAtoms^2 x
avgNeighbors)` double loop in `calculateForces()`'s `HDNNP_4G` block
that calls `Atom::calculateDChidr()` for every atom pair, which
internally does a **linear neighbor-list search per pair** -- the same
shape of cost (dense linear algebra is cheap, an unported O(N^2) pair
loop is not) that made the 2G `calculateForces()` port worth `~12x`
above.

Design: `lambdaTotal(j)`/`lambdaElec(j)` are constant per *owner* atom
`j`, so pre-multiplying them into `dChidG_j[k]` once per call turns the
whole computation into the exact same self-term + owner-centric-edge-
list shape `GpuForces.cu` already uses for the 2G short-range port,
plus one extra embarrassingly-parallel dense `O(numAtoms^2)` reduction
for the `dAdrQ` term. No factorization anywhere in this particular
port -- it's a pure reduction/scatter -- so none of `GpuKalmanFilter`'s
"GPU LU/inverse silently disagrees with Eigen's on real ill-conditioned
data" risk applies here; still validated against real production data
anyway (`gpu/gemm/elecforces_test.cu`, real `fElec` dumped from an
actual `nnp-train` stage-2 run, max abs diff `1.1e-19`) rather than
relying on that argument alone.

New module: `src/libnnpgpu/GpuElecForces.h/.cu`, same conventions as
`GpuForces.h/.cu` (persistent per-structure topology keyed by
`Structure::index`, `CUDA_CHECK` macro, plain-C++ header). Wired into
`Mode::calculateForces()`'s `HDNNP_4G` block behind `#ifdef N2P2_GPU` /
`!N2P2_FULL_SFD_MEMORY`, guarded by a `doneOnGpu` flag so the original
CPU loop is untouched as the fallback.

**End-to-end result, honestly:** correctness holds -- epoch 0 (a
deterministic full-dataset evaluation with no Kalman-filter candidate
selection involved) is bit-identical GPU vs. CPU; epochs 1-2 diverge in
raw per-epoch training-candidate RMSE by an amount consistent with the
Kalman filter's already-documented chaotic sensitivity to
floating-point reduction order (see this file's very first 2G
`learning-curve` discussion), not a correctness problem. But the speed
win is modest: total training-loop time `262.8s -> 234.6s` (`~12%`,
with an MPS daemon for the 4 ranks sharing 1 GPU in this small
5-structure test; `247.5s` without MPS) -- far short of what the
isolated kernel's cost share suggested. Most likely cause, not yet
fixed: `dAdrQ` (`9.5MB` for this 630-atom structure) is re-uploaded via
`cudaMemcpy` on **every single call**, unlike the topology, which is
genuinely cached. `dAdrQ` only changes when charges do, and stage 2
keeps charges frozen across many calls per `Structure::hasAMatrix`'s
existing validity window (same reasoning that already justifies caching
the electrostatics solve itself) -- caching `dAdrQ` the same way is the
natural next step, not yet done.

### Follow-up: tried caching `dChidG`/`dAdrQ`/`pEelecpr` too -- found a real correctness bug via the exact discipline this file keeps preaching, reverted

Attempted the natural next step above: split `GpuElecForces`'s upload
into two tiers (`gpuElecForcesUpdateChargeData()` for `dChidG`/`dAdrQ`/
`pEelecpr`, called only when they'd actually changed, vs.
`gpuElecForcesCompute()`'s per-call `lambdaTotal`/`lambdaElec` upload).
Staleness was tracked with a `Mode.cpp`-file-scope `set<size_t>`,
marked in `Mode::chargeEquilibration()` whenever its `derivativesElec`
block runs (the only place these three are actually recomputed) and
consumed in `Mode::calculateForces()`'s GPU block. Verified every call
site in `Training.cpp` pairs the elec-NN forward pass with
`chargeEquilibration(..., true)` under the same flag -- a real,
grep-confirmed invariant, not a guess.

First end-to-end run already looked suspicious: raw training-candidate
RMSE diverged from the CPU baseline by more than the already-known
Kalman-chaos noise floor (`1.84e-2` vs. the `1.50e-2`/`1.56e-2` range
every other GPU/CPU comparison in this file has shown). Per this
project's standing rule (verify against real data, don't trust a
number that merely looks plausible -- the exact lesson `GpuKalmanFilter`
already paid for once), added a temporary debug cross-check: on every
`calculateForces()` call, recompute the CPU-reference `f`/`fElec` from
the structure's *current* live data and diff against the GPU result,
printing `structureId`/`needChargeDataUpload`/max-abs-diff.

**Confirmed real, not noise.** Every call with `needChargeDataUpload=1`
(fresh upload) matched to float noise (`~1e-18`), as expected. Calls
with `needChargeDataUpload=0` (cache reuse) matched too -- for the
*first* reuse after each upload -- then diverged, and the divergence
**settled into a fixed, per-structure constant** rather than growing
unboundedly (e.g. structure 3's `fElec` error sat at exactly
`2.722e-06` across dozens of later calls, `f`'s error scaled with the
still-fresh, still-correctly-uploaded `lambdaTotal`). That shape --
one missed refresh, then a stable offset -- points at `pEelecpr`
specifically (added directly, unscaled by any lambda, so a fixed delta
in it shows up as a fixed delta in the output) going stale through a
path the `derivativesElec`-only marker doesn't see: `chargeEquilibration()`
also gets called with `derivativesElec=false` from the energy-training
branch (`Training.cpp`, gated on `!s.hasCharges`, not `!s.hasAMatrix`),
which still flips `hasAMatrix` true (`calculateElectrostaticEnergy()`
sets it unconditionally) without refreshing `pEelecpr`/`dAdrQ`/`dChidG`
-- a cross-k-branch interaction a single-function staleness marker
can't safely observe from outside.

**Reverted the caching change entirely** (`Mode.cpp`,
`GpuElecForces.h/.cu` back to the always-reupload version this section
already validated correct) rather than ship a partial fix without being
certain -- training on silently-wrong forces is a far worse outcome
than a missed 12%-ish-more-of-a-modest-win. A safe version of this
optimization needs a staleness signal that can't be fooled by
cross-branch mutation, which really means `Structure` itself exposing
an explicit, incrementing generation counter for `dChidG`/`dAdrQ`/
`pEelecpr` (bumped at their one real write site, immune to guessing
which call sites might indirectly trigger it) rather than inferring
staleness from *outside* by pattern-matching caller code -- a more
invasive change than fits as a quick follow-up, left for a dedicated
pass. `dAdrQ`'s per-call re-upload cost stands as the next real
optimization target, just not solved yet.

### Follow-up: stage-1's Jacobian-assembly loop -- the biggest remaining cost, and it didn't need GPU at all

Went back to what was actually still unaddressed from the original
stage-1/stage-2 profiling: the Jacobian-assembly loop (`"Finally sum up
Jacobian"` in `Training.cpp`'s `k == "charge"` branch), ~60% of
stage-1 epoch time (`15.86s` of `~26.6s`), never ported or fixed --
the two CPU bugs earlier in this section fixed the dense solves and
the redundant-`calculateForces()` cost, not this.

Before reaching for CUDA, looked at the loop itself: `i` (atoms) x `k`
(atoms) x `j` (weight connections for atom `k`'s element), accumulating
`jacobian[offset[l]+j] += (1/QErrorNorm) * QError(i) * dQdChi[k](i) * dChidc[k][j]`.
`dChidc[k][j]` doesn't depend on `i` at all -- so the `i`-sum factors
out algebraically:
`dChidc[k][j] * sum_i QError(i)*dQdChi[k](i) = dChidc[k][j] * QError.dot(dQdChi[k])`.
Swapping loop order (`k` outer, `i` only inside a dot product) turns
`O(numAtoms^2 * numWeightsPerElement)` into `O(numAtoms^2)` (the dot
products, one per atom, using Eigen's vectorized `VectorXd::dot()`) +
`O(numAtoms * numWeightsPerElement)` (the final scatter) -- the same
reassociation applies to the hardness term. A pure CPU refactor, no
GPU code at all, and a much bigger win than a straight GPU port of the
original triple loop would likely have delivered for the same effort
-- this project's "measure before porting" rule paying off again, this
time by finding there was nothing left to port once the algorithm
itself was fixed.

Verified on the same real `temp/H2O_4G` 60-structure/630-atom dataset,
same config as the earlier `AConstrainedQr` fix: epoch time
`26.6s -> 8.6s` (`~3.1x`), and -- better than expected, since
reassociating a floating-point sum isn't generally bit-exact --
`learning-curve.out.stage-1` came out **bit-identical** across every
epoch and every column, not just agreeing to float64 precision.
Within the new epoch time, `Q_err` (the phase this loop lives in) went
from `22.6s` (`97.5%` of epoch) to `5.2s` (`91.2%`) -- still the
dominant phase, but the loop itself is no longer the outsized cost it
was.

Net effect of all three stage-1/stage-2 fixes this section covers,
stage 1 specifically: `270s -> 26.6s -> 8.6s` per epoch on the real
dataset -- roughly `31x` from the original O(N^4) bug through to here,
entirely on the CPU side, before any GPU code touched stage 1 at all.

### Follow-up: profiled the remaining stage-1 "error" phase -- 32.6% of epoch time, no code change needed at all

The Jacobian-assembly fix above left `Q_err` at `91.2%` of a smaller
epoch, but `printEpoch()`'s own `TIMING` line already reported a
SEPARATE `error` phase (`calculateErrorEpoch()`, computes the
train/test RMSE that actually gets written to `learning-curve.out` --
distinct from `Training::update()`'s per-candidate work) at `32.6%` of
epoch time, unexplained by anything measured so far. Same temporary-
Stopwatch methodology, this time bracketing `calculateErrorEpoch()`'s
per-structure loop (`Training.cpp:1422-1492`): two components,
`chargeEquilibration()` (`~0.90s/epoch`, same dense Qeq-solve cost
already characterized elsewhere in this section) and
`calculateSymmetryFunctionGroups()` (`~1.46s/epoch`, `53%` of the
error phase on its own) -- together `85%` of the phase, the rest file
I/O and MPI reduction.

The symmetry-function recomputation is the interesting one: every
`calculateErrorEpoch()` call frees each structure's `G`/`dGdr` right
after use (`Training.cpp:1494`, `if (freeMemory) it->freeAtoms(...)`)
and `freeMemory` is set from a single existing n2p2 keyword --
`freeMemory = !settings.keywordExists("memorize_symfunc_results")`
(`Training.cpp:936`) -- **already present, commented out, in
`temp/H2O_4G/input.nn`** (`#memorize_symfunc_results`). No source
change needed at all; just uncomment it.

Verified with the flag enabled (real `temp/H2O_4G` data, otherwise
identical setup): epoch time `8.6s -> ~4.9s` (a further `~1.75x`),
`learning-curve.out.stage-1` bit-identical to the flag-disabled run.
Stacking every stage-1 fix in this section: `270s -> 26.6s -> 8.6s ->
~4.9s` per epoch, `~55x`, with the last step being a one-line
config change rather than a code change. The obvious tradeoff --
memory: with the flag on, every structure's symmetry-function data
stays resident for the whole run instead of being freed between uses,
so this is a real memory-for-speed trade whose cost scales with
dataset size (not yet measured at the full 1254-structure production
scale) -- worth deciding deliberately, not enabling blindly.

### Follow-up: `memorize_symfunc_results` at full production scale -- ~2.4GB, enabled

The memory tradeoff flagged above was measured directly rather than
guessed: full `temp/H2O_4G` dataset (1254 structures, 790,020 atoms
total, 630 atoms/structure average), 64 ranks, one dcgp node,
`sacct` cgroup-wide peak RSS across the whole job (`nnp-scaling` +
`nnp-train 1`, both `memorize` on and off).

| | epoch time | peak RSS |
|---|---|---|
| `memorize` off (baseline) | 48.99s | 194.5 GB |
| `memorize` on | 27.8-27.95s | 196.9 GB |

Extra memory: **~2.4GB, ~1.2% over baseline** -- far smaller than
expected; the per-atom `G`/`dGdr`/neighbor-cache arrays this setting
keeps resident turn out to be a rounding error next to the dataset's
baseline footprint (dominated by neighbor lists and the dense
electrostatics matrices, both always resident regardless of this
flag). Speedup held at full scale too: `~1.76x`, matching the
subset-60 measurement almost exactly. Given a node has 500GB+
available and the cost is ~2.4GB, `memorize_symfunc_results` is now
**enabled in the real `temp/H2O_4G/input.nn`** (single line
uncommented, not git-tracked -- `temp/` is gitignored).

### Follow-up: `GpuQeqSolver` -- the dense charge-equilibration solve family, `~2.06x` epoch / `~2.32x` train phase

With `memorize_symfunc_results` eliminating the symmetry-function
recompute cost, re-profiling stage 1's `Training::update("charge")`
(same temporary-Stopwatch methodology as this section's earlier
entries, `sf`/`fwd`/`qeq`/`dq`/`jac` brackets around the "charge"
branch and the candidate-selection loop's `calculateSymmetryFunctionGroups()`
call) found `chargeEquilibration()` + `calculateDQdChi()`/`calculateDQdJ()`
(`qeq`+`dq`) now **44.3% of epoch time** -- `sf` collapsed to ~0
(confirming the fix above), `dq` 25.7%, `qeq` 18.6%, `fwd` (the chi
forward-pass loop, originally the suspected target) only 7.5%.

All five of `Structure::AConstrainedQr`'s downstream solve call sites
(`calculateDQdChi`, `calculateDQdJ`, `calculateDQdr`,
`calculateForceLambdaTotal`, `calculateForceLambdaElec`) share one
factorization per `calculateElectrostaticEnergy()` call -- and the
last two are stage-2 call sites, so this port helps both stages, not
just stage 1 like `GpuElecForces` above.

**Numerical safety, checked before writing any CUDA** (the discipline
that would have caught the Kalman-filter GPU regression earlier in
this file): cuSOLVER's fast solvers are LU-based (`getrf`/`getrs`),
not Eigen's rank-revealing `ColPivHouseholderQR` used today.
- Condition number on real `AConstrained` matrices (60 real 631x631
  structures from `temp/H2O_4G`): `cond ~ 3.27e5`, essentially
  identical across all 60 (consistent with same-density MD snapshots
  -- the extremal singular values are dominated by aggregate
  hardness/density, not fine geometry).
- Direct QR-vs-LU solve comparison, first with Eigen
  `ColPivHouseholderQR` vs `PartialPivLU` as a CPU stand-in, then with
  the actual `cusolverDnDgetrf`/`getrs` API (`gpu/gemm/qeq_solver_test.cu`,
  validated against the same real matrices, dumped via a temporary
  `N2P2_DUMP_QEQ` hook in `Structure.cpp`, reverted after use): relative
  error in the solved charges **~1e-15, worst case ~7.6e-14** -- float64
  machine precision, no measurable accuracy loss from switching to LU.

**Design** (`src/libnnpgpu/GpuQeqSolver.h/.cu`): a `static`
`unordered_map<size_t, State>` cache keyed by `Structure::index`,
exactly like `GpuForces.h`/`GpuElecForces.h`'s topology cache --
*not* a `Structure`-owned handle, because `calculateForceLambdaTotal()`/
`calculateForceLambdaElec()` are `const` member functions and
`Structure` has no user-defined destructor/copy-constructor (adding
one to manage a raw GPU handle risks a shallow-copy/double-free bug
when the training-set `vector<Structure>` reallocates). A `static`
cache sidesteps this entirely -- it's freely mutable from `const`
methods since staticness bypasses the enclosing object's constness.
Unlike `GpuForces`' topology (purely geometric, cached forever),
`AConstrained` depends on the current per-element `hardness` (a
trainable weight), so `gpuQeqFactorize()` refactorizes unconditionally
every `calculateElectrostaticEnergy()` call, exactly like the CPU
code's unconditional `AConstrainedQr.compute()` today -- only the
device buffer allocation and LU factors are reused, not the
factorization result across calls.

`calculateDQdChi()`/`calculateDQdJ()` originally called `.solve()` in
a loop (`numAtoms`=630 times, `numElements`=2 times). Batched into
**one** multi-RHS `gpuQeqSolve()` call each instead -- doing 630
separate small GPU round trips would very plausibly have repeated
`GpuForces.cu`'s documented first-pass MPS-contention regression (see
above in this file). `calculateDQdr()`/`calculateForceLambdaTotal()`/
`calculateForceLambdaElec()` stay `nrhs=1` (their actual call sites
only ever pass one right-hand side) -- `GpuElecForces` already proved
small unbatched per-call GPU round trips are an acceptable, real net
win in this exact codebase.

**Correctness caught something real, and it wasn't a bug.** End-to-end
multi-epoch verification (isolated worktree, `GPU=1` build,
`temp/H2O_4G` subsets, `nnp-scaling` + `nnp-train 1` *and* `2`):
stage 1's `learning-curve.out.stage-1` came out **bit-identical**
between CPU and GPU builds. Stage 2 did not -- `learning-curve.out.stage-2`
diverged by up to ~2x in RMSE from epoch 1 onward, while epoch 0 (a
deterministic evaluation pass, no Kalman recursion yet) matched
exactly. Rather than assume this away as "expected Kalman chaos"
(the same phrase that would have rationalized away a real bug), added
a live cross-check: temporarily ran the CPU `AConstrainedQr` solve
*alongside* the GPU one during an actual `N2P2_QEQ_XCHECK`-gated stage-2
run and diffed every call. All 60 live per-call relative errors came
back `~1e-15` to `~1e-18` -- matching the standalone fixture validation
exactly, confirming every individual GPU solve agrees with CPU to
machine precision during real training, not just in isolation. The
~2x aggregate RMSE divergence is therefore the Kalman filter's
already-documented chaotic sensitivity to floating-point order (a
~1e-15 per-call perturbation compounding through ~673 recursive
updates/epoch) amplifying a machine-precision difference, not a
correctness defect -- confirmed by measurement, not assumed.
Instrumentation reverted after use.

**Wall-clock, real data, MPS** (60-structure/630-atom subset, 8 ranks
sharing 1 GPU, stage 1):

| | epoch | train phase |
|---|---|---|
| CPU | 4.94s | 3.78s |
| GPU (`GpuQeqSolver`) | 2.40s | 1.63s |

`~2.06x` epoch, `~2.32x` on the `train` phase specifically (where
`qeq`/`dq` live) -- no MPS-contention regression, and a substantially
bigger win than `GpuElecForces`' ~12%, consistent with `qeq`+`dq`
having been the single largest remaining cost (44.3% of epoch) before
this port.

### Follow-up: full production-scale comparison (1254 structures) -- CPU core-count scaling vs 4-GPU, and a self-inflicted MPS bug

Requested comparison: CPU wall time on `dcgp_usr_prod` (32/64/112
cores) vs the fully GPU-ported path on all 4 A100s of one
`boost_usr_prod` node, on the **real, full** `temp/H2O_4G` dataset
(1254 structures, 790,020 atoms), 1 epoch, `memorize_symfunc_results`
on.

**CPU baseline, real 4G code as it stands today** (i.e. with every
CPU-side fix from this whole effort already included -- the O(N^4)
charge-equilibration fix, the Jacobian-assembly fix, the SM_THRESHOLD
redundant-`calculateForces()` fix, `memorize_symfunc_results` -- *not*
a "before this session" baseline, since building that would have meant
rebuilding from a commit predating all of the above just for this
comparison):

| dcgp cores | Stage 1 epoch | Stage 2 epoch |
|---|---|---|
| 32 | 23.51s | 2686s (44.8 min) |
| 64 | 12.97s | 1859s (31.0 min) |
| 112 | 9.39s | 1121s (18.7 min) |

**First GPU attempt: 32 ranks spread across all 4 GPUs, one MPS control
daemon PER GPU (each started with its own `CUDA_VISIBLE_DEVICES`
restriction).** This is a legitimate MPS topology in principle -- but
on this cluster it hard-failed: every rank bound to GPU 1/2/3 got
`no CUDA-capable device is detected`, only GPU 0's daemon ever accepted
connections. A serialized-startup diagnostic (`gpu/gemm/run_mps_serialized_diag.slurm`
-- each daemon started alone, 3s apart, with an immediate single-process
connectivity test right after each) ruled out a startup race: GPU 1/2/3
failed even in total isolation, with no other daemon or client anywhere
near them. Falling back to plain `CUDA_VISIBLE_DEVICES` rank binding
with **no MPS at all** (`gpu/gemm/run_nomps_gpu_diag.slurm`) worked
cleanly on all 4 GPUs, so that's what the first full-scale run used --
giving 32-ranks/1-GPU=21.48s and 32-ranks/4-GPUs(8/GPU)=15.63s for
stage 1 (a real but modest `~1.37x` from spreading across GPUs, far
from the `~4x` naive scaling would suggest) and a disappointing
**2658s** for stage 2 -- barely different from the 32-core CPU number.

**The mistake, found by checking this file first.** Multi-GPU MPS had
already been solved for the 2G port, earlier in this same document (the
"spread the 32 ranks across all 4 GPUs" section above): **one single,
UNRESTRICTED MPS control daemon serving all 4 GPUs**, with each
*client* rank picking its own GPU via its own `CUDA_VISIBLE_DEVICES` --
the pattern NVIDIA's own docs describe as the default for multi-GPU
nodes, and already implemented in `gpu/e2e_train_check/rank_gpu_wrapper.sh`.
That section even already documents hitting the *same* per-GPU-daemon
failure once before ("An earlier attempt ran 4 *separate* per-GPU
daemons and hit intermittent `cublasCreate()` failures on some ranks
... switching to one shared daemon fixed it outright"). This 4G
investigation re-discovered the identical failure mode from scratch
before checking whether it had already been solved -- a real process
mistake, not a new cluster limitation. Re-verified via
`gpu/gemm/run_single_daemon_mps_diag.slurm` (one unrestricted daemon,
8 ranks across all 4 GPUs, 2/GPU): clean success on every rank.

**Corrected result, same full-scale run, single shared MPS daemon**
(`gpu/gemm/gpu_rank_wrapper.sh`, `RANKS_PER_GPU=8`):

| Config | Stage 1 epoch | Stage 2 epoch |
|---|---|---|
| CPU, 112 cores | 9.39s | **1121s (18.7 min)** |
| CPU, 64 cores | 12.97s | 1859s (31.0 min) |
| **GPU, 4xA100, 32 ranks (correct MPS)** | **14.15s** | **2643s (44.1 min)** |
| CPU, 32 cores | 23.51s | 2686s (44.8 min) |

MPS fixed stage 1 modestly (`15.67s -> 14.15s`, now clearly ahead of
32-core CPU) but barely touched stage 2 (`2658s -> 2643s`) -- direct
confirmation that stage 2's cost was never an MPS/contention problem in
the first place.

### Follow-up: why stage 2 stays slow -- verified, not guessed

Rather than leave "stage 2 is slow" as an unexplained number, profiled
it directly with the same temporary-Stopwatch methodology used
throughout this file, bracketing `Training::update("force")`'s PART 1
(candidate/trial selection) and PART 2 (Jacobian assembly), on a real
5-structure `temp/H2O_4G` subset:

| bracket | share of `F_err` |
|---|---|
| `calcForces` (`Mode::calculateForces()`, the O(numAtoms^2) electrostatics loop) | ~68-70% |
| `nnFallback` (force-Jacobian NN forward+`calculateDFdc`, CPU-only loop) | ~30% |
| `shortNN` (short-range NN forward pass) | ~5% |
| `dQdr` (this session's `GpuQeqSolver` work) | **~0.3%, negligible** |

Two real, already-understood reasons, not a bug:

1. **`calculateForces()` is already GPU-accelerated (`GpuElecForces`,
   confirmed via a call-count check that `doneOnGpu=true` on every
   call) but was only ever a documented ~12% win** (see this section's
   earlier entry) -- capped by having to re-upload an O(numAtoms^2)
   `dAdrQ` array (~9.5MB for a 630-atom structure) on every single
   weight update, since it depends on the current charges and can't be
   cached across calls. A caching fix for exactly this was attempted
   earlier this session, caught a real correctness bug via cross-
   checking, and was deliberately reverted rather than shipped
   uncertain -- so this ~12% ceiling is accepted, not overlooked.
2. **The force-Jacobian NN backward pass (`calculateDFdc`) was never
   GPU-ported for 4G at all.** `Training.cpp`'s own comment says why:
   `gpuNnForceDFdcSum` (the 2G equivalent) explicitly scopes out
   HDNNP_4G's extra charge-input neuron and `dQdxia` handling. This is
   a genuine, well-scoped, still-open follow-up -- not a defect in
   anything built this session.

Confirmed with a live MPS-vs-no-MPS comparison on the same small
subset that `calcForces`'s cost is flat either way (~23s vs ~26s,
within noise) -- ruling out contention as an explanation for its size
and pointing squarely at the re-upload cost and kernel/data-marshaling
overhead instead.

**Net assessment:** this session's `GpuQeqSolver` port is fully
validated and doing exactly what it was built for (stage 1, and the
`calculateDQdr`/`calculateForceLambdaTotal`/`calculateForceLambdaElec`
slice of stage 2) -- it just isn't where stage 2's time goes. The
natural next step for anyone picking this back up is porting 4G's
force-Jacobian NN backward pass, following `gpuNnForceDFdcSum`'s
existing pattern with the extra charge-input-neuron/`dQdxia` handling
this time -- likely the single biggest remaining opportunity in the
whole 4G pipeline, given it's ~30% of stage 2's dominant `F_err` phase
and has had zero GPU attention so far.

### Follow-up: porting `calculateDFdc` to GPU for 4G -- a real, modest win, and why it's smaller than hoped

Picked up the follow-up above. The surprise: no new device code was
needed at all. `gpuNnForceDFdcSum()` (`src/libnnpgpu/GpuNeuralNetwork.cu`,
built for the 2G port) was already fully generic in `numIn` -- it never
assumes an "input" is a symmetry function rather than 4G's extra charge
input neuron, and `NeuralNetwork::hasGpuCompatibleArchitecture()` never
checks `numIn` either (only layer count, activation functions, and
output size). The only thing scoping it to `HDNNP_2G` was
`Training.cpp`'s dispatch code itself.

The fix (`src/libnnptrain/Training.cpp`, `Training::update()`'s
`"force"` branch) is a small, surgical extension of the existing
2G-only GPU dispatch: `nnpType == NNPType::HDNNP_2G` widened to also
accept `HDNNP_4G`, plus two array-construction fixes so the extra
column reaches the (unchanged) GPU function correctly --

- **G's last column** set to each atom's charge (`a.charge`), matching
  the CPU fallback loop's `nn.setInput(it->G.size(), it->charge)`.
- **dGdxyz's last column** set to `dQdxia =
  s.atoms.at(sC->a).dQdr.at(ia)[sC->c]`, matching the CPU fallback
  loop's `dGdxia.back() = dQdxia` (this value was already being
  computed for every atom via `Structure::calculateDQdr()`, called
  earlier in the same branch -- this session's `GpuQeqSolver` work, so
  no new cost here either).

`Atom::dEdG` and `NeuralNetwork::getNumNeuronsInLayer(0)` both already
account for the extra charge neuron (`Atom.cpp`'s
`dEdG.resize(numSymmetryFunctions + 1, ...)` when `useChargeNeuron` is
set), so nothing downstream needed touching.

**Correctness**, validated two ways:
1. `gpu/gemm/libnnpgpu_dfdc_test.cu` already fuzzes `gpuNnForceDFdcSum()`
   across architectures with random `numIn`/weights/inputs against the
   real CPU `calculateDFdc()` -- since the function is unchanged, this
   coverage carries over directly to the 4G case (a 4G net is just
   "one more `numIn`" to that test).
2. A temporary, env-var-gated live cross-check (`N2P2_DFDC_XCHECK`,
   the same pattern used for `GpuQeqSolver` above) ran the real CPU
   `calculateDFdc()` alongside the new GPU dispatch for every single
   force-update candidate of a full 2-epoch stage-2 run on real 4G
   data, diffing the two Jacobian sums. Every single call, across the
   whole run: max absolute error ~1E-14 to 1E-19, max relative error
   ~1E-11 to 1E-14 (the rare higher relative-error samples all traced
   to a near-zero true value in the denominator, not a real
   discrepancy) -- machine-precision agreement throughout. Reverted
   after validating, per this file's established practice.

**Performance**, full `temp/H2O_4G` dataset (1254 structures), 1
epoch, 32 ranks / 4 GPUs, single shared MPS daemon (the corrected
setup from the section above):

| | Before this port | After this port |
|---|---|---|
| Stage 2 epoch time | `2643s` (44.1 min) | `2455s` (40.9 min) |
| `F_err` (the force error/Jacobian phase) | `2554s` | `2355s` |
| Stage 1 epoch time (unaffected, sanity check) | `14.15s` | `14.37s` (within noise) |

A real **~7.1%** stage-2 speedup, **~7.8%** on `F_err` specifically --
correct, and in the right direction, but far short of the ~30%
`F_err` share the earlier CPU-time profiling attributed to this exact
code path. The likely reason, consistent with a lesson already learned
elsewhere in this file (`GpuForces.cu`'s documented first-pass
regression from many small GPU calls under MPS contention): this
dispatch is called once per (atom, coordinate) force-update candidate,
not batched across candidates, so each of the many calls per epoch
pays its own H2D/D2H transfer and kernel-launch overhead on a
comparatively small amount of work (`numAtoms` ~210-420,
`numIn` ~36-43) -- multiplied by 32 concurrent MPI ranks all sharing
the same 4 GPUs under MPS. The CPU-time share this replaced was real,
but a large fraction of the saved CPU time was spent instead on
transfer/launch overhead rather than idle GPU compute.

**Net assessment:** correct, shipped, and a genuine (if modest)
stage-2 win -- worth keeping. `F_err` is still stage 2's dominant cost
and 4x-A100 stage 2 is still slower than 64-core or 112-core CPU at
this problem size (`2455s` vs `1859s`/`1121s`); closing that gap
further would mean batching this dispatch across multiple force-update
candidates per call (amortizing the transfer/launch overhead across
more work per call, the same insight that motivated
`GpuQeqSolver`'s multi-RHS batching earlier in this file) rather than
further per-call kernel tuning.

### Follow-up: batching the force-Jacobian dispatch -- built, validated, reverted (a real hyperparameter wall, not a bug)

Picked up the follow-up above. Designed and implemented
`gpuNnForceDFdcSumBatch()`: rather than just amortizing per-call
transfer/launch overhead, it exploits that most of
`gpuNnForceDFdcSum()`'s per-input (`k0`) loop body doesn't actually
depend on `dGdxyz` at all -- only the final accumulation step per `k0`
does. Splitting each `k0` iteration into a "producer" half (shared
weights/forward-pass only, computed ONCE per `k0` for a whole group of
candidates) and a "consumer" half (the `dGdxyz`-dependent part, still
once per candidate) cuts the k0-loop's kernel-launch count by roughly
1.8x as group size grows, on top of amortizing the weight/`G` upload
and forward pass. `Training.cpp`'s HDNNP_4G force branch was
restructured to stage each candidate's `dGdxyz` instead of dispatching
immediately, then flush one batched call per (structure, element)
group after the whole `batchSize` loop (safe because every candidate
in one `update()` call shares the same not-yet-updated weights -- PART
3 applies exactly one weight update per call).

**Correctness**, validated two ways, both clean: (1) a standalone test
(`gpu/gemm/libnnpgpu_dfdc_batch_test.cu`) comparing the batched
function against repeated single calls across 7 cases (group sizes
1-20, growing/shrinking atom counts to exercise the capacity-growth
logic) came back **bit-identical** (`0.000E+00` in every case -- not
just close, exactly the same floating-point operations in the same
order). (2) A live cross-check
(`N2P2_DFDC_BATCH_XCHECK`) against the real CPU `calculateDFdc()`
through the actual staged/flushed `Training.cpp` code path, over 7232
samples of a real stage-2 run, held machine precision throughout.

**The wall:** every single one of those 7232 samples had `nCand=1`.
`temp/H2O_4G/input.nn` sets `task_batch_size_force 1` -- every
`Training::update("force")` call processes exactly ONE candidate
before applying its Kalman-filter weight update. Since weights change
between every single candidate, there is *structurally* never more
than one force-training candidate available to batch within a
dispatch opportunity under this training configuration -- not a bug
anywhere in this session's code, just what `task_batch_size_force=1`
means. The `useSubCandidates`/`numGroupedSubCand` machinery this
follow-up hoped to piggyback on turned out to group which *candidate*
several consecutive (separately weight-updated) calls draw from the
same structure, not multiple candidates evaluated under the same
weights -- an easy mechanism to misread from its name alone.

With `nCand` pinned at 1, the batched path is provably no faster than
the original per-candidate dispatch (same work, same GPU calls) and
strictly *more* overhead (duplicate `G`/`atomsByElement` construction
at both staging and flush time, plus the staging map itself) -- a pure
regression risk for zero benefit. Reverted in full (`Training.cpp`,
`GpuNeuralNetwork.cu`/`.h`, and the standalone test) rather than kept
as unreachable dead code.

The only way to get more than one force-training candidate under the
same weights would be raising `task_batch_size_force` itself -- a
training-methodology change (fewer, larger-batch Kalman updates
instead of many single-candidate ones) that affects convergence
dynamics, not a performance-only knob, and explicitly out of scope for
a "speed up the existing GPU port" session. **Net assessment:** the
diagnosis that motivated this (many small GPU calls, ~30% of `F_err`
theoretically available) was correct; the fix built for it was correct
too; the training hyperparameters this dataset actually uses simply
don't create an opportunity for it to help. Stage 2's remaining time
is not further addressable by batching this call site without first
deciding, separately and deliberately, to change how many force
samples go into each Kalman update.

### Follow-up: a fresh re-profile found `calculateForces()` was being called TWICE per force candidate -- a real, ~40-48% stage-2 win, no GPU code involved

Started a new profiling pass from scratch (temporary fine-grained
`Stopwatch` brackets, same methodology as every other entry in this
file, on a real 60-structure `temp/H2O_4G` subset, reverted after use)
to re-check where stage-2 time goes now that `calculateDFdc` is ported
and the batching attempt above is reverted. `calculateForces()`
(PART 1's cached call) still accounted for roughly half of `F_err`, as
expected -- but tallying every bracket against `F_err`'s own total left
**~50% of `F_err` unaccounted for**, on top of the `calculateForces()`
share already measured.

The missing time: `Training::update()`'s `"force"` branch calls
`calculateForces(s)` in **two** places --

1. PART 1's trial loop (`if (!forcesValid) { ...; calculateForces(s);
   forcesValid = true; }`), added by an earlier fix in this file (the
   "two failed attempts, then a real ~12x win" / SM_THRESHOLD
   redundant-trial fix) specifically to avoid recomputing forces across
   repeated trials of the *same* candidate.
2. PART 2's "Sum up total potential energy or calculate force" block,
   **unconditionally**, with no `forcesValid` guard at all -- pre-dating
   the fix above, never updated when that fix was added.

Under this project's real config (`selection_mode 2` = `SM_THRESHOLD`,
confirmed in `temp/H2O_4G/input.nn`), PART 1 always runs at least once
per candidate, so PART 2's call is *always* a second, complete
recomputation of the exact same structure under the exact same
not-yet-updated weights -- pure duplicate work on `calculateForces()`,
the single most expensive operation in stage 2 (see the `dAdrQ`
re-upload discussion earlier in this section).

**Why this is safe to fix, checked two ways before touching anything
based only on a plausible-looking timing number:**
- **By reading the code**: `Mode::calculateForces()` starts with an
  unconditional `ai.f = Vec3D{};` reset for every atom (`Mode.cpp:2149`)
  before accumulating short-range and electrostatic contributions --
  every call is a complete, independent recomputation from zero, never
  an accumulation across calls. Nothing between PART 1's cached call
  and PART 2's runs changes the weights, charges, or geometry (no
  `s.clearElectrostatics()` for the `useSubCandidates` path this
  branch always takes, no `freeAtoms()` effect on `f`), so the two
  calls are computing the identical thing.
- **By measurement, live, on real training candidates**: added a
  temporary `N2P2_FORCES_DETERMINISM_CHECK` probe that calls
  `calculateForces(s)` twice in a row on the same unchanged state and
  diffs the raw per-atom output bit-for-bit. Across 82 real candidates
  on the GPU build, every single one showed nonzero differences between
  the two calls -- up to ~1000 of 1890 force components differing, max
  `8.9E-16`. This is exactly the known, already-documented
  `atomicAdd`-on-doubles non-reproducibility of the `GpuElecForces`/
  `GpuForces` scatter kernels (this file's `force/` section: "`atomicAdd`
  on doubles is not run-to-run bit-reproducible"), not a logic error --
  confirming the two calls really were computing the same thing to
  within float noise, on both counts.

That float noise explains an apparent puzzle when verifying the fix:
`learning-curve.out.stage-1` came out bit-identical before/after (stage
1 never touches this branch), but `learning-curve.out.stage-2` visibly
diverged from epoch 1 onward. This is *not* a correctness regression --
it's the Kalman filter's already well-documented chaotic sensitivity to
floating-point perturbations (the same phenomenon that made the
reverted `dAdrQ`/`dChidG`/`pEelecpr` caching attempt and `GpuQeqSolver`'s
stage-2 divergence both look alarming at first) amplifying *which* of
two ~1e-16-different, equally-valid roundings ends up feeding the
filter. Removing the redundant call changes that choice; it does not
change the physics.

**Fix**: guard PART 2's call the same way PART 1's already is --
`if (!forcesValid) calculateForces(s);`. `forcesValid` is declared
fresh per candidate and is only ever set `true` by the
`HDNNP_4G`/`SM_THRESHOLD` branch in PART 1, so `HDNNP_2G` and any
non-`SM_THRESHOLD` selection mode are completely unaffected (the flag
stays `false`, PART 2's call still runs exactly as before).

**Measured win**, same 60-structure subset, 3 epochs, before vs. after
(cumulative-since-start `Stopwatch` brackets and `timing.out.stage-2`
both agree):

| | Stage-2 epoch time | `calculateForces()` calls/rank (3 epochs) |
|---|---|---|
| GPU, 4xA100, 32 ranks (MPS) | `123.3s` -> `64.8s` | 500 -> 250 |
| CPU, 32 ranks (dcgp) | (not separately measured unfixed) | 500 -> 250 (est. ~160s -> 99.5s epoch) |

**~47.5%** stage-2 epoch-time reduction on GPU, **~38%** estimated on
CPU (from the now-near-zero `prof_calcForcesPart2` bracket vs. the
still-present `prof_calcForces` bracket) -- a bigger win than any GPU
kernel work in this whole port, delivered by a four-line CPU-only
change. This does not change the earlier finding that stage 2's
`calculateForces()` GPU port is capped by the `dAdrQ` re-upload cost
(still true, still unfixed) -- it changes how many times that capped
cost gets paid per candidate, from two down to the necessary one.

Verified: `learning-curve.out.stage-1` bit-identical, confirming
`HDNNP_2G` and stage 1 are untouched by construction (`forcesValid` is
declared but never set outside the `HDNNP_4G`/`SM_THRESHOLD` force
branch). All temporary instrumentation (`Stopwatch` brackets,
`N2P2_FORCES_DETERMINISM_CHECK`) reverted before commit; only the
one-line `forcesValid` guard ships.

### Follow-up: confirmed at full production scale (1254 structures) -- stage 2 GPU time halved, CPU-112 gap narrows from 2.36x to 1.90x

Reran the same CPU-112-vs-4xA100 comparison this file's "full
production-scale comparison" section already established, on the exact
same real, full `temp/H2O_4G` dataset (1254 structures, 790,020 atoms),
1 epoch, `memorize_symfunc_results` on -- this time on top of the
`forcesValid` fix above, to confirm the subset-60 result wasn't an
artifact of the smaller test:

| | Stage 1 epoch | Stage 2 epoch |
|---|---|---|
| CPU, 112 cores (dcgp) | `9.44s` (was `9.39s`, unaffected as expected) | `691.2s` (was `1121s`) -- **~38.4% faster** |
| GPU, 4xA100, 32 ranks (MPS) | `14.15s` (unchanged, as expected) | `1311s` (was `2643s`) -- **~50.4% faster** |

Stage 1 is bit-for-bit unaffected on both hardware paths, exactly as
the fix's scope predicts. Stage 2 dropped by roughly half on GPU (a
bigger relative win than the CPU side, consistent with GPU calls paying
proportionally more per-call transfer/launch overhead that a halved
call count directly removes) and by over a third on CPU. Net effect:
the CPU-112-vs-GPU gap this file flagged as "stage 2 stays slow" is
real but meaningfully smaller now -- **`1.90x`** (GPU slower) instead
of the pre-fix **`2.36x`**. GPU is still behind CPU-112 on both stages
(`14.15s` vs. `9.44s` stage 1, `1311s` vs. `691.2s` stage 2), so this
alone doesn't flip the headline conclusion, but it's the single biggest
step towards closing that gap found in this whole port -- and the only
one so far that required no GPU code at all.

### Follow-up: found the next lever -- HDNNP_4G's short-range force loop was never GPU-ported, and it now dwarfs the electrostatics block by ~55-60x

With `calculateForces()`'s redundant call gone, re-checked what's left
inside a single call. Earlier profiling in this file attributed
`calculateForces()`'s cost mostly to the electrostatics `dChidr`/`dAdrQ`
loop -- the piece `GpuElecForces` ported -- since that was the obvious
`HDNNP_4G`-specific addition on top of the (already GPU-ported for
`HDNNP_2G`) short-range term. That assumption was never directly
checked with its own bracket. It was wrong.

`Mode::calculateForces()` has two top-level phases for `HDNNP_4G`: the
short-range self+pair loop (`Mode.cpp:2138-2192`, shared, unmodified
code also used by `HDNNP_2G` -- the exact computation `GpuForces.cu`
already ported for a documented `~12x` win, see this file's `force/`
section) followed by the `HDNNP_4G`-only electrostatics block
(`GpuElecForces`-ported). A direct temporary `Stopwatch` split of
these two phases (60-structure subset, 1 epoch, GPU build) found:

| Phase | Cumulative time (1 epoch, per rank) |
|---|---|
| Short-range self+pair loop (**not** GPU-ported for `HDNNP_4G`) | `~55-60s` |
| Electrostatics block (`GpuElecForces`-ported) | `~0.8-1.1s` |

**The un-ported short-range loop is ~55-60x more expensive than the
entire already-GPU-ported electrostatics block.** `Mode.cpp`'s
`GpuForces.h` dispatch is gated `if (nnpType == NNPType::HDNNP_2G)`
only, with a comment on the `HDNNP_4G` electrostatics port above it
noting "`HDNNP_4G` is out of scope" -- true for the electrostatics
addition that comment was actually about, but it silently left the
*shared* short-range term unported for `HDNNP_4G` too, since `HDNNP_4G`
never takes the `HDNNP_2G`-gated early-return path at all and falls
straight through to the plain CPU loop instead. This single loop is
now, by a wide margin, `calculateForces()`'s dominant cost -- and by
extension one of stage 2's dominant costs, given `calculateForces()`
is `F_err`'s dominant phase.

**Why this looks like a low-risk, high-value port, not a new one:**
`HDNNP_4G`'s short-range term uses the exact same
`calculateAtomicNeuralNetworks(s, derivatives, "short")` forward pass
and the same `Atom::dEdG`/`dGdr`/neighbor-list structures `HDNNP_2G`
already uses -- `GpuForces.h/.cu` was written against that shared
representation, not anything `HDNNP_2G`-specific in the math itself
(`NeuralNetwork::hasGpuCompatibleArchitecture()`, the same
architecture gate `calculateDFdc`'s widening relied on, doesn't check
`nnpType` either). The module is already built, already validated
(`gpu/gemm/libnnpgpu_forces_test.cu`, real data), and already proved
worth `~12x` on this identical computation for `HDNNP_2G`.

**What the port actually needs**, unlike `calculateDFdc`'s widening,
isn't a one-line dispatch-condition change: `HDNNP_2G`'s current path
(`Mode.cpp:2037`) `return`s immediately after `gpuForcesCompute()`
sets `ai.f`, but `HDNNP_4G` must *continue* into the electrostatics
block afterward (which adds onto the same `ai.f` via `+=`, not
overwrites it) rather than returning. So this needs restructuring the
early-return into a fall-through -- run `gpuForcesCompute()` for both
`HDNNP_2G` and `HDNNP_4G` when the architecture check passes, populate
`ai.f` (and, for `HDNNP_2G`, `return` as today; for `HDNNP_4G`, don't),
then let the existing `HDNNP_4G` electrostatics block run on top as it
already does today for CPU-computed short-range forces. Moderate,
well-understood scope -- not a new algorithm, no new numerical-safety
question (pure reduction/scatter, same class as `GpuElecForces`, not a
factorization like `GpuQeqSolver`) -- but real engineering, not a
one-liner, and needs its own real-data validation pass before shipping
(same discipline as every port in this file).

**Expected impact, not yet realized**: if this port gets anywhere near
`GpuForces.cu`'s existing `~12x` for the identical `HDNNP_2G`
computation, `calculateForces()`'s cost would drop from
`~56-61s`/epoch to roughly `~5-6s`/epoch at this problem size (the
already-cheap electrostatics block barely moves the total either way)
-- and since `calculateForces()` is `F_err`'s dominant phase and `F_err`
is stage 2's dominant phase, this is very plausibly enough to flip the
CPU-112-vs-GPU comparison the "confirmed at full production scale"
entry above left at `1.90x` (GPU slower) into GPU actually winning.
This is the single biggest lever identified in this whole file that
hasn't been pulled yet.

**On the single-node multi-GPU question this profiling pass set out to
answer**: the deployment shape already in use (4xA100 on one Booster
node, 32 MPI ranks, one shared unrestricted MPS control daemon) is not
the bottleneck and doesn't need to change -- every scaling experiment
in this file (single-daemon vs. per-GPU-daemon MPS, rank-per-GPU
counts, DCGP core-count scaling) already confirmed that topology is
sound. What was actually missing was completeness of the port itself:
a large fraction of stage 2's real compute was still silently running
on the CPU inside a function whose other half looked, from the
outside, like it had already been GPU-accelerated. The right "multi-GPU
strategy" here is not more GPUs, a different batching topology, or a
new kernel-launch pattern -- it's finishing this specific, already
well-scoped, low-numerical-risk port.

Not yet implemented -- this section documents the finding and the
plan, following this file's practice of writing up what was found
before deciding whether/how to act on it.

### Follow-up: implemented -- HDNNP_4G's short-range force loop now uses GpuForces.cu, ~8.8x stage-2 win, GPU flips from 1.90x slower than CPU-112 to 4.63x faster

Implemented the plan above. `Mode::calculateForces()`'s GPU dispatch
(`Mode.cpp:2057`) widened from `if (nnpType == NNPType::HDNNP_2G)` to
`if (nnpType == NNPType::HDNNP_2G || nnpType == NNPType::HDNNP_4G)` --
zero changes to the topology/dEdG-building code or to `GpuForces.cu`
itself, exactly as expected (confirmed by directly reading
`Atom::calculateSelfForceShort()`/`calculatePairForceShort()`: both
loop strictly over `[0, numSymmetryFunctions)`, so `HDNNP_4G`'s extra
charge-neuron `dEdG` element -- which the shared force-loop code never
touches -- doesn't change the topology/`dEdG` array sizing at all).

The one real code change: a new `shortRangeDoneOnGpu` flag replaces
`HDNNP_2G`'s unconditional early `return` with a conditional one --
`HDNNP_2G` still returns immediately (nothing left to do), but
`HDNNP_4G` falls through into the CPU loop's `#pragma omp parallel`
region with `if (shortRangeDoneOnGpu) continue;` guarding the per-atom
work (so the loop body never re-runs), reaching the existing
electrostatics block afterward exactly as before -- that block adds
onto `ai.f` (`+=`) rather than resetting it, so it composes correctly
with either the GPU-computed or CPU-computed short-range result
underneath.

**Verified two ways**, following the "test before touching anything"
discipline this bug already needed:
1. **Live cross-check** (`N2P2_SHORTFORCE_XCHECK`, same pattern as
   every other port in this file): inside the GPU dispatch, saved the
   GPU result, recomputed the exact CPU reference into the same
   slots, diffed, then restored the GPU result before continuing (so
   the check never changes what the real run uses). 60-structure
   subset, 2 epochs, real training candidates: **5428 evaluations,
   every one matching to machine precision, max abs diff `3.3E-14`**,
   zero failures.
2. **Clean performance run** (no cross-check -- the diagnostic itself
   redoes the full CPU computation for verification, which would
   otherwise mask any speedup): full `temp/H2O_4G` dataset (1254
   structures), 1 epoch, 32 ranks / 4 GPUs (MPS).

**Full-scale result**:

| | Stage 1 epoch | Stage 2 epoch |
|---|---|---|
| CPU, 112 cores (dcgp) | `9.44s` | `691.2s` |
| GPU, 4xA100, 32 ranks (MPS), before this port | `14.15s` | `1311s` |
| **GPU, 4xA100, 32 ranks (MPS), after this port** | `14.17s` (unchanged, expected) | **`149.3s`** |

**~8.8x** stage-2 speedup from this one port (`1311s -> 149.3s`) --
close to `GpuForces.cu`'s original `~12x` for the identical
computation on `HDNNP_2G`, and by far the single biggest win in this
whole 4G porting effort. Stage 1 is unaffected, exactly as expected
(it never calls `calculateForces()`).

**This flips the headline CPU-vs-GPU comparison.** The "confirmed at
full production scale" entry above left GPU `1.90x` slower than
CPU-112 on stage 2; with this port, **GPU is now `4.63x` faster than
CPU-112 on stage 2** (`691.2s` vs. `149.3s`), and roughly **`4.3x`
faster overall** across both stages combined (`700.6s` vs. `163.5s`
total). Stage 1 remains GPU's one weak point (`14.17s` vs. `9.44s`,
`~1.5x` slower) -- `GpuQeqSolver`'s territory, not this port's, and a
much smaller absolute cost than stage 2 either way.

All temporary instrumentation (`N2P2_SHORTFORCE_XCHECK`) reverted
before commit; the shipped change is the `shortRangeDoneOnGpu`
restructuring only, no new device code.

**Net assessment for the single-node multi-GPU question this whole
profiling pass set out to answer**: the deployment shape (4xA100, one
Booster node, 32 ranks, one shared MPS daemon) was correct all along
and needed no change. What was missing was completing an already
well-scoped, already-validated port that had silently stopped halfway
at the `HDNNP_2G`/`HDNNP_4G` boundary. With this port shipped, GPU is
now the clearly faster option for 4G stage-2 training at this problem
size, reversing the conclusion every full-scale comparison in this
file had shown until now.

### Follow-up: a fresh post-port profile found a third gap -- HDNNP_4G's "short" NN forward pass was also HDNNP_2G-only, ~15% more stage-2 win

With the short-range force loop no longer dominant, re-profiled stage 2
from scratch (same temporary-`Stopwatch`-bracket methodology, 60-structure
subset) to see what the *new* `F_err` composition looks like, rather than
assume the job was finished:

| Component | Share of `F_err` (post short-range-loop port) |
|---|---|
| `calculateDFdc` GPU dispatch | 47.6% |
| `calculateForces()` (both terms now GPU-ported) | 21.3% |
| `calculateAtomicNeuralNetworks(..., "short")` | **17.1%** |
| elec-NN forward + `chargeEquilibration` gate | 10.9% |
| `calculateDQdr` | 3.0% |

`calculateDFdc` is already GPU-ported and already at its documented
ceiling (the `task_batch_size_force=1` batching wall from the reverted
batching attempt above -- a training-methodology change, out of scope).
But `calculateAtomicNeuralNetworks()`'s GPU dispatch
(`gpuNnForwardDEdG`, the batched forward pass already validated at
`~25-27x` for `HDNNP_2G` in `src/libnnpgpu/`'s Phase 6 work) turned out
to have the *exact same* `if (nnpType == NNPType::HDNNP_2G)` gate as
the two bugs already fixed this session (`Mode.cpp:1725`) --
`HDNNP_4G` gets its own separate branch (`id == "short"` / `id ==
"elec"`) that always falls back to the one-atom-at-a-time CPU loop.

**Ported the `"short"` case** (the bigger of the two, `"elec"` is a
separate follow-up below). Unlike `HDNNP_2G`'s branch, `HDNNP_4G`'s
`"short"` NN has an extra charge-neuron input (`nn.setInput(a.G.size(),
a.charge)` in the CPU fallback) and a correspondingly-widened `dEdG`
output (`numSymmetryFunctions + 1`, last element `dEdQ`). No new device
code needed -- `gpuNnForwardDEdG` is already `numIn`-generic and
`hasGpuCompatibleArchitecture()` doesn't check `numIn` either, same
"no new device code" pattern as `calculateDFdc`'s and the short-range
loop's `HDNNP_4G` widenings. The added GPU branch mirrors `HDNNP_2G`'s
almost exactly, with two differences: `G`'s extra column set to
`a.charge` (matching the CPU fallback), and `dEdG`'s full `numIn`-wide
result copied back *including* the trailing `dEdQ` element -- unlike
`calculateForces()`'s short-range loop (which never reads it),
`calculateForceLambdaTotal()`/`Elec()` depend on `dEdQ` being correct,
so it can't be truncated the way the short-range-loop port could
ignore it.

**Verified live** (`N2P2_SHORTNN_XCHECK`, same pattern as every other
port here): compared GPU-computed `energy`/`dEdG` against the real CPU
reference for every real training candidate, 60-structure subset, 2
epochs -- **5588 evaluations, every one matching to machine precision**
(max abs diff `6.6E-14` energy, `6.8E-13` `dEdG`), zero failures.
Reverted before commit; only the GPU dispatch branch itself ships.

**Full-scale result** (clean run, no cross-check overhead):

| | Stage 1 epoch | Stage 2 epoch |
|---|---|---|
| CPU, 112 cores (dcgp) | `9.44s` | `691.2s` |
| GPU, before this port (short-range loop port only) | `14.17s` | `149.3s` |
| **GPU, after this port** | `14.27s` (unchanged, expected) | **`127.2s`** |

**~14.8%** further stage-2 speedup (`149.3s -> 127.2s`), `F_err`
itself dropping `131.7s -> 109.4s` (`~16.9%`, matching the subset-60
share estimate almost exactly). Stage 1 unaffected, as expected (this
fix only touches `id == "short"`, and stage 1 only ever calls
`calculateAtomicNeuralNetworks()` with `id == "elec"`/the charge NN).
**GPU is now `5.44x` faster than CPU-112 on stage 2** (`691.2s` vs.
`127.2s`), up from `4.63x` after the short-range-loop port alone.

The `"elec"` case (`id == "elec"`, the charge-equilibration NN's own
forward pass, part of the remaining `10.9%` bucket above) is a
separate, smaller follow-up -- different output target (`dChidG`
instead of `dEdG`, no extra charge-input column since the elec NN's
own input is just symmetry functions) and a `normalize`
post-processing step the `HDNNP_2G` branch doesn't have, so it isn't a
copy-paste of this same diff. Tracked as its own commit.

### Follow-up: ported the "elec" NN forward pass too -- correct, but a modest, honestly-reported ~1% win, not another big one

Implemented the `"elec"` case following the same pattern as `"short"`
above: a new `HDNNP_4G`-only GPU branch mirroring `HDNNP_2G`'s dispatch,
writing into `a.chi`/`a.dChidG` instead of `a.energy`/`a.dEdG`, no extra
charge-input column (confirmed via `Atom.cpp`:
`dChidG.resize(numSymmetryFunctions, ...)`, no `+1` -- the elec NN's
own input is just symmetry functions, unlike the "short" NN), and the
"negativity" `normalize()` post-processing step (when configured)
replicated after the batched dispatch.

**Verified live** (`N2P2_ELECNN_XCHECK`): compared GPU-computed
`chi`/`dChidG` against the real CPU reference across both stages
(stage 1 trains charges and calls this heavily; stage 2's force branch
calls it once per structure per `hasAMatrix` validity window), 60-structure
subset, 2 epochs each -- **575 evaluations total (244 stage 1 + 331
stage 2), `chi` matching to `~7-9E-13`**. Stage 1's `dChidG` diff came
back exactly `0.000E+00` -- not a red flag: stage 1's charge branch
calls this with `derivatives=false` (it only needs the forward `chi`
value there; the actual charge-training weight-Jacobian goes through a
separate `calculateDEdc()` call, not this function's `dChidG` output),
so both the GPU and CPU-reference paths skip `dChidG` entirely on
those calls, matching trivially. Stage 2's non-trivial `dChidG` match
(`2.0E-12`) confirms the `derivatives=true` path is independently
exercised and correct.

**Performance, full scale, honestly reported.** The first clean run
looked like a wash-to-slight-regression (stage 1 `14.27s -> 15.13s`,
stage 2 `127.2s -> 126.2s`) -- rather than accept a single one-epoch
number the way earlier sections in this file already learned not to, reran
under identical conditions. The second run landed at stage 1 `14.01s`,
stage 2 `125.8s` -- the first run's stage-1 number was noise (an
`Q_com`/MPI-communication spike unrelated to this change, `0.030s ->
1.197s` between the two runs on an otherwise-identical code path), not
a real regression. Taking the reproducible second run:

| | Stage 1 epoch | Stage 2 epoch |
|---|---|---|
| GPU, before this port ("short" NN port only) | `14.27s` | `127.2s` |
| **GPU, after this port** | `14.01s` | **`125.8s`** |

A real but modest **~1-2%** win on both stages -- far smaller than the
`"short"` case's `~14.8%`, consistent with what was flagged when this
follow-up was scoped ("a separate, smaller opportunity"). Most likely
explanation: unlike `"short"` (called once per force-training
candidate, i.e. very frequently), `"elec"`'s forward pass with
`derivatives=true` only runs once per structure per `hasAMatrix`
validity window -- a much lower call volume for the GPU dispatch's
fixed per-call overhead (the `hasGpuCompatibleArchitecture()` sweep,
topology/`G`-array construction, kernel launch) to amortize against.
Still a net positive with no measured downside, so kept rather than
reverted -- unlike the batching attempt and `dAdrQ`-caching attempt
elsewhere in this file, this one didn't hit a wall, it just had less
room to matter.

**Net effect of both `calculateAtomicNeuralNetworks()` widenings
together** (`"short"` + `"elec"`, full scale): stage 2 `149.3s ->
125.8s` (`~15.7%` combined, on top of the short-range-loop port's own
`8.8x`), stage 1 essentially unchanged (`14.15s -> 14.01s`, this
dataset's stage-1 cost is dominated by the already-GPU-ported
`chargeEquilibration()`/`GpuQeqSolver` solve, not NN forward passes).
**GPU is now `5.49x` faster than CPU-112 on stage 2** (`691.2s` vs.
`125.8s`).

### Follow-up: one more fresh full-scale profile with all three ports together, to confirm the numbers hold and see what's left

Re-ran the same fine-grained `Stopwatch`-bracket profiling one more
time, this time at the fully-committed state (short-range force loop +
`"short"` NN + `"elec"` NN, all three ports from this session), on the
full 1254-structure dataset in a single job -- both to confirm the
incrementally-measured numbers reproduce together (not just
individually) and to see the current `F_err` composition in one place.

| | Stage 1 epoch | Stage 2 epoch |
|---|---|---|
| CPU, 112 cores (dcgp) | `9.44s` | `691.2s` |
| GPU, 4xA100, 32 ranks (MPS), all three ports | `14.07s` | `126.5s` |

Matches the incrementally-committed numbers closely (`125.8s` /
`14.01s` from the last two commits individually) -- **`5.46x` faster
than CPU-112 on stage 2**, consistent with the `5.49x` reported above
to within normal run-to-run variance.

`F_err` composition, freshly measured (not inferred from earlier,
now-stale subset-60 numbers):

| Component | Share of `F_err` | Status |
|---|---|---|
| `calculateDFdc` GPU dispatch | 60.3% | GPU-ported, at its `task_batch_size_force=1` ceiling |
| `calculateForces()` | 28.6% | GPU-ported (both terms), capped by the `dAdrQ` re-upload cost |
| elec-NN forward + `chargeEquilibration` gate | 6.5% | GPU-ported |
| `calculateDQdr` | 2.9% | GPU-ported, negligible |
| `"short"` NN forward (PART 1 candidate scoring) | 1.8% | GPU-ported |

`calculateDFdc` and `calculateForces()` together are now `~89%` of
`F_err`, and **both are already at documented ceilings** -- `dfdc`'s
batching wall (`task_batch_size_force=1`, a training-methodology
change, out of scope) and `calculateForces()`'s `dAdrQ` re-upload cost
(the caching fix for this was attempted, caught a real correctness bug
via cross-checking, and was deliberately reverted rather than shipped
uncertain -- still true, still unfixed). Everything else this session
found and ported is now under `7%` combined. Further stage-2 GPU work
from here would mean revisiting one of those two known walls
specifically (a `Structure`-level generation-counter for `dAdrQ`
staleness, or a deliberate, separately-scoped decision to raise
`task_batch_size_force`) rather than another "found an unported
function" pass -- this session's low-risk, high-value gaps of that
kind appear to be exhausted.

### Follow-up: the "ceilings" were both bundling something real underneath -- collectDGdxia()'s O(numAtoms x numNeighbors) CPU scan, ~27% more stage-2 win, no batching wall involved

Asked to double-check the two "ceilings" above rather than accept them
at face value. Good call -- both turned out to bundle a genuinely
unrelated, addressable cost together with the piece that actually was
capped. A direct split (temporary `Stopwatch` brackets around `dfdc`'s
two halves, and around `calculateForces()`'s short-range/electrostatics
sub-phases) found:

- Within `dfdc` (`59.4%` of `F_err`): `collectDGdxia()` -- called once
  per atom, each atom linearly scanning its own full neighbor list for
  one fixed target atom -- was **`34.0%` of `F_err` on its own, bigger
  than the GPU kernel dispatch it feeds (`25.3%`)**. This has nothing
  to do with `task_batch_size_force`; it never touches the GPU.
- Within `calculateForces()` (`29.9%` of `F_err`): the CPU-side
  `dAdrQ` flatten (rebuilding the flat array from `Atom::dAdrQ` every
  call) was `11.0%` on its own, comparable to the `14.2%` GPU
  re-upload cost the "ceiling" framing had named as the whole story.

`collectDGdxia()`'s pattern -- each atom scanning its own neighbor list
for one fixed target -- is the exact same shape `GpuForces.cu`/
`GpuElecForces.cu` already fixed elsewhere via a precomputed
owner-centric edge list instead of a live per-call scan. Built a CPU-side
(no new device code) equivalent: `Training::collectDGdxiaAllAtoms()`
precomputes, once per structure and cached thereafter (purely
geometric -- doesn't depend on weights or which atom is being
perturbed), a *target-grouped* reverse index (CSR: `dGdxiaTopologyCache`,
keyed by `Structure::index`). A single call for a given perturbed atom
then only visits the edges that actually touch it -- O(edges touching
that atom), not O(numAtoms x numNeighbors) -- and fills every atom's
`dGdxia` array in one pass instead of `numAtoms` separate calls.

**Verified live** (`N2P2_DGDXIA_XCHECK`): unlike every GPU port in this
file, this rewrite does the *exact same arithmetic in the exact same
order* as the original (each owner atom's accumulation is independent
of every other, so cross-owner reordering can't matter, and the
target-grouped edges for one atom preserve the original's
owner-then-neighbor-then-symmetry-function visitation order) -- so
the correctness bar here is **bit-identical**, not float-noise
agreement. Confirmed three times across implementation iterations, on
real training data (60-structure subset, 2 epochs, both stages):
**5248 evaluations, `maxAbsDiff=0.000E+00`, zero differing elements,
every single time.**

**A real memory cost, found and fixed, not hidden.** The first working
version stored each edge as `(ownerAtom, ownerTableIndex, dGdr)` --
40 bytes, built via a temporary `vector<vector<DGdxiaEdge>>` bucketed
by target. Measured on the full 1254-structure/32-rank dataset:
cgroup peak RSS **197GB -> 324GB (+127GB)**, uncomfortably close to
the 400GB node budget. First fix attempt (two-pass exact-size CSR
build, no temporary per-target vectors, `uint32_t` fields instead of
`size_t`: 32 bytes/edge) only brought it to **309GB** -- the
hypothesis that construction overhead was the dominant cost was
*wrong*; the temporary vectors were already being freed correctly, and
the real driver was simply the size of the cached data itself (each
630-atom structure's edge list is genuinely large, confirmed by this
measurement, not assumed). Second fix: store a *reference* into data
already resident in `Atom`/`Structure` (`ownerAtom`, `neighborSlotIndex`,
`dGdrIndex` -- 12 bytes, confirmed via a standalone `sizeof()` check)
instead of a copy of `dGdr`/the table index, re-deriving both from
already-resident storage (`Element::getSymmetryFunctionTable()`, an
inline getter with no computation) at read time. This brought peak RSS
to **272GB** -- smaller than the naive 2.67x-from-struct-size
prediction would suggest (some other overhead remains, not fully
explained), but a real, comfortable reduction from the original
`324GB`, and confirmed bit-identical all over again after the rewrite.

**Full-scale result** (clean run, no cross-check overhead):

| | Stage 1 epoch | Stage 2 epoch | Peak RSS (cgroup, whole job) |
|---|---|---|---|
| GPU, before this fix | `14.15s` | `126.5s` | not separately measured |
| **GPU, after this fix** | `14.13s` (unchanged, expected) | **`91.81s`** | `272GB` |

**~27.4%** further stage-2 speedup (`126.5s -> 91.81s`), on top of
everything else this session already shipped. Stage 1 is unaffected,
exactly as expected (`collectDGdxiaAllAtoms()` is only reached from the
`"force"` branch's PART 2 Jacobian assembly, which stage 1 never runs).
**GPU is now `7.53x` faster than CPU-112 on stage 2** (`691.2s` vs.
`91.81s`), up from `5.46x` before this fix -- and up from `1.90x`
*slower* at the very start of this whole profiling pass.

All temporary instrumentation (`Stopwatch` brackets, `MODE_PROF_CEILING`,
`N2P2_DGDXIA_XCHECK`) reverted before commit; the shipped change is
`collectDGdxiaAllAtoms()`, `DGdxiaEdge`, and `dGdxiaTopologyCache` (all
new, in `Training.h`/`.cpp`) plus the one call-site swap in
`Training::update()`'s `"force"` branch. No GPU code at all -- same
category as the stage-1 Jacobian-assembly reassociation fix earlier in
this file: a real algorithmic fix that happened to need no CUDA.

### Follow-up: stage 1 double-checked -- found a real unaccelerated loop, but it's not the dominant cost; the two already-GPU-ported pieces still are

Stage 2 is now clearly faster than CPU-112 (`7.53x`), but stage 1
remains the other way around (`14.13s` GPU vs. `9.44s` CPU-112,
`~1.5x` *slower*) -- asked to double-check the code for stage 1 the
same way, rather than accept that as a given.

**Code review found one real, genuine gap**: `Training::update()`'s
`"charge"` branch, PART 2 (`Training.cpp`, the loop building `dChidc`)
computes every atom's `chi` forward pass *and* its full weight-Jacobian
(`NeuralNetwork::calculateDEdc()`) one atom at a time through the plain
CPU `propagate()`/`calculateDEdc()` path -- never touches the GPU.
Structurally identical to `calculateDFdc`'s pattern (forward +
weight-Jacobian per atom), which this file already GPU-ported for
forces. Unlike that fix, though, this one can't reuse the existing
production dispatch as-is: `gpuNnEnergyDEdcSum()` only returns the
atom-*summed* result (fine for energy training, which only ever needs
the sum), but stage 1's charge Jacobian needs each atom's `dChidc`
*individually* -- each gets scaled by a per-atom weight `Sk` that isn't
known until *after* this loop runs (it depends on the charge-
equilibration solve, which itself depends on this loop's `chi`
output). The underlying batched math for a genuine per-atom
(non-summed) version already exists and was validated standalone
(`gpu/gemm/nn_dedc_gemm_test.cu`) -- it was just never wired into
`GpuNeuralNetwork.h/.cu` as a callable dispatch, because until now
nothing needed per-atom results. Real opportunity, but a different
risk/effort class than today's earlier fixes: needs genuinely new
device code (a per-atom dEdc variant), not just widening an existing
dispatch's condition -- though with much lower correctness risk than
e.g. `dAdrQ` caching, since it's a pure batched computation, not
stateful caching.

**Measured its actual current size before deciding whether it's worth
that effort** (fresh `Stopwatch` brackets, full 1254-structure
dataset, 3 epochs) -- and the result reframes the priority:

| Bracket | s/epoch | % of epoch | % of `Q_err` | GPU status |
|---|---|---|---|---|
| `part1charge` (PART 1 elec-NN forward, `calculateAtomicNeuralNetworks("elec")`) | `3.66s` | `26.5%` | `38.2%` | **already GPU-ported** (this session) |
| `qeq` (`chargeEquilibration()`) | `3.61s` | `26.2%` | `37.7%` | **already GPU-ported** (`GpuQeqSolver`, prior session) |
| `fwd` (the loop described above) | `2.00s` | `14.5%` | `20.9%` | **not GPU-ported** |
| `dq` (`calculateDQdChi`/`calculateDQdJ`) | `0.21s` | `1.5%` | `2.2%` | already GPU-ported |
| `jac` (Jacobian assembly) | `0.10s` | `0.7%` | `1.0%` | CPU-optimized, already small |

(Sum `9.57s` matches `Q_err`'s measured `9.574s/epoch` almost exactly
-- fully accounted for, nothing else hiding in stage 1's train phase.)

**The real finding here isn't the unaccelerated loop -- it's that the
two *already-GPU-ported* pieces (`part1charge` + `qeq`, `76%` of
`Q_err` combined) are still the two largest costs**, bigger than the
genuinely-unaccelerated `fwd` loop. This is the same shape of result
`calculateDFdc`'s original port and `GpuElecForces` both hit earlier in
this file: a real GPU port that's real but capped, most likely by the
same cause already documented there (many small per-candidate GPU
dispatch calls under 32-way MPS contention paying transfer/launch
overhead disproportionate to the work per call) -- not confirmed by a
fresh sub-split of `part1charge`/`qeq` themselves yet, so stated as the
leading hypothesis, not a verified conclusion.

**Not yet investigated further or implemented -- stopping here to
write this down.** Fair next steps, in rough order of how well-scoped
they are: (1) split `qeq`/`part1charge` the same way `calculateForces()`
and `dfdc` were split earlier in this file, to confirm whether MPS
per-call overhead is really the cause before assuming it; (2) if so,
the fix would likely mean reducing per-call count/overhead rather than
porting more code, a different kind of problem than everything fixed
in this stage-2 pass; (3) the `fwd` loop's per-atom dEdc port remains
a real, independently-worthwhile `~14.5%`-of-epoch opportunity
regardless of what (1)/(2) find, using the already-validated
`nn_dedc_gemm_test.cu` math as a starting point.

### Follow-up: did (1) -- the MPS-contention hypothesis was WRONG for `chargeEquilibration()`, right for the elec-NN forward call, and points at a real, unmeasured lever

Split both already-GPU-ported functions into their CPU-side and
GPU-dispatch phases directly (temporary `Stopwatch` brackets, full
1254-structure dataset, 3 epochs), rather than trust the "MPS
contention" guess from the entry above.

**`chargeEquilibration()` -> `Structure::calculateElectrostaticEnergy()`**
splits into three phases: the CPU-side dense `(numAtoms+1)x(numAtoms+1)`
matrix assembly (Ewald real+reciprocal-space summation for this
periodic dataset, `Structure.cpp:609-807`, never touched by any GPU
work), the actually-GPU-ported `gpuQeqFactorize`/`gpuQeqSolve` call,
and the final charge-assignment loop:

| Phase | s/epoch | % of function | GPU status |
|---|---|---|---|
| matrix assembly (Ewald sum) | `11.35s` | `95.1%` | **never GPU-ported** |
| `gpuQeqFactorize`/`gpuQeqSolve` | `0.58s` | `4.9%` | GPU-ported |
| charge assignment | `~0s` | `0.0%` | n/a, trivial |

**The MPS-contention hypothesis is wrong here.** The piece that's
actually GPU-ported (the factorize+solve) is cheap -- `19.6x` cheaper
than the CPU-side matrix assembly it's bundled with. `GpuQeqSolver`
did exactly what it was built for; the "ceiling" was never there. This
is the same shape of surprise as `calculateForces()`'s `dAdrQ` flatten
and `dfdc`'s `collectDGdxia()` earlier in this file: a "GPU-ported"
function whose *un-ported* CPU setup dominates. Note this split
measures *every* call to `calculateElectrostaticEnergy()`, including
the ones from `calculateErrorEpoch()`'s unconditional per-structure
pass (not just `Training::update()`'s PART 1/2, which only account for
part of this total) -- `calculateErrorEpoch()`'s own contribution isn't
separately isolated yet.

**The elec-NN forward call is the opposite story.** Split
`calculateAtomicNeuralNetworks("elec")`'s GPU branch into the CPU-side
`G` array construction vs. the `gpuNnForwardDEdG` dispatch call itself:

| Phase | s/epoch | % of function |
|---|---|---|
| CPU `G` array build | `0.011s` | `2.4%` |
| `gpuNnForwardDEdG` dispatch | `0.468s` | `97.6%` |

Here the GPU call genuinely dominates its own function (`41.3x` over
the CPU build) -- consistent with the per-call transfer/launch
overhead under 32-way MPS contention already documented for
`calculateDFdc`/`GpuElecForces` elsewhere in this file. But this
function's total (`~0.48s/epoch`) is small next to `chargeEquilibration`'s
matrix assembly (`11.35s/epoch`) -- it was never the main story.

**Bottom line**: the real, still-unaddressed lever in stage 1 is
`Structure::calculateElectrostaticEnergy()`'s CPU-only Ewald matrix
assembly -- an O(numAtoms^2 x k-vectors) real+reciprocal-space sum,
never GPU-accelerated, currently ~`11.35s` of every training epoch on
this rank (dwarfing everything else measured in this stage-1
investigation, including the `fwd`-loop gap found earlier). Porting it
would need genuinely new device code (a dense-matrix-assembly kernel,
not a dispatch widening) -- same effort/risk class as the `fwd`-loop
`calculateDEdc` port, but larger in magnitude. Not yet scoped or
started; `calculateErrorEpoch()`'s separate contribution to this same
cost is also not yet isolated. Both are natural next steps if this
gets picked up.

**Is this only a stage-1 cost? No, but the call pattern differs a lot
between the two stages** (checked directly against every
`chargeEquilibration()` call site in `Training.cpp`):

- **Stage 1** (`"charge"` branch): called *unconditionally*, twice per
  candidate, no caching at all -- `Training.cpp:2437` (PART 1
  trial-loop scoring) and `Training.cpp:2970` (PART 2 Jacobian
  assembly). Neither is gated by `hasAMatrix`/`hasCharges`. Every
  single training candidate rebuilds the full Ewald matrix from
  scratch, twice.
- **Stage 2** (`"energy"`/`"force"` branches): called only when the
  cached electrostatics state is stale -- `Training.cpp:2374`/`2407`
  (PART 1) and `2580`/`2721` (PART 2), all gated by
  `if (!s.hasCharges)` or `if (!s.hasAMatrix)`. Force training reuses
  the same structure across several sub-candidates before a weight
  update (the `useSubCandidates` grouping already relevant to the
  `calculateForces()` redundancy fix earlier in this file), so stage 2
  hits this function far less often per candidate processed -- most
  calls find `hasAMatrix` already `true` and skip past it.
- `calculateErrorEpoch()` (the separate error-evaluation pass) also
  calls it for every structure, every epoch -- unconditionally for
  stage 1 (`Training.cpp:1436`), gated for stage 2 (`Training.cpp:1440`).

So a fix here would help **both stages** (this is the same
`AConstrained`/`GpuQeqSolver` machinery already documented above as
shared across both), but stage 1's exposure is direct and predictable
(unconditional, every candidate) while stage 2's benefit scales with
how often `hasAMatrix` actually gets invalidated during training --
still real, just proportionally smaller than stage 1's.

**Status: analysis only, nothing implemented.** This and the two
entries above it are a complete, self-contained writeup of the
investigation (what's expensive, why, how it could be fixed, and where
else the fix would matter) for whoever picks this up next.

### Follow-up: implemented the CPU-first pass on the Ewald matrix assembly -- the reciprocal-sum hypothesis was wrong, but it led to a real redundant-call bug worth ~16-20% of stage 1

Went CPU-first on `calculateElectrostaticEnergy()`'s matrix assembly,
per the earlier feasibility analysis: the reciprocal-space double loop
(`A(i,j) += 2*coeff*cos(k.(ri-rj))/fourPiEps`, summed over all
k-vectors for every atom pair) has an exact algebraic reformulation via
`cos(a-b) = cos(a)cos(b) + sin(a)sin(b)`, turning an `O(numAtoms^2 x
numKvectors)` sum of `cos()` calls into two GEMMs (`Cw.Cwᵀ + Sw.Swᵀ`
from precomputed per-atom `cos(k.r)`/`sin(k.r)` projections, weighted
by `sqrt(2*coeff/fourPiEps)`).

**Implemented and validated bit-exact, but caught a real bug on the
way.** Cross-checked the new GEMM path against the original triple
loop on real `temp/H2O_4G` data (env-gated, reverted before commit):
the first version's `relDiff` was `~57x`, not floating-point noise.
Root cause: the *removed* loop's `A(j,i) = A(i,j)` mirror step wasn't
just handling the reciprocal term -- it was also propagating the
real-space erfc contribution (which the atom loop only ever writes
into the upper triangle, `j >= i`) into the lower triangle. The
replacement has to rank-update the *same* (upper) triangle the
real-space loop already populated, then mirror at the end -- rank-
updating the lower triangle first (the more "natural" Eigen default)
silently drops the real-space term from half the matrix. Once fixed,
`relDiff ~5.7e-9` against the direct sum (pure roundoff).

**The reciprocal sum turned out not to be the lever.** Split
`calculateElectrostaticEnergy()`'s periodic branch into three
brackets (diagonal+real-space, cos/sin projection build, GEMM) with
temporary `Stopwatch`s, full 1254-structure dataset, 1 epoch:

| Phase | s/epoch (per rank) |
|---|---|
| diagonal + real-space erfc loop | `~6.0-6.4s` |
| cos/sin projection build | `~0.009s` |
| GEMM (both ranks, old or new) | `~0.08-0.2s` |

`numK` for this dataset's Ewald parameters is only **3** -- the
reciprocal sum was always cheap here (the `eta` for this system pushes
essentially all the weight into real space, needing a correspondingly
large `rCut`), so the "port the reciprocal sum" hypothesis from the
earlier analysis undershot the real cost by roughly two orders of
magnitude. The GEMM fix is still correct, still a real (if small)
win, and would matter more for a system with a larger k-space grid --
kept, not reverted.

**Instrumented call count + neighbor-pair count to find the real
lever**, since real-space cost = calls x neighbor-pairs-per-call x
erfc-cost: `148` calls/rank/epoch, `~202M` neighbor-pairs/rank/epoch
(`~1.37M` per call -- large, but genuinely neighbor-list-bounded, not
`O(numAtoms^2)`), `~15ns`/erfc() call. This is real, unavoidable-by-
caching compute for 148 *distinct* calls -- except a look at
`Training.cpp` showed not all 148 are actually distinct work.

**Found the actual redundant-call bug -- same shape as the
`calculateForces()` fix earlier in this file.** Stage 1's PART 1
trial loop (`Training.cpp`, `SM_THRESHOLD` candidate scoring) calls
`chargeEquilibration(s, false)` per trial, then either `break`s
immediately if the candidate clears the RMSE threshold (skipping
`s.clearElectrostatics()` entirely) or falls through to
`s.clearElectrostatics()` and tries the next candidate. So for the
*winning* candidate specifically, `s.hasAMatrix` is still `true` when
PART 2 runs -- yet PART 2's `chargeEquilibration(s, false)` call was
unconditional, unlike every stage-2 call site in this same file (all
gated by `hasAMatrix`/`hasCharges`). Nothing between PART 1's call and
PART 2's runs changes the weights or geometry (PART 2 does recompute
`ak.chi` via a direct NN forward pass, needed regardless for `dChidc`,
but under unchanged weights/inputs it's numerically identical to
PART 1's) -- so PART 2 was reliably re-running the entire ~6s/epoch
real-space+reciprocal matrix assembly for a structure whose `A`/`Q`/
`lambda` were already valid. Fix: `if (!s.hasAMatrix)
chargeEquilibration(s, false);`, mirroring the exact gating stage 2
already uses.

**Validated bit-exact, not just plausible.** Temporary env-gated
cross-check: whenever `s.hasAMatrix` was `true` at PART 2, force the
recompute anyway and diff the resulting charges against the cached
ones. `maxChargeDiff` / `lambdaDiff` = `0.0` across all `112` cache-hit
occurrences over 2 epochs on a 60-structure subset -- exact
determinism, as expected from a deterministic NN forward pass under
unchanged weights.

**Combined effect, full 1254-structure dataset, 6 epochs:** steady-
state stage-1 epoch time `~22.6s -> ~19s` (`~16-20%`), `Q_err` bucket
`~15.7s -> ~10.4s` (`~34%`, since that's where most `chargeEquilibration()`
calls live). Multi-epoch `learning-curve.out.stage-1` matches the
baseline closely -- epochs 0-1 bit-identical, epochs 2+ diverge only
at the `~1e-4` relative level, consistent with the reciprocal GEMM's
different floating-point summation order (same caveat already
documented for `atomicAdd` reordering elsewhere in this file, not a
correctness regression).

**What's left.** The real-space erfc loop itself (`~6s/epoch/rank`,
`~202M` neighbor-pairs/rank/epoch even after the redundant-call fix)
is now the clear, unaddressed dominant cost -- not the reciprocal sum
this investigation started from. It's neighbor-list-bounded (not
`O(numAtoms^2)`), which makes it structurally similar to `GpuForces.cu`'s
owner-centric edge-list pattern: a natural next GPU candidate if this
gets picked up again, now correctly scoped by measurement rather than
by the original (wrong) hypothesis. Not started.

### Follow-up: ported the real-space erfc loop to GPU (`GpuEwaldReal`) -- correctly scoped this time, and the payoff shows it

Followed `GpuForces.cu`'s exact template, since the real-space sum has
the identical shape: an owner-centric edge list (owner atom, target
atom, distance, element pair), purely geometric (atom positions never
change during training) so uploaded to the GPU exactly **once** per
structure and cached there keyed by `Structure::index`, then a
one-thread-per-edge kernel that `atomicAdd`s each edge's
`(erfc(rij/sqrt2eta) - erfc(rij/gammaSqrt2(ei,ej))) / (rij*fourPiEps)`
contribution into a dense `numAtoms x numAtoms` output buffer.
`atomicAdd` is required, not optional here (same as `GpuForces.cu`'s
pair-force kernel): this dataset's real-space cutoff pulls in multiple
periodic images of the same neighbor (consistent with `numK=3` --
large `eta`, real-space-dominated regime, see above), so several edges
can legitimately target the same `(i, j)` output slot. Only
`gammaSqrt2` (depends on the trainable per-element `Qsigma`) is
re-uploaded every call, mirroring `GpuForces.h`'s "topology once,
small per-call data every time" convention exactly.

**One correctness wrinkle caught before it mattered**: `GpuEwaldReal`'s
contract is row-major `gammaSqrt2`, but Eigen's `MatrixXd::data()` is
column-major. `gammaSqrt2` happens to be symmetric (`gammaSqrt2(i,j) ==
gammaSqrt2(j,i)` by construction in `Mode::chargeEquilibration()`), so
passing the column-major buffer directly would have happened to read
back the *correct* value anyway -- a coincidence not worth relying on.
Built an explicit small row-major copy instead (tiny, `numElements^2`,
negligible cost) rather than leave a landmine for the day `gammaSqrt2`
stops being symmetric.

**Validated bit-exact.** Same env-gated cross-check pattern as the
rest of this file: computed the CPU reference (still compiled in even
under `N2P2_GPU`, gated by a runtime env check rather than excluded
via `#ifndef`) alongside the GPU result on real `temp/H2O_4G` data.
`relDiff ~4.5e-16` -- pure floating-point roundoff between the CPU's
and CUDA's `erfc()` implementations, on the first try (unlike the
reciprocal GEMM earlier in this file, no triangle-placement or other
bug this time -- the math is a direct one-to-one transcription of the
CPU loop, with no reformulation to get subtly wrong).

**Full-scale result, 1254-structure dataset:** no GPU errors, no OOM
(host `MaxRSS ~198GB` of a `400GB` budget -- the edge-list cache's
memory footprint was checked, not assumed, given this file's
`collectDGdxiaAllAtoms` memory-blowup history earlier). Full-scale
stage-1 epoch time: `~19s` (CPU, both earlier fixes in this
investigation applied) `-> ~4.4s` -- a further **`~4.3x`**. Combined
with this investigation's two CPU-side fixes (`~22.6s -> ~19s`), stage
1's epoch time has gone `~22.6s -> ~4.4s` end to end
(**`~5.1x`**) since this Ewald-matrix-assembly investigation started.
Multi-epoch `learning-curve.out.stage-1` matches the baseline closely,
same floating-point-reassociation-level caveat as the reciprocal GEMM
fix (different `atomicAdd`/reduction order, not a correctness
regression).

**Status: implemented, validated, and merged.** The original
hypothesis this whole investigation started from (port the reciprocal
sum) turned out to be a red herring for this dataset; measuring before
committing to it is what found the real lever twice in a row here (the
redundant `chargeEquilibration()` call, then this). Stage 1's Ewald
matrix assembly is now GPU-accelerated end to end -- both the
reciprocal sum (GEMM) and the real-space sum (`GpuEwaldReal`) -- with
only the trivial `O(numAtoms)` diagonal/RHS setup and the already-fast
`GpuQeqSolver` factorize/solve left on their existing paths.

### Follow-up: refreshed the CPU-112 comparison -- stage 1 flips too, same shape as every stage-2 flip in this file

Every number in this file's stage-1-vs-CPU-112 story up to now
(`14.13s` GPU vs `9.44s` CPU-112, GPU `~1.5x` *slower*) predates this
whole investigation. That `9.44s` CPU-112 baseline is now stale in a
specific way worth calling out: the two CPU-side fixes above
(reciprocal GEMM, redundant-call skip) are pure algorithmic changes
with no GPU dependency, so they *also* speed up a CPU-only run --
comparing today's GPU number against yesterday's CPU-112 number would
overstate the GPU win. Rebuilt CPU-only (no `GPU=1`) at current HEAD
and reran the exact same 112-rank/full-node methodology used for
every other CPU-112 number in this file (`dcgp_usr_prod`, `112` ranks,
full 1254-structure dataset):

| | Stage 1 epoch |
|---|---|
| CPU, 112 cores (dcgp), pre-investigation | `9.44s` |
| **CPU, 112 cores (dcgp), current HEAD (both CPU fixes)** | `~7.6-10.5s` (`Q_com`-noisy, avg `~8.8s`) |
| **GPU, 4xA100, 32 ranks (MPS), current HEAD** | `~4.4s` |

The CPU-112 number barely moved and got noisier, not cleanly faster --
unlike the clean `22.6s -> 19s` win measured earlier in this file at
32 ranks. `Q_com` (communication time) swung `0.11s` to `3.05s` across
3 epochs on this run, while `Q_err` (the actual compute) stayed tight
at `4.24-4.26s` -- consistent with the interpretation that 112-way MPI
synchronization overhead dominates enough at full-node scale to mask
the compute-side win visible at 32 ranks. Reported as a range rather
than a cherry-picked single epoch, matching this file's "rerun before
trusting a single number" practice.

**Even against the noisier end of that range, GPU wins.** `~4.4s` vs
`~7.6-10.5s` is **`~1.7x-2.4x` faster** (`~2.0x` against the `8.8s`
average) -- the same *shape* of result as every stage-2 flip in this
file (`GpuElecForces`, the short-range force loop, the "short"/"elec"
NN forward ports, `collectDGdxiaAllAtoms`): GPU was behind CPU-112 on
stage 1 for this whole file until this investigation, and now isn't.

**Stage 2's CPU-112 comparison (`691.2s`, GPU `7.53x` faster) is
carried forward unchanged, not re-verified this session.** It's not
entirely unaffected in principle -- stage 2 also calls
`chargeEquilibration()` (gated by `hasAMatrix`/`hasCharges`, see the
"Is this only a stage-1 cost?" entry above), so both of today's CPU
fixes and the `GpuEwaldReal` port apply there too whenever that gate
lets a call through. But stage 2's dominant costs are
`calculateDFdc`/`calculateForces`/the NN forward passes/
`collectDGdxiaAllAtoms` -- charge equilibration was never more than a
minor contributor there, gated to run far less often than stage 1's
unconditional-every-candidate calls. Expected to move slightly, not
enough to be worth a dedicated rerun right now.

**Net picture, both stages, against a full 112-core CPU node:**

| | Stage 1 | Stage 2 |
|---|---|---|
| CPU-112 (dcgp, full node) | `~7.6-10.5s`/epoch | `691.2s`/epoch |
| GPU (4xA100, 32 ranks, MPS) | `~4.4s`/epoch | `91.81s`/epoch |
| GPU vs CPU-112 | **`~2.0x` faster** | **`7.53x` faster** |

GPU is now the clearly faster option on **both** stages of `HDNNP_4G`
training at this problem size, on the single-node-multi-GPU deployment
shape (4xA100, one Booster node, one shared MPS daemon) this whole
profiling effort validated as correct from the start. Stage 1's flip
is narrower than stage 2's (`~2.0x` vs `7.53x`) -- consistent with
stage 1 being the smaller absolute cost of the two throughout this
file, and with today's fix being control-flow/algorithmic wins on top
of an already fairly light workload, not a from-scratch GPU port of a
previously-dominant, previously-CPU-only cost the way stage 2's
`GpuForces.cu` port was.

### Follow-up: fresh stage-1 reprofile -- the next lever, and it's a big one

Asked "what else could we optimize" after the `~5.1x` stage-1 result
above -- rather than guess from the pre-investigation percentages
(which are now stale, since `chargeEquilibration` collapsed from the
dominant cost to a minor one), re-profiled from scratch with the same
temporary-`Stopwatch`-bracket methodology used throughout this file,
splitting `Training::update("charge")`'s PART 1/PART 2 into six named
brackets. Full 1254-structure dataset, 3 epochs, current HEAD (GPU
build):

| Bracket | s/epoch (per rank) | % of `Q_err` | GPU status |
|---|---|---|---|
| `fwd` (PART 2's per-atom `chi`/`dChidc` forward + weight-Jacobian loop) | `~2.04s` | **`~62%`** | **not GPU-ported** |
| `part1qeq` (PART 1's `chargeEquilibration()`, all trials) | `~0.84s` | `~26%` | GPU-ported (real-space + solve), CPU (reciprocal GEMM) |
| `dq` (`calculateDQdChi`/`calculateDQdJ`) | `~0.26s` | `~8%` | GPU-ported |
| `jac` (Jacobian summation) | `~0.10s` | `~3%` | CPU-optimized, already small |
| `part1nn` (PART 1's `calculateAtomicNeuralNetworks`) | `~0.03s` | `~1%` | GPU-ported |
| `part2qeq` (PART 2's now-conditional `chargeEquilibration()`) | `~0.00002s` | `~0%` | confirms the redundant-call fix above works as intended |

(Sum `~3.27s/epoch` matches `Q_err`'s measured `~3.27s/epoch` almost
exactly -- fully accounted for.)

**`fwd` is now the dominant remaining cost by a wide margin** --
`~62%` of `Q_err`, roughly `~46%` of the whole stage-1 epoch. This is
`Training.cpp`'s PART 2 computing, one atom at a time on CPU, the
electronegativity NN's forward pass (`chi`) and its full per-atom
weight-Jacobian (`NeuralNetwork::calculateDEdc()`, needed for
`dChidc`) -- never GPU-ported. It's structurally identical to
`calculateDFdc`'s pattern (forward + weight-Jacobian per atom), which
this file already GPU-ported for stage 2 at `~47.6%` of `F_err` back
when it was profiled -- the same underlying batched-GEMM math
(`gpu/gemm/nn_dedc_gemm_test.cu`) should apply here directly, no new
device-code design needed, just a new dispatch function for the
"elec" NN mirroring `calculateDFdc`'s.

`part1qeq`'s `~26%` is worth noting too: it's the PART-1 trial-loop's
repeated `chargeEquilibration()` calls (`SM_THRESHOLD`'s rejected
trials, not just the accepted candidate -- `part2qeq`'s near-zero
number above confirms the redundant-call fix is working), and it's
*already* GPU-accelerated end to end (real-space via `GpuEwaldReal`,
solve via `GpuQeqSolver`). Its remaining cost is most likely per-call
GPU dispatch/transfer overhead under 32-way MPS contention -- the same
shape this file has documented for `calculateDFdc` and others -- not
unaccelerated compute. A secondary, smaller-payoff target if `fwd`
gets addressed first (which it should be, given the size difference).

**Status: profiled, not yet implemented.** `fwd`'s port looks like a
well-scoped, comparatively low-risk next step given the directly
reusable validated math, but a real GPU-code change nonetheless --
next action if this gets picked up. All temporary instrumentation
(`STAGE1_REPROFILE`, the six `Stopwatch` brackets) reverted, nothing
shipped from this entry beyond the write-up.

### Follow-up: implemented the `fwd` port -- correct, individually fast, but a net regression at full 32-rank MPS scale; reverted

Built `gpuNnChargeDEdc()` (`GpuNeuralNetwork.h/.cu`), mirroring
`gpuNnEnergyDEdcSum()`'s forward-pass GEMM pipeline exactly but keeping
every atom's `dEdc` separate instead of contracting the atom axis --
the same per-atom batched-outer-product math `gpu/gemm/nn_dedc_gemm_test.cu`
already validated standalone (batched `cublasDgemmStridedBatched`,
`k=1`, for `dE/dW1`/`dE/dW2`, plus a small packing kernel for the
`dE/db1`/`dE/db2`/`dE/dW3`/`dE/db3` alias pieces already sitting in
the forward pass's own intermediates -- see that file's header comment
for the derivation). Wired into `Training.cpp`'s PART 2 `"charge"`
branch, grouping atoms by element and dispatching one batched call per
element per structure, mirroring `calculateAtomicNeuralNetworks()`'s
`"elec"`-branch GPU dispatch in `Mode.cpp` -- but returning the
per-atom breakdown instead of a sum, since PART 2 weights each atom's
`dChidc` differently (by `dQdChi`) before summing into the Jacobian,
unlike `"energy"` training which only ever needs the total.

**Validated bit-exact, no correctness issues at any point.** Cross-checked
`chi`/`dEdc` against the CPU reference on real data (env-gated, forced
recompute after the GPU result, diffed, then restored): `relDiff` at
machine precision (`maxChiDiff ~4.4e-15`, `maxDEdcDiff ~8.9e-16`) across
112 occurrences, 60-structure subset, 2 epochs -- correct on the first
attempt, no triangle-placement-style bug this time.

**But full-scale timing was a real, reproducible regression, not a
win.** Confirmed via multiple controlled back-to-back A/B runs (same
job, same node, before/after built and run sequentially, ruling out
cluster-load noise): full 1254-structure dataset, stage-1 steady-state
epoch time went `~4.39s -> ~4.76-4.78s` -- a **`6-9%` slowdown**, not
the expected win, despite the port being individually fast and exactly
correct.

**Diagnosed by bracketing every remaining component of
`Training::update("charge")`'s `HDNNP_4G` branch, not just the new
code** (temporary `Stopwatch`s on `part1nn`, `part1qeq`, `fwd`'s three
phases, `part2qeq`, `dq`, `jac` -- literally everything PART 1/PART 2
do for this property):

| Bracket | s/epoch (per rank) |
|---|---|
| `part1nn` | `0.026s` |
| `part1qeq` | `0.848s` (unchanged from before -- rules out shared-GPU-contention slowdown of *this* piece specifically) |
| `fwd` prep (CPU-side G/connections build) | `0.070s` |
| `fwd` call (`gpuNnChargeDEdc()` itself) | `0.172s` |
| `fwd` copyback (chi/dChidc into host arrays) | `0.137s` |
| `part2qeq` | `~0.000s` (confirms the earlier redundant-call fix still works) |
| `dq` | `0.242s` |
| `jac` | `0.115s` |
| **Sum** | **`~1.61s`** |
| **Measured `Q_err`** | **`~3.58s`** |
| **Unaccounted gap** | **`~1.97s`** |

Every single piece of compute this branch does is bracketed above, and
they sum to less than half of measured `Q_err`. The missing `~1.97s`
isn't hiding in any code path -- it's the same failure mode this file
already documented for `calculateForces()`'s first GPU pass: a
Stopwatch around a GPU call measures *this rank's* wall time for that
call, but under 32-way MPS sharing across only 4 GPUs, adding more
small per-structure GPU dispatches (now two more per structure per
candidate, on top of everything already contending for the same GPUs)
increases *timing variance across ranks* -- some ranks' calls queue
behind others'. That desync is invisible to any single rank's own
bracket; it shows up as every rank blocking at the next synchronization
point waiting for the slowest one. `Q_com` (the explicit communication
column) stays small throughout because this isn't communication, it's
an implicit stall at a barrier -- real wall-clock cost with no
corresponding line in any per-rank instrumentation.

**Reverted, not shipped.** Unlike `GpuForces.cu`'s first-pass
regression (fixed by caching topology once per structure, cutting
repeat upload cost directly), there's no equivalent fix available
here without a genuinely different design: `gpuNnChargeDEdc()`'s
inputs (the current weights) change every call by construction, so
there's no topology to cache. The plausible next move is batching
*across* candidates/structures into fewer, larger calls (amortizing
per-call MPS contention the way `GpuElecForces`/`GpuForces` amortize
per-call transfer cost) -- but that's new design work, not a quick
follow-up, and not attempted here. `GpuNeuralNetwork.h/.cu`'s
`gpuNnChargeDEdc()`, `GpuNnChargeState`, `outerProductBatched()`, and
`packDEdcKernel()`, plus the `Training.cpp` PART 2 dispatch change,
were all reverted before commit -- this entry is the complete record
for whoever picks this up next, including the exact numbers that ruled
out the two most obvious explanations (a correctness bug, and
part1qeq-specific contention) before landing on the real one.

### Follow-up: generalized the GPU-ported NN kernels beyond tanh -- unlocks real in-repo examples, no depth work yet

Separate task, not part of the 4G electrostatics effort above: while
surveying `examples/` for what else could benefit from this port,
`NeuralNetwork::hasGpuCompatibleArchitecture()` turned out to silently
fall every element's NN forward/backward pass back to CPU-only unless
the network had *exactly* two hidden layers using tanh -- including two
real, already-in-repo datasets (`examples/nnp-train/Cu2S_PBE`,
`examples/nnp-train/QM9`, both `global_activation_short p p l`, i.e.
softplus). `GpuForces` (the force scatter-reduction kernel) already
dispatched for them regardless, since it's gated independently and only
needs `dEdG` as input -- this follow-up is specifically about the NN
forward/Jacobian kernels catching up.

Checked the n2p2 docs and the CPU reference directly: 10 activation
functions are supported (`AF_IDENTITY/TANH/LOGISTIC/SOFTPLUS/RELU/
GAUSSIAN/COS/REVLOGISTIC/EXP/HARMONIC`), no documented cap on hidden-layer
depth or width. Scoped this pass to activation only -- depth (more than
two hidden layers) stays out of scope, a separate and materially bigger
rewrite (the three dispatch functions are hand-unrolled for exactly two
hidden layers, not loop-based).

**Where the tanh assumption actually lived.** Grepped all of
`src/libnnpgpu/` for `AF_TANH`/`tanh(` -- exactly two kernels in the
whole tree hardcoded it (`biasTanhKernel`, forward-only; `biasTanhD2Kernel`,
also needs `d2fdx2` for the force-Jacobian pass), called twice each
(once per hidden layer) inside the three public dispatch functions.
Everything else (the GEMMs, elementwise mul/add, transpose, scale-by-
row/col) was always activation-agnostic. Replaced both with
`biasActivationKernel`/`biasActivationD2Kernel`, generic device helpers
implementing all 10 formulas verbatim from `NeuralNetwork::propagateLayer()`
(including the `EXP_LIMIT=35.0` overflow clamp on LOGISTIC/SOFTPLUS
only -- no clamp added anywhere the CPU reference doesn't have one),
selected via a `switch` on a plain `int activation` kernel parameter
(uniform across every thread in one launch, so this is a predictable
branch, not warp divergence). The per-architecture device buffer cache
(`ArchKey = numIn/numHidden1/numHidden2`) deliberately does *not* get
activation added to its key -- nothing activation-dependent is ever
cached, weights are re-uploaded and activations recomputed every call,
so two elements sharing hidden-layer sizes but using different
activations correctly share cached buffers and just pass different
activation arguments per call.

**Validated kernel-level first**, same discipline as everything else in
this file. Generalized `gpu/gemm/nn_forward_gemm_test.cu` and
`nn_dfdc_gemm_test.cu` (previously tanh-only) to loop over all 10
activations against the real `nnp::NeuralNetwork` CPU class as ground
truth -- not synthetic hand-rolled formulas, the actual class this port
has to match. Softplus used real weight/G magnitudes from
`Cu2S_PBE/scaling.data`; the other 8 (no real in-repo example) used
weight/G ranges drawn from that same real dataset's scale rather than
arbitrary values. All 10 activations PASS to ~1e-14 on both the forward
(`dEdG`) and force-Jacobian (`dFdc`, exercises `d2fdx2`) paths.

**Caught one real numerical artifact along the way, not a port bug.**
`AF_EXP` initially failed catastrophically (`max|E_gemm-E_cpu|` ~1e229).
Root cause: `AF_EXP` has no overflow clamp on the CPU side either
(unlike LOGISTIC/SOFTPLUS's `EXP_LIMIT`) -- with the test's `[-1,1]`
random weight init, two chained unclamped `exp()` layers overflow
toward `double`'s range limit, where GPU and CPU `exp()` diverge at the
ULP level and that gets amplified exponentially. Both implementations
were doing the right thing; the random weights just aren't a
numerically realistic point for this specific activation (no real
trained network would survive to such weights -- the Kalman update
would diverge to NaN long before). Fixed by scaling that one
activation's test weights down (`×0.05`) to a regime a real optimizer
could actually reach, not by adding a clamp neither implementation has.

**End-to-end validation on the real softplus example.** Built CPU and
GPU binaries from the same commit in a scratch worktree
(`wt_activation_cu2s`), ran `examples/nnp-train/Cu2S_PBE` (2G, softplus
hidden layers, 20 structures, 144 atoms/structure, 4 MPI ranks) for its
full 10 configured epochs on both. `learning-curve.out` matches to
~8 significant figures through epoch 5, then diverges the way every
other GPU-vs-CPU comparison in this file already does past one full
epoch (recursive Kalman-filter update, floating-point reduction-order
sensitivity -- not a correctness issue, see the H2O_2G report's
identical footnote), converging to the same order of magnitude by
epoch 10 (`E_test` 1.65E-3 CPU vs 8.93E-4 GPU, both down from 1.11E+1
at epoch 0). Confirms the softplus path is wired correctly, not just
numerically correct in isolation.

**Honest timing note, both directions.** Kernel-level (step above):
all 10 activations landed within 0.048-0.055ms/call in the batched
forward pass, no meaningful difference between tanh and any other
activation -- confirms these are memory-bound elementwise kernels where
the specific transcendental function is a rounding error, exactly as
expected going in. End-to-end on `Cu2S_PBE`, though: GPU total training
time was *slower* than CPU (37.5s vs 12.6s for the 10-epoch run) --
expected and consistent with this file's own repeated finding that
GPU dispatch needs enough problem size to amortize per-call overhead
under MPS; `Cu2S_PBE` (20 structures, 144 atoms/structure, 4 ranks) is
far below the ~1254-structure/630-atom scale this project's real wins
were measured at. This generalization makes the softplus/relu/etc. path
*available* and *correct* for smaller real datasets -- it doesn't by
itself make GPU dispatch worth using at every scale, same caveat as
always in this file.

### Follow-up: ported the LAMMPS interface (2G) to GPU -- turned out to be a link-flag fix, not new code, and the honest performance story has a real twist

New, separate phase (`gpu/lammps_e2e/`, plan at
`soft-percolating-jellyfish.md`): n2p2 ships a LAMMPS pair style
(`pair_style hdnnp`, the ML-HDNNP package) for running MD with
n2p2-trained potentials. Goal: get it GPU-accelerated, 2G first.

**Two findings up front.** (1) LAMMPS's own built-in `PKG_GPU` package
does not cover custom/external pair styles like `hdnnp` -- confirmed via
LAMMPS's own docs -- so building it would add zero acceleration here;
the only real axis is CPU-vs-GPU inside n2p2's own interface code, not
"LAMMPS's GPU build" as a separate thing. (2) `InterfaceLammps`
(`src/libnnpif/LAMMPS/InterfaceLammps.cpp`) is `class InterfaceLammps :
public Mode`, and its `process()` (what `pair_hdnnp::compute()` calls
every timestep) calls `calculateAtomicNeuralNetworks()` directly -- the
exact same GPU-dispatch-gated method the `nnp-train` port already built
and validated. `InterfaceLammps::getForces()` is a pure CPU reduction
over `atom->dEdG`/`atom->dGdr`, already backend-agnostic. So **no new
kernel code was needed** -- porting this was a build/link task: get
LAMMPS's generated `Makefile.lammps-extra` (produced by
`src/libnnpif/makefile`'s `lammps-mf` rule) to also carry
`-lnnpgpu`/CUDA link flags when `GPU=1`, mirroring the existing pattern
in `application/makefile`. Also fixed a real, unrelated gap hit along
the way: `hdnnp_SYSINC` never carried the Eigen include path, so LAMMPS's
own compile of `pair_hdnnp.cpp` failed to find `Eigen/Core` on any
system without Eigen installed system-wide (this cluster included).

**Built LAMMPS from the official stable tarball**
(`download.lammps.org/tars/lammps-stable.tar.gz`, `22Jul2025`) via its
own recommended package/make path (`make yes-ml-hdnnp && make lib-hdnnp
args="-p n2p2_devel" && make mpi`), not n2p2's own discouraged dev-build
script -- the ML-HDNNP package is already upstream in stock LAMMPS,
unmodified, so nothing from `src/interface/LAMMPS/` needed copying in.

**Correctness validated before any performance claim**, same rule as
every prior GPU step here: GPU-linked `lmp_mpi` reproduced a CPU
2000-step H2O_2G run's thermo trajectory (temperature, potential energy)
*exactly* (`0.0` max diff) at every one of 21 printed steps, with
per-atom forces matching to `1e-10` across all dumped frames.

**Honest performance story -- and it depends entirely on what CPU
config you compare against.** A 4-config rank/GPU-count scan on Booster
(2000-step H2O_2G, 630 atoms) found:

| Config | Pair time (2000 steps) | Loop time |
|---|---|---|
| 4 ranks / 1 GPU | 174.4s | 186.4s |
| 8 ranks / 2 GPU | 88.5s | 99.5s |
| 16 ranks / 4 GPU | 45.3s | 55.6s |
| **32 ranks / 4 GPU (8/GPU)** | **25.1s** | **34.6s** |

32/4 (matching the `nnp-train` MPS precedent exactly) won, Comm share
rising with rank count (6%->27%) but never overtaking the Pair-time
gain. So far this looks like the familiar "GPU wins, scale it up"
story -- except a same-hardware CPU comparison at the same 32 ranks
(no GPU) measured **25.45s Pair / 35.06s loop**, essentially tied with
GPU's 25.1s/34.6s. The earlier "2.8x GPU win" (4-rank comparison) was
really "GPU beats a lightly-parallel CPU run," not "GPU beats CPU at
full node utilization."

Full 200000-step production runs confirmed this holds at scale, plus a
third data point requested to separate hardware from rank-count effects:

| Config | Ranks | Loop time | Pair (% total) | Comm (% total) |
|---|---|---|---|---|
| CPU, DCGP | 112 | 4356s (1:12:36) | 2231s (51%) | 2110s (48%) |
| GPU, Booster | 32 (4 GPU, 8/GPU) | 3462s (0:57:43) | 2490s (72%) | 955s (28%) |
| CPU, Booster | 32 (no GPU) | 3510s (0:58:29) | 2503s (71%) | 990s (28%) |

DCGP-112 -- despite 3.5x more ranks than either Booster config -- is the
*slowest* of the three: 630 atoms split 112 ways is badly
over-decomposed, and Comm balloons to 48% of wall-clock, erasing the
extra parallelism (its Pair time alone, 2231s, is genuinely the smallest
of the three, consistent with finer-grained decomposition -- Comm is
what sinks it). On matched Booster hardware, GPU beats CPU by only
**1.4%** (3462s vs 3510s) -- Pair time itself is nearly identical
(2490s vs 2503s, ~0.5% apart). All three runs' thermo statistics agree
(mean temp 299.3-300.1K against the 300K NVT target, mean PotEng
-105.3 to -105.6) -- exact trajectory agreement isn't expected or
checked for at this length (chaotic MD trajectory, same reasoning as
every long-run comparison in this file), just statistical consistency.

**Why the GPU edge is so small here, and the real next target.** The
NN-forward GPU dispatch this reuses is only part of what LAMMPS's "Pair"
bucket measures. `InterfaceLammps::process()` calls
`calculateSymmetryFunctionGroups()` *before*
`calculateAtomicNeuralNetworks()` -- confirmed zero `N2P2_GPU`
references in either symmetry-function evaluation function
(`Mode::calculateSymmetryFunctions`/`calculateSymmetryFunctionGroups`,
~40 `SymFnc*`/`SymGrp*` classes) -- entirely CPU-only, same cost for
both the CPU and GPU `lmp_mpi` builds. `GPU_PORTING_PLAN.md`'s Phase 0
profiling measured this at only ~4.1% of `nnp-train`'s wall-clock and
deprioritized it on that basis -- but that profiling run's `input.nn`
had `memorize_symfunc_results` active, which caches symmetry-function
results across epochs since training revisits the same fixed structures
repeatedly; the 4.1% figure is the *amortized*, near-free-after-epoch-1
cost. LAMMPS/MD has no such luxury -- atoms move every timestep, so
every symmetry function and derivative is recomputed from scratch, in
full, every single step, with nothing to memorize. That's the leading
(not yet profiled) explanation for why CPU and GPU converge to the same
Pair time at 32 ranks: both pay the same unamortized SF cost, and only
a shrinking NN slice actually differs. Real prototype CUDA kernels for
symmetry functions already exist and were never wired into production
(`gpu/soa/symfnc_exprad_soa_test.cu`, `gpu/soa/symfnc_exprad_group_test.cu`,
`gpu/soa/symfnc_expangn_group_test.cu`, `gpu/smoke/symfnc_exprad_test.cu`,
`gpu/smoke/symfnc_family_test.cu`) -- this is the planned next phase
(Phase 5 in the plan doc), starting with a real `perf` profile of the
MD case to confirm the SF-vs-NN-vs-Comm split before implementing
anything, same "measure before assuming" discipline as every phase in
this file.

### Follow-up: DCGP core-count scaling scan (112 is optimal, no interior peak) -- and a `perf` profiling dead end

Two side-quests while Phase 5's real work was queued.

**DCGP scaling**: scanned 4/8/16/32/56/64/112 ranks (2000-step H2O_2G,
`gpu/lammps_e2e/scan_cpu_dcgp_ranks.slurm`) to answer "what's the
optimal core count on DCGP" directly, rather than assuming the full
node. Loop time falls monotonically the entire way -- 527.0s -> 278.5s
-> 152.0s -> 85.5s -> 60.4s -> 54.3s -> 43.6s -- with parallel efficiency
dropping from 95% (8 ranks) to 43% (112 ranks) as Comm share grows from
11% to 49%. No interior optimum in this range: more cores always won on
wall-clock, right up to the physical limit of one node, even though
efficiency kept falling. So 112 (the full node) is the answer, which is
exactly what the already-completed full-scale production run
(job 52496567, 1:12:36) used -- no rerun needed. Worth remembering this
isn't "efficient" in a core-hours sense (8-16 ranks would be 87-95%
efficient) -- it's optimal only if wall-clock is what's being optimized,
which is what was asked.

**`perf` profiling dead end**: attempted Phase 5 step 1 (profile a
single-rank CPU LAMMPS run to split SF-vs-NN-vs-force-assembly time,
matching `GPU_PORTING_PLAN.md`'s Phase 0 methodology) twice --
`gpu/lammps_e2e/profile_cpu_lammps.slurm`, then
`profile_cpu_lammps_debug.slurm` after rebuilding n2p2 with `-g` added
(kept `-O3`/`-march=native` unchanged) when the first attempt resolved
almost no n2p2 symbols. Adding `-g` didn't help -- `perf report` still
shows the same ~50% of samples as dozens of near-identical ~1%-each
raw addresses, and checking with `--sort=dso,symbol` shows they're not
even attributed to a known shared object (`[unknown]` DSO, not just an
unresolved symbol within a known one). Real tooling limitation on this
cluster, not a missing-debug-info problem -- the recording itself
warned about lost chunks/IO overload, consistent with missed `MMAP`
events for dynamically-loaded libraries. The ~16% of samples that *did*
resolve (`libm`'s `tanhf32x`/`expf32x`/`expm1f32x`) doesn't help isolate
SF from NN anyway, since both use these (NN's tanh activation,
symmetry functions' exponential radial/angular terms). Not pursuing
`perf` further here -- manual `Stopwatch` instrumentation around the
`calculateSymmetryFunctionGroups()`/`calculateAtomicNeuralNetworks()`
call sites in `Mode.cpp` (matching the pattern `Training.cpp` already
uses for its `_err`/`_com`/`_upd` buckets) is the likely next attempt,
since it doesn't depend on `perf`'s symbol resolution working at all.

### Follow-up: why DCGP loses to Booster for this MD case, when it consistently won for `nnp-train` -- not a contradiction, a different bottleneck

Fair question raised: this file has repeatedly shown DCGP CPU beating
Booster CPU for `nnp-train`, sometimes decisively even at matched core
count (see the earlier "DCGP CPU, 32 cores (same core count, newer
silicon)... 14.1x" entry). So why did the DCGP-112 production run above
come in slowest of the three? Checked directly rather than assumed.

**Compilation ruled out.** A dedicated check job on each partition
(`lscpu`, `mpicxx --version`, `gcc --version`) confirmed identical
toolchain: GCC 12.2.0 (Spack), OpenMPI 4.1.6, same modules on both.
Both CPU `lmp_mpi` binaries were freshly, natively rebuilt on their own
partition (`-march=native` requires this -- see the earlier
cross-partition `.o`-file cleanup notes), so this isn't a stale-binary
artifact either.

**Hardware topology is the real answer, and it's stark:**

| | DCGP | Booster |
|---|---|---|
| CPU | Xeon Platinum 8480+ (Sapphire Rapids) | Xeon Platinum 8358 (Ice Lake) |
| Sockets | 2 | 1 |
| NUMA nodes | **8** (14 cores/domain) | **2** (16 cores/domain) |

DCGP's cores are individually newer/faster -- that's exactly why it
still wins `nnp-train` at matched core count. But `nnp-train`'s
communication is one large `MPI_Gatherv`/`Allgatherv` of the weight
Jacobian **once per mini-batch update** -- infrequent, throughput-bound,
so DCGP's per-core speed advantage comes through cleanly. LAMMPS/MD on
this 630-atom system exchanges small ghost-atom messages **every single
timestep** -- frequent, latency-bound -- and crossing DCGP's 2 sockets
and up to 8 NUMA domains costs real, repeated latency that Booster's
much tighter single-socket/2-NUMA layout doesn't pay.

**Matched-rank-count scan confirms it's not close, and not a crossover
effect** (`gpu/lammps_e2e/scan_cpu_booster_ranks.slurm`, companion to
the DCGP scan, same 2000-step H2O_2G case):

| Ranks | DCGP loop time | Booster loop time | Booster advantage |
|---|---|---|---|
| 4 | 527.0s | 194.1s | 2.7x |
| 8 | 278.5s | 104.0s | 2.7x |
| 16 | 152.0s | 55.8s | 2.7x |
| 32 | 85.5s | 35.1s | 2.4x |

Booster wins by a remarkably *consistent* ~2.5-2.7x at every rank count
tested -- not DCGP-catches-up-eventually, a persistent gap. Even DCGP's
full 112-core node (43.6s, scaled from the production run) never
catches Booster's 32 cores (35.1s) -- DCGP's 3.5x core-count advantage
isn't enough to overcome the per-message latency penalty on this
workload.

**Conclusion**: not a contradiction -- `nnp-train` is compute-throughput-
bound (DCGP wins), this small 630-atom MD case is communication-
latency-bound (Booster wins). A small system spread across many ranks
is close to a worst case for exposing NUMA/topology latency, precisely
because there's so little compute per rank to amortize communication
against. **This is why the 8640-atom `examples/interface-LAMMPS/H2O_RPBE-D3`
case is designated the final check** (Phase 6, plan doc) before drawing
any final conclusion about GPU-vs-CPU or DCGP-vs-Booster for this
project -- a larger system should have a much better compute-to-comm
ratio at any given rank count, giving a cleaner and more representative
read closer to what a real production MD run's economics would actually
look like. Deferred, not urgent -- to be run once Phase 5 (or whatever
else is worth a comprehensive final validation) is far enough along.

### Follow-up: Stopwatch profiling confirms it -- symmetry functions are ~95% of the per-timestep cost, NN evaluation only ~3.3%

Phase 5 step 1, attempt 3 (after `perf`'s dead end above): manual
`Stopwatch` instrumentation, mirroring `Training.cpp`'s `_err`/`_com`/
`_upd` pattern exactly. Added three temporary timers to
`InterfaceLammps` (`swProfile["sf"]`/`["nn"]`/`["forces"]`, wrapping
`calculateSymmetryFunctionGroups()`, `calculateAtomicNeuralNetworks()`,
and `getForces()` respectively), printed to stderr every 500 calls to
`process()`. One bug caught and fixed along the way: the header edit
initially dropped the class's closing `};`, producing a cascading
"extra qualification"/Eigen-template pile of unrelated-looking errors
downstream -- not a `perf`-style environment issue this time, just a
missed brace, fixed by re-adding it.

Single-rank, 2000-step H2O_2G run (CPU, Booster), 4 printouts across
the run, remarkably stable throughout:

| Calls | SF | NN | Forces |
|---|---|---|---|
| 500 | 169.10s (94.87%) | 5.85s (3.28%) | 3.29s (1.84%) |
| 1000 | 337.67s (94.85%) | 11.72s (3.29%) | 6.60s (1.86%) |
| 1500 | 506.73s (94.84%) | 17.61s (3.30%) | 9.95s (1.86%) |
| **2000** | **676.36s (94.84%)** | **23.49s (3.29%)** | **13.30s (1.87%)** |

Sums to 713.15s against LAMMPS's own reported "Pair" bucket of 730.82s
for the same run (~17.7s/2.4% unaccounted, presumably other overhead in
`process()`/`PairHDNNP::compute()` not wrapped by these three timers) --
close enough to trust the split.

**This is a much more decisive confirmation than the earlier indirect
reasoning** (CPU-vs-GPU converging at 32 ranks). It directly answers
the question the whole Phase 5 motivation was built on: symmetry-function
evaluation is ~95% of the real per-timestep cost, not just "the likely
dominant piece" -- NN evaluation (the part already GPU-accelerated) is
capped at ~3.3% of the total, meaning the existing GPU port's ceiling on
this workload was always going to be small, exactly as observed (1.4%
at matched Booster hardware). Force assembly, despite being flagged as
the single largest cost for `nnp-train` (46.5% there), is only ~1.9%
here -- a reminder that these fractions are workload-shape-dependent,
not universal constants, and confirming this directly rather than
assuming it carried over was the right call.

Instrumentation reverted after collecting this data -- it was marked
temporary in the code from the start, not meant to become permanent
production logging.

### Follow-up: ported symmetry-function evaluation to GPU -- correct, and a real win, but only at low MPI rank counts; shipped as a separate opt-in flag, off by default

Phase 5's remaining steps (3-8, `soft-percolating-jellyfish.md`), following
directly from the ~95%-of-per-timestep-cost finding above. Two
already-validated CUDA kernels (`gpu/soa/symfnc_exprad_group_test.cu`/
`symfnc_expangn_group_test.cu`, covering the only two symmetry-function
types (`SymGrpExpRad`/`SymGrpExpAngn`, type 2/3) and the only cutoff
type (`CT_TANHU`) this project's real datasets use) were promoted into
`src/libnnpgpu/GpuSymmetryFunction.h/.cu`, gated by a new
`Element::hasGpuCompatibleSymmetryFunctions()` compatibility check (same
all-or-nothing-per-structure shape as the existing NN dispatch gate),
and wired into `Mode::calculateSymmetryFunctionGroups()`. Kernels return
unscaled values by design -- the real `SymFnc::scale()`/
`getScalingFactor()` is applied by the caller, same "don't re-derive
validated production math" precedent as the NN port.

**Kernel-level correctness validated first, standalone, before touching
production** (`gpu/e2e_symfnc_check/`, matching `gpu/e2e_predict_check`/
`e2e_train_check`'s precedent): GPU vs. CPU `G`/`dGdr` agreement on real
H2O_2G neighbor geometry, both elements, all group-filter variants --
**ALL PASS**, max errors 1.96E-15 (`G`, 23520 values), 1.49E-16 (owner
derivative), 6.94E-17 (neighbor derivative, 2.5M values).

**Two real, separate bugs found and fixed while validating end to end
through LAMMPS** (neither was a symmetry-function math bug -- the
kernels were correct from the first standalone validation onward):

1. **A genuine ODR violation/ABI mismatch between `libnnp` and
   `libnnpif`.** The GPU symmetry-function path requires
   `-DN2P2_FULL_SFD_MEMORY` (the uncompacted per-neighbor derivative
   layout, needed because the kernel writes `Atom::Neighbor::dGdr`
   indexed by global symmetry-function index directly). `libnnp/makefile`
   added this define under `GPU=1`; `libnnpif/makefile` -- which compiles
   `InterfaceLammps.cpp`, the code that actually creates/manipulates
   `Atom`/`Structure` objects -- did not. Since `N2P2_FULL_SFD_MEMORY`
   guards actual member declarations in `Atom.h` (not just runtime
   behavior), this was a real binary-layout mismatch between two
   separately-compiled libraries: one side read/wrote members at offsets
   the other side never allocated. It manifested as `structure.atoms.size()`
   returning garbage (`0x3333333333333333`) well after the actual
   mismatch site -- the same "corruption appears downstream of the real
   bug" shape this file has hit before with stale `-march=native`
   binaries, but a genuinely different root cause this time, found via
   the same disciplined `fprintf`+`fflush` bisection debugging.
2. **A pre-existing macro-name typo in `InterfaceLammps.cpp`.**
   `getForces()`/`getForcesLambda()`/`getdChidxyz()` checked
   `#ifndef NNP_FULL_SFD_MEMORY` (missing "2P2") instead of the real
   `N2P2_FULL_SFD_MEMORY` used everywhere else in the codebase -- a dead
   macro name that was always false, so these functions always took the
   *compact*-layout branch regardless of the real build flag. Never
   triggered before because nobody had built the LAMMPS interface with
   `N2P2_FULL_SFD_MEMORY` defined until this port -- `Element::symmetryFunctionTable`
   (the compact-layout lookup table these functions read) is never
   populated when the full layout is active
   (`setupSymmetryFunctionMemory()` is skipped), so indexing into it
   threw exactly the observed `vector::_M_range_check` exceptions on
   empty vectors. Fixed by correcting the macro name (6 occurrences).

**With both fixed, full correctness validated end to end at LAMMPS
level**, same two-step discipline as the NN-only port: full 2000-step
H2O_2G MD trajectory matches the CPU baseline exactly -- `Max |Temp_cpu
- Temp_gpu| = 0`, `Max |PotEng_cpu - PotEng_gpu| = 0`, forces agree to
1E-7 across all dumped atoms/timesteps.

**But the first honest performance measurement was a severe
regression, not a win: ~10x slower than CPU**, not the 98%-of-workload
win the ~95%-SF-cost finding above seemed to promise. `Loop time of
5280.36s` for the same 2000-step smoke test (4 ranks/1 GPU, Booster) vs.
~490s CPU-only Pair time for the equivalent NN-only-GPU-era baseline.

**Root-caused via the same measure-don't-guess discipline used
throughout this file**, in three rounds:

1. **Isolated GPU contention from raw dispatch cost** with a 1-rank vs.
   4-rank probe (630-atom H2O_2G, no MPS): per-rank cost scaled from
   0.702s/step (1 rank) to 2.626s/step (4 ranks) -- almost exactly
   linear with rank count, meaning the 4 ranks were essentially fully
   serialized on the shared GPU. Real, but only part of the story: even
   the *zero-contention* single-rank number was already ~2.85x slower
   than the old CPU-only per-rank baseline.
2. **First hypothesis (wrong, but a reasonable one): blocking
   synchronous CUDA calls.** Each group call issued ~16 sequential
   *synchronous* `cudaMemcpy`/`cudaDeviceSynchronize` calls (H2D
   transfers, memsets, D2H transfers) on ordinary pageable host memory --
   `cudaMemcpyAsync` only actually avoids the per-call blocking
   round-trip with *pinned* host memory, so both a pinned-staging-buffer
   rewrite and async transfers on one dedicated stream (collapsing ~16
   blocking round-trips into 1 `cudaStreamSynchronize` per call) were
   implemented together in `GpuSymmetryFunction.cu`. Re-validated
   correct (`gpu/e2e_symfnc_check` still ALL PASS, same tolerances) --
   but made **zero measurable timing difference** (0.702s -> 0.707s/step).
   Worth recording as a real negative result: the obvious "too many
   small synchronous driver calls" diagnosis was plausible and easy to
   reach for, but wrong here.
3. **Real root cause, found by adding actual `Stopwatch` phase
   instrumentation** (CSR build / GPU call / scatter-back) instead of
   guessing again:

   | Phase | Time/step (1 rank, H2O_2G) | Share |
   |---|---|---|
   | CSR build (host) | 0.0038s | 0.5% |
   | **GPU call (kernel + transfer)** | **0.66s** | **93%** |
   | Scatter-back (host) | 0.019s | 2.7% |

   The dispatch call itself dominates, and the reason is visible
   directly in the launch config: `gridSize = ceil(numAtoms/128)` -- for
   this system's ~210-420 atoms per (element, group) call, that's only
   **2-4 CUDA thread blocks**, on a GPU with ~108 SMs. Over 95% of the
   GPU sits idle for the entire kernel call. Combined with the angular
   kernel's O(neighbors²) per-atom serial inner loop (~109 neighbors ->
   ~5900 pairs x up to 26 members, branchy scalar code) and
   `MAX_MEMBERS=64`-sized per-thread local arrays (likely spilling out
   of registers), this is a kernel shaped for throughput on a large
   batch, not latency on a small one -- validated for *correctness*
   early (as it should be) but never benchmarked for *raw kernel speed*
   against CPU before being wired into production, unlike the NN-forward
   port (which had a documented 25.6x/27.9x prototype speedup measured
   first).

**Before attempting a real kernel-parallelization redesign (a much
bigger, uncertain undertaking), checked whether the same unmodified
kernel already does better at 8640-atom scale** (`examples/interface-LAMMPS/H2O_RPBE-D3`,
Phase 6's previously-deferred "final check" case -- confirmed
GPU-compatible first: `input.nn` uses only types 2/3 with `cutoff_type 2`
= `CT_TANHU`, same as H2O_2G). The result flips completely, and the
mechanism confirms the diagnosis: the GPU call's cost per step is
**nearly unchanged** across a 14x increase in atoms (0.64-0.67s/step at
both 630 and 8640 atoms) -- exactly what "underutilized, idle capacity
absorbing the extra work for free" predicts. Single-rank, same Booster
hardware:

| System | Config | Pair time/step |
|---|---|---|
| H2O_2G (630 atoms) | GPU, 1 rank | 0.702s |
| H2O_RPBE-D3 (8640 atoms) | GPU, 1 rank | 1.202s |
| H2O_RPBE-D3 (8640 atoms) | CPU, 1 rank | 2.859s |

GPU is **2.4x faster** than CPU at this scale, single rank. Correctness
re-validated at this scale too, same discipline: full 41-step trajectory,
exact match (`0` diff Temp/E_pair/Press, `0` diff forces).

**But real production runs use far more than 1 rank, and a multi-rank
scan tells a materially different story** -- 4/8/16/32 ranks, both
backends, same 40-step H2O_RPBE-D3 config, Booster (GPU side using the
established MPS + `rank_gpu_wrapper.sh` multi-GPU-sharing pattern from
the `nnp-train` production runs):

| Ranks | Atoms/rank | GPU s/step | CPU s/step | Result |
|---|---|---|---|---|
| 1 | 8640 | 1.202 | 2.859 | GPU 2.4x faster |
| 4 | 2160 | 0.502 | 0.715 | GPU 1.4x faster |
| 8 | 1080 | 0.403 | 0.359 | CPU 1.1x faster |
| 16 | 540 | 0.349 | 0.186 | CPU 1.9x faster |
| 32 | 270 | 0.322 | 0.104 | CPU 3.1x faster |

CPU's Pair time scales down almost linearly with rank count (good
parallel efficiency, as expected for ordinary per-atom CPU work).
GPU's barely moves at all across the whole scan -- splitting the same
total work across more ranks just means fewer atoms per GPU call,
pushing the kernel straight back into the same underutilization regime
that hurt it at H2O_2G's scale. The crossover from win to loss lands
between 4 and 8 ranks, roughly 1000-2000 atoms/rank -- and this
project's actual production LAMMPS runs (Phase 3 above) have
consistently used 32 ranks, not 1-4.

**Conclusion: correct, and a real win in a narrow regime (large system,
low rank count), but a net loss at the rank counts this project
actually deploys with.** Rather than reverting the feature (the
underlying kernels and dispatch logic are real, working, validated
code, and genuinely faster in the regime where GPU occupancy is
adequate), it's shipped behind a **separate opt-in build flag,
`GPU_SF=1`, deliberately distinct from the existing `GPU=1`** --
`libnnp/makefile`/`libnnpif/makefile` only add `-DN2P2_GPU_SF`
(and the `-DN2P2_FULL_SFD_MEMORY` it requires) when `GPU_SF=1` is
explicitly passed, and `Mode.cpp`'s dispatch is gated on
`defined(N2P2_GPU) && defined(N2P2_GPU_SF)` together. Plain `GPU=1`
keeps exactly the NN-only dispatch (a genuine, if modest, win at any
scale tested so far) without silently inheriting a feature that would
regress a normal, higher-rank-count production run. Anyone deliberately
running a large system at a small rank count (e.g. one powerful node,
few ranks) can opt in and get a real ~1.4-2.4x speedup; everyone else's
build is unaffected.

### Follow-up: redesigned the SF kernels around the diagnosed occupancy problem -- fixed the short/medium-run regression completely, but the real production-length measurement still says no

The rank-scan crossover above pointed at one specific, fixable cause:
`gridSize = ceil(numAtoms/blockSize)` in the old one-thread-per-atom
kernels meant a real MPI rank's local atom count (as low as ~20-270 at
this project's actual 32-rank production configuration) launched only
2-4 CUDA thread blocks against a 108-SM GPU. The fix: **warp-per-atom**
kernels instead of thread-per-atom. Each atom now gets one warp (32
threads); each lane processes a disjoint stride of that atom's neighbors
(`ExpRad`) or outer pair-index `j` (`ExpAngn`), and the owner-atom
accumulators (`result`/`dResultX/Y/Z`) are combined with a
shuffle-based warp reduction -- no atomics needed for that part, since
the reduction stays within one warp. This multiplies the number of
concurrently active parallel units by up to 32x at the same atom count,
directly targeting the diagnosed bottleneck. `ExpAngn`'s neighbor-side
derivative writes are the one place real synchronization is required:
a neighbor slot can receive contributions from multiple `(j,k)` pairs
owned by different lanes, so those specifically use `atomicAdd` (native
on this project's target compute capability, sm_80); `ExpRad` has no
such problem, since each neighbor slot is touched by exactly one lane.
Also reduced `MAX_MEMBERS` from 64 to 32 (real datasets need at most
26) to ease the per-thread register/local-memory footprint that was
plausibly compounding the occupancy problem.

**A real bug was caught by the existing standalone harness, not missed
by it.** The first version of the rewrite failed `gpu/e2e_symfnc_check`
with `max|dG_own_gpu-dG_own_cpu| = 3.246E-02` while `G` (6.69E-15) and
the neighbor-side derivative (6.94E-17) stayed at the established
tight tolerance -- a strong, specific signal, since it isolated the bug
to exactly one accumulator. The original serial code applies the
`pow(2.0, 1.0-zeta[m])` normalization to `result[m]` (-> `G`) *only*,
post-loop -- `dResultX/Y/Z[m]` already have it baked in earlier via
`fgF = fg * pnorm` inside the pair loop, and must not be re-multiplied.
The warp rewrite's final write-out applied that normalization to all
four outputs, double-counting it for the three derivative components.
Fixed by only scaling `G`'s output; re-ran the harness -- **ALL PASS**,
same tolerances as before (6.69E-15 / 1.35E-16 / 6.94E-17).

**Every short- and medium-duration measurement after the fix looked
like a clean, complete reversal.** The identical H2O_RPBE-D3 rank scan
that showed the crossover above, rerun with the new kernel (same 40-step
config, same rank counts):

| Ranks | Atoms/rank | GPU s/step | CPU s/step | Result |
|---|---|---|---|---|
| 4 | 2160 | 0.687 | 0.713 | GPU 1.04x faster |
| 8 | 1080 | 0.345 | 0.360 | GPU 1.04x faster |
| 16 | 540 | 0.179 | 0.187 | GPU 1.04x faster |
| 32 | 270 | 0.101 | 0.104 | GPU 1.03x faster |

No crossover at all -- GPU tracks CPU's scaling closely while staying
consistently ahead across the whole range. H2O_2G's most extreme case
(32 ranks, ~20 atoms/rank, previously ~30x slower) flipped too: 23.77s
vs. 24.82s Pair time over 2000 steps, GPU **~4.4% faster**. A GPU-only
rank scan for H2O_2G specifically (300-step, 4/8/16/32 ranks, always
4 GPUs) found Pair time decreasing monotonically with rank count --
0.0329 -> 0.0259 -> 0.0221 -> 0.0198 s/step -- confirming 32 ranks is
GPU's best config too, matching CPU's already-established best
single-node config, so "best vs. best" and "matched rank count" turned
out to be the same comparison.

**A two-tier validation (short-window exact match, full-trajectory
statistical match, plus a GPU-vs-GPU repeat run to separate expected
chaos from a real bug) confirmed correctness cleanly on a real 2000-step
H2O_2G LAMMPS run, 32 ranks.** `ExpAngn`'s `atomicAdd` makes GPU results
not bit-reproducible run-to-run (atomic completion order varies), and
MD is chaotic, so some visible trajectory divergence over a long run
was expected -- in practice, forces matched exactly (`0` diff) through
step 1800 of 2000, with a single 1E-7 blip at step 2000 (right at the
dump file's print precision) shared identically by *both* independent
GPU runs against CPU, while the two GPU runs matched each other exactly
throughout -- consistent with ordinary floating-point reassociation
(GPU's warp-reduction sums in a different order than CPU's serial loop),
not a bug. Temperature and PotEng means/stdevs matched to 4 decimal
places across all three runs.

**But the one measurement that matters most -- a real, full-length
200,000-step production run at each backend's best configuration
(32 ranks, Booster, H2O_2G) -- reversed again, back to a loss:**

| | Loop time | Pair time/step | Comm time/step |
|---|---|---|---|
| GPU (32 ranks/4 GPUs) | 4445.8s (74.1 min) | 19.82 ms | 2.33 ms |
| CPU (32 ranks) | 3458.2s (57.6 min) | 12.35 ms | 4.87 ms |

GPU is **28.6% slower overall**, with Pair time specifically **1.6x
higher** than CPU -- directly contradicting both the 2000-step
validation run (GPU ahead) and, in the opposite direction, largely
matching what the shorter 300-step scan already hinted at (GPU behind,
though by less). Correctness held up regardless: temperature and
PotEng means stayed statistically consistent between GPU and CPU
(299.52 vs. 299.35 K, both near the 300 K target; -104.99 vs. -105.58,
both well within a stdev) -- the expected chaotic divergence of a real
200000-step trajectory, not a physics bug.

**Three measurements at three different durations, in three different
directions is a genuine, currently-unresolved puzzle** -- 300-step scan
pessimistic, 2000-step run optimistic, 200000-step run pessimistic
again, and by more than the 300-step scan alone predicted. The most
likely explanation is sustained-load GPU clock/thermal behavior: a
33-second run has no time to either reach full sustained boost clocks
or trigger thermal throttling, while a 74-minute run has plenty of time
for both effects to matter, potentially in opposite directions at
different points in the run -- but this is *not confirmed*, since no
GPU telemetry (clocks, temperature, power) was captured during the
production run to distinguish throttling from shared-cluster contention
from some other long-run-specific effect (e.g. driver-level bookkeeping
across the ~57 million total kernel launches -- 32 ranks x 200000 steps
x ~9 SF group calls -- that a 2000-step run would never approach).

**Conclusion: the occupancy diagnosis and warp-per-atom fix were real
and correct** -- they fixed the specific short/medium-run regression
they targeted, cleanly and completely, and the standalone kernel-level
and short-trajectory correctness validation is solid. **But the only
measurement taken at this project's actual intended production
duration is the 200000-step run, and it says GPU loses to CPU by a
substantial margin (28.6%) even with the improved kernel.** That is the
number that should be trusted over the shorter ones, precisely because
it is the only one measured at real use-case scale. The `GPU_SF=1`
opt-in-flag decision (kept separate from, and off by default relative
to, plain `GPU=1`) stands -- if anything this result reinforces it more
strongly than the original crossover finding did, since even the fixed
kernel doesn't deliver a production-length win at this project's actual
deployment configuration. The broader lesson: for GPU work in this
project, a short or medium benchmark run is not a reliable substitute
for one run at real intended length -- both directions of error (too
pessimistic before clocks/occupancy settle, potentially too optimistic
before sustained-load effects appear) are live risks, and only the
full-length run resolved which one actually applied here.

### Follow-up: reproducibility-checked the 200000-step result and ruled out the two obvious environmental explanations via live GPU telemetry -- candidate next-phase approaches recorded, not yet attempted

Before trusting the 28.6%-slower result as a real property of the
workload rather than a one-off artifact of that particular run (the
home filesystem happened to be at its 50GB capacity, 0 bytes free, at
around the same time -- see below), reran the identical 200000-step
H2O_2G GPU-vs-CPU comparison at the same best configuration (32
ranks, Booster), from freshly rebuilt LAMMPS trees (the originals had
been deleted as part of a separate output-cleanup pass) and a freshly
regenerated MPS session:

| | First run | Rerun |
|---|---|---|
| GPU Loop time | 4445.8s (74.1 min) | 4426.8s (73.8 min) |
| CPU Loop time | 3458.2s (57.6 min) | 3426.6s (57.1 min) |
| GPU vs. CPU | 28.6% slower | 29.2% slower |
| GPU Pair time | 3964.9s | 3967.9s |
| CPU Pair time | 2469.1s | 2465.1s |

Essentially identical -- confirms the result is a real, robustly
reproducible property of the current kernel at production scale, not
noise, a stale build, or an artifact of whatever else was happening on
the system during the first run.

**The near-full home filesystem (50GB total, 0 bytes free at one
point this session) was a real, separate problem -- but not this one.**
It's the most plausible explanation for several odd tool-call
timeouts/hiccups earlier in the session (filesystem I/O operations
failing/stalling near capacity), and cleanup recovered ~21GB of
headroom. But `Pair` time is pure GPU/CPU compute, not disk I/O, so
disk capacity was never a plausible mechanism for the timing
regression specifically -- worth being precise about, since the two
issues surfaced around the same time and are easy to conflate.

**Checked live GPU telemetry during the rerun** (`nvidia-smi` via
`srun --overlap` into the running job's node) instead of continuing to
guess -- this ruled out both obvious environmental explanations
cleanly:

| GPU | Temp | SM clock | Power draw | Utilization |
|---|---|---|---|---|
| 0 | 46°C | 1395 MHz (= max, no throttle) | 105W / 550W limit | 69% |
| 1 | 45°C | 1395 MHz (= max) | 101W / 550W limit | 85% |
| 2 | 46°C | 1395 MHz (= max) | 101W / 500W limit | 87% |
| 3 | 46°C | 1395 MHz (= max) | 103W / 500W limit | 75% |

**Thermal/clock throttling: ruled out.** Cold (throttle threshold is
~85°C+), clocks pinned at their full rated maximum, power draw ~20% of
budget -- the opposite of what sustained-load throttling would look
like. This was the leading hypothesis in the previous entry;
telemetry falsifies it directly. **Cluster contention: also ruled
out.** `squeue -w <node>` showed only this job scheduled on the node,
and the GPU process list showed exactly 8 client processes + 1 shared
MPS server process per GPU (32 ranks / 4 GPUs, matching the configured
`RANKS_PER_GPU=8`) -- all belonging to this job, no foreign PIDs.

**What the telemetry does suggest**: utilization sitting at 69-87%
(not saturated) with clocks/power/temp all showing headroom points at
the GPUs sitting idle *between* dispatches rather than being
compute-bound -- consistent with some form of per-call or
per-launch overhead that compounds over a very long run (32 ranks x
200000 steps x ~9 SF group calls ≈ 57 million total dispatches through
MPS, vs. ~1.1 million for the 2000-step run that looked fast), rather
than a raw compute-throughput problem. Not confirmed without deeper
profiling.

**Candidate approaches for the next phase, recorded but not yet
attempted** -- roughly in order of how promising/cheap-to-check each
is:

1. **The bottleneck may have shifted from GPU to host CPU, and this
   hasn't actually been checked.** The warp-per-atom fix made the GPU
   kernel itself much faster; the per-timestep pipeline still
   round-trips through host CPU multiple times (CSR build -> H2D ->
   kernel -> D2H -> scatter into `Atom::G`/`dGdr` -> a separate NN-GPU
   dispatch -> `InterfaceLammps::getForces()`, still pure CPU). That
   host-side glue code was ~3% of Pair time against the *old*, slow
   kernel -- against the *new*, much faster kernel, the same fixed
   cost could now be a much larger fraction of a much smaller total,
   and 32 independent MPI processes each paying it, sustained over 74
   minutes, is a different regime than a 33-second burst. GPU
   clocks/temps were checked directly this round; CPU-side clock/
   frequency-scaling behavior during the long run was not. Cheapest
   next check: resurrect the phase-level `Stopwatch` instrumentation
   used for the original occupancy diagnosis, but sample it throughout
   a long run rather than only early on, to see whether the
   CPU-glue-to-GPU-call time ratio grows over the course of the run.
2. **Batch multiple MPI ranks' local work into one shared-GPU call per
   node**, instead of each of the 8 ranks/GPU independently dispatching
   through MPS. Directly reduces the *count* of independent dispatches
   (the likely-relevant quantity per the utilization pattern above).
   A real architecture change (a GPU-owning worker process per node,
   ranks feed it via shared memory/IPC) -- this is the "cross-rank
   pooling" idea considered and set aside earlier as disproportionate
   given the warp-per-atom fix looked sufficient at the time; worth
   reconsidering now that the full-length result reopened the question.
3. **Direct profiling with `nsys`/`ncu`** instead of continuing to
   infer host-vs-device behavior from aggregate telemetry -- would show
   the actual gap structure directly. Not yet set up.

**Open question this raises for H2O_RPBE-D3 (the larger system)**: all
of Phase 5's short/medium-duration measurements on both systems, and
the *only* long-duration (200000-step) measurement, which is H2O_2G-only
(H2O_RPBE-D3 at that length was estimated at ~11-12 GPU-hours and
descoped for cost, see the 2K-comparison-report follow-up above). The
mechanism diagnosed across this whole investigation (fixed per-call
overhead amortized better by more atoms/call) predicts a larger system
should be *more* robust against whatever causes H2O_2G's long-run
reversal, not less -- but that is a reasoned prediction, not a
validated one, and this session's own experience is a direct
caution against trusting short-run-based predictions here. A
moderate-duration (e.g. 20000-50000 step) H2O_RPBE-D3 run -- a
fraction of full-200000-step cost -- would be the proportionate way to
check whether the same short-run-optimistic/long-run-pessimistic
divergence shows up there too, before treating "GPU_SF is fine for big
systems" as anything more than an informed guess.

### Follow-up: candidate approach #1 ruled out cleanly -- the real cause is a time-correlated environment change, not run length, node identity, or a within-run degradation

Implemented candidate #1 from the previous entry: resurrected the
phase-level `Stopwatch` instrumentation (CSR build / GPU call /
scatter-back), but reporting *windowed* deltas (last 2000 calls) every
2000 timesteps throughout a full 200000-step H2O_2G run, instead of a
single early snapshot -- directly testing whether the CPU-glue-to-GPU
ratio grows over the course of a long run.

**Result: flat.** First 10% of the run vs. last 10%: GPU-call time
-0.5%, CSR+scatter time +2.4%, ratio +2.9% -- noise-level, no trend.
**Candidate #1 is ruled out** -- nothing about the SF-GPU dispatch's
own internal phase balance changes over the course of a long run.

That same data pointed somewhere more useful, though: the *first* 2000
steps of this 200000-step run were *already* running at the slow rate
(SF-only ~0.0186s/step) -- not something that develops gradually. That
reopened the node-identity question from the previous entry's
telemetry check, since "slow from step 1, on whichever node this
particular job happened to land on" is exactly what node-to-node
hardware variance would look like too.

**Directly tested node identity by forcing repeat runs onto specific
nodes (`--nodelist`).** The original fast measurement (job 52576480,
2000 steps, 0.0119s/step aggregate SF+NN Pair time) ran on `lrdn3271`.
Forcing an identical 2000-step run onto `lrdn1600` (the node behind
the slow 200000-step runs) reproduced the slow rate exactly
(0.0198s/step) -- consistent with node identity mattering. But forcing
the *same* identical config back onto `lrdn3271` itself, the
originally "fast" node, **also came back slow** (0.0198s/step) --
four days after the original fast measurement. A single physical node
cannot be fast once and slow later while every other node stays
uniformly slow in between if the cause were a fixed hardware property
of that node. **Node identity is therefore also ruled out.**

**The real pattern is time-correlated, not node-correlated or
duration-correlated:**

| Job | Time (Aug 2026) | Node | Result |
|---|---|---|---|
| 52576480 | 17th, 11:40 | `lrdn3271` | fast (0.0119s/step) |
| 52588528 | 17th, 13:46 | `lrdn1818` | slow (0.0198s/step) |
| 52597485 | 17th, 17:08 | `lrdn1700` | slow (0.0198s/step) |
| 52603120 | 17th, 19:59 | `lrdn1600` | slow (0.0199s/step) |
| 52664763 | 19th, 14:45 | `lrdn1600` | slow (0.0198s/step) |
| 52787816 | 22nd, 00:18 | `lrdn3271` | slow (0.0198s/step) |

Five different nodes, spanning five days, all uniformly slow --
except the single earliest measurement. Something changed, cluster-wide
or in the immediate run environment, in the roughly two-hour window
between 11:40 and 13:46 on the 17th, after which every subsequent
measurement on every tested node has been consistently ~67% slower per
step than that one early result. No git commits landed in that window
(checked directly), so it isn't attributable to a code change on this
project's side that's visible in version control. One job ran in that
window whose log was later deleted during an unrelated output-cleanup
pass, so the boundary can't be narrowed past that ~2-hour range with
data currently available.

**Conclusion: the root cause of the short-run/long-run divergence is
an environment change correlated with time, not with run length, node
identity, or GPU thermal/clock/contention state (all directly ruled
out across this and the previous entry).** What actually changed is
not identified -- a driver/MPS update, a scheduler or cluster
configuration change, or something else outside this project's
visibility are all plausible, none confirmed. Not pursued further this
session given the compute and wall-clock already spent chasing it;
flagged here for anyone revisiting this with access to cluster-side
change logs for that window. The practical conclusion for `GPU_SF=1`
is unchanged either way: every measurement taken since that window --
which is to say, the environment this code will actually run in going
forward -- shows a real, substantial loss at this project's production
configuration, so it stays a separate, off-by-default opt-in flag.

### RETRACTED -- everything from here to the "corrected understanding" entry at the end of this file is INVALID. Do not cite, trust, or act on any specific number, structure count, or "X vs Y" claim in this range.

**Read this box first if you're jumping into the file anywhere below.**
Every entry from this line through the final "### Follow-up: corrected
understanding" entry at the end of the file describes an investigation
built on a broken comparison script. The specific claims below --
"38 of 140 structures disagree," "median force diff 0.43," "systemic
across all three systems," "root cause is normalization-constant
magnitude," the whole "race condition" framing, all of it -- are
**not real findings**. They are artifacts of a scripting bug, described
in full at the end of this file. The bug: every `nnp-dataset`
CPU-vs-GPU force comparison in this range joined the two `forces.comp`
files using `join -j 1` on the key `structure-index_atom-index` --
but each atom appears on **three** consecutive lines in that file (one
per force *x/y/z* component), so the key is not unique. POSIX `join`
on a non-unique key emits the **cartesian product** of every matching
line on both sides, which silently cross-paired components (e.g.
CPU's *x* value against GPU's *y* or *z* value) and manufactured large
"disagreements" that reflect nothing more than the fact that force
components have different values from each other -- not any actual
CPU-vs-GPU discrepancy.

**What actually happened when this was redone correctly** (row-aligned
comparison instead of the broken join, plus an independent manual
Python recomputation of the force from the raw values fed into
`GpuForces.cu`'s kernels): CPU and GPU produce **bit-identical-to-
floating-point-precision forces** for water, magnetite, feldspar, and
H2O_2G, under every normalization configuration tested. `GpuForces.cu`
is correct. There is no per-structure GPU force corruption, no
activation-function sensitivity, no normalization-magnitude
sensitivity, and no cross-system pattern -- none of that was ever
real.

**What IS still real and unaffected by this bug**: water's actual
`nnp-train` GPU run (full iterative Kalman-filter training over
multiple epochs) genuinely diverges to `NaN` -- that observation came
directly from `nnp-train`'s own printed per-epoch log output, never
touched by the flawed comparison script. With the force computation
itself now conclusively cleared, the real open question is why water's
training *loop* (not the force kernel) diverges over epochs -- most
likely chaotic amplification, over many sequential Kalman updates, of
some ordinary and otherwise harmless CPU-vs-GPU floating-point
reassociation difference (already documented as expected, bounded
behavior for H2O_2G elsewhere in this file) that for water's specific
configuration runs away instead of staying bounded. Not yet
investigated with a correct methodology -- treat this as the next
thing to look at, not as an answered question.

The entries below are preserved, unedited, exactly as originally
written, so the mistake and how it was found are on the record --
**not** because any conclusion in them should be trusted.

---

### Follow-up: benchmarking three new real-world systems (water, magnetite,
feldspar) found a genuine race condition in the GPU force-computation path

A colleague (Pablo) provided three new 2G-HDNNP training sets to
benchmark CPU-vs-GPU on: `water` (H/O, softplus activation, 1495
structures), `magnetite` (Fe/O/H, tanh, 26789 structures, plus
`committee_mode`/`committee_data`/`committee_cutoff` settings this n2p2
fork doesn't implement -- confirmed via grep across `src/libnnp`/
`src/application`, harmless since `Settings.cpp` only warns on unknown
keywords, doesn't error), and `feldspar` (H/O/K/Al/Si, tanh, 12575
structures). Three hardware configs per system (Booster 32-core CPU,
Booster 32-core+4-GPU, DCGP 112-core), 10-epoch preliminary runs first,
100-epoch full runs to follow.

**Checked whether the recent LAMMPS-interface/SF-GPU work (this file's
last several entries) affected `nnp-train`'s code path before running
anything** -- it doesn't: every change since the H2O_2G 100-epoch
reference artifact (`8784c3e`/`567d14e`) that touches
`Mode.cpp`/`Element.{h,cpp}`/the two `libnnp*/makefile`s is inside
`#if defined(N2P2_GPU_SF)`-gated blocks (confirmed by diffing
`Mode.cpp` against the pre-LAMMPS-work commit and reading every hunk),
which plain `GPU=1` builds never enable. The one thing that *did*
change since that reference artifact -- the activation-function
generalization (`4ee5b0f`/`cd5de23`) -- is unrelated to the LAMMPS work
and is required here anyway: water uses softplus, and pre-generalization
`hasGpuCompatibleArchitecture()` silently fell back to CPU-only for
anything but tanh. So: current `HEAD` used as-is, no revert.

**Host-memory OOM risk, checked before running anything expensive.**
Naive scaling from structure counts alone suggested magnetite/feldspar
(21x/10x more structures than H2O_2G) would blow past a node's ~514GB
RAM -- but `nnp-scaling`'s own built-in memory estimator (it prints
this unprompted) gave much more reassuring, dataset-composition-aware
numbers: water 12.6 GiB, magnetite 253.9 GiB, feldspar 177.4 GiB --
both comfortably under one node's capacity. Real measured peak RSS
during the 10-epoch runs ran ~25-30% over that estimate (magnetite
317GB, feldspar 223GB) -- consistent enough to trust the estimator as
a planning tool, not exact. Practical lesson: `--mem=` should be sized
per system from this estimate (with margin), not copy-pasted from
H2O_2G's ~75GB baseline -- one job (feldspar's Booster-GPU run, capped
at a copy-pasted 280GB) hit a real `slurmstepd oom_kill` this way.

**A second, different kind of OOM: `GpuForces.cu`'s persistent
per-structure device-memory cache is unbounded** (topology uploaded
once per structure, never evicted -- a design gap deliberately deferred
back when H2O_2G, ~1254 structures, was the only dataset this ever ran
on). Per-GPU cached-structure count works out to
`(total dataset structures) / (total GPUs)`, independent of how many
MPI ranks share each GPU. Magnetite's single-node (4-GPU) run hit this
directly -- `GPU error at GpuForces.cu:114/117: out of memory`, crashing
during the very first pass over the data. Fix (resource-level, no code
change): spread across more nodes/GPUs -- 2 nodes (8 GPUs) halves the
per-GPU load and both magnetite's and feldspar's Booster-GPU 10-epoch
runs completed cleanly at that setting. `gpu/pablo_benchmark/
train_booster_gpu_multinode.slurm` generalizes the single-node GPU
training script: rank/GPU counts computed from `$SLURM_NTASKS`/
`$SLURM_JOB_NUM_NODES` instead of hardcoded, and (the part the
single-node script didn't need) an MPS control daemon started via
`srun --ntasks-per-node=1` across the *whole* allocation, since MPS is
per-node and the original script only started one on the launch node.

**The important finding: water's GPU training run doesn't just run
slower or faster -- it silently diverges to NaN.** Epoch-by-epoch
energy RMSE: `5.3E-4` (epoch 1, CPU, converging normally) vs. `1.29E+34`
(epoch 1, GPU) -> `INF` (epoch 4) -> `NaN` (epoch 6 onward). Both CPU
variants (Booster, DCGP; identical `input.nn`) converge smoothly from
epoch 1. Magnetite/feldspar's GPU runs show no such thing -- their
CPU-vs-GPU differences are modest and look like ordinary Kalman-filter
run-to-run chaos (different rounding, different but still-converging
trajectory), not corruption. Water is also the only one of the three
using softplus rather than tanh, which was the first suspect --
checked `GpuNeuralNetwork.cu`'s `AF_SOFTPLUS` forward/backward
formulas directly against `NeuralNetwork.cpp`'s reference (including
the `EXP_LIMIT=35.0` overflow clamp): bit-for-bit identical. Not the
activation math.

**Isolated via a dedicated `gpu/pablo_benchmark/debug_water_force/`
harness** (built CPU and GPU `nnp-dataset` in the already-built worktrees,
compared predicted forces structure-by-structure against a known-good
CPU baseline on water's *epoch-0* weights -- CPU and GPU matched
exactly at epoch 0 with 32 ranks each, so this is a clean, apples-to-
apples starting point, not confounded by different random weight
initialization):
- Energy always matched exactly, at every scale tested -- the bug is
  isolated to `GpuForces.cu`'s force path specifically, not the NN
  forward pass.
- A 20-structure subset at 4 ranks/1 GPU (no MPS) matched CPU exactly.
  The full 140-structure test set at 4, 8, or 32 ranks (with or
  without the MPS wrapper) showed **every single structure** disagreeing.
- Bisecting ruled out rank count and GPU-sharing density directly:
  holding rank/GPU config fixed at the known-good 4-ranks/1-GPU/no-MPS
  setting and only swapping in the full 140-structure set still broke
  *all* 140, including the first 20 that matched perfectly in isolation.
  So it isn't rank count, GPU count, or MPS/wrapper presence.
- At 1 rank processing all 140 structures sequentially (no MPI
  complexity at all), only 38 of 140 disagreed -- always the *first*
  38 in file order, not a fixed set tied to specific structure content.
- Wrapping the 1-rank/140-structure case in `compute-sanitizer`
  (`--tool memcheck`, after working around two rounds of unrelated
  `CUDA_ERROR_INVALID_CONTEXT` noise from OpenMPI's own UCX transport
  probing CUDA devices during `MPI_Init` -- fixed with
  `OMPI_MCA_pml=ob1 OMPI_MCA_osc=^ucx UCX_TLS=^cuda,...`) reported
  **zero errors, and the bug disappeared** -- forces matched CPU
  exactly under the sanitizer.

**That combination -- correct under `memcheck`'s heavy serialization,
wrong under normal unsynchronized execution, and triggered by
processing enough structures in one process -- is the standard
signature of a genuine race condition**, not a formula bug or an
indexing mistake. `GpuForces.cu`'s topology upload/compute both use
synchronous `cudaMemcpy` and default-stream kernel launches, which
*should* already serialize correctly within one rank; the actual
missing synchronization point hasn't been pinned to a line yet --
`memcheck` isn't the right sanitizer tool for this (`racecheck`/
`synccheck` are, or manual `cudaDeviceSynchronize()` bisection), and
that's the natural next step if this gets picked back up.

**Correction (superseded by the next follow-up entry): magnetite and
feldspar's GPU results are NOT trustworthy either.** The paragraph
below originally said the opposite -- wrong, and left visible so the
mistake and its correction are both on the record. The
"no divergence, differences look like normal Kalman chaos" read was
based only on aggregate epoch-RMSE trends over 12M+/2.9M+ force
components; a direct per-structure check (same methodology as water's,
done right after this entry was first written) found the identical
100%-of-structures-wrong pattern in both, just diluted below visibility
at that aggregate scale. See the next entry for the full correction and
what's now known.

10-epoch timing summary (wall-clock, Booster CPU / Booster GPU / DCGP-112):

| System | Booster CPU (32 cores) | Booster GPU | DCGP-112 |
| --- | --- | --- | --- |
| water | 481.7s (48.2s/ep) | 77.8s (7.8s/ep) -- **diverged, not valid** | 283.9s (28.4s/ep) |
| magnetite | 32514s / 9.04h | 12045s / 3.35h (2 nodes/8 GPUs) | 15695s / 4.36h |
| feldspar | 81825s / 22.7h | 11049s / 3.07h (2 nodes/8 GPUs) | 46848s / 13.0h |

GPU's magnetite/feldspar numbers used 2 nodes (8 GPUs) vs. CPU/DCGP's 1
node, per the OOM fix above -- not a resource-matched comparison, real
single-node-equivalent speedup would be smaller. 100-epoch runs (with
`write_trainpoints`/`write_trainforces` disabled -- Pablo's water/
feldspar `input.nn` had these at `1`, which would have written a full
per-epoch force-comparison file, 425MB for feldspar alone, every
epoch; ~42GB for feldspar over 100 epochs. Matches H2O_2G's/magnetite's
already-sensible `0` convention) follow, results routed directly to
`/leonardo_work/L-AUT_Giane_26/acoretti/NEURALCPM/gpu_porting/
pablo_benchmark/` instead of home -- home hit its 50GB cap during the
10-epoch runs and had to be freed by moving feldspar/magnetite's
results there mid-session.

(All six magnetite/feldspar 100-epoch jobs were subsequently cancelled
before running -- extrapolating from the 10-epoch numbers above, every
one of them exceeds the cluster's 24h single-job limit, from `1.3
days` (feldspar GPU) up to `9.5 days` (feldspar CPU, needing ~10
chained restart jobs). Water's 100-epoch CPU/DCGP jobs were also
cancelled per explicit direction, in favor of a direct per-epoch
timing comparison from the 10-epoch data instead -- see the numbers
above.)

### Follow-up: correction -- the GpuForces divergence is NOT water-
specific or softplus-specific. It's systemic across all three systems,
and it's real corruption, not floating-point noise

Two direct questions from the user prompted this: (1) had softplus
actually been tested against tanh, not just checked at the formula
level, and (2) given the divergence turned out activation-independent,
were magnetite/feldspar's GPU results -- only checked at the aggregate
epoch-RMSE level -- actually clean, or just diluted?

**Both were real gaps, and both changed the picture.**

**Tanh control, same architecture/dataset as water, isolates the
variable cleanly**: generated fresh epoch-0 weights for water's exact
network (2 hidden layers, 25/25 nodes, H/O) with `global_activation_short`
changed from `p p l` to `t t l`, then ran the same CPU-vs-GPU
`nnp-dataset` force comparison on the full 140-structure test set.
**Identical result to softplus** -- energy matches exactly, all 140
structures show force disagreements. Rules out softplus/the activation
function entirely, which fits the code-level picture: `GpuForces.cu`
only consumes already-computed `dEdG`, with zero dependency on which
activation produced it.

**Applying the same per-structure force check to magnetite and
feldspar (never done before -- their 10-epoch runs were only checked
via aggregate epoch-RMSE trends) found the identical pattern**: fresh
epoch-0 weights, CPU-vs-GPU `nnp-dataset` on a 150-structure subset of
each system's real test set -- **150/150 structures disagree for both**,
energy exact, same shape as water. The earlier read ("differences look
like normal Kalman-filter chaos") wasn't wrong about what the
*aggregate* epoch-RMSE trend showed -- it was wrong to trust that
aggregate as sufficient evidence of correctness. Averaged over
12M+/2.9M+ force components, a systematic per-structure corruption is
easy to miss; averaged over water's 140-structure test set, it isn't.

**Checked whether this is genuine corruption or expected numerical
noise before concluding anything** -- `pairForceKernel`'s `atomicAdd`
accumulation has a real, known, harmless non-deterministic summation
order (order of floating-point addition affects the last few ULPs),
so a naive ">1e-6 absolute difference" threshold could in principle be
flagging normal noise, not a bug. Checked the actual *magnitude* of
disagreements directly: median absolute force-component difference is
`0.43` (water/tanh), `0.21` (magnetite), `0.41` (feldspar), with maxes
around `4-5` -- the same order of magnitude as the force values
themselves. This is real corruption, not ULP-level noise.

**Two more hypotheses tested and ruled out, both empirically:**
- Added `cudaDeviceSynchronize()` after each kernel launch in
  `GpuForces.cu::gpuForcesCompute()` (testing whether default-stream
  ordering wasn't actually providing the serialization it should) --
  rebuilt, reran the reliably-failing 4-rank/140-structure config:
  **no change**, still 140/140 wrong.
- Added host-side bounds checking on the edge-list indices
  (`edgeTarget`/`edgeOwnerDEdGIndex`) that `Mode.cpp` builds before
  uploading topology to the GPU -- testing whether a bad index sends a
  force contribution into a *different* structure's device buffer
  (memory-layout-dependent, which would explain why `compute-sanitizer
  --tool memcheck`'s redzone padding "fixes" it: padding moves buffers
  apart, breaking the accidental aliasing). **Zero violations found** --
  every index is correctly within range for its own structure.

**Where this leaves things**: the race-condition read from the
previous entry (correct under `memcheck`'s heavy instrumentation,
wrong under normal execution, wrong under the lighter `racecheck`/
`synccheck` tools too) still stands as the best-supported observation,
but two of the more obvious concrete mechanisms for it (missing
kernel-completion sync, cross-structure buffer aliasing via bad
indices) are now ruled out directly rather than just suspected. The
actual mechanism remains unidentified.

**Practical consequence, corrected from the previous entry**: none of
the three systems' GPU force numbers from this benchmark should be
trusted -- not just water's. All three GPU timing numbers (wall-clock)
remain valid as *timing* measurements, since they don't depend on force
correctness, but the trained-model quality/force-error side of every
GPU run in this benchmark is unverified and likely wrong. CPU (Booster)
and DCGP-112 are the only currently-trustworthy comparison for all
three systems.

**Checked directly (was left as an open question, now settled): H2O_2G
does NOT have this bug.** Same exact methodology -- fresh epoch-0
weights (already on hand from the original 100-epoch comparison),
CPU-vs-GPU `nnp-dataset` on the full 118-structure `test.data`, reusing
the already-built binaries (H2O_2G is the same architecture family,
tanh/2x25 nodes, no rebuild needed):

| System | Median \|CPU-GPU\| force diff | Max | Verdict |
| --- | --- | --- | --- |
| H2O_2G | `0.00027` | `0.0041` | Consistent with ordinary floating-point noise |
| water (softplus or tanh) | `0.43` | `3.83` | Real corruption |
| magnetite | `0.21` | `4.85` | Real corruption |
| feldspar | `0.41` | `4.68` | Real corruption |

H2O_2G's differences are ~1000x smaller than the other three -- the
right order of magnitude for `pairForceKernel`'s known, harmless
`atomicAdd` summation-order non-determinism, not the same O(1)
corruption. **The original "17.8x, no drift" H2O_2G validation earlier
in this file stands; it was never affected by this bug.**

This is also a useful negative clue for root-causing, not just a relief:
H2O_2G's test set (118 structures) is a similar *count* to water's
(140), so "number of structures processed in one call" alone doesn't
explain the split between clean and corrupted -- ruling out (or at
least weakening) dataset-size-in-structure-count as the real trigger.
What differs is per-structure *size*: H2O_2G's structures are ~630
atoms each vs. water's ~190/magnetite's ~171/feldspar's ~233 -- fewer,
much bigger structures vs. many smaller ones. Something about
per-structure edge-list/topology *shape* (not the number of structures
visited) is the more likely axis to chase next, though this hasn't
been tested directly yet either.

### Follow-up: ruled out a global-normalization-corruption mechanism; input.nn diffing and normalization tracing found real differences but not the cause

Comparing water's and H2O_2G's `input.nn` directly surfaced a real
structural difference worth checking: H2O_2G has `mean_energy`/
`conv_energy`/`conv_length` pre-baked as fixed constants (computed once,
offline, by `nnp-norm` -- the older, optional workflow, per n2p2's own
docs), while water/magnetite/feldspar all use `normalize_data_set
force`, which computes this calibration *on-the-fly* inside `nnp-train`
itself every run (the newer, documented-as-preferred path -- no
`nnp-norm` step needed).

Reading `Training.cpp`'s handling of `normalize_data_set == "force"`
directly confirms the calibration depends on GPU-dispatched force
predictions: `convLength = sigmaForceNnp` (std. dev. of the *NNP's own
predicted* forces from an initial pass over the whole training set --
literally the same `calculateForces()`/`GpuForces.cu` path already
isolated as producing wrong per-structure forces), `convEnergy =
sigmaForceNnp / sigmaForceRef`. `meanEnergy` comes from reference data
only, unaffected -- fitting neatly with energy always matching exactly
while forces don't. A very plausible causal chain: `GpuForces.cu`
corrupts raw forces -> corrupts the `sigmaForceNnp` statistic -> a
wrong global scaling constant gets baked in and applied to everything
downstream for the rest of the run.

**Tested directly and ruled out**: ran a GPU-linked `nnp-train` and a
CPU-linked `nnp-train` on identical fresh `input.nn` (same
`random_seed`, tanh, 1 epoch) and diffed the resulting computed
`mean_energy`/`conv_energy`/`conv_length`. **Identical to all 16
significant digits.** The aggregate `sigmaForceNnp` statistic --
computed from ~337 structures/rank of the real training set -- comes
out the same between backends, even though the same-scale, same-rank-
count `nnp-dataset` comparison on the 140-structure *test* set showed
every single structure's individual force prediction disagreeing.
Whatever's wrong doesn't corrupt this particular aggregate statistic
detectably, which rules out the clean "one wrong global constant
explains everything" story -- the actual per-call force corruption in
`GpuForces.cu` is still the real site of the bug, unexplained mechanism
still not identified.

### Follow-up: root cause narrowed to the magnitude of the normalization constants -- not units, not calibration correctness, the actual numeric scale

Pushed back on directly (rightly) when the working theory drifted
toward "different physical unit systems" -- units alone shouldn't
matter to unit-agnostic code, and a re-read of `input.nn` confirmed
water/magnetite/feldspar's data is in Angstrom while H2O_2G's is in
Bohr (confirmed via characteristic O-H bond lengths: water ~0.93,
H2O_2G ~1.82, ratio matching the Bohr/Angstrom conversion factor almost
exactly), but this alone doesn't explain a real bug in correctly-
normalized code. Redirected to comparing `input.nn` more carefully
instead, specifically around normalization.

**That comparison surfaced the real lead**: water's `normalize_data_set
force`-computed `conv_energy`/`conv_length` (`~2.75`/`~2.29`) are
727x/14x *smaller* in magnitude than H2O_2G's fixed, `nnp-norm`-computed
values (`~1997`/`~32.5`) -- not just "present vs. absent" as the
previous entry treated it, but genuinely different numeric scale. This
makes sense mechanistically: `force` mode calibrates `convLength` from
`sigmaForceNnp`, the standard deviation of the *untrained, randomly-
initialized* network's own force predictions -- an inherently small,
somewhat arbitrary number, unlike H2O_2G's presumably `ref`-mode-or-
similar calibration from actual reference-data statistics.

**Tested directly, and it's decisive**: substituted H2O_2G's actual
`conv_energy`/`conv_length` values into water's `input.nn` (same
weights, same data, same everything else) and reran the CPU-vs-GPU
`nnp-dataset` force comparison:

| Config | conv_energy | conv_length | Median \|force diff\| | Max |
| --- | --- | --- | --- | --- |
| water, own on-the-fly calibration | `2.75` | `2.29` | `0.43` | `3.83` |
| water, H2O_2G-scale constants substituted in | `1997` | `32.5` | `0.00059` | `0.0053` |
| H2O_2G, own fixed calibration | `1997` | `32.5` | `0.00027` | `0.0041` |

Same weights, same dataset, same code, same `GpuForces.cu` -- changing
only the normalization scale takes water from catastrophic corruption
to the same clean, floating-point-noise-level agreement H2O_2G already
had. **This is the real trigger.**

**Why this magnitude would matter mechanistically**: `Mode.cpp:739`
(`it->changeLengthUnitSymmetryFunctions(convLength)`) rescales every
symmetry function's length parameters (`eta`, `rs`, `rc`) by
`convLength` at setup time -- this is the *internal*, normalized
representation that actually flows through the rest of the
computation, including into `GpuForces.cu`'s kernels. Computed the
actual internal-unit cutoff radius for both systems (same *physical*
cutoff, per the earlier Bohr/Angstrom-conversion finding):
`rc_physical / convLength` gives water `2.77` vs. H2O_2G `0.369` --
water's internal geometric quantities are **~7.5x larger in magnitude**
for the identical physical system, purely because of its much smaller
`convLength`. Something in the GPU force-computation path is evidently
sensitive to this internal magnitude -- the exact operation/line
hasn't been pinned yet (this is a scale *trigger*, not yet the
mechanism itself), but the trigger is now solid, reproducible, and
mechanistically explained rather than mysterious.

**Practical mitigation, validated**: using fixed, appropriately-scaled
`mean_energy`/`conv_energy`/`conv_length` (the traditional `nnp-norm`-
style approach H2O_2G already uses, or `normalize_data_set ref` instead
of `force`) avoids the bug entirely, at least for water -- worth
verifying on magnetite/feldspar too before treating this as a general
fix. `normalize_data_set force`'s specific calibration -- deriving
`conv_length` from an untrained network's own arbitrary force-
prediction scale -- looks like the actual footgun, independent of
whether a deeper GPU code fix is ever pursued.

### Follow-up: the clean "magnitude alone" story doesn't survive further corroboration -- there's a real second, system-specific factor still unidentified

Two direct challenges (rightly skeptical of a single-experiment
conclusion) led to two more corroborating tests, and the picture got
messier, not cleaner -- worth recording exactly as found rather than
forcing a tidy narrative.

**Test 1: a genuine, water-specific `nnp-norm` calibration** (not
H2O_2G's borrowed numbers) -- built and ran `nnp-norm` on water's own
dataset, giving `conv_energy=61.6`/`conv_length=51.4` (derived from
reference/DFT statistics, unlike `force` mode's untrained-network
statistics). Verified directly (via the program's own "SETUP:
NORMALIZATION" printout, in both the CPU and GPU runs) that these
values were actually parsed and used, not a test-harness artifact.
Result: median force diff `0.0192`, max `0.171` -- a real 22x
improvement over the `force`-mode baseline (`0.43`), but ~30x worse
than the H2O_2G-borrowed test's `0.00059`, even though `nnp-norm`'s own
`conv_length` (`51.4`) is *larger* than H2O_2G's (`32.5`). That already
breaks a clean "`conv_length` magnitude alone" story.

**Test 2: the missing complementary direction** -- take H2O_2G (the
clean system) and switch *it* to `normalize_data_set force` instead of
its fixed constants, same 1-epoch-then-`nnp-dataset` methodology.
Result: H2O_2G's `force`-mode calibration gives `conv_energy=41.7`/
`conv_length=0.679` (smaller than water's own `force`-mode
`conv_length=2.29`!), and its force disagreement is median `0.00748`,
max `0.082` -- **measurably worse than H2O_2G's own clean state
(`0.00027`, ~28x)**, so H2O_2G is not fully immune to this bug either.
But it's dramatically *milder* than water's `force`-mode corruption
(`0.43`, ~57x better), despite H2O_2G's `conv_length` being smaller.

**All five data points together:**

| System | Calibration | conv_energy | conv_length | Median \|force diff\| |
| --- | --- | --- | --- | --- |
| water | `force` mode (own) | `2.75` | `2.29` | `0.43` |
| water | `nnp-norm` (own) | `61.6` | `51.4` | `0.0192` |
| water | H2O_2G-borrowed | `1997` | `32.5` | `0.00059` |
| H2O_2G | `force` mode | `41.7` | `0.679` | `0.00748` |
| H2O_2G | own fixed (`nnp-norm`) | `1997` | `32.5` | `0.00027` |

Neither `conv_length` alone nor `conv_energy` alone is monotonic across
all five rows (e.g. H2O_2G's `force`-mode `conv_length` is smaller than
water's, yet its error is 57x better; `nnp-norm`'s `conv_energy` for
water, `61.6`, is *larger* than H2O_2G `force`-mode's `41.7`, yet gives
a *worse* result, `0.0192` vs `0.00748`). **Conclusion, stated
honestly**: normalization-constant magnitude is real and reproducibly
matters *within* a given system (water's own three tests are cleanly
ordered: smaller constants -> worse), and `force` mode measurably hurts
even H2O_2G -- but there is a second, still-unidentified factor tied to
the system itself (water vs. H2O_2G -- structure size/count,
composition, symmetry function set, or something else not yet isolated)
that determines *how much* a given normalization scale hurts. Single-
variable explanations (magnitude alone, `force`-mode-vs-fixed alone)
are each falsified by at least one row in this table. Not yet resolved;
the honest state is "two real, interacting factors, second one
unidentified" rather than a clean single root cause.

---

### Follow-up: corrected understanding -- every entry above since the "RETRACTED" marker was chasing a scripting bug, not a real GPU bug

Everything from the "RETRACTED" box earlier in this file through the
entry directly above this one is invalid. This entry explains exactly
how that was found, what was actually re-verified, and what's really
still true and still open. Read this entry, not the retracted ones,
for the current state of knowledge.

**How the bug was found.** Debugging this "systemic GPU force
corruption" properly required tracing actual numbers, not just
aggregate output -- so `Mode.cpp` was temporarily instrumented (gated
behind an env var, never committed to the shared build) to dump the
exact inputs `GpuForces.cu`'s kernels receive (the CSR self-term
`dGdrSelf`, the neighbor edge list `edgeTarget`/`edgeOwnerDEdGIndex`/
`edgeDGdr`, and `dEdG`) and its raw output force, for one chosen
structure at a time. The plan: manually recompute the force from the
dumped inputs in Python (double precision, the exact same summation
`selfForceKernel`/`pairForceKernel` do), and compare that recomputation
against both the GPU's actual dumped output and the CPU-trusted force
for the same structure -- if the bug were real, this would show
exactly where it lives (CPU-side topology construction vs. the GPU
kernel itself).

Traced structure 0 for both water and H2O_2G first, as a sanity check
before hunting the "worst" structures the retracted entries had
identified. Both matched the GPU's actual output to the recomputation
to machine precision (~1e-16) -- already suspicious, since structure 0
"shouldn't" have been clean if the corruption were as pervasive as
claimed. Went straight for the previously-identified worst offender
instead: water structure 86, atom 163, reported earlier (in the now-
retracted entries) as showing a `0.171` disagreement -- the single
largest in the whole dataset. Traced it directly, twice (once at 1
rank, once at 4 ranks, matching the exact rank count the original
"bad structure" identification used). **Both traces showed perfect
agreement** -- GPU actual, Python recomputation, and CPU-trusted force
all matched to machine precision, for the *exact* atom that had been
reported as the worst-disagreeing case in the whole investigation.

That result -- a structure independently and specifically flagged as
badly wrong, found to be perfectly correct under direct inspection --
was the signal that the measurement itself, not the GPU code, was
broken. Went back to the comparison script and found it: every
CPU-vs-GPU `forces.comp` comparison in the retracted entries used

```
join -j 1 <(awk '{printf "%s_%s %s\n",$1,$2,$4}' cpu.forces.comp | sort) \
          <(awk '{printf "%s_%s %s\n",$1,$2,$4}' gpu.forces.comp | sort)
```

keyed on `structure-index_atom-index`. `forces.comp` prints **three**
consecutive rows per atom (the *x*, *y*, and *z* force components),
all sharing that same key. `join` on a non-unique key emits the
cartesian product of every matching line on each side -- for 3
matching rows per file per key, that's 9 output lines per atom instead
of 3, most of them cross-pairing *mismatched* components (CPU's *x*
joined to GPU's *y*, etc.). The resulting "differences" were mostly
just the size of ordinary inter-component variation in a force vector,
not any real CPU-vs-GPU disagreement -- and large enough, sampled
across enough atoms, to look exactly like a genuine, reproducible,
structure-dependent corruption pattern.

**Redid every key comparison correctly** (row-aligned via `paste`,
after verifying both files list the same structures/atoms in the same
order -- true whenever `<shuffle>=0` and rank counts match, which
every retracted test used) instead of the broken `join`:

| System | Retracted (broken) result | Corrected result |
| --- | --- | --- |
| water, `normalize_data_set force` | median `0.43`, max `3.83` | max `0` (bit-identical) |
| water, `nnp-norm` calibration | median `0.019`, max `0.171` | max `1e-10` |
| magnetite | 150/150 structures "disagree" | max `1e-11` |
| feldspar | 150/150 structures "disagree" | max `0` |
| H2O_2G, `normalize_data_set force` | median `0.00748`, max `0.082` | max `1e-10` |

Every single one is clean. `GpuForces.cu` produces the same forces as
the CPU implementation, full stop -- across every system, every
normalization configuration, every activation function, every rank
count tested in this whole investigation. There was never a per-
structure corruption pattern, never an activation-function dependence,
never a normalization-magnitude dependence, never a "second system-
specific factor." All of that was the shape of noise from a broken
join, mistaken for a signal because it was consistent enough (across
many re-runs of the *same broken script*) to look reproducible.

**What survives, because it never touched the broken script**: water's
real `nnp-train` GPU run -- full iterative Kalman-filter training, not
a single-shot force evaluation -- genuinely diverges to `NaN` by epoch
6 (energy RMSE `1.29E+34` already at epoch 1). That number came
straight from `nnp-train`'s own log, printed independently of any of
this investigation's comparison tooling. It's real.

**Where that leaves the actual open question**: given `GpuForces.cu` is
now conclusively correct, water's training-time divergence can't be a
force-computation bug. The far more likely explanation, and one this
file already documented as expected behavior in H2O_2G's own 100-epoch
validation earlier on ("Kalman-filter training is a recursive,
chaotically-sensitive process where tiny floating-point order-of-
operation differences between CPU and GPU math compound fast over many
sequential updates"): ordinary, harmless run-to-run floating-point
noise (different summation order between CPU and GPU reductions,
present in *every* run, GPU or not) gets chaotically amplified over
many sequential weight updates -- bounded and harmless for H2O_2G
(its trajectory diverges from CPU's but both still converge), but for
water's specific configuration the amplification runs away to `NaN`
instead of staying bounded. This is a genuine, unresolved question
about Kalman-filter training-loop numerical stability for water's
configuration specifically -- not a GPU correctness bug, and not yet
investigated with a methodology anyone should trust without triple-
checking the comparison script first this time.

**Practical takeaway for anyone touching this in the future**: never
compare two `forces.comp` files with a join/merge keyed on anything
less specific than a genuinely unique row identifier (structure index
+ atom index + component index, or just row order if both files are
verified to list rows in identical order first). A non-unique-key join
silently produces a cartesian product with no error or warning --
exactly the trap this whole investigation fell into.

### Follow-up: went digging in the Kalman-filter training loop instead of the (now-cleared) force kernel -- found and fixed a real, pre-existing n2p2 bug, but it's not the (or not the whole) cause of water's divergence

With `GpuForces.cu` conclusively cleared, the next step was tracing the
actual *training-loop* GPU code water's divergence runs through --
`gpuNnEnergyDEdcSum`/`gpuNnForceDFdcSum` (the energy/force Jacobian-sum
kernels, used only during weight updates) and `GpuKalmanFilter` (the
Kalman gain/covariance update itself). Neither of these had been
examined before; both are completely separate code paths from
`GpuForces.cu`, which is only used for force *prediction* (evaluation),
never for the Jacobians a weight update actually needs.

**Traced the real weight-update sequence directly**, this time with a
methodology built to avoid the earlier mistake: temporarily instrumented
`Training.cpp` to dump `pu.error`/`pu.jacobian`/`weights` immediately
before and after each of the first several weight updates, for both CPU
and GPU builds, compared with a small hand-written Python script doing
plain array indexing (no text-key joins at all). Because
`force_energy_ratio=10`, force updates vastly outnumber energy updates,
so "force update 0" is the genuinely first-ever weight update of the
whole run -- confirmed directly: `weightsBefore` matches to `0.0` between
CPU and GPU. For that exact update: the **Jacobian matches to ~13
significant digits** (ordinary rounding), but the **error vector is
wildly different** (CPU `[0.277, 35.19, 6.37, -19.74]` vs. GPU
`[-41.18, 140.99, 69.54, -123.48]`, different in magnitude and even
sign on the first component) -- despite identical starting weights.

Cross-checking against `train-log.out` (which independently records
each update's selected structure/atom/component per rank) showed why:
of the 4 per-rank candidates in that first update, 3 matched CPU's
candidates exactly, but **one rank picked a genuinely different
structure/atom** than CPU did (structure 919/atom 223 vs. CPU's
154/29). So part of what's going on is CPU and GPU builds selecting
different training candidates from the very first update, not
(only) computing the same candidate differently.

**Separately, `train-log.out` also showed something stranger**: the
"count" column (an update counter, `Training::countUpdates` -- a class
*member*, not a stack-local) printed as `4437379212764330956` on the
GPU build instead of a small integer like CPU's `1`. Reinterpreting
that exact 64-bit pattern as an IEEE-754 double gives `4.7152e-12` -- an
entirely unremarkable floating-point value, not "random" garbage. That
strongly suggested a stray *double* being written into a `size_t`
class member via an out-of-bounds write, not simple uninitialized
memory. `compute-sanitizer` found nothing (0 errors across all ranks)
-- but it only instruments CUDA/device memory, not plain host C++ heap
or stack corruption, so that result doesn't clear a host-side bug the
way it would a device-side one.

**Rebuilt with AddressSanitizer instead** (the right tool for host-side
corruption; confirmed it doesn't touch `libnnpgpu`'s separate nvcc
build, only the plain `.cpp` files) and reran the identical
reproduction. It caught a real, unambiguous bug on the first try: a
**stack-buffer-overflow** in `Dataset::sendStructure()`
(`src/libnnptrain/Dataset.cpp:256`), inside an `MPI_Pack` call packing
the local variable `ts`. The bug: `ts` is declared `int` (4 bytes) but
packed with the `MPI_SIZE_T` datatype (8 bytes) --

```cpp
int ts = 0;                              // Dataset.cpp:256
ts = s.comment.length() + 1;
MPI_Pack(&ts, 1, MPI_SIZE_T, buf, bs, &p, comm);   // reads 8 bytes from a 4-byte int
```

-- three times over in `sendStructure()` (comment length,
`numAtomsPerElement` size, atom count) and mirrored identically in
`recvStructure()` (`Dataset.cpp:468`), which unpacks the same fields
with the same 4-byte-`int`-vs-8-byte-`MPI_SIZE_T` mismatch. This is
genuine undefined behavior -- reading/writing 4 bytes past a 4-byte
stack variable -- on *every* structure sent to or received by any
non-rank-0 MPI process, in both directions, for a fixed field every
single call.

**`git blame` traces this to `9769786`, "v2.0.0 release candidate,"
2018-10-29 -- upstream n2p2's original author, years before this
project's GPU port started.** It is not a bug this project introduced.
Its visible impact is undefined-behavior-dependent on incidental stack
layout, which is exactly consistent with everything observed: harmless
in the CPU build's particular compilation/stack layout, consequential
in the GPU build's (different flags, different linked object code,
different layout) -- and invisible to any single-rank run, since rank 0
never calls `sendStructure`/`recvStructure` for its own structures.
This is very plausibly a contributor to more of this investigation's
"rank-count/build-dependent, hard to pin down" flakiness than just
water's case specifically, though that's not separately confirmed.

**Fixed** (`int ts` -> `size_t ts`, both functions, matching the
`MPI_SIZE_T` datatype already used to pack/unpack it -- a strictly
widening, uses-`.length()`/`.size()`-consistently change; syntax-
checked clean). **Rebuilt with the fix still under AddressSanitizer and
confirmed the overflow is gone: 0 ASan errors**, versus a caught
overflow on every prior run. This is a real, validated bug fix, kept
regardless of what else this investigation finds.

**But it does not resolve water's divergence.** Rerunning the exact
same reproduction with the fix applied: `ENERGY` still explodes to
`1.64E+41` by epoch 1 -- same order of magnitude, same severity as
before the fix. So this bug, while real, is not the (or not the whole)
explanation for water's training-time blowup. The candidate-selection
divergence found in the per-update trace above is still open and
unexplained by this fix. **Status at the point of pausing to
consolidate**: one real, confirmed, fixed bug (kept); the actual
mechanism behind water's NaN divergence is still not identified.
Next steps, not yet started: trace *why* CPU and GPU select different
candidates from the very first update (a control-flow/RNG-consumption
question, not obviously related to the `ts` bug just fixed), and
check whether the fixed `ts` bug changes anything about magnetite/
feldspar's (so far apparently healthy) GPU training given it affects
every multi-rank run, not just water's.

### Follow-up: found and fixed the real cause -- a stale GPU force-topology cache, populated once with the wrong cutoff radius and never invalidated

Picked the candidate-selection thread back up and ruled it out cleanly
before finding the real mechanism.

**Dead ends, each tested directly, not assumed:**
- **Candidate selection isn't it.** Water's `input.nn` uses
  `selection_mode 2` (`SM_THRESHOLD`), which *recomputes* energy/force
  live (via GPU-dispatched `calculateAtomicNeuralNetworks`/
  `calculateForces`) as part of deciding the next update candidate --
  a real, GPU-sensitive feedback path in principle. But switching to
  `selection_mode 0` (`SM_RANDOM`, a deterministic, pre-shuffled
  round-robin with zero live-error-based decisions, hence provably
  identical candidate selection on CPU and GPU) **did not stop the
  explosion** -- if anything it was worse (`ENERGY` to `6.5E+39` by
  epoch 1 vs. the baseline's `7E+29`). Candidate selection was a red
  herring.
- **Not MPI/rank-count.** The exact same divergence reproduces at a
  single rank (`FORCE` epoch-0 RMSE `1.22489` CPU vs. `10.3339` GPU,
  `ENERGY` NaN by epoch 1) -- actually faster/worse than at 4 ranks.
  Rules out `Dataset::sendStructure`/`recvStructure` (the already-fixed
  `ts` bug and anything else in that path) entirely, since a 1-rank run
  never calls them.
- **Not a fresh memory-safety bug.** Rebuilt both CPU and GPU with
  AddressSanitizer again and reran the 1-rank reproduction: **0 ASan
  errors on both**, divergence still present and identical in
  magnitude. Whatever this is, it's stale-but-valid data, not an
  out-of-bounds access.
- **Not `memorize_symfunc_results`.** Disabling this cache (which
  memoizes each structure's symmetry-function `G`/`dGdr` across calls)
  changed nothing -- GPU `FORCE` at epoch 0 stayed at `10.3339`.

**The break: compared inputs, not just outputs, for the single worst
atom found in the original (correctly, `paste`-based) per-atom trace**
(water structure 164, atom 110 -- `Fnnp` CPU `-4.008` vs. GPU `-27.03`
in that trace). Instrumented `Training::calculateError`'s own loop
(temporary, env-var-gated dump, not part of the fix) to print, right
after `calculateSymmetryFunctionGroups()` and right after
`calculateForces()`, for this exact atom: the symmetry-function vector
`G`, the neighbor count, `Mode::maxCutoffRadius`, the NN backprop
vector `dEdG`, and the final force. Result, CPU vs. GPU, same weights,
same 1-rank run:

| Quantity | CPU | GPU | Agreement |
|---|---|---|---|
| neighbor count | 89 | 89 | exact |
| `G` (27 values) | -3.2118385392636684E-01 ... | -3.2118385392636639E-01 ... | ~13 significant digits |
| `dEdG` (27 values) | -1.3127141953446186E+00 ... | -1.3127141953446220E+00 ... | ~13 significant digits |
| structure energy | 5.3775638709580016E+02 | 5.3775638709580085E+02 | ~14 significant digits |
| **force** | **(0.9909, -0.0553, 0.7572)** | **(11.698, -0.6527, 8.940)** | **~11.8x off, all 3 components, same ratio** |

Every quantity that feeds `calculateForces()` matched to ordinary
floating-point precision -- **except the force itself**, which was
wrong by a strikingly *uniform* scalar multiple across all three
components. A uniform multiplicative error from otherwise-correct
inputs is the signature of a stale scaling/geometry factor baked into
one specific number, not a scattered numerical bug.

**Root cause, found by reading `Mode::calculateForces()`'s GPU branch
directly (`src/libnnp/Mode.cpp`, short-range self+pair force kernel
added by this project's own GPU port, documented in the "GpuForces.cu"
follow-up entry above):**

```cpp
// Mode.cpp, calculateForces(), original code:
static set<size_t> gpuForceTopologyUploaded;
bool const firstCallForThisStructure =
    gpuForceTopologyUploaded.insert(structure.index).second;
...
if (firstCallForThisStructure)
{
    // build dGdrSelf/edge list from the CURRENT neighbor list...
    gpuForcesUploadTopology(...);   // uploaded ONCE per structure.index, ever
}
// dEdG rebuilt and re-uploaded every call, unconditionally
gpuForcesCompute(...);
```

This cache is a real, deliberate, and previously-validated performance
optimization (topology -- `dGdrSelf`/the pair edge list, derived from
the neighbor list -- is one to two orders of magnitude bigger than
`dEdG`, and re-uploading it every call measured markedly worse). Its
premise: geometry never changes during training, so upload the
expensive part once per structure and just refresh the cheap,
weights-dependent `dEdG` every call. **True for the actual training
loop -- false exactly once, at startup.**

`Training::dataSetNormalization()` (only for `normalize_data_set force`
or `ref`, water's config) calibrates `conv_energy`/`conv_length` by
running one real forward+force pass per structure *using the network's
fresh random weights*:

```cpp
// Training.cpp, dataSetNormalization(), the calibration loop:
s.calculateNeighborList(maxCutoffRadius);   // pre-rescale cutoff
calculateSymmetryFunctionGroups(s, true);
calculateAtomicNeuralNetworks(s, true);
calculateEnergy(s);
if (useForcesLocal) calculateForces(s);     // <- first-ever call: caches PRE-rescale topology
s.clearNeighborList();
```

This is the very first time `calculateForces()` is called for each
structure in the process -- so it's the call that populates the
topology cache, using whatever cutoff radius was in effect *before*
normalization. Later in that same function, once the force statistics
are in hand, `conv_length` gets set and every symmetry function's
length parameters are rescaled (`if (normalize) { ... setupSymmetryFunctions(); ... }`),
which changes the cutoff radius and hence the *real* neighbor
list/topology going forward. `Training::calculateNeighborLists()`
(called once more, right after, from `nnp-train.cpp`'s driver) rebuilds
the CPU-side neighbor list correctly with the new cutoff -- but the
GPU-side topology cache, keyed only by `structure.index` with no
notion of "this structure's cutoff changed," has no way to know its
cached `dGdrSelf`/edge list are now stale. Every `calculateForces()`
call for the rest of the entire run -- all of real training --
silently combines **fresh, correctly-rescaled `dEdG`** with **stale,
pre-rescale topology**, for every structure that was part of the
calibration pass (i.e. all of them). Whether a given atom's force
comes out visibly wrong depends on whether that atom's actual
neighbor set differs between the old and new cutoff radius -- which is
exactly why atom 0 of structure 0 (unaffected, topology happened to be
identical either way) showed no divergence at all while atom 110 of
structure 164 showed an 11.8x error: same bug, different local
geometry.

This also explains, retroactively and correctly this time, the
earlier-retracted observation that `normalize_data_set force` looked
implicated and `ref`/`stats-only` looked clean: `force` and `ref` both
run this same force-calibration loop (only `stats-only` skips it,
reusing already-correct normalization from a previous run) -- it was
never about the *magnitude* of `conv_length`, it was about which
normalization modes call `calculateForces()` before the one-time
cutoff rescale.

**Fixed**: added `Mode::resetForceTopologyCache()` (`Mode.h`/`Mode.cpp`,
`#ifdef N2P2_GPU`-gated like the rest of this port) which simply clears
the cache; `Training::dataSetNormalization()` calls it once, right
after the rescale block, so the *next* `calculateForces()` call per
structure -- the first one in real training -- re-uploads the
corrected topology. (The cache itself moved from a function-local
`static` to a small anonymous-namespace file-scope variable in
`Mode.cpp`, purely so a class method can reach it to clear it; no
behavior change for the already-validated once-per-structure caching
itself.)

**Verified, not just plausible:**

| Config | CPU | GPU before fix | GPU after fix |
|---|---|---|---|
| 4 ranks, water's real `input.nn` (`selection_mode 2`), `FORCE` epoch 0 | 1.24088 | 5.95323 | **1.24088** |
| 4 ranks, same, `ENERGY` epoch 1 | 5.89E-04 | 1.64E+41 (NaN by later epochs) | **5.89E-04** |
| 1 rank, `selection_mode 0`, `FORCE` epoch 0 | 1.22489 | 10.3339 | **1.22489** |
| 1 rank, same, `ENERGY`/`FORCE` epoch 1 | 6.97E-04 / 4.229E-02 | NaN / NaN | **6.97E-04 / 4.229E-02** |

GPU now matches CPU to the same ordinary floating-point precision seen
everywhere else in this project once the stale cache is gone -- across
both the original reproducing configuration and the isolation
configuration used throughout this investigation. Water's real GPU
training divergence, open since the "benchmark three real-world
systems" entry much earlier in this file, is resolved.

**Provenance and scope**: unlike the `ts`/`MPI_SIZE_T` bug (pre-existing
upstream n2p2, 2018), this bug was introduced by this project's own GPU
force-kernel port (the topology-caching optimization documented in the
"GpuForces.cu" follow-up above) -- it only exists because that
optimization's "geometry never changes" premise has one real exception
this codebase actually exercises. It affects any GPU-enabled `nnp-train`
run using `normalize_data_set force` or `ref` (both call
`calculateForces()` during calibration); `stats-only` is unaffected.
Magnetite/feldspar's GPU training was never observed to diverge like
water's, most likely because their geometry/cutoff-rescale interaction
doesn't happen to flip any atom's neighbor set the way water's does --
not because they're immune to the same stale-cache bug. Worth a
follow-up check once resources allow: rerun magnetite/feldspar with the
fix and confirm their (already-healthy-looking) results are unchanged,
and don't rule out subtler, currently-invisible force errors of the
"atom 0" kind (same bug, but topology happens not to change) lurking in
runs that looked clean only because nothing forced a comparison at the
per-atom level.

**Left deliberately unfixed and flagged in code**: `Mode::calculateForces()`'s
separate 4G-electrostatics topology cache (`gpuElecForcesTopologyUploaded`,
same exact pattern) is not cleared by `resetForceTopologyCache()`. 4G/HDNNP_Q
training is out of scope this pass and untested this session
(`dataSetNormalization()` itself refuses to run for 4G/HDNNP_Q at
stage 1); if 4G training work resumes and uses `force`-based
normalization, this cache needs the identical fix and verification
before its forces can be trusted.

### Follow-up: honest, resource-matched DCGP comparison for magnetite (2 nodes/224 ranks, matching GPU's 2-node/8-GPU footprint)

The original 10-epoch timing table above compared GPU's 2-node/8-GPU
magnetite run against a **1-node** DCGP-112 run -- not resource-matched
(GPU's OOM fix, spreading `GpuForces.cu`'s per-structure device cache
across 2 nodes, was a correctness necessity, not a choice). Reran
magnetite on DCGP at 2 nodes/224 ranks (`train_dcgp_112.slurm`,
generalized earlier to read `$SLURM_NTASKS`/`$SLURM_JOB_NUM_NODES`
dynamically) for a real apples-to-apples comparison, mirroring the
feldspar request:

| Config | Per-epoch time |
| --- | --- |
| Booster CPU, 32 cores, 1 node | 3244.2s (54.07 min) |
| Booster GPU, 2 nodes/8 GPUs | 1201.1s (20.02 min) |
| DCGP-112, 1 node/112 ranks | 1562.5s (26.04 min) |
| **DCGP-112, 2 nodes/224 ranks** | **1180.7s (19.68 min)** |

At matched 2-node scale, **DCGP is marginally faster than GPU** (19.68
vs. 20.02 min/epoch, ~2% -- within noise, effectively a tie), not the
2.6x-3x GPU lead the 1-node-DCGP comparison implied. DCGP's own 1-node
&rarr; 2-node scaling (26.04 &rarr; 19.68 min, ~1.32x from 2x the ranks)
shows the same sub-linear-but-real improvement already established for
feldspar and the earlier core-count scan (Phase 3b) -- consistent, not
a new finding. This 2-node DCGP run predates the stale-force-topology-
cache fix documented above; since magnetite's GPU forces were never
observed to diverge like water's (this fix's correctness impact on
magnetite, if any, is a still-open follow-up noted above), these timing
numbers stand on their own regardless -- timing is unaffected by force
correctness either way, same reasoning as the original 10-epoch table.
Feldspar's matching 2-node DCGP run was still in progress at the time
of writing; see the next entry (or `sacct -j 53821293`) for its result
once complete.

### Follow-up: feldspar's 2-node DCGP result, and water's first healthy GPU 10-epoch run since the topology-cache fix

Feldspar's 2-node/224-rank DCGP run completed: **61.62 min/epoch**,
vs. Booster GPU's 2-node/8-GPU **18.36 min/epoch** -- unlike magnetite,
GPU still wins clearly here (~3.4x), even at matched node count.
Feldspar and magnetite give genuinely different answers to "does DCGP
catch GPU once node count is matched" -- not a contradiction, just a
real difference between the two systems (dataset/structure size,
presumably, though not separately isolated).

Also reran water's full 10-epoch production benchmark on GPU
(`train_booster_gpu.slurm`, 32 ranks/4 GPUs, the real dataset and
`input.nn` -- not the reduced debug subset used throughout the topology-
cache investigation) with the fix in place. **Clean, healthy
convergence, matching CPU almost exactly** -- `FORCE` RMSE at epoch 10:
`2.80304E-02` (GPU, this run) vs. `2.80532E-02` (CPU, the existing
10-epoch baseline). No divergence, no NaN, first time water's GPU
training has produced a trustworthy result since it was first flagged
as broken.

Full 10-epoch wall-clock recap, all three systems, every configuration run:

| System | Booster CPU (32 cores, 1 node) | Booster GPU (32 ranks) | DCGP-112 (1 node) | DCGP-112 (2 nodes/224 ranks) |
| --- | --- | --- | --- | --- |
| water | 47.79 s/ep (477.9s total) | **7.28 s/ep (80.9s total, 4 GPUs/1 node, fixed)** | 27.32 s/ep (273.2s total) | -- |
| magnetite | 54.07 min/ep | 20.02 min/ep (2 nodes/8 GPUs) | 26.04 min/ep | 19.68 min/ep -- ties GPU |
| feldspar | 136.27 min/ep | **18.36 min/ep** (2 nodes/8 GPUs) | 77.98 min/ep | 61.62 min/ep -- GPU still ~3.4x faster |

Water needed only 1 node/4 GPUs (no OOM issue, unlike magnetite/
feldspar's larger structure counts) and is now GPU's clearest win of
the three: 6.6x faster than CPU, 3.75x faster than DCGP-112. All three
systems' GPU numbers are now trustworthy end to end -- water via this
session's topology-cache fix, magnetite/feldspar via the earlier
per-structure force-correctness re-verification (the "corrected
understanding" follow-up much earlier in this file). Magnetite/
feldspar have not yet been rerun with the topology-cache fix
specifically (their GPU forces were never observed to diverge the way
water's did, so this is a should-still-check, not a known problem --
see the fix's own follow-up entry above).
