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

Next steps (not yet done): wire more symmetry-function types through this
same grouped-kernel + `G`/`dGdx`/`neighborDGdx`-storage pattern, and/or start
on the actual force-assembly scatter-add that consumes `neighborDGdx,Dy,Dz`
(Phase 3).

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

Next steps (not yet done): batch the forward/backward/Jacobian passes
properly with `cuBLAS gemmStridedBatched` instead of one-thread-per-atom
sequential math (these steps' version, matching Phase 2's kernels' style,
proves correctness first). Force assembly (the single biggest cost center)
is covered next, in `force/`.

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

Next steps (not yet done): wire real `dEdG`/`dGdx`/`neighborDGdx` values
through (from `nn/` and `soa/`'s symmetry-function kernels) instead of
synthetic ones, for an actual end-to-end energy+force pipeline on one
structure; determinism (`atomicAdd` on doubles is not run-to-run
bit-reproducible — the plan flags this explicitly, an alternative would be
a neighbor-major segmented reduction); and, more broadly, Phase 4 (Kalman
filter weight updater) or Phase 6 (build system integration, without which
none of this is reachable from the real `nnp-train` binary).
