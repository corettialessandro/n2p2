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
`H`/`xi`) is now done — see `e2e/`'s Stage 5 below.

Next steps (not yet done): a force-fit Kalman update (this project's config
also does force updates, just far less often — `short_force_fraction
0.0041` — and force-fit uses `calculateDFdc`'s output as its Jacobian
instead of `calculateDEdc`'s, a straightforward extension of the same
wiring); batched-GEMM kernels instead of one-thread-per-atom for Phase 3's
NN forward/backward and this phase's `updateP`; and Phase 6 (build system
integration — a `makefile.cuda`/`N2P2_GPU` flag actually linking this code
into `nnp-train`, without which none of it is reachable from the real
training binary, whatever else gets validated standalone).
