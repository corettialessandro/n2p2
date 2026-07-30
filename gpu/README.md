# GPU port work

Code for the `nnp-train` GPU port described in `../GPU_PORTING_PLAN.md`. Kept
separate from `src/` so it doesn't touch the shared, upstream-tracked build
tree until it's ready to be integrated (see the plan's Phase 6) -- as of
Phase 6's first pass, one real, narrow integration point now exists in
`src/` too: `src/libnnpgpu/` and a `Mode::calculateAtomicNeuralNetworks()`
call site gated behind `N2P2_GPU`/`make GPU=1`, described at this file's end.

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
