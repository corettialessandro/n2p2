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
  and why**: the `update()` step itself measurably sped up — `timing.out`'s
  `F_upd` column (the force-branch Kalman update time, which dominates
  since force candidates vastly outnumber energy candidates) dropped from
  `12.14s` (CPU) to `0.39s` (GPU), a genuine `~31×`. But the *energy*-branch
  update column, `E_upd`, got **slower** (`1.58s → 5.37s`), and — more
  importantly — `F_com` (MPI communication time for the force branch)
  ballooned from `2.96s` to `50.45s`, and the **overall epoch got slower
  overall** (`156.0s → 211.2s`, about `1.35×` slower), even though the
  piece of code this pass actually touched got dramatically faster in
  isolation. The likely explanation: this cluster's job allocates a
  **single physical GPU shared by all 32 MPI ranks** (`--gres=gpu:1`,
  `--ntasks-per-node=32`, no MPS configured) — each rank runs the *exact
  same* redundant Kalman computation (by design: every rank
  independently recomputes the identical update from identical
  `MPI_Allgatherv`'d data, avoiding a separate broadcast step), so 32
  separate CUDA contexts now contend for one device far more often per
  epoch than either Jacobian call site did. `MPI_Allgatherv`'s wall time
  is set by whichever rank is slowest to arrive, so uneven GPU-context
  scheduling delays across ranks plausibly show up as inflated
  "communication" time even though the actual payload didn't change. This
  is a single epoch's measurement, not a scientifically thorough timing
  study, but the direction is clear enough not to overclaim: correctness
  is solid, but realizing this call site's speedup as a net `nnp-train`
  win likely needs a different execution model (e.g. only rank 0
  computing the update and broadcasting `w`, instead of 32-way redundant
  GPU computation) rather than more of what worked for the embarrassingly
  parallel Jacobian branches.

Next steps (not yet done): investigate the MPI/GPU-contention effect above
properly (multi-epoch timing to separate one-time CUDA context setup from
a recurring per-call cost; consider a rank-0-computes-and-broadcasts
redesign for the Kalman step specifically, since redundant 32-way
computation is what turns "32 processes, 1 GPU" from a non-issue into a
bottleneck at this call site's calling frequency); extend all three
dispatches above from `HDNNP_2G` to `HDNNP_4G` (needs the extra charge
input neuron and electrostatics coupling handled); and, now that all three
of Phase 4's profiled cost centers are wired into the real binary, a
proper multi-epoch wall-clock benchmark of `nnp-train` as a whole would
be a meaningful exercise — though per the finding above, the answer isn't
a foregone "faster" until the GPU-contention question is resolved.
