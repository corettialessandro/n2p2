# GPU Porting Plan: `nnp-train` on LEONARDO Booster

Branch: `gpu-portability` | Status: design/brainstorm + Phase 0 profiling done, no GPU code changes yet | Author: drafted with Claude, 2026-07-17 | Updated: 2026-07-22 (Phase 0 profiling, see §3a)

## 1. Goal

Make `nnp-train` (and the shared `libnnp`/`libnnptrain` compute path it depends on) run efficiently
on LEONARDO's Booster partition: 3456 nodes, each with a single 32-core Intel Ice Lake CPU (Xeon
Platinum 8385) and four NVIDIA A100 SXM4 64GB GPUs connected by NVLink 3.0 (source: Turisini et al.
2024, attached). This document is a brainstorm/design doc only — it does not change any compute code.

## 2. Target hardware, in numbers that matter for this port

| Property | Value | Why it matters |
|---|---|---|
| GPUs/node | 4x A100 64GB SXM4 | 1 MPI rank per GPU is the natural mapping |
| GPU-GPU (NVLink 3.0) | 200 GB/s per pair, 600 GB/s aggregate/GPU | Cheap to all-reduce weight gradients within a node |
| CPU-GPU (PCIe4) | 32 GB/s per GPU, 128 GB/s total | Host↔device transfer of structures/symmetry functions is not free — batch it |
| GPU memory | 64 GB HBM2e, 1.6 TB/s BW | Plenty for weights (n2p2 nets are KB–MB); dataset/symmetry-function cache is the thing that can blow this budget |
| CPU cores/node | 32 (64 threads) Ice Lake, AVX-512 | Still useful for I/O, neighbor lists, orchestration — don't strand them |
| Inter-node | 2x HDR100 IB, ~400 Gbps/node | NCCL/CUDA-aware MPI both work well here |
| Node RAM | 512 GB | Comfortable for staging full datasets |
| Software stack | RHEL8, SLURM, cuDNN + NCCL preinstalled | No need to hand-roll multi-GPU collectives |

Practical consequence: the natural parallel decomposition is **1 MPI rank = 1 GPU = 1/4 node**,
i.e. `srun --ntasks-per-node=4 --gpus-per-task=1`, layered on top of the existing MPI structure-based
domain decomposition n2p2 already has. This reuses `training.setupMPI()` almost unchanged; the new
work is what happens *inside* each rank.

## 3. What `nnp-train` actually does today (call graph, from reading `src/application/nnp-train.cpp`
and `src/libnnptrain/Training.cpp`)

```
nnp-train.cpp
 └─ Training::setupGeneric / setupSymmetryFunction* / setupRandomNumberGenerator
 └─ Training::distributeStructures()      MPI: structures split across ranks
 └─ Training::selectSets()                random train/test split
 └─ Training::calculateNeighborLists()    per-structure, CPU, O(N·k)
 └─ Training::loop()                      epoch loop
     └─ Training::update("energy"/"force"/"charge")   [Training.cpp:2256, ~900 lines]
         for each pattern in mini-batch:
           Mode::calculateSymmetryFunctionGroups()     <- SymGrp*/SymFnc* (~40 files, libnnp/)
           Mode::calculateAtomicNeuralNetworks()        <- NeuralNetwork::propagate() per atom
           NeuralNetwork::calculateDEdG / calculateDFdc <- backward pass, per-weight Jacobian
         MPI_Gatherv/Allgatherv of error + Jacobian to build the update
         Updater::update()  →  GradientDescent  OR  KalmanFilter
```

Everything downstream of "per pattern" is per-atom, per-symmetry-function work that today runs
serially on one CPU core per MPI rank (OpenMP exists behind `#ifdef _OPENMP` but is **disabled by
default** in `makefile.gnu` — `#-fopenmp` is commented out). MPI parallelizes across *structures*,
not within one.

### Where the time goes (structural argument, not yet profiled on real hardware)

1. **Symmetry function evaluation + derivatives** (`libnnp/SymFnc*.cpp`, `SymGrp*.cpp`, ~40 classes,
   ~9k lines total) — for every atom, loop over neighbors (radial terms) and neighbor pairs (angular
   terms), evaluate cutoff functions, accumulate `G` values and `dG/dx` derivatives. This is the
   classic O(N·k) / O(N·k²) NNP cost center and is **embarrassingly parallel across atoms** — no
   dependency between atoms of the same structure. This is the single best GPU target.
2. **NN forward pass** (`NeuralNetwork::propagate()`, `NeuralNetwork.cpp`) — small dense MLP per
   atom (per-element weights, typically O(10²–10³) parameters). Individually tiny, but thousands of
   atoms × same-element weights per batch turns into a batched GEMM — a textbook cuBLAS/cuDNN use case
   if atoms are grouped by element.
3. **NN backward pass** (`calculateDEdG`, `calculateDEdc`, `calculateDFdc`) — same shape as the
   forward pass, same batching opportunity.
4. **Jacobian assembly + MPI communication** (`Training.cpp:2906` "PART 3: Communicate error and
   Jacobian") — currently `MPI_Gatherv`/`MPI_Allgatherv` to rank 0 or all ranks. This is already a
   known scaling bottleneck on CPU and will be *more* visible once the per-rank compute is GPU-fast;
   it needs to become an NCCL/CUDA-aware-MPI reduction, not a gather.
5. **Weight update** (`GradientDescent.cpp`, 223 lines vs `KalmanFilter.cpp`, 353 lines, uses
   `Eigen/LU`) — see §5.3, this is the trickiest piece, not the easiest.

### 3a. Phase 0 results (2026-07-22): real profiling, and a correction to §3

§3 above was a structural argument, written before any real measurement — it predicted symmetry
function evaluation would be "the single best GPU target." Real profiling **contradicts the
ranking**, though not the overall list of cost centers. This matters because it changes which phase
should be implemented first.

**Setup**: `temp/H2O_2G` dataset (1254 structures, 630 atoms/structure, water), CPU build (this
branch's baseline, no GPU code), 32 MPI ranks on one `boost_usr_prod` node, 2 training epochs,
`updater_type=1` (Kalman) with `update_strategy=0` (Combined/global — the harder case per §5.4).
Two independent measurements, both from the same run:

1. **Coarse, exact wall-clock split** (`timing.out`, built-in per-property Stopwatch instrumentation
   already in `Training.cpp` — `sw[k+"_err"]`/`"_com"`/`"_upd"`, no code changes needed): force updates
   dominate (275/epoch vs. 35 energy updates), and within force updates: **~79% of total epoch time**
   in the combined "compute" bucket (`_err`: symmetry functions + NN forward + NN backward + Jacobian
   assembly, PARTS 1–2 of `Training::update()`), **~9%** in MPI communication (`_com`, PART 3), **~8%**
   in the weight updater (`_upd`).
2. **Fine split within that 79% bucket** (`perf record -g --call-graph=dwarf` on MPI rank 0 only —
   representative since all ranks run the same per-atom code on different structures; 1.3M samples,
   `cycles` event, 0 lost samples): resolves the `_err` bucket by symbol:

   | Bucket | Function(s) | % of **total** wall time |
   |---|---|---|
   | **Force assembly** (dE/dG · dG/dx scatter-reduce over neighbors) | `Mode::calculateForces` (+`Atom::calculatePairForceShort`) | **~46.5%** |
   | **NN backward / Jacobian** (dE/dG, d²E/dGdc, dF/dc) | `NeuralNetwork::calculateDFdc` (+`calculateD2EdGdc`, `calculateDxdG`) + `calculateDEdG` | **~30.8%** |
   | NN forward pass | `Mode::calculateAtomicNeuralNetworks` (+`propagate`/`propagateLayer`) | ~7.7% |
   | Symmetry function evaluation | `Mode::calculateSymmetryFunctionGroups` (+`SymGrpExpAngn::calculate` etc.) | ~4.1% |

**The correction**: symmetry function evaluation is cheap (~4%), not the dominant cost §3 assumed.
NN forward pass is also modest (~8%). The two real dominant costs are **force assembly** (the
dE/dG · dG/dx scatter-reduce, §5 Phase 3's "force assembly" bullet — previously an implementation
footnote, not flagged as a priority) and the **NN backward/Jacobian pass** (§5 Phase 3's main
content) — together **~78% of total wall time**. Practical consequence: **Phase 3 (NN
forward/backward + force assembly) should be prioritized over Phase 2 (symmetry functions)** for
this dataset/config — Phase 2 is still worth doing (it's still embarrassingly parallel, still a clean
first GPU kernel for the reasons §7 gives), but it is not where the wall-clock time is.

**Kalman filter attribution — resolved (2026-07-22, follow-up pass):** re-profiled with a much larger
DWARF unwind buffer (`--call-graph=dwarf,65528` vs. the default 8KB) to test whether the unattributed
Eigen GEMM/LU kernel samples (`gebp_kernel`, `product_selfadjoint_matrix`, etc.) were mis-nested under
`KalmanFilter::update` due to unwind truncation. The larger buffer changed nothing (`KalmanFilter::
update`'s reported Children went 5.2% → 7.8%, consistent with run-to-run noise, not a fix), so we
inspected raw call chains directly with `perf script`. They resolve cleanly and completely:
`gebp_kernel → product_selfadjoint_matrix::run → KalmanFilter::update → Training::update →
Training::loop → main` — a 6-frame chain, trivially within even the original 8KB buffer. **The Eigen
kernels are confirmed children of `KalmanFilter::update`**; `Structure.cpp`'s only other dense-Eigen
usage in this codebase is the 4G charge-equilibration path, inactive for this 2G/H2O config, so there
is no other candidate origin. The mismatch was `perf report`'s flat `--sort=overhead,symbol -g none`
summary view under/over-counting `Children` in a way that doesn't reflect the real call tree (a report
accounting quirk, not a data problem) — combined with `perf`'s `cycles` event being sensitive to
per-region CPU frequency scaling (e.g. AVX-512 downclocking differs between the SF/force-assembly-
dominated majority of the run and Kalman's more bursty calls), which makes a cycles-based percentage
an unreliable stand-in for a wall-clock-time percentage when comparing across code regions with
different vectorization profiles. **Conclusion: the original `timing.out` wall-clock measurement
(~9% total, `_upd` Stopwatch bracket) was correct all along** — a wall-clock timer wrapped tightly
around `Updater::update()` cannot under-measure what happens inside it, Eigen internals included, by
construction. Phase 4 (Kalman) should be sized against **~9% of current wall time**, not the ~21%
floated as a possibility in the first pass of this doc; that concern is retracted.

**MPI communication, unresolved**: *its active-CPU share (~2%) is much lower than its wall-clock share (~9%,
  `timing.out`'s `_com`)*: a blocked/waiting rank doesn't burn CPU cycles, so cycle-sampling is blind
  to time lost waiting on stragglers in the Allgatherv. That gap is most likely rank load imbalance,
  not communication throughput — a different problem than what NCCL/CUDA-aware MPI (§5 Phase 5) fixes
  on its own; worth checking per-rank structure/atom-count balance before assuming Phase 5's plan
  addresses it.

**Note for future profiling runs on this cluster**: `perf record --call-graph=dwarf` at default
sampling frequency produced an **11GB** `perf.data` file for a single rank over 2 epochs (~330s). The
home filesystem quota here is only 50GB and filled to 100% mid-run; point `perf record -o` directly
at `/leonardo_work/L-AUT_Giane_26/acoretti/NEURALCPM/porting/` (997GB free at time of writing), not
the home directory. `-F 200` (200Hz, vs. the ~4000Hz adaptive default) plus a larger unwind buffer
(`--call-graph=dwarf,65528`) brought a comparable 2-epoch capture down to ~4.3GB with still-ample
statistics (66K samples) — a reasonable default for future runs on this codebase. Also useful:
`perf script -i <file>` dumps raw per-sample call chains, which is the reliable way to confirm a
symbol's true caller when `perf report`'s flat summary view looks inconsistent with the source.

## 4. Existing groundwork already in this repo: `libnnpif/CabanaMD`

`src/libnnpif/CabanaMD/{ModeCabana,ElementCabana}*.h` already implement a **Kokkos-based**,
GPU-portable version of symmetry-function scaling and `calculateAtomicNeuralNetworks` — but only for
the *inference* path used inside LAMMPS/CabanaMD molecular dynamics (`Kokkos::View` weight/SF tensors,
`Kokkos::parallel_for` kernels at `ModeCabana_impl.h:828` and `:923`). This is valuable prior art:
it proves the data layout (flattened `View<T***>`/`View<T****>` tensors keyed by
[atom][element][neuron]) works, and it's a second pair of eyes' worth of design decisions we don't
have to re-litigate. It is **not** sufficient for training, because:

- It's forward-only — no backward pass (`dEdG`, `dEdc`), no per-weight Jacobian, no force-consistent
  chain rule assembly for the loss gradient.
- It targets a single evaluation of a frozen potential, not a training loop with weight updates,
  mini-batching, and an optimizer.

Recommendation: use `ElementCabana`/`ModeCabana`'s tensor layout and kernel style as a **reference**
for the training-side symmetry-function and forward-pass kernels (same View shapes, same per-element
batching idea), but write dedicated CUDA/cuBLAS kernels for `libnnptrain` rather than trying to bolt
training onto the Cabana/Kokkos MD interface — the user's stated preference is direct CUDA C++, and
the training data flow (mini-batches of structures, not an MD neighbor list) is different enough that
reuse would mostly be copy-adaptation anyway.

## 5. Phased porting strategy

### Phase 0 — Baseline & instrumentation (no GPU code yet)
- Build with `-DEIGEN_USE_MKL_ALL`/OpenBLAS enabled and `-fopenmp` turned on as a CPU-only upper bound.
- Profile a representative `nnp-train` run (existing `examples/` inputs, e.g. the water or Cu
  examples already in the repo) with `perf`/Nsight Systems to confirm the §3 hypothesis quantitatively
  before writing kernels. Concretely: what fraction of `Training::update()` wall time is symmetry
  functions vs. NN forward/backward vs. Jacobian communication vs. updater? This determines whether
  Phase 2 or Phase 5 is actually the bottleneck worth prioritizing.
- Establish correctness baselines: the existing `test/cpp` unit tests plus energy/force regression
  values from a small system, used later to validate GPU kernels bit-for-bit-ish (double precision,
  tolerance-based).

### Phase 1 — Data layout redesign (prerequisite for everything else)
`Structure`/`Atom` (`libnnp/Structure.cpp`, `Atom.cpp`) are array-of-structs, one `std::vector<Atom>`
per structure, with per-atom `std::vector` neighbor lists and symmetry-function derivative storage.
This is fine on CPU, hostile to GPU (pointer-chasing, no coalescing, no batching). Needed:
- A struct-of-arrays "batch" representation: flatten N structures worth of atoms into contiguous
  `double*`/`Kokkos::View`-style arrays (positions, per-SF values `G`, per-SF-per-neighbor
  derivatives `dGdx`), grouped/sorted by element type so per-element GEMMs are contiguous.
- Pinned host staging buffers + CUDA streams for async H2D transfer overlapped with the next batch's
  CPU-side neighbor list construction (neighbor lists likely stay on CPU initially, see Phase 2 note).
- This phase touches `Training::allocateArrays`, `Dataset.cpp`, and the symmetry function memory
  layout flag `N2P2_FULL_SFD_MEMORY` (already an alternate memory layout switch in the existing
  makefile — worth reading closely, it may be the closest existing analog to what a GPU layout needs).

### Phase 2 — Symmetry functions on GPU
- **Update (§3a, 2026-07-22 profiling)**: real profiling puts this at only ~4% of wall time on the
  test dataset — lower priority than Phase 3 below, which accounts for ~78%. Still worth doing first
  as a *proof-of-concept* (lowest-risk, cleanest kernel, see §7's suggested first PR), but don't expect
  it to move the needle on end-to-end training time by itself.
- Highest-value, lowest-risk target: purely local per-atom(-pair/-triple) math, no cross-atom
  dependencies except within a fixed neighbor list.
- Keep neighbor list construction on CPU initially (`Training::calculateNeighborLists()`) — it's
  cheap relative to symmetry-function evaluation and avoids the complexity of GPU cell lists in v1.
  Revisit only if Phase 0 profiling shows it matters.
- One CUDA kernel per symmetry-function *family* (radial exp, angular narrow/wide, compact/weighted
  variants — the ~13 `SymFnc*`/`SymGrp*` class pairs collapse into far fewer numerical kernels once
  virtual dispatch is removed; today's polymorphism is a compile-time template/switch dispatch on
  GPU, not runtime virtual calls). One thread (or one warp) per atom, looping over that atom's
  neighbor list; derivatives (`dGdx`) computed in the same kernel to avoid a second neighbor-list walk.
- Validate numerically against the existing CPU `SymFnc::calculate()` implementations per symmetry
  function type before moving on — this is the highest-value regression test to write first.

### Phase 3 — Neural network forward/backward on GPU
- **Update (§3a, 2026-07-22 profiling)**: this is the actual priority, not Phase 2. Real profiling
  attributes ~78% of total wall time to this phase's two pieces combined — force assembly
  (`Mode::calculateForces`/`calculatePairForceShort`, ~46.5% alone) and the NN backward/Jacobian pass
  (`calculateDFdc`/`calculateD2EdGdc`/`calculateDEdG`, ~30.8%). NN forward pass itself is cheap (~8%).
  Recommend implementing/validating force assembly and the backward pass before or alongside the
  forward-pass batching described below, not after it.
- Group atoms by element (weights differ per element type but are shared across all atoms of that
  type in the batch) → this turns "many tiny per-atom MLPs" into a handful of batched GEMMs, one set
  per element, per layer. cuBLAS `gemmBatched`/`gemmStridedBatched` or cuDNN's dense/RNN-adjacent
  batched primitives are the right tool; hand-written kernels only where activation functions
  (`CoreFunction`, `CutoffFunction`) need custom, non-GEMM elementwise ops.
- `calculateDEdG`/`calculateDEdc`/`calculateDFdc` follow the same batched-GEMM pattern (they're
  reverse-mode passes through the same layer structure).
- Force assembly (`dE/dG · dG/dx`, summed over neighbors, `Training::collectDGdxia`) is a
  scatter-add — implement with atomics or, better, a per-atom neighbor-major layout that turns it
  into a segmented reduction instead of atomics, for reproducibility (atomics on doubles can make
  results run-to-run non-deterministic, which matters for regression testing against CPU results).

### Phase 4 — Weight updaters: the actually-hard part
- **Update (§3a, resolved 2026-07-22)**: Kalman's real cost is confirmed at **~9% of total wall
  time** (`timing.out`, exact wall-clock). A first profiling pass suggested it might be as high as
  ~21% due to unattributed Eigen GEMM/LU kernel samples, but raw call-chain inspection (`perf script`)
  confirmed those Eigen kernels genuinely are `KalmanFilter::update`'s children — the discrepancy was
  a `perf report` flat-summary accounting quirk plus cycles-vs-wall-clock frequency-scaling
  sensitivity, not a real hidden cost. Size Phase 4 against ~9%, not ~21% — it's real and worth doing
  (still the highest-*risk* phase per the numerical-stability concerns below), but it is not competing
  with Phase 3 for priority the way symmetry functions turned out not to.
- **`GradientDescent`** (Adam-style, `GradientDescent.cpp`): trivially GPU-friendly — elementwise
  vector ops over the full weight vector. Straightforward cuBLAS axpy/elementwise kernel, or even
  just keep weights resident on GPU and do the update there without a round trip.
- **`KalmanFilter`** (`KalmanFilter.cpp`, uses `Eigen/LU`): this is the actual risk in the whole
  plan. It's an extended Kalman filter over the *full weight vector* with an O(P²) covariance matrix
  `P` and an O(P³) matrix solve per update, where P = number of weights. Two mitigating facts found
  in the code: (a) `Training.h` already has `numUpdaters` — confirms n2p2 already supports one
  Kalman filter **per element** rather than one global filter (`KalmanType`, decoupled mode), which
  caps P per updater at a single element's weight count instead of the whole network's; (b) A100 has
  cuSOLVER for batched Cholesky/LU. Plan: port the per-element decoupled case first (batched
  cuSOLVER, one batch entry per element/per-GPU-rank), and treat the fully-global Kalman mode as
  either (i) explicitly out of scope for GPU in v1 (fall back to CPU/Eigen path, documented as a
  known limitation), or (ii) a stretch goal once the rest of the pipeline is validated. This should
  be called out explicitly to whoever picks up implementation — it's the one place where "port to
  GPU" doesn't obviously mean "faster."

### Phase 5 — Multi-GPU / multi-node scaling
- **Update (§3a, 2026-07-22 profiling)**: measured MPI communication's *active* CPU cost is small
  (~2%, `perf`), far below its ~9% wall-clock share (`timing.out`'s `_com`). That gap looks like
  rank load imbalance (idle time waiting on stragglers), not communication throughput — check
  per-rank structure/atom-count balance before assuming NCCL/CUDA-aware MPI alone fixes it.
- 1 MPI rank ↔ 1 GPU ↔ 1/4 node, matching Booster's 4×A100 layout (`srun --ntasks-per-node=4
  --gpus-per-task=1`).
- Replace today's `MPI_Gatherv`/`MPI_Allgatherv` Jacobian collection (`Training.cpp:2906`, "PART 3")
  with NCCL allreduce (intra-node, over NVLink) composed with inter-node CUDA-aware MPI/NCCL — this
  is the piece most likely to dominate wall time once per-rank compute is GPU-accelerated, per §3
  point 4. Needs its own profiling pass once Phases 2–4 exist.
- Decide precision policy explicitly: n2p2 trains in `double` throughout for physical accuracy of
  energies/forces. A100 FP64 is ~9.7 TFLOPS (half of FP32, per the attached datasheet numbers) —
  still far above any CPU path, so there's no forced need to drop to mixed precision, but it's worth
  a flagged experiment (e.g. FP32 symmetry functions + FP64 accumulation) once correctness is
  established, not before.

### Phase 6 — Build system & CI
- New `makefile.cuda` (mirroring the existing `makefile.gnu`/`.intel`/`.llvm` pattern) or a CMake
  CUDA-language target; link `cublas`, `cusolver`, `nccl`; new compile-time switch (e.g.
  `N2P2_GPU`, following the existing `N2P2_NO_SF_GROUPS`/`N2P2_NO_MPI`-style flags in
  `src/makefile.gnu`) so the CPU path stays the default and fully intact — this is a fork with
  active upstream tracking (`upstream/CompPhysVienna/n2p2`), so minimizing footprint on the shared
  code paths matters for future merges.
- `.github/workflows/n2p2-ci.yaml` has no GPU runner; CI can only validate that GPU code *compiles*
  (nvcc available on a CPU-only runner) — actual correctness/perf runs need to happen on Booster
  itself via SLURM batch jobs, which should be scripted (`sbatch` template) as a deliverable, not
  left as tribal knowledge.

### Phase 7 — Validation
- Numerical: GPU vs CPU energies/forces/Jacobians on the existing small `examples/` systems, to
  double-precision tolerance.
- Scaling: strong/weak scaling on Booster from 1 GPU up to a meaningful multi-node allocation,
  measured against the CPU+MPI baseline from Phase 0.
- Regression: extend `test/cpp` with GPU-path equivalents of existing tests, gated behind
  `N2P2_GPU` so the suite still runs on CPU-only CI.

## 6. Effort estimate (order of magnitude, not a commitment)

This is realistically **several person-months** of HPC/CUDA engineering for a solid, validated port,
not a single session — GPU rewrites of physics simulation codes routinely take this long even for
teams that do this for a living. Rough phase weights: Phase 1 (layout) and Phase 2 (symmetry
functions) are the biggest single chunks of new code; Phase 4 (Kalman filter) is the biggest *risk*;
Phase 5 (communication) is the part most likely to need a second pass after real profiling on
Booster. Phase 0 should happen first regardless of how the rest is sequenced — it's cheap and it
tells you whether this whole plan's priority ordering is even right for your actual datasets (system
sizes, element counts, and batch sizes in electrochemical-systems workloads may shift the balance
between symmetry-function cost and NN cost versus what's assumed here).

## 7. Suggested first concrete PR (if/when this moves to implementation)

Smallest slice that proves the approach end-to-end: pick one symmetry function type (e.g.
`SymFncExpRad`, the simplest radial one, `libnnp/SymFncExpRad.cpp`) and one small example system,
implement Phase 1's SoA layout + a single CUDA kernel for that one symmetry function + host/device
round trip, validate numerically against `SymFncExpRad::calculate()` on CPU. This exercises the
build-system change, the data layout, and the validation harness without committing to the full
kernel library up front — everything after that is "more of the same pattern," which is a much
easier thing to parallelize across contributors or sessions than the open-ended version of this task.

## 8. Open questions for whoever picks this up

1. What are realistic problem sizes for electrochemical-systems datasets (atoms/structure, number of
   elements, dataset size, batch size)? This directly determines whether GPU occupancy is achievable
   (GPUs need large batches to be worth it; if typical structures are small, minibatching *across*
   structures becomes mandatory, not optional).
2. Is the global (non-decoupled) Kalman filter actually used in current workflows, or is
   element-decoupled Kalman/plain gradient descent the practical default? This changes whether Phase
   4's biggest risk is in scope at all.
3. Does CINECA provide A100 allocation/dev access for iterative testing during implementation, or
   would development need to happen against a local single-GPU proxy first?
