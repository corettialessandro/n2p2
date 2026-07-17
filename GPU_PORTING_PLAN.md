# GPU Porting Plan: `nnp-train` on LEONARDO Booster

Branch: `gpu-portability` | Status: design/brainstorm, no code changes yet | Author: drafted with Claude, 2026-07-17

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
