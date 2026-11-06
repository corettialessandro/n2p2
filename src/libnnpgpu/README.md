# libnnpgpu: GPU backend of n2p2

CUDA implementations of the most expensive parts of n2p2, built only with
`make GPU=1` (see `src/makefile.gnu` for the CUDA settings). This file states
the rules the GPU backend follows. Every change to the GPU code must keep
them, and every newly ported feature goes through the checklist at the end.

## Rules

### 1. The build decides where n2p2 runs

- A `GPU=1` build uses the GPU for everything that has a GPU implementation.
  A plain build never uses it.
- There is no keyword in `input.nn` (or any other input file) to choose the
  hardware. Input files describe the model and the training; the same files
  must give the same results with any build.
- `GPU=1` compiles all code (libraries, applications, tests, interfaces) with
  `-DN2P2_GPU`. Mixing objects compiled with and without it is not supported.

### 2. Building never checks for a GPU

Building with `GPU=1` on a machine without a GPU is allowed: on clusters the
code is usually compiled on login nodes without GPUs, and continuous
integration compiles the GPU code on machines without GPUs. `nvcc` does not
need a device, the target architecture is set by `PROJECT_NVCCFLAGS`.
Building without `GPU=1` on a machine with a GPU is of course allowed too.

### 3. Running a GPU build without a usable GPU is an error

A `GPU=1` build never falls back to the CPU silently. When a component that
runs on the GPU is set up, it checks the device and throws an error with a
clear message if

- no CUDA device is visible, or
- the device cannot run the compiled code (architecture mismatch).

The check happens at setup, before any expensive work, not at the first use
deep inside a calculation. Programs that never reach GPU code (for example
`nnp-convert` or `nnp-select`) run without a GPU, also in a `GPU=1` build.

Each process uses the device given by its local rank modulo the number of
devices on its node (`gpuSelectDevice()` in `GpuDevice.h`).

### 4. Model features without a GPU implementation run on the CPU, loudly

If the device is fine but a feature of the model has no GPU implementation
(a network architecture, a symmetry function type, a Kalman filter variant,
...), that part runs on the CPU and a prominent warning in the log names it.
The rest still runs on the GPU. The log always states which parts run on the
GPU and which on the CPU.

### 5. Errors are exceptions

GPU errors throw `std::runtime_error` (checks in `GpuError.h`), like all other
errors in n2p2. Library code never calls `exit()`.

### 6. GPU state is owned and freed

Device memory and caches belong to an object (for example the structure or
the Kalman filter they serve), are invalidated when what they cache changes,
and are freed in its destructor. No process-global caches keyed by indices.
The one process-wide object is the pool of released device memory of the
symmetry functions (`GpuSymmetryFunctions.cu`): it holds no data, only free
blocks, reused by size, limited to a quarter of the cache budget of the
process.

### 7. CPU and GPU results agree within tolerance

The CPU code path stays unchanged by the GPU port: a build without `GPU=1`
gives bit-identical results to upstream. Tests compare the GPU path against
the CPU path in the same test binary (through an internal switch that is not
visible to users) and against the stored CPU references, with the GPU
tolerance profile of `test/cpp/tolerance.h`.

### 8. Choices for performance only between equivalent paths

A GPU build may still run part of a ported feature on the CPU when that is
faster, but only between two implementations of the same calculation (same
results up to round-off), by a fixed, documented rule, and never in a way
that depends on user input. Example: the force assembly copies the
symmetry function derivatives of a structure to the GPU only at the second
force calculation with the same geometry and only within a memory budget;
one-off geometries and structures beyond the budget are assembled on the
CPU, which is faster for them. Such choices are counted (e.g.
`Mode::getForceAssemblyCounts()`) and listed in the table below. The same
applies to launch configurations: the symmetry function kernel chooses its
block size from the number of atoms of the launch (fixed threshold); its
results do not depend on the block size at all.

## Running with several MPI ranks per GPU

Each MPI rank works on its own structures and sends many small pieces of
work to the GPU. Without NVIDIA's Multi-Process Service (MPS) the processes
sharing a GPU take turns on it, and this contention can make a GPU build
slower than a CPU build. Start the MPS daemon in the job script whenever
several ranks share a GPU:

```
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$SLURM_JOB_ID
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
mpirun -np 32 -x CUDA_MPS_PIPE_DIRECTORY -x CUDA_MPS_LOG_DIRECTORY nnp-train
echo quit | nvidia-cuda-mps-control
```

Measured on one Leonardo Booster node (32 cores, A100), one training epoch
with 32 ranks (water, 1254 structures for 2G, 250 for 4G stage 2):

| Setup | 2G | 4G stage 2 |
|---|---|---|
| CPU build | 33.3 s | 48.9 s |
| GPU build, 1 GPU, no MPS | 29.2 s | 49.4 s |
| GPU build, 1 GPU, MPS | 16.9 s | 29.5 s |
| GPU build, 4 GPUs, MPS | 15.6 s | 27.6 s |
| GPU build with GPU force assembly, 1 GPU, MPS | 15.9 s | 16.1 s |
| GPU build with GPU force assembly, 4 GPUs, MPS | 8.9 s | 13.9 s |

(The rows without MPS and the first two MPS rows were measured before the
force assembly was ported.)

Work-item symmetry function kernel (several chunks of an atom in parallel),
2G training throughput on two oxide data sets with many elements and
symmetry functions per element, 32 ranks on 4 GPUs (MPS), updates per
second in the first epoch: 343 to 507 (330-atom structures, 5 elements) and
493 to 647 (up to 384 atoms, 2 elements).

Ranks are spread over the GPUs of a node automatically (rule 3).

## Performance

Leonardo Booster node (32 cores, 4 A100, MPS) against a full CPU node of
Leonardo DCGP (112 cores) or the CPUs of the same Booster node.

4G training, SPC/Fw water (630 atoms, 1254 structures, 12 bohr cutoff),
32 ranks:

| Stage | CPU | GPU |
|---|---|---|
| Stage 1, one epoch | 48.7 s (same Booster node) | 8.6 s |
| Stage 2, one epoch | 300 s (DCGP node) | 117 s |

2G MD with LAMMPS (`pair_style hdnnp`, water H2O_RPBE-D3, ns/day; GPU
rows gpu-port 4ca8737, 233,280 atoms at f80aaae, i.e. without the atom reuse
of the LAMMPS interface, about 6% at this size; the work-item symmetry
function kernel raised the 32-rank GPU rate to 4.40 for 8,640 atoms and
0.679 for 69,120 atoms):

| Atoms | DCGP node, 112 ranks | Booster CPUs, 32 ranks | 4 GPUs, 32 ranks (MPS) | 4 GPUs, 4 ranks |
|---|---|---|---|---|
| 8,640 | 1.47 | 0.39 | 4.04 | 1.89 |
| 69,120 | 0.19 | 0.055 | 0.668 | 0.227 |
| 233,280 | 0.064 | 0.016 | 0.190 | |

The GPU part per step is limited by the symmetry function kernel; with few
ranks per GPU the host work of each rank (neighbor lists) matters more,
which is why the LAMMPS interface reuses its atoms between steps
(`Atom::reset()`, no reallocation of the neighbor lists).

4G MD (`pair_style hdnnp/4g`, SPC/Fw water, Ewald 1e-3, time per step):
630 atoms 0.126 s (Booster CPUs, 32 ranks) vs 0.027 s (4 GPUs, 32 ranks);
5040 atoms 1.23 s vs 0.144 s. `nnp-predict` of 5040 atoms of the same model:
78 s on one core vs 24 s on one GPU.

## Checklist for porting a feature to the GPU

1. The CPU code path is unchanged; the GPU path is a dispatch around it, not
   a copy of it.
2. **Device check:** decide where this feature is set up and make sure the
   device check (rule 3) runs there, before expensive work. Check which
   applications and interfaces now reach GPU code and therefore need a GPU;
   update the list below. Re-check earlier features: the right place for the
   check can move when a new feature is ported.
3. Unsupported variants of the feature fall back to the CPU with a warning in
   the log (rule 4).
4. GPU state has an owner, is invalidated on change and freed (rule 6).
5. Errors use the checks of `GpuError.h` (rule 5).
6. Tests compare GPU against CPU in the same binary, plus the full suite with
   `GPU=1` (rule 7).

## Ported features and their device checks

| Feature | GPU part | Device check | Fallback to CPU |
|---|---|---|---|
| Kalman filter (`KalmanFilter`, both types) | P on the device; X = P . H and P = scale (P - K . X^T) + q I | `KalmanFilter` constructor (via `GpuKalmanFilter`) | none needed |
| Atomic neural networks (`NeuralNetwork`, `Mode::calculateAtomicNeuralNetworks()`): 2G, 4G (elec, short), HDNNP_Q | forward pass and derivatives with respect to the inputs, all atoms of an element as one batch; neuron statistics | when the weights of a network type are set up: `Mode::readNeuralNetworkWeights()`, `Training::randomizeNeuralNetworkWeights()` (via `GpuNeuralNetwork`); not in `setupGeneric()`, which tools without network evaluation use too | none needed (any depth, all activation functions, normalized neurons) |
| Training Jacobians (`Training::update()`): energy and force updates (2G, 4G stage 2), charge updates (4G stage 1) | derivatives with respect to all connections summed (charges: weighted by dError/dChi) over the atoms of an element (`NeuralNetwork::calculateDEdcBatchGpu()`, `calculateDFdcBatchGpu()`) | same as the atomic neural networks (they use the same GPU twins) | not ported yet, warning in the training log: force updates with `N2P2_FULL_SFD_MEMORY` |
| Force assembly (`Mode::calculateForces()`): short-range forces and the 4G electrostatic dChi/dr sums | sums of weight * dG/dr over all terms of each atom (deterministic, one warp per atom); the topology (`GpuForceTopology`) is kept with its structure (`Structure::gpuData`) for the current geometry (`Structure::geometryStamp`) and freed with the symmetry function derivatives | together with the "short" networks (`Mode::setNeuralNetworkGpuEnabled()`) | performance choice (rule 8): first force calculation of a geometry, or topology beyond the memory budget (half the device memory, shared by the processes using it); not needed for symmetry function derivatives calculated on the GPU (see below) |
| Symmetry functions (`Mode::evaluateNNP()`, LAMMPS `pair_style hdnnp` and `hdnnp/4g`, training errors and update candidates): types 2, 3 and 9, all cutoff functions and scaling types | values and derivatives of all atoms in one kernel (one block per atom, neighbors grouped by element in shared memory; work items of 32 neighbors of a chunk's elements, taken by whichever warp of the block is free, one lane per neighbor; cutoff function once per neighbor and distinct cutoff; item sums added per chunk in a fixed order, so the results do not depend on the block size; 128 threads per atom below 1024 atoms per launch, 64 from 1024 on, at most 128 registers per thread); the derivatives stay on the GPU (`GpuSymmetryFunctionData`, kept with its structure in `Structure::gpuData` under the geometry stamp, its device memory reused through a process-wide pool) and are contracted there with dE/dG (and lambda * dChi/dG for 4G) for the forces (`Mode::calculateForces()`, `InterfaceLammps::getForces()` and `getForcesChi()`; the two passes of a 4G-HDNNP in LAMMPS share them); the derivatives with respect to the coordinate of a force update are gathered there for the training Jacobian and dQ/dr (`GpuSymmetryFunctionData::derivativesOfTarget()`); only values, weights and forces cross to the host | together with the networks of either type (`Mode::setSymmetryFunctionGpuEnabled()`; 4G stage 1 sets up only the "elec" networks) | other symmetry function types or more than 32 elements: all symmetry functions of the model on the CPU, warning in the log; performance choice (rule 8): only where forces follow right away or the derivatives are only contracted (`evaluateNNP()`, LAMMPS interface, error calculations, update candidates and force updates of the training via `Training::calculateSymmetryFunctionsTraining()`), while the tools use the derivatives on the host and get them from the CPU, as does training with `memorize_symfunc_results` (derivatives of all structures kept); structures with an atom with more than 2048 neighbors on the CPU |
| Charge equilibration and force Lagrange multipliers of the 4G LAMMPS interface (`fix hdnnp`, `pair_style hdnnp/4g`) | the dense matrix of the whole structure is built (real space over periodic images, reciprocal space as one matrix product) and LU-factorized once per step on the GPU of rank 0 (`GpuChargeEquilibration`), charges and multipliers are two solves with it, broadcast to all ranks; replaces the iterative minimizations and their communication | when the solver is first used (`PairHDNNP4G::solveChargeEquilibrationGpu()`), on rank 0 | performance and feature choice (rule 8): PPPM, triclinic boxes and matrices beyond the memory budget use the iterative solvers, stated in the log |
| Charge equilibration of `Mode` (4G-HDNNP: `nnp-predict`, training stages 1 and 2, tools) | the A matrix built on the CPU is LU-factorized on the GPU instead of the QR factorization on the CPU (`GpuChargeEquilibration`, the same direct solution up to round-off); all solves with it run there (`Structure::solveA()`): charges, dQ/dChi and dQ/dJ (one solve with N right-hand sides), dQ/dr and the force Lagrange multipliers; one solver per process, refactorizing from the stored matrix when it was used for another structure since | together with the "elec" networks (`Mode::setChargeEquilibrationGpuEnabled()`) | performance choice (rule 8): structures whose matrix and N right-hand sides exceed a quarter of the device memory per process on the CPU (size limit stated in the log) |

## Programs that require a GPU in a GPU build

- All programs evaluating neural networks, from the moment they read or
  initialize the weights: `nnp-train`, `nnp-norm2`, `nnp-predict`,
  `nnp-checkf`, `nnp-checkdw`, `nnp-comp2`, `nnp-dataset` (when predicting
  with weights) and the LAMMPS pair styles (`InterfaceLammps`).
- `nnp-train` with the Kalman filter updater (`updater_type 1`) also for the
  updaters, created in `Training::setupTraining()`.

Programs that never set up weights run without a GPU: `nnp-scaling`,
`nnp-atomenv`, `nnp-convert`, `nnp-cutoff`, `nnp-dist`, `nnp-norm`,
`nnp-prune`, `nnp-select`, `nnp-symfunc`, `nnp-dataset` without weights.
