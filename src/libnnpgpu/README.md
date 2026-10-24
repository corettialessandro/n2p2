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

### 7. CPU and GPU results agree within tolerance

The CPU code path stays unchanged by the GPU port: a build without `GPU=1`
gives bit-identical results to upstream. Tests compare the GPU path against
the CPU path in the same test binary (through an internal switch that is not
visible to users) and against the stored CPU references, with the GPU
tolerance profile of `test/cpp/tolerance.h`.

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

Ranks are spread over the GPUs of a node automatically (rule 3).

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
