.. _gpu:

Running on GPUs
===============

n2p2 can run the expensive parts of training, prediction and LAMMPS MD on
NVIDIA GPUs (CUDA). The build decides: a build with ``GPU=1`` uses the GPU, a
normal build never does. There is no input keyword for it, and the same input
files give the same results with both builds, up to round-off.

What runs on the GPU
--------------------

* Symmetry functions (types 2, 3 and 9) and their derivatives.
* The atomic neural networks (2G and 4G) and their derivatives.
* The force assembly.
* The training Jacobians and the Kalman filter.
* The 4G charge equilibration, in ``nnp-predict``, in training, and in LAMMPS
  (``fix hdnnp``, ``pair_style hdnnp/4g``).

If a model uses a feature without a GPU implementation (for example another
symmetry function type), that part runs on the CPU and the log says so with a
warning. ``src/libnnpgpu/README.md`` lists every ported feature, its fallbacks
and the rules the GPU code follows.

A ``GPU=1`` build stops with an error if no GPU is visible or if the GPU cannot
run the compiled code. Programs that do not evaluate neural networks (for
example ``nnp-convert`` or ``nnp-select``) run without a GPU.

Building
--------

Build as usual with ``GPU=1`` added, for example:

.. code-block:: none

   cd src
   make COMP=gnu GPU=1
   make COMP=gnu GPU=1 lammps-hdnnp

The CUDA settings are in ``src/makefile.gnu``:

* ``PROJECT_NVCCFLAGS``: set ``-arch`` to the compute capability of your GPU,
  for example ``sm_80`` for A100, ``sm_90`` for H100, ``sm_100`` for B200
  (``nvidia-smi --query-gpu=compute_cap --format=csv`` shows it).
* ``PROJECT_CUDA_HOME``: defaults to ``$CUDA_HOME``, which ``module load cuda``
  sets on most clusters.

Building does not need a GPU, so login nodes are fine. Two pitfalls seen on
clusters:

* Some MPI compiler wrappers add ``-L/usr/lib64``, so the system GSL is linked
  even when the GSL of a module was used for compiling. Check the result with
  ``ldd`` and, if needed, give the library path explicitly, for example
  ``PROJECT_LDFLAGS_BLAS="-L<gsl>/lib -Xlinker -rpath -Xlinker <gsl>/lib -lopenblas -lgsl -lgslcblas"``.
* Do not use commas in ``PROJECT_LDFLAGS_BLAS`` (such as ``-Wl,-rpath,...``):
  the LAMMPS build step uses this variable inside a ``sed`` expression.

Running
-------

Each MPI process uses one GPU, chosen by its rank on the node modulo the number
of GPUs. Usually several processes share one GPU. Then start NVIDIA's
Multi-Process Service (MPS) in the job script, otherwise the processes take
turns on the GPU:

.. code-block:: none

   export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$SLURM_JOB_ID
   export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$SLURM_JOB_ID
   mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
   nvidia-cuda-mps-control -d
   mpirun -np 32 -x CUDA_MPS_PIPE_DIRECTORY -x CUDA_MPS_LOG_DIRECTORY nnp-train
   echo quit | nvidia-cuda-mps-control

How many processes per GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^

Part of the work still runs in the CPU processes that feed the GPUs, so the
number of processes per GPU matters as much as the GPU itself:

* **Training and large MD systems:** use as many processes per GPU as the node
  has cores, up to the MPS limit of 48 processes per GPU.
* **Small MD systems** (about 10,000 atoms or fewer): about 8 processes per GPU.
  More processes split the system too finely.

Measured on a node with 4 H100 and 176 cores, speed relative to 8 processes per
GPU:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20

   * - Benchmark
     - 16 per GPU
     - 44 per GPU
   * - 2G MD, 8,640 atoms
     - 0.83
     - 0.47
   * - 2G MD, 233,280 atoms
     - 1.63
     - 2.43
   * - 4G MD, 17,010 atoms
     - 1.43
     - 1.48
   * - 2G training
     - 1.24
     - 1.52
   * - 4G training, stage 1
     - 1.66
     - 3.11
   * - 4G training, stage 2
     - 1.46
     - 1.91

Start-up time
^^^^^^^^^^^^^

Each process creates its own GPU context at the start. On some systems this
takes up to about 0.8 s per process, one after the other, so a run with
176 processes can spend 2 to 3 minutes before the first step, with or without
MPS. This does not depend on n2p2 and does not affect the speed of the run
itself.

Performance
-----------

Best tested configuration per node, ``nnp-train`` and LAMMPS, water models:

.. list-table::
   :header-rows: 1
   :widths: 32 17 17 17 17

   * - Benchmark
     - 4 H100, 176 cores
     - 4 A100, 32 cores
     - CPU, 190 cores
     - CPU, 112 cores
   * - 2G MD, 233,280 atoms (ns/day)
     - 0.66
     - 0.24
     - 0.11
     - 0.059
   * - 4G MD, 17,010 atoms (s/step)
     - 0.33
     - 0.46
     - 0.78
     - 1.35
   * - 2G training (s per epoch)
     - 8.1
     - 19
     - 56
     - 107
   * - 4G training, stage 1 (s per epoch)
     - 1.8
     - 7.9
     - 5.2
     - 11.8
   * - 4G training, stage 2 (s per epoch)
     - 30
     - 80
     - 181
     - 314

Machines: ASC MUSICA (H100 node and AMD EPYC 9654 CPU node) and CINECA Leonardo
(Booster node with A100, DCGP node with Intel Xeon 8480+). 2G: water
H2O_RPBE-D3; 4G: SPC/Fw water; training on 1254 structures. The benchmark
script is ``tools/benchmark/n2p2_benchmark.py``.

Correctness
-----------

The test suite compares the GPU results with the CPU results in the same test
binary and with the stored CPU references. In MD with the recommended LAMMPS
settings (see :ref:`if_lammps`), GPU runs conserve energy as well as CPU runs:
the drift over 50 ps is below 1e-4 meV/atom/ps for 2G and 4G water models.
