# Benchmarks of n2p2 builds

`n2p2_benchmark.py` times LAMMPS MD and `nnp-train` on the examples of this
repository, so that builds (CPU and GPU) and run configurations (ranks,
GPUs, MPS) can be compared on the same machine. It uses only the Python
standard library and contains nothing specific to a cluster: binaries, MPI
launcher and work directory are arguments. Call it from your own job script.

## MD with LAMMPS

```
python3 n2p2_benchmark.py md --lmp /path/to/lmp_mpi \
    --example examples/interface-LAMMPS/H2O_RPBE-D3 \
    --replicate 1 2 3 --ranks 32 --label cpu --csv md.csv
```

The example directory must contain the LAMMPS input (`--input`, default
`md.lmp`) with `variable numSteps ...` and `run ${numSteps}`. For each
replication factor the example is copied to a run directory, the system is
replicated in all three directions, dumps and restart files are removed, and
a warm-up run (`--warmup`, default 10 steps) precedes the timed run
(`--steps`, default 200 / replicate^3, at least 10). Reported: atoms, loop
time and ns/day of the timed run.

## Training

```
python3 n2p2_benchmark.py train --nnp-train /path/to/nnp-train \
    --example examples/nnp-train/H2O_RPBE-D3 --epochs 2 --ranks 4 \
    --label cpu --csv train.csv
```

For 4G-HDNNPs pass `--stage 1` or `--stage 2` (stage 2 needs the stage-1
results in the example directory). Reported: the time of each epoch from
the `TIMING` lines of the training log and their mean.

## GPU builds

GPU builds select the device of each rank automatically (see
`src/libnnpgpu/README.md`). With several ranks per GPU add `--mps`: the
script starts an NVIDIA MPS daemon for each benchmark (pipe directory in
the temporary directory, `TMPDIR` or `/tmp`, as the daemon's socket paths
are limited to about 100 characters; log directory in the work directory)
and forwards its variables with
OpenMPI's `-x`; with another launcher make sure the variables
`CUDA_MPS_PIPE_DIRECTORY` and `CUDA_MPS_LOG_DIRECTORY` reach the ranks.

## Example job script

```
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
# (account, partition, modules: as required by your site)
B=/path/to/benchmark/n2p2_benchmark.py
EX=/path/to/n2p2/examples/interface-LAMMPS/H2O_RPBE-D3
python3 $B md --lmp /path/to/cpu-build/bin/lmp_mpi --example $EX \
    --replicate 1 2 3 --ranks 32 --label cpu --csv md.csv --workdir runs
python3 $B md --lmp /path/to/gpu-build/bin/lmp_mpi --example $EX \
    --replicate 1 2 3 --ranks 32 --mps --label gpu --csv md.csv --workdir runs
```

The exit status is nonzero if any run failed; its output is in `out.log`
of the run directory.
