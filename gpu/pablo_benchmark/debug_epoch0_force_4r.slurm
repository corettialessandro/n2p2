#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-epoch0-4r
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Single-rank nnp-dataset (no MPI structure gather at all) showed ZERO
# CPU-vs-GPU force disagreement with epoch-0 weights, and swapping in
# the GPU's ULP-different conv_energy/conv_length header alone only
# moved forces by ~1e-10 -- nowhere near nnp-train's own internally
# observed divergence (median 4.7, max 23) at 4 MPI ranks. This test
# isolates whether the MPI multi-rank structure-gather path itself
# (Dataset::sendStructure/recvStructure, already patched once for the
# `ts` int/size_t MPI_SIZE_T mismatch) is where the real divergence
# lives: same weights, same (identical, CPU-derived) header on both
# sides, but now -np 4 with shuffle=0 for deterministic distribution,
# matching nnp-train's own rank count.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

TEST=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force/kf_trace/nnpdataset_epoch0_test
CPU_BIN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water/bin/nnp-dataset
GPU_BIN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water/bin/nnp-dataset

echo "=== CPU nnp-dataset, epoch-0 weights, 4 ranks, shuffle=0 ==="
cd "$TEST/cpu_4r"
mpirun -np 4 "$CPU_BIN" 0 > run.out 2>&1
tail -5 run.out

echo "=== GPU nnp-dataset, epoch-0 weights, 4 ranks, shuffle=0 ==="
cd "$TEST/gpu_4r"
mpirun -np 4 "$GPU_BIN" 0 > run.out 2>&1
tail -5 run.out

echo "Done"
