#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --job-name=debug-epoch0-force
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# nnp-train's own epoch-0 log (before ANY training update, with
# bit-identical initial weights confirmed via md5sum) shows CPU FORCE
# RMSE 1.24 vs GPU FORCE RMSE 5.95 -- energy matches exactly, force
# does not. Isolate this with the standalone nnp-dataset tool (single
# force evaluation, no training loop involved) using those exact
# epoch-0 weights, to confirm/refute whether this is the NN-forward
# GPU kernel's backprop (dE/dG) being wrong specifically for small,
# near-zero freshly-initialized weights -- a regime never tested by
# the earlier GpuForces.cu validation, which only used converged,
# post-training weights.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

TEST=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force/kf_trace/nnpdataset_epoch0_test
CPU_BIN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water/bin/nnp-dataset
GPU_BIN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water/bin/nnp-dataset

echo "=== CPU nnp-dataset, epoch-0 weights, shuffle=0 ==="
cd "$TEST/cpu"
"$CPU_BIN" 0 > run.out 2>&1
tail -5 run.out

echo "=== GPU nnp-dataset, epoch-0 weights, shuffle=0 ==="
cd "$TEST/gpu"
"$GPU_BIN" 0 > run.out 2>&1
tail -5 run.out

echo "Done"
