#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time 03:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-kf-sanitizer
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# The Kalman-filter trace found a garbage Training::countUpdates value
# (a class member, not a stack local) in the GPU build's training log
# output -- a classic out-of-bounds-write corruption symptom. Never
# examined gpuNnForceDFdcSum/gpuNnEnergyDEdcSum (the Jacobian-sum GPU
# kernels used only during training weight updates) before -- run the
# same reproduction under compute-sanitizer to try to catch the actual
# faulting write directly.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force
KF="$DBG/kf_trace"

mkdir -p "$KF/sanitizer_run"
cp "$KF/input.nn" "$KF/scaling.data" "$KF/sanitizer_run/"
cp "$KF/input_small.data" "$KF/sanitizer_run/input.data"
cd "$KF/sanitizer_run"

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader
export OMPI_MCA_osc=^ucx
export UCX_TLS=^cuda,gdr_copy,cuda_copy,cuda_ipc

echo "=== Running nnp-train (4 ranks, matching the reproducing config) under compute-sanitizer --tool memcheck ==="
mpirun -np 4 compute-sanitizer --tool memcheck --error-exitcode 0 --log-file "sanitizer_%p.log" \
    "$N2P2_GPU/bin/nnp-train" > run.out 2>&1 || true

echo "--- ERROR SUMMARY ---"
grep "ERROR SUMMARY" sanitizer_*.log || echo "(none found)"
echo "--- first error block, if any ---"
grep -n -B2 -A25 "^========= Invalid\|^========= Program hit" sanitizer_*.log | head -80

echo "--- countUpdates in this run's training log (sanity check) ---"
awk '$1=="F" && $3+0>0 && $3+0<1000000' train-log.out 2>/dev/null | head -3
awk '$1=="F"' train-log.out 2>/dev/null | head -3

echo "Done"
