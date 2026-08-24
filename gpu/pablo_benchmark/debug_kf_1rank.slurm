#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --job-name=debug-kf-1rank
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# nnp-dataset (standalone tool, uses Mode::evaluateNNP) shows ZERO
# CPU-vs-GPU force disagreement with epoch-0 weights, at both 1 and 4
# MPI ranks. But nnp-train's OWN internal epoch-0 RMSE evaluation
# (Training::calculateError, a hand-rolled calculateSymmetryFunction
# Groups/calculateAtomicNeuralNetworks/calculateEnergy/calculateForces
# sequence -- NOT evaluateNNP, which Training.cpp's own code has a
# "TODO: Can we use evaluateNNP?" comment questioning) shows large
# force disagreement (median 4.7, max 23) at 4 ranks. This test checks
# whether nnp-train itself diverges even at 1 rank (isolating whether
# this is an MPI-rank-count effect at all, or purely a property of
# Training::calculateError's own code path vs evaluateNNP's).

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_CPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water
N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
KF=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force/kf_trace
TEST="$KF/1rank_test"

mkdir -p "$TEST/cpu" "$TEST/gpu"
cp "$KF/input.nn" "$KF/scaling.data" "$TEST/cpu/"
cp "$KF/input_small.data" "$TEST/cpu/input.data"
cp "$KF/input.nn" "$KF/scaling.data" "$TEST/gpu/"
cp "$KF/input_small.data" "$TEST/gpu/input.data"
sed -i 's/^selection_mode\s\+2/selection_mode                   0/' "$TEST/cpu/input.nn" "$TEST/gpu/input.nn"

echo "=== CPU nnp-train, 1 rank, selection_mode=0 ==="
cd "$TEST/cpu"
mpirun -np 1 "$N2P2_CPU/bin/nnp-train" > run.out 2>&1 || true
echo "--- ENERGY/FORCE trajectory (cpu, 1 rank) ---"
grep -E "^ENERGY|^FORCE" run.out

echo "=== GPU nnp-train, 1 rank, selection_mode=0 ==="
cd "$TEST/gpu"
mpirun -np 1 "$N2P2_GPU/bin/nnp-train" > run.out 2>&1 || true
echo "--- ENERGY/FORCE trajectory (gpu, 1 rank) ---"
grep -E "^ENERGY|^FORCE" run.out

echo "Done"
