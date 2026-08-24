#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --job-name=debug-kf-nocache
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Hypothesis: `memorize_symfunc_results` (active in water's input.nn)
# caches symmetry-function G/dGdr results per structure. Training's
# driver order is: initializeWeights() -> dataSetNormalization()
# (force-based calibration, which itself does a full forward+force
# pass over the data using the FRESH random weights, computing SF
# results BEFORE the cutoff-radius rescaling that normalization
# applies at Mode.cpp:739) -> setupTraining() -> calculateNeighborLists()
# (rebuilds the neighbor list with the FINAL, rescaled cutoff). If the
# SF-result cache from the calibration pass isn't invalidated before
# the epoch-0 error evaluation, that evaluation could be silently
# reusing STALE (pre-rescale) G/dGdr values -- a genuine, likely
# pre-existing n2p2 bug, not necessarily GPU-specific on its own, but
# one that could make the downstream GPU-dispatched NN/force kernels
# far more sensitive to ordinary CPU/GPU floating-point differences.
# Test: disable memorize_symfunc_results and rerun the exact 1-rank
# comparison that showed FORCE RMSE 1.22 (CPU) vs 10.3 (GPU) at epoch 0.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_CPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water
N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
KF=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force/kf_trace
TEST="$KF/nocache_test"

mkdir -p "$TEST/cpu" "$TEST/gpu"
cp "$KF/input.nn" "$KF/scaling.data" "$TEST/cpu/"
cp "$KF/input_small.data" "$TEST/cpu/input.data"
cp "$KF/input.nn" "$KF/scaling.data" "$TEST/gpu/"
cp "$KF/input_small.data" "$TEST/gpu/input.data"
sed -i 's/^selection_mode\s\+2/selection_mode                   0/' "$TEST/cpu/input.nn" "$TEST/gpu/input.nn"
sed -i 's/^memorize_symfunc_results/#memorize_symfunc_results/' "$TEST/cpu/input.nn" "$TEST/gpu/input.nn"
grep -n "^selection_mode\|memorize_symfunc" "$TEST/cpu/input.nn"

echo "=== CPU nnp-train, 1 rank, selection_mode=0, memorize_symfunc_results DISABLED ==="
cd "$TEST/cpu"
mpirun -np 1 "$N2P2_CPU/bin/nnp-train" > run.out 2>&1 || true
echo "--- ENERGY/FORCE trajectory (cpu, no cache) ---"
grep -E "^ENERGY|^FORCE" run.out

echo "=== GPU nnp-train, 1 rank, selection_mode=0, memorize_symfunc_results DISABLED ==="
cd "$TEST/gpu"
mpirun -np 1 "$N2P2_GPU/bin/nnp-train" > run.out 2>&1 || true
echo "--- ENERGY/FORCE trajectory (gpu, no cache) ---"
grep -E "^ENERGY|^FORCE" run.out

echo "Done"
