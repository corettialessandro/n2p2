#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-kf-selmode
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Decisive test for the candidate-selection-divergence lead: water's
# input.nn uses selection_mode=2 (SM_THRESHOLD), which LIVE-recomputes
# energy/forces (via Mode::calculateAtomicNeuralNetworks/calculateForces,
# which dispatch to GPU kernels under N2P2_GPU) as part of deciding which
# structure/atom becomes the next update candidate. Any GPU-vs-CPU FP
# divergence in that recomputation can push a candidate across
# short_energy/force_error_threshold differently, picking a genuinely
# different candidate -- unlike SM_RANDOM (mode 0), which just walks a
# pre-shuffled list with no live error computation at all, so it cannot
# diverge this way. If switching to selection_mode=0 removes water's
# NaN explosion (with everything else identical: ts-fixed Dataset.cpp,
# same weights/seed/data), that confirms the selection feedback loop --
# not the Kalman filter or GPU kernels themselves -- is the amplifying
# mechanism.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force
KF="$DBG/kf_trace"
TEST="$KF/selmode_test"

echo "=== Clean rebuild: GPU nnp-train, plain (no ASan), ts-fixed Dataset.cpp ==="
(cd "$N2P2_GPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_GPU/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-train > "$TEST.build.log" 2>&1) || \
    (echo "--- build failed ---" && tail -60 "$TEST.build.log" && exit 1)
git -C "$N2P2_GPU" checkout -- src/libnnp/version.h

mkdir -p "$TEST/sm2_baseline" "$TEST/sm0_random"
cp "$KF/input.nn" "$KF/scaling.data" "$TEST/sm2_baseline/"
cp "$KF/input_small.data" "$TEST/sm2_baseline/input.data"
cp "$KF/input.nn" "$KF/scaling.data" "$TEST/sm0_random/"
cp "$KF/input_small.data" "$TEST/sm0_random/input.data"

# sm0_random: only change selection_mode 2 -> 0, everything else identical
sed -i 's/^selection_mode\s\+2/selection_mode                   0/' "$TEST/sm0_random/input.nn"
grep -n "^selection_mode" "$TEST/sm2_baseline/input.nn" "$TEST/sm0_random/input.nn"

echo "=== GPU run, selection_mode=2 (baseline, reconfirm explosion with plain non-ASan build) ==="
cd "$TEST/sm2_baseline"
mpirun -np 4 "$N2P2_GPU/bin/nnp-train" > run.out 2>&1 || true
echo "--- ENERGY/FORCE trajectory (sm2_baseline) ---"
grep -E "^ENERGY|^FORCE" run.out

echo "=== GPU run, selection_mode=0 (test: pure round-robin, no live-error-based selection) ==="
cd "$TEST/sm0_random"
mpirun -np 4 "$N2P2_GPU/bin/nnp-train" > run.out 2>&1 || true
echo "--- ENERGY/FORCE trajectory (sm0_random) ---"
grep -E "^ENERGY|^FORCE" run.out

echo "Done"
