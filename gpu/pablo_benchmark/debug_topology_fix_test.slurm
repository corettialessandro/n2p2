#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --job-name=debug-topo-fix
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# THE bug, found via direct code reading: Mode::calculateForces()'s GPU
# path (Mode.cpp ~2477) caches force topology (dGdrSelf/edge list) in a
# process-lifetime `static set<size_t> gpuForceTopologyUploaded`, keyed
# only by structure.index, uploaded ONCE and never invalidated -- on
# the assumption that geometry "never changes during training". But
# Training::dataSetNormalization() (Training.cpp ~512-522) calls
# calculateForces() ONCE per structure using the PRE-rescale cutoff
# radius, as part of its "force"-based conv_energy/conv_length
# calibration -- BEFORE that same function rescales the cutoff later
# on. Every subsequent calculateForces() call for the rest of training
# then silently reuses this now-STALE (wrong-cutoff) topology while
# dEdG gets freshly recomputed each time -- a genuine geometry/weights
# mismatch. This test verifies the diagnosis directly: temporarily
# force topology to always re-upload (env-var gated, Mode.cpp) and
# check whether water's epoch-0 divergence (CPU FORCE RMSE 1.22 vs GPU
# 10.3, from debug_kf_1rank.slurm) disappears, and whether the actual
# multi-epoch training explosion (ENERGY -> NaN by epoch 1) is
# resolved.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
KF=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force/kf_trace
TEST="$KF/topofix_test"

echo "=== Clean rebuild: GPU nnp-train (topology-reupload debug flag added) ==="
(cd "$N2P2_GPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_GPU/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-train > "$TEST.build.log" 2>&1) || \
    (echo "--- build failed ---" && tail -60 "$TEST.build.log" && exit 1)
git -C "$N2P2_GPU" checkout -- src/libnnp/version.h

mkdir -p "$TEST/baseline" "$TEST/fixed"
for d in baseline fixed; do
  cp "$KF/input.nn" "$KF/scaling.data" "$TEST/$d/"
  cp "$KF/input_small.data" "$TEST/$d/input.data"
  sed -i 's/^selection_mode\s\+2/selection_mode                   0/' "$TEST/$d/input.nn"
done

echo "=== GPU nnp-train, 1 rank, BASELINE (stale topology cache, current behavior) ==="
cd "$TEST/baseline"
mpirun -np 1 "$N2P2_GPU/bin/nnp-train" > run.out 2>&1 || true
echo "--- ENERGY/FORCE (baseline) ---"
grep -E "^ENERGY|^FORCE" run.out

echo "=== GPU nnp-train, 1 rank, FIXED (N2P2_FORCE_TOPOLOGY_REUPLOAD=1) ==="
cd "$TEST/fixed"
export N2P2_FORCE_TOPOLOGY_REUPLOAD=1
mpirun -np 1 "$N2P2_GPU/bin/nnp-train" > run.out 2>&1 || true
unset N2P2_FORCE_TOPOLOGY_REUPLOAD
echo "--- ENERGY/FORCE (fixed) ---"
grep -E "^ENERGY|^FORCE" run.out

echo "Done"
