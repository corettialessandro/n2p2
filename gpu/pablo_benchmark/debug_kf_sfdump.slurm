#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --job-name=debug-kf-sfdump
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ASan found nothing (no memory corruption), yet Training::calculateError
# (used internally by nnp-train, not Mode::evaluateNNP which nnp-dataset
# uses and is confirmed bit-clean) shows a large, deterministic CPU-vs-GPU
# FORCE divergence at epoch 0, even at 1 rank. This dumps the first
# training structure's atom-0 G vector, neighbor count, and Mode's
# maxCutoffRadius right after calculateSymmetryFunctionGroups() inside
# Training::calculateError's own loop, for both CPU and GPU builds, to
# see whether the SYMMETRY FUNCTION inputs themselves (not just the
# downstream NN/force GPU kernels) already differ between builds within
# nnp-train's own run -- which would point at neighbor-list/cutoff
# consistency (Training::calculateNeighborLists(), called once at setup)
# rather than at GpuForces.cu or the NN-forward kernel (both already
# independently confirmed correct via nnp-dataset).

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_CPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water
N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
KF=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force/kf_trace
TEST="$KF/sfdump_test"

echo "=== Clean rebuild: CPU nnp-train (plain, SF-dump instrumented) ==="
(cd "$N2P2_CPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_CPU/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" nnp-train > "$TEST.build-cpu.log" 2>&1) || \
    (echo "--- CPU build failed ---" && tail -60 "$TEST.build-cpu.log" && exit 1)
git -C "$N2P2_CPU" checkout -- src/libnnp/version.h

echo "=== Clean rebuild: GPU nnp-train (plain, SF-dump instrumented) ==="
(cd "$N2P2_GPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_GPU/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-train > "$TEST.build-gpu.log" 2>&1) || \
    (echo "--- GPU build failed ---" && tail -60 "$TEST.build-gpu.log" && exit 1)
git -C "$N2P2_GPU" checkout -- src/libnnp/version.h

mkdir -p "$TEST/cpu" "$TEST/gpu"
cp "$KF/input.nn" "$KF/scaling.data" "$TEST/cpu/"
cp "$KF/input_small.data" "$TEST/cpu/input.data"
cp "$KF/input.nn" "$KF/scaling.data" "$TEST/gpu/"
cp "$KF/input_small.data" "$TEST/gpu/input.data"
sed -i 's/^selection_mode\s\+2/selection_mode                   0/' "$TEST/cpu/input.nn" "$TEST/gpu/input.nn"

echo "=== CPU nnp-train, 1 rank, SF dump ==="
cd "$TEST/cpu"
export N2P2_DUMP_SF_FILE="$TEST/cpu/sfdump.txt"
mpirun -np 1 "$N2P2_CPU/bin/nnp-train" > run.out 2>&1 || true
unset N2P2_DUMP_SF_FILE
echo "--- SF dump (CPU) ---"
cat "$TEST/cpu/sfdump.txt"
echo "--- ENERGY/FORCE (CPU) ---"
grep -E "^ENERGY|^FORCE" run.out

echo "=== GPU nnp-train, 1 rank, SF dump ==="
cd "$TEST/gpu"
export N2P2_DUMP_SF_FILE="$TEST/gpu/sfdump.txt"
mpirun -np 1 "$N2P2_GPU/bin/nnp-train" > run.out 2>&1 || true
unset N2P2_DUMP_SF_FILE
echo "--- SF dump (GPU) ---"
cat "$TEST/gpu/sfdump.txt"
echo "--- ENERGY/FORCE (GPU) ---"
grep -E "^ENERGY|^FORCE" run.out

echo "Done"
