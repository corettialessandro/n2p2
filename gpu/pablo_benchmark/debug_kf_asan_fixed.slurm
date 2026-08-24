#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-kf-asan-fixed
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Validates the fix: Dataset.cpp's `ts` (int -> size_t, matching the
# MPI_SIZE_T datatype it's packed/unpacked with) in both
# sendStructure() and recvStructure(). Rebuild with ASan still enabled
# (confirms the overflow is gone) and rerun the exact same reproducing
# config to see if water's training divergence is actually resolved.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force
KF="$DBG/kf_trace"

echo "=== Clean rebuild: GPU nnp-train with the ts fix + ASan ==="
(cd "$N2P2_GPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_GPU/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-train > "$KF/build-asan-fixed.log" 2>&1) || \
    (echo "--- build failed ---" && tail -60 "$KF/build-asan-fixed.log" && exit 1)
git -C "$N2P2_GPU" checkout -- src/libnnp/version.h

mkdir -p "$KF/asan_fixed_run"
cp "$KF/input.nn" "$KF/scaling.data" "$KF/asan_fixed_run/"
cp "$KF/input_small.data" "$KF/asan_fixed_run/input.data"
cd "$KF/asan_fixed_run"

export ASAN_OPTIONS=protect_shadow_gap=0:halt_on_error=1:abort_on_error=1
mpirun -np 4 "$N2P2_GPU/bin/nnp-train" > run.out 2>&1 || true

echo "--- ASan reports (should be none now) ---"
grep -c "ERROR: AddressSanitizer" run.out || echo "0"

echo "--- ENERGY/FORCE trajectory ---"
grep -E "^ENERGY|^FORCE" run.out

echo "Done"
