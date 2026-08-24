#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-kf-asan
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# The countUpdates corruption looks like a stray double from a
# host-side buffer overflow, not a GPU-device memory error --
# compute-sanitizer only instruments CUDA/GPU memory, not plain C++
# host heap corruption. Rebuilds with AddressSanitizer (-fsanitize=
# address), the right tool for a host-side overflow, and reruns the
# same reproducing config.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force
KF="$DBG/kf_trace"

echo "=== Clean rebuild: GPU nnp-train with AddressSanitizer ==="
(cd "$N2P2_GPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_GPU/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-train > "$KF/build-asan.log" 2>&1) || \
    (echo "--- build failed, tail of log ---" && tail -60 "$KF/build-asan.log" && exit 1)
git -C "$N2P2_GPU" checkout -- src/libnnp/version.h

mkdir -p "$KF/asan_run"
cp "$KF/input.nn" "$KF/scaling.data" "$KF/asan_run/"
cp "$KF/input_small.data" "$KF/asan_run/input.data"
cd "$KF/asan_run"

export ASAN_OPTIONS=protect_shadow_gap=0:halt_on_error=1:abort_on_error=1
mpirun -np 4 "$N2P2_GPU/bin/nnp-train" > run.out 2>&1 || true

echo "--- ASan reports (if any) ---"
grep -n -A40 "ERROR: AddressSanitizer\|heap-buffer-overflow\|stack-buffer-overflow\|global-buffer-overflow" run.out | head -150

echo "Done"
