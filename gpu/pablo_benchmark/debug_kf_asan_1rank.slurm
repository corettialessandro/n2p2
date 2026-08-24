#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --job-name=debug-kf-asan-1r
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Training::calculateError (nnp-train's own epoch-0 evaluation code
# path, distinct from Mode::evaluateNNP which nnp-dataset uses and
# which is confirmed bit-clean) shows a large, reproducible CPU-vs-GPU
# FORCE RMSE divergence even at 1 rank -- ruling out MPI, candidate
# selection, and the SF-results cache as causes (all tested and
# eliminated). Given the shape of this bug (invisible to one code
# path, present in another, using nominally the same underlying
# computation), this rebuilds BOTH CPU and GPU water with
# AddressSanitizer -- the same tool that caught the real ts/MPI_SIZE_T
# bug earlier -- to check for host-side memory corruption (heap/stack
# overflow, use-after-free) specifically in this code path.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_CPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water
N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
KF=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force/kf_trace
TEST="$KF/asan_1rank_test"

echo "=== Clean rebuild: CPU nnp-train with ASan ==="
(cd "$N2P2_CPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_CPU/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" nnp-train > "$TEST.build-cpu.log" 2>&1) || \
    (echo "--- CPU build failed ---" && tail -60 "$TEST.build-cpu.log" && exit 1)
git -C "$N2P2_CPU" checkout -- src/libnnp/version.h

echo "=== Clean rebuild: GPU nnp-train with ASan ==="
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

export ASAN_OPTIONS=protect_shadow_gap=0:halt_on_error=0:abort_on_error=0:exitcode=0

echo "=== CPU nnp-train, 1 rank, under ASan ==="
cd "$TEST/cpu"
mpirun -np 1 "$N2P2_CPU/bin/nnp-train" > run.out 2>&1 || true
echo "--- ASan reports (CPU) ---"
grep -n -A25 "ERROR: AddressSanitizer" run.out | head -150
echo "--- ENERGY/FORCE trajectory (CPU) ---"
grep -E "^ENERGY|^FORCE" run.out

echo "=== GPU nnp-train, 1 rank, under ASan ==="
cd "$TEST/gpu"
mpirun -np 1 "$N2P2_GPU/bin/nnp-train" > run.out 2>&1 || true
echo "--- ASan reports (GPU) ---"
grep -n -A25 "ERROR: AddressSanitizer" run.out | head -150
echo "--- ENERGY/FORCE trajectory (GPU) ---"
grep -E "^ENERGY|^FORCE" run.out

echo "Done"
