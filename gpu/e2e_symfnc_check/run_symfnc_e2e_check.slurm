#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --job-name=symfnc-e2e-check
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Phase 5 step 6 (soft-percolating-jellyfish.md): kernel-level validation
# of the promoted GPU symmetry-function kernels (GpuSymmetryFunction.cu)
# against the REAL production pipeline (Prediction::predict(), the same
# calculateSymmetryFunctionGroups() call InterfaceLammps uses), not the
# gpu/soa prototypes' own self-referential check. Must happen before
# touching Mode.cpp.
#
# -DN2P2_FULL_SFD_MEMORY is required for both libnnp and this test
# binary -- this port's chosen GPU-path memory layout (see
# GpuSymmetryFunction.h's doc comment), needed here so
# Atom::Neighbor::dGdr is indexed by global symmetry-function index
# directly, matching the GPU kernels' output layout.
#
# Full clean rebuild on THIS execution node -- gpu/e2e_predict_check's
# mode_gpu_call_site_test.cpp header comment documents a real, previously
# root-caused (commit 1477941) crash from reusing a stale -march=native
# library built on a different node; this script never reuses one.

set -e
module purge
module load cuda/12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
CHK="$N2P2_DEVEL/gpu/e2e_symfnc_check"
TEMP="$N2P2_DEVEL/temp/H2O_2G"
TRAIN100="$N2P2_DEVEL/gpu/e2e_train_check/results/run-100ep-cpu"
cd "$CHK"

echo "=== Staging input.nn/scaling.data/weights.*.data (converged 100-epoch potential) ==="
cp "$TRAIN100/input.nn" "$TRAIN100/scaling.data" .
cp "$TRAIN100/weights.001.000100.out" weights.001.data
cp "$TRAIN100/weights.008.000100.out" weights.008.data

echo ""
echo "=== [1/3] Build libnnpgpu.a (GPU=1) ==="
(cd "$N2P2_DEVEL/src/libnnpgpu" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpgpu" && make COMP=gnu MODE=static GPU=1)
cp "$N2P2_DEVEL/lib/libnnpgpu.a" .

echo ""
echo "=== [2/3] Build libnnp.a (N2P2_FULL_SFD_MEMORY only -- no GPU=1 here,"
echo "    the test harness calls GpuSymmetryFunction.h directly and doesn't"
echo "    need Mode.cpp's own N2P2_GPU dispatch branches, which would pull"
echo "    in unrelated 4G electrostatics GPU code needing -lcusolver) ==="
(cd "$N2P2_DEVEL/src/libnnp" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnp" && make COMP=gnu MODE=static \
    PROJECT_EIGEN="$EIGEN" \
    PROJECT_OPTIONS="-DEIGEN_DONT_PARALLELIZE -DN2P2_FULL_SFD_MEMORY")
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
cp "$N2P2_DEVEL/lib/libnnp.a" .

echo ""
echo "=== [3/3] Compile and link symfnc_e2e_check ==="
CUDA_LIB="$CUDA_HOME/lib64"
g++ -O3 -std=c++11 -DN2P2_FULL_SFD_MEMORY \
    -I"$N2P2_DEVEL/include" -I"$EIGEN" -DEIGEN_DONT_PARALLELIZE \
    -o symfnc_e2e_check.o -c symfnc_e2e_check.cpp
g++ -O3 -std=c++11 -o symfnc_e2e_check symfnc_e2e_check.o libnnp.a libnnpgpu.a \
    -L"$CUDA_LIB" -lcudart -lcublas -lcusolver

echo ""
echo "=== Running (real H2O_2G structure, converged 100-epoch weights) ==="
./symfnc_e2e_check "$TEMP"

echo ""
echo "Done"
