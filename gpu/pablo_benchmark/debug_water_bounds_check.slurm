#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-water-bounds
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Host-side bounds check on the edge-list indices GpuForces' topology
# upload builds (Mode.cpp) -- tests whether edgeTarget/edgeOwnerDEdGIndex
# ever go out of range for THIS structure, which would mean force
# contributions get written into a DIFFERENT structure's GPU buffer
# (memory-layout-dependent, not a timing race -- would explain why
# compute-sanitizer's redzone padding "fixes" it).

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force

echo "=== Rebuild nnp-dataset with the bounds-check instrumentation ==="
(cd "$N2P2_GPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_GPU/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-dataset > "$DBG/build-bounds.log" 2>&1)
git -C "$N2P2_GPU" checkout -- src/libnnp/version.h

mkdir -p "$DBG/gpu_run_bounds" && cd "$DBG/gpu_run_bounds"
cp "$DBG/gpu_run_1r_sanitize"/{input.nn,scaling.data,weights.001.data,weights.008.data,full_test.data} .

echo "=== Running (4 ranks, 1 GPU, full 140 structures) -- watch stderr for BOUNDS-CHECK lines ==="
mpirun -np 4 "$N2P2_GPU/bin/nnp-dataset" 0 full_test.data 2> stderr.log || true

echo ""
echo "=== BOUNDS-CHECK violations found ==="
grep -c "BOUNDS-CHECK" stderr.log || echo "0 (none found)"
grep "BOUNDS-CHECK" stderr.log | head -20

echo "Done"
