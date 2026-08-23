#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-water-force
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Diagnostic for water's GPU-training force divergence: CPU and GPU
# nnp-train both matched exactly at epoch 0 (identical initial weights,
# 32 ranks both sides), so this compares CPU vs GPU force output on
# THOSE SAME (already known-identical) weights via nnp-dataset, which
# calls the exact same Mode::calculateForces()/GpuForces path as
# training. Small (20-structure) subset for a fast turnaround.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_CPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water
N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force

echo "=== Build CPU nnp-dataset ==="
(cd "$N2P2_CPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_CPU/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" nnp-dataset > "$DBG/build-cpu.log" 2>&1)
git -C "$N2P2_CPU" checkout -- src/libnnp/version.h

echo "=== Build GPU nnp-dataset (GPU=1) ==="
(cd "$N2P2_GPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_GPU/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-dataset > "$DBG/build-gpu.log" 2>&1)
git -C "$N2P2_GPU" checkout -- src/libnnp/version.h

mkdir -p "$DBG/cpu_run" "$DBG/gpu_run"
cp "$DBG/input.nn" "$DBG/scaling.data" "$DBG/weights.001.data" "$DBG/weights.008.data" "$DBG/small.data" "$DBG/cpu_run/"
cp "$DBG/input.nn" "$DBG/scaling.data" "$DBG/weights.001.data" "$DBG/weights.008.data" "$DBG/small.data" "$DBG/gpu_run/"

echo ""
echo "=== Running CPU nnp-dataset on small.data (epoch-0 weights) ==="
cd "$DBG/cpu_run"
mpirun -np 4 "$N2P2_CPU/bin/nnp-dataset" 0 small.data

echo ""
echo "=== Running GPU nnp-dataset on small.data (epoch-0 weights) ==="
cd "$DBG/gpu_run"
mpirun -np 4 "$N2P2_GPU/bin/nnp-dataset" 0 small.data

echo ""
echo "=== energy.comp diff ==="
diff <(grep -v '^#' "$DBG/cpu_run/energy.comp") <(grep -v '^#' "$DBG/gpu_run/energy.comp") | head -20
echo "(no output above = identical)"

echo ""
echo "=== forces.comp diff (first differing lines) ==="
diff <(grep -v '^#' "$DBG/cpu_run/forces.comp") <(grep -v '^#' "$DBG/gpu_run/forces.comp") | head -40

echo "Done"
