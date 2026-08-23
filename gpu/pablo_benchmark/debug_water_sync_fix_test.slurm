#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-water-syncfix
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Empirical test: does adding cudaDeviceSynchronize() after each kernel
# launch in GpuForces.cu::gpuForcesCompute() fix the divergence? Uses
# the reliably-failing config (4 ranks/1 GPU, no wrapper, full
# 140-structure test.data -- confirmed 100% of structures disagree at
# this config without the fix).

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force

echo "=== Rebuild nnp-dataset with the cudaDeviceSynchronize() fix ==="
(cd "$N2P2_GPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_GPU/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-dataset > "$DBG/build-syncfix.log" 2>&1)
git -C "$N2P2_GPU" checkout -- src/libnnp/version.h

mkdir -p "$DBG/gpu_run_syncfix" && cd "$DBG/gpu_run_syncfix"
cp "$DBG/gpu_run_1r_sanitize"/{input.nn,scaling.data,weights.001.data,weights.008.data,full_test.data} .

echo "=== Running GPU nnp-dataset WITH the sync fix (4 ranks, 1 GPU, full 140 structures) ==="
mpirun -np 4 "$N2P2_GPU/bin/nnp-dataset" 0 full_test.data

echo ""
echo "=== energy.comp diff vs CPU baseline ==="
diff <(grep -v '^#' "$DBG/cpu_run_4r1g_full/energy.comp" | sort) <(grep -v '^#' energy.comp | sort) | head -10
echo "(no output above = identical)"

echo ""
echo "=== forces.comp: distinct structure indices with a disagreement (>1e-6) ==="
join -j 1 \
  <(grep -v '^#' "$DBG/cpu_run_4r1g_full/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) \
  <(grep -v '^#' forces.comp | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) | \
awk '{d=$2-$3; if (d<0) d=-d; if (d>1e-6) {split($1,a,"_"); print a[1]}}' | sort -n -u | wc -l
echo "(count of disagreeing structures out of 140 -- 0 means the sync fix worked)"

echo "Done"
