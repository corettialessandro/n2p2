#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4
#SBATCH --mem=32GB
#SBATCH --job-name=debug-water-8r-full
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Bisection: full 140-structure test.data (which failed pervasively at
# 32 ranks/8-per-GPU), now at 8 ranks/2-per-GPU, same wrapper+MPS setup.
# Isolates whether the trigger is "any multi-rank GPU sharing" or
# specifically the higher 8-ranks/GPU density.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_ORIG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
N2P2_CPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water
N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force
WRAPPER="$N2P2_ORIG/gpu/e2e_train_check/rank_gpu_wrapper.sh"

echo "=== CPU nnp-dataset on full_test.data (140 structures, 8 ranks) ==="
cd "$DBG/cpu_run_8r_full"
mpirun -np 8 "$N2P2_CPU/bin/nnp-dataset" 0 full_test.data

echo ""
echo "=== Starting MPS control daemon ==="
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-pipe-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
sleep 2

echo "=== GPU nnp-dataset on full_test.data (140 structures, 8 ranks / 4 GPUs, 2/GPU) ==="
cd "$DBG/gpu_run_8r_full"
export RANKS_PER_GPU=2
mpirun -np 8 "$WRAPPER" "$N2P2_GPU/bin/nnp-dataset" 0 full_test.data

echo ""
echo "=== Stopping MPS control daemon ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2

echo ""
echo "=== energy.comp diff ==="
diff <(grep -v '^#' "$DBG/cpu_run_8r_full/energy.comp" | sort) <(grep -v '^#' "$DBG/gpu_run_8r_full/energy.comp" | sort) | head -20
echo "(no output above = identical)"

echo ""
echo "=== forces.comp: distinct structure indices with a disagreement (>1e-6) ==="
join -j 1 \
  <(grep -v '^#' "$DBG/cpu_run_8r_full/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) \
  <(grep -v '^#' "$DBG/gpu_run_8r_full/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) | \
awk '{d=$2-$3; if (d<0) d=-d; if (d>1e-6) {split($1,a,"_"); print a[1]}}' | sort -n -u | wc -l
echo "(count of distinct structures with disagreement, out of 140)"

echo "Done"
