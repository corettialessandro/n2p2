#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --mem=32GB
#SBATCH --job-name=debug-water-32r-small
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Isolates rank count vs. dataset size as the real variable: same 20
# structures that matched cleanly at 4 ranks, now at 32 ranks (matching
# the real training job's rank count) with the proper wrapper+MPS setup.

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

echo "=== CPU nnp-dataset on small.data (20 structures, 32 ranks) ==="
cd "$DBG/cpu_run_32r_small"
mpirun -np 32 "$N2P2_CPU/bin/nnp-dataset" 0 small.data

echo ""
echo "=== Starting MPS control daemon ==="
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-pipe-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
sleep 2

echo "=== GPU nnp-dataset on small.data (20 structures, 32 ranks / 4 GPUs) ==="
cd "$DBG/gpu_run_32r_small"
export RANKS_PER_GPU=8
mpirun -np 32 "$WRAPPER" "$N2P2_GPU/bin/nnp-dataset" 0 small.data

echo ""
echo "=== Stopping MPS control daemon ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2

echo ""
echo "=== energy.comp diff ==="
diff <(grep -v '^#' "$DBG/cpu_run_32r_small/energy.comp" | sort) <(grep -v '^#' "$DBG/gpu_run_32r_small/energy.comp" | sort) | head -20
echo "(no output above = identical)"

echo ""
echo "=== forces.comp: rows where predicted force (col 4) disagrees by > 1e-6 ==="
join -j 1 \
  <(grep -v '^#' "$DBG/cpu_run_32r_small/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) \
  <(grep -v '^#' "$DBG/gpu_run_32r_small/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) | \
awk '{d=$2-$3; if (d<0) d=-d; if (d>1e-6) print}' | sort -t_ -k1,1n

echo "Done"
