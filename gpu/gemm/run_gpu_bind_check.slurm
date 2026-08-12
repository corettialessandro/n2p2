#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4
#SBATCH --mem=16GB
#SBATCH --job-name=gpu-bind-check
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel

cd "$N2P2_DEVEL/gpu/gemm"
chmod +x gpu_rank_wrapper.sh

echo "=== nvidia-smi -L (GPUs visible to the job) ==="
nvidia-smi -L

echo ""
echo "Compiling gpu_bind_check.cu..."
nvcc -O3 -std=c++14 -arch=sm_80 -o gpu_bind_check gpu_bind_check.cu

export N_GPUS_PER_NODE=4
export MPS_BASE_DIR="/tmp/mps-bindcheck-$SLURM_JOB_ID"
mkdir -p "$MPS_BASE_DIR"

echo ""
echo "=== 8 ranks, 4 GPUs -- expect 2 ranks per physical GPU (distinct pciBusID pairs) ==="
mpirun -np 8 ./gpu_rank_wrapper.sh ./gpu_bind_check

echo "Done"
