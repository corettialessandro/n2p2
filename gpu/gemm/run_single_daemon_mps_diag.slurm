#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4
#SBATCH --mem=16GB
#SBATCH --job-name=single-daemon-mps-diag
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
cd "$N2P2_DEVEL/gpu/gemm"
chmod +x gpu_rank_wrapper.sh

echo "Compiling mps_gpu_diag.cu..."
nvcc -O3 -std=c++14 -arch=sm_80 -lcusolver -o mps_gpu_diag mps_gpu_diag.cu

export RANKS_PER_GPU=2
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-pipe-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
echo "--- control.log ---"; cat "$CUDA_MPS_LOG_DIRECTORY/control.log" 2>&1

echo ""
echo "=== 8 ranks / 4 GPUs (2/GPU), single unrestricted MPS daemon ==="
mpirun -np 8 ./gpu_rank_wrapper.sh ./mps_gpu_diag

echo quit | nvidia-cuda-mps-control || true
echo "Done"
