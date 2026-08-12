#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4
#SBATCH --mem=16GB
#SBATCH --job-name=nomps-gpu-diag
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

export N_GPUS_PER_NODE=4
# No MPS daemons this time -- MPS_BASE_DIR still needs to exist since the
# wrapper unconditionally points CUDA_MPS_PIPE/LOG_DIRECTORY at it, but
# without a daemon listening there, CUDA just falls back to ordinary
# (non-MPS) per-process context creation on the bound GPU.
export MPS_BASE_DIR="/tmp/nomps-diag-$SLURM_JOB_ID"
mkdir -p "$MPS_BASE_DIR/pipe-gpu0" "$MPS_BASE_DIR/pipe-gpu1" "$MPS_BASE_DIR/pipe-gpu2" "$MPS_BASE_DIR/pipe-gpu3"
mkdir -p "$MPS_BASE_DIR/log-gpu0" "$MPS_BASE_DIR/log-gpu1" "$MPS_BASE_DIR/log-gpu2" "$MPS_BASE_DIR/log-gpu3"

echo "=== 8 ranks / 4 GPUs, NO MPS ==="
mpirun -np 8 ./gpu_rank_wrapper.sh ./mps_gpu_diag

echo "Done"
