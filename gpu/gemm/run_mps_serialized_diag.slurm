#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4
#SBATCH --mem=16GB
#SBATCH --job-name=mps-serialized-diag
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
cd "$N2P2_DEVEL/gpu/gemm"

echo "Compiling mps_gpu_diag.cu..."
nvcc -O3 -std=c++14 -arch=sm_80 -lcusolver -o mps_gpu_diag mps_gpu_diag.cu

MPS_BASE_DIR="/tmp/mps-serdiag-$SLURM_JOB_ID"
mkdir -p "$MPS_BASE_DIR"

for gpu in 0 1 2 3; do
    echo ""
    echo "=== Starting MPS daemon for GPU $gpu ==="
    mkdir -p "$MPS_BASE_DIR/pipe-gpu${gpu}" "$MPS_BASE_DIR/log-gpu${gpu}"
    CUDA_VISIBLE_DEVICES=$gpu \
    CUDA_MPS_PIPE_DIRECTORY="$MPS_BASE_DIR/pipe-gpu${gpu}" \
    CUDA_MPS_LOG_DIRECTORY="$MPS_BASE_DIR/log-gpu${gpu}" \
    nvidia-cuda-mps-control -d
    sleep 3
    echo "--- control.log ---"
    cat "$MPS_BASE_DIR/log-gpu${gpu}/control.log" 2>&1 || echo "(none)"

    echo "--- immediate single-rank test against GPU $gpu's daemon ---"
    CUDA_VISIBLE_DEVICES=$gpu \
    CUDA_MPS_PIPE_DIRECTORY="$MPS_BASE_DIR/pipe-gpu${gpu}" \
    CUDA_MPS_LOG_DIRECTORY="$MPS_BASE_DIR/log-gpu${gpu}" \
    ./mps_gpu_diag || echo "*** FAILED for GPU $gpu ***"
done

echo ""
echo "=== Now test all 4 daemons again with 2 ranks each (8 total), after full serialized startup ==="
export N_GPUS_PER_NODE=4
export MPS_BASE_DIR
chmod +x gpu_rank_wrapper.sh
mpirun -np 8 ./gpu_rank_wrapper.sh ./mps_gpu_diag || echo "*** mpirun reported failures ***"

for gpu in 0 1 2 3; do
    CUDA_MPS_PIPE_DIRECTORY="$MPS_BASE_DIR/pipe-gpu${gpu}" \
    bash -c 'echo quit | nvidia-cuda-mps-control' || true
done

echo "Done"
