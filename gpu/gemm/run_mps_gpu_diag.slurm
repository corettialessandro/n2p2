#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4
#SBATCH --mem=16GB
#SBATCH --job-name=mps-gpu-diag
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
cd "$N2P2_DEVEL/gpu/gemm"
chmod +x gpu_rank_wrapper.sh

echo "=== nvidia-smi -L ==="
nvidia-smi -L

echo ""
echo "Compiling mps_gpu_diag.cu..."
nvcc -O3 -std=c++14 -arch=sm_80 -lcusolver -o mps_gpu_diag mps_gpu_diag.cu

export N_GPUS_PER_NODE=4
export MPS_BASE_DIR="/tmp/mps-diag-$SLURM_JOB_ID"
mkdir -p "$MPS_BASE_DIR"
for gpu in 0 1 2 3; do
    mkdir -p "$MPS_BASE_DIR/pipe-gpu${gpu}" "$MPS_BASE_DIR/log-gpu${gpu}"
    CUDA_VISIBLE_DEVICES=$gpu \
    CUDA_MPS_PIPE_DIRECTORY="$MPS_BASE_DIR/pipe-gpu${gpu}" \
    CUDA_MPS_LOG_DIRECTORY="$MPS_BASE_DIR/log-gpu${gpu}" \
    nvidia-cuda-mps-control -d
done
sleep 2
echo "--- MPS control.log per GPU ---"
for gpu in 0 1 2 3; do
    echo "-- gpu $gpu --"
    cat "$MPS_BASE_DIR/log-gpu${gpu}/control.log" 2>&1 || echo "(no control.log)"
done
echo "--- ps aux | grep mps ---"
ps aux | grep -i mps | grep -v grep

echo ""
echo "=== 8 ranks / 4 GPUs, WITH MPS ==="
mpirun -np 8 ./gpu_rank_wrapper.sh ./mps_gpu_diag || echo "mpirun exited non-zero (expected if reproducing the bug)"

for gpu in 0 1 2 3; do
    CUDA_MPS_PIPE_DIRECTORY="$MPS_BASE_DIR/pipe-gpu${gpu}" \
    bash -c 'echo quit | nvidia-cuda-mps-control' || true
done

echo "Done"
