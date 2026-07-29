#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:05:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4GB
#SBATCH --job-name=check-mps
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load cuda/12.2

echo "=== nvidia-smi ==="
which nvidia-smi 2>&1
nvidia-smi 2>&1
echo ""
echo "=== nvidia-smi -q for compute mode ==="
nvidia-smi -q | grep -A2 "Compute Mode" 2>&1
echo ""
echo "=== nvidia-cuda-mps-control ==="
which nvidia-cuda-mps-control 2>&1
find / -xdev -iname "nvidia-cuda-mps-control" 2>/dev/null
find /usr -iname "nvidia-cuda-mps*" 2>/dev/null
find /opt -iname "nvidia-cuda-mps*" 2>/dev/null
echo ""
echo "=== CUDA driver version / MPS-related env ==="
nvidia-smi --query-gpu=driver_version,name,compute_cap --format=csv 2>&1
echo ""
echo "=== checking for any MPS log/pipe dirs already set up by site admins ==="
env | grep -i "MPS\|CUDA_VISIBLE"
ls -la /tmp/nvidia-mps* 2>&1
echo "Done"
