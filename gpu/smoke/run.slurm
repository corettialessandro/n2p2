#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:10:00        # format: HH:MM:SS
#SBATCH -N 1                   # 1 node
#SBATCH --ntasks-per-node=1    # 1 task (this is a single-process smoke test)
#SBATCH --gres=gpu:1           # 1 gpu
#SBATCH --mem=8GB
#SBATCH --job-name=gpu-smoke
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load cuda/12.2

cd /leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/smoke
echo "Compiling..."
nvcc -O3 -std=c++14 -arch=sm_80 -o symfnc_exprad_test symfnc_exprad_test.cu
echo "Running..."
./symfnc_exprad_test
echo "Done"
