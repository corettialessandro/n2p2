#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:10:00        # format: HH:MM:SS
#SBATCH -N 1                   # 1 node
#SBATCH --ntasks-per-node=1    # 1 task
#SBATCH --gres=gpu:1           # 1 gpu
#SBATCH --mem=8GB
#SBATCH --job-name=expangn-neighbor-deriv-test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load cuda/12.2
# No GSL/BLAS/MPI modules needed -- same reasoning as the other gpu/*/run.slurm scripts.

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
cd "$N2P2_DEVEL/gpu/soa"

echo "Compiling AtomBatch (shared host-side object)..."
g++ -O3 -std=c++14 -c -I"$N2P2_DEVEL/include" -I"$EIGEN" -DEIGEN_DONT_PARALLELIZE \
    -o AtomBatch.o AtomBatch.cpp

echo "=== ExpAngn neighbor-side derivatives, grouped over real H2O_2G instances ==="
nvcc -O3 -std=c++14 -arch=sm_80 -I"$N2P2_DEVEL/include" -I"$EIGEN" \
    -DEIGEN_DONT_PARALLELIZE \
    -o symfnc_expangn_group_test AtomBatch.o symfnc_expangn_group_test.cu \
    "$N2P2_DEVEL/lib/libnnp.a"
./symfnc_expangn_group_test "$N2P2_DEVEL/temp/H2O_2G/input.data" "$N2P2_DEVEL/temp/H2O_2G/input.nn"

echo "Done"
