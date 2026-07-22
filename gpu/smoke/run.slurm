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

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
cd "$N2P2_DEVEL/gpu/smoke"

echo "Compiling real-neighbor dumper (links against libnnp.a)..."
# No GSL/BLAS needed here -- confirmed libnnp.a has zero undefined gsl_*
# symbols for this ElementMap/Structure-only usage (nm -D found none), and
# the system linker chokes on libgsl.so's compressed debug sections anyway.
g++ -O3 -std=c++11 -I"$N2P2_DEVEL/include" -I"$EIGEN" -DEIGEN_DONT_PARALLELIZE \
    -o dump_real_neighbors dump_real_neighbors.cpp \
    "$N2P2_DEVEL/lib/libnnp.a"

echo "Dumping real H-H neighbor data from H2O_2G/input.data..."
./dump_real_neighbors "$N2P2_DEVEL/temp/H2O_2G/input.data" real_neighbors_HH.txt

echo "Compiling CUDA smoke test..."
nvcc -O3 -std=c++14 -arch=sm_80 -o symfnc_exprad_test symfnc_exprad_test.cu

echo "Running..."
./symfnc_exprad_test real_neighbors_HH.txt
echo "Done"
