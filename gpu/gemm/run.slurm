#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:10:00        # format: HH:MM:SS
#SBATCH -N 1                   # 1 node
#SBATCH --ntasks-per-node=1    # 1 task
#SBATCH --gres=gpu:1           # 1 gpu
#SBATCH --mem=8GB
#SBATCH --job-name=nn-forward-gemm-test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load cuda/12.2
# No GSL/BLAS/MPI modules needed -- ElementMap/Structure/NeuralNetwork have
# no undefined gsl_* symbols for this usage (same reasoning as
# gpu/smoke/dump_real_neighbors.cpp and gpu/nn/run.slurm). cuBLAS comes from
# the CUDA toolkit itself (-lcublas), not a separate module.

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
cd "$N2P2_DEVEL/gpu/gemm"

echo "Compiling AtomBatch (shared host-side object, from ../soa)..."
g++ -O3 -std=c++14 -c -I"$N2P2_DEVEL/include" -I"$EIGEN" -DEIGEN_DONT_PARALLELIZE \
    -o AtomBatch.o "$N2P2_DEVEL/gpu/soa/AtomBatch.cpp"

echo "=== Performance work: cuBLAS-batched NN forward+dEdG vs. one-thread-per-atom ==="
nvcc -O3 -std=c++14 -arch=sm_80 -I"$N2P2_DEVEL/include" -I"$EIGEN" \
    -DEIGEN_DONT_PARALLELIZE \
    -o nn_forward_gemm_test AtomBatch.o nn_forward_gemm_test.cu \
    "$N2P2_DEVEL/lib/libnnp.a" -lcublas
./nn_forward_gemm_test "$N2P2_DEVEL/temp/H2O_2G/input.data"

echo "Done"
