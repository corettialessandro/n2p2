#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:10:00        # format: HH:MM:SS
#SBATCH -N 1                   # 1 node
#SBATCH --ntasks-per-node=1    # 1 task
#SBATCH --gres=gpu:1           # 1 gpu
#SBATCH --mem=16GB             # production case needs an 89 MB dense P (x2,
                                # one GEMM copy + one naive-kernel copy) plus
                                # host-side copies for the CPU reference
#SBATCH --job-name=kalman-gemm-test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load cuda/12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
MPI_INC=/opt/mpi/openmpi/4.1.5.4/include
cd "$N2P2_DEVEL/gpu/gemm"

# Same debug-info-stripping workaround as ../kalman/run.slurm (KalmanFilter.o's
# compressed .debug_info section breaks this cluster's binutils 2.30 ld).
echo "Extracting and stripping needed objects from libnnptrain.a/libnnp.a..."
ar p "$N2P2_DEVEL/lib/libnnptrain.a" KalmanFilter.o > KalmanFilter.o
ar p "$N2P2_DEVEL/lib/libnnptrain.a" Updater.o > Updater.o
ar p "$N2P2_DEVEL/lib/libnnp.a" Stopwatch.o > Stopwatch.o
ar p "$N2P2_DEVEL/lib/libnnp.a" utility.o > utility.o
strip --strip-debug KalmanFilter.o Updater.o Stopwatch.o utility.o

echo "=== Performance work: cuBLAS-batched Kalman updateP vs. one-thread-per-element ==="
nvcc -O3 -std=c++14 -arch=sm_80 -I"$N2P2_DEVEL/include" -I"$EIGEN" -I"$MPI_INC" \
    -DEIGEN_DONT_PARALLELIZE \
    -o kalman_gemm_test kalman_gemm_test.cu \
    KalmanFilter.o Updater.o Stopwatch.o utility.o -lcublas
./kalman_gemm_test

echo "Done"
