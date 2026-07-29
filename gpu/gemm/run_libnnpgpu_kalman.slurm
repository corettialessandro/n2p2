#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB             # production case needs an 89 MB dense P plus
                                # host-side copies for the CPU reference
#SBATCH --job-name=libnnpgpu-kalman-test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load cuda/12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
MPI_INC=/opt/mpi/openmpi/4.1.5.4/include

# Rebuild libnnp.a fresh on THIS node (stale cross-node -march=native
# libraries caused a real crash earlier in this port -- see gpu/README.md's
# Phase 6 section). libnnptrain.a needs GSL/MPI to build, so load those
# only for this step then purge back to a plain cuda module (nvcc below
# doesn't want mpic++'s wrapper flags).
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
(cd "$N2P2_DEVEL/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" libnnp libnnptrain)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
module purge
module load cuda/12.2

cd "$N2P2_DEVEL/gpu/gemm"

# Same debug-info-stripping workaround as ../kalman/run.slurm and this
# directory's own run_kalman.slurm (KalmanFilter.o's compressed
# .debug_info section breaks this cluster's binutils 2.30 ld when linking
# the full archive -- extracting+stripping the handful of objects actually
# needed avoids it).
echo "Extracting and stripping needed objects from libnnptrain.a/libnnp.a..."
ar p "$N2P2_DEVEL/lib/libnnptrain.a" KalmanFilter.o > KalmanFilter.o
ar p "$N2P2_DEVEL/lib/libnnptrain.a" Updater.o > Updater.o
ar p "$N2P2_DEVEL/lib/libnnp.a" Stopwatch.o > Stopwatch.o
ar p "$N2P2_DEVEL/lib/libnnp.a" utility.o > utility.o
strip --strip-debug KalmanFilter.o Updater.o Stopwatch.o utility.o

echo "Compiling src/libnnpgpu/GpuKalmanFilter.cu..."
nvcc -O3 -std=c++14 -arch=sm_80 -I"$N2P2_DEVEL/include" \
    -c -o GpuKalmanFilter.o "$N2P2_DEVEL/src/libnnpgpu/GpuKalmanFilter.cu"

echo "=== Validating gpuKalman* persistent-state API (KalmanFilter::update() call site) ==="
nvcc -O3 -std=c++14 -arch=sm_80 -I"$N2P2_DEVEL/include" -I"$EIGEN" -I"$MPI_INC" -I"$N2P2_DEVEL/src/libnnpgpu" \
    -DEIGEN_DONT_PARALLELIZE \
    -o libnnpgpu_kalman_test GpuKalmanFilter.o libnnpgpu_kalman_test.cu \
    KalmanFilter.o Updater.o Stopwatch.o utility.o -lcublas
./libnnpgpu_kalman_test

echo "Done"
