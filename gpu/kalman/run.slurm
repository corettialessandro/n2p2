#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:10:00        # format: HH:MM:SS
#SBATCH -N 1                   # 1 node
#SBATCH --ntasks-per-node=1    # 1 task
#SBATCH --gres=gpu:1           # 1 gpu
#SBATCH --mem=16GB             # production case needs an 89 MB dense P plus
                                # host-side copies for the CPU reference
#SBATCH --job-name=kalman-test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load cuda/12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
MPI_INC=/opt/mpi/openmpi/4.1.5.4/include
cd "$N2P2_DEVEL/gpu/kalman"

# This test links the REAL nnp::KalmanFilter class (lib/libnnptrain.a) as
# ground truth (see kalman_test.cu's header comment). KalmanFilter.o's
# .debug_info section is zlib-compressed in a way this cluster's binutils
# 2.30 `ld` can't decompress ("unable to initialize decompress status for
# section .debug_info"), which then makes `ld` reject the *entire* archive
# ("File format not recognized") -- same underlying class of problem noted
# elsewhere in this project for libgsl.so's compressed debug sections, just
# hitting our own object this time instead of a system library. Since
# KalmanFilter.o/Updater.o have zero undefined GSL/MPI runtime symbols
# (confirmed via `nm -u`), the fix is to extract just the four objects we
# actually need (KalmanFilter.o/Updater.o from libnnptrain.a, Stopwatch.o/
# utility.o from libnnp.a) and strip their debug info before linking,
# rather than linking the full archives.
echo "Extracting and stripping needed objects from libnnptrain.a/libnnp.a..."
ar p "$N2P2_DEVEL/lib/libnnptrain.a" KalmanFilter.o > KalmanFilter.o
ar p "$N2P2_DEVEL/lib/libnnptrain.a" Updater.o > Updater.o
ar p "$N2P2_DEVEL/lib/libnnp.a" Stopwatch.o > Stopwatch.o
ar p "$N2P2_DEVEL/lib/libnnp.a" utility.o > utility.o
strip --strip-debug KalmanFilter.o Updater.o Stopwatch.o utility.o

echo "=== Phase 4 step 1: Kalman filter update() on GPU vs. CPU reference vs. real KalmanFilter ==="
nvcc -O3 -std=c++14 -arch=sm_80 -I"$N2P2_DEVEL/include" -I"$EIGEN" -I"$MPI_INC" \
    -DEIGEN_DONT_PARALLELIZE \
    -o kalman_test kalman_test.cu \
    KalmanFilter.o Updater.o Stopwatch.o utility.o
./kalman_test

echo "Done"
