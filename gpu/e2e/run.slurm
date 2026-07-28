#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:10:00        # format: HH:MM:SS
#SBATCH -N 1                   # 1 node
#SBATCH --ntasks-per-node=1    # 1 task
#SBATCH --gres=gpu:1           # 1 gpu
#SBATCH --mem=16GB             # Stage 5's dense N=3327 P matrix (~89 MB) plus
                                # host-side copies for the CPU reference
#SBATCH --job-name=e2e-single-structure-test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load cuda/12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
MPI_INC=/opt/mpi/openmpi/4.1.5.4/include
cd "$N2P2_DEVEL/gpu/e2e"

echo "Compiling AtomBatch (shared host-side object, from ../soa)..."
g++ -O3 -std=c++14 -c -I"$N2P2_DEVEL/include" -I"$EIGEN" -DEIGEN_DONT_PARALLELIZE \
    -o AtomBatch.o "$N2P2_DEVEL/gpu/soa/AtomBatch.cpp"

# Stage 5 links the REAL nnp::KalmanFilter class (lib/libnnptrain.a) for a
# ground-truth cross-check. KalmanFilter.o's .debug_info section is
# zlib-compressed in a way this cluster's binutils 2.30 `ld` can't
# decompress, which makes `ld` reject the whole archive ("File format not
# recognized") -- same class of problem as the libgsl.so compressed-debug-
# sections issue noted elsewhere in this project, just in our own object
# this time (see ../kalman/run.slurm, where this was first hit and fixed).
# Fix: extract just the needed objects and strip debug info before linking,
# instead of linking the full libnnptrain.a archive.
echo "Extracting and stripping needed objects from libnnptrain.a/libnnp.a..."
ar p "$N2P2_DEVEL/lib/libnnptrain.a" KalmanFilter.o > KalmanFilter.o
ar p "$N2P2_DEVEL/lib/libnnptrain.a" Updater.o > Updater.o
strip --strip-debug KalmanFilter.o Updater.o

echo "=== End-to-end: real symmetry functions -> NN forward/dEdG -> force assembly -> Kalman energy update ==="
nvcc -O3 -std=c++14 -arch=sm_80 -I"$N2P2_DEVEL/include" -I"$EIGEN" -I"$MPI_INC" \
    -DEIGEN_DONT_PARALLELIZE \
    -o e2e_single_structure_test AtomBatch.o e2e_single_structure_test.cu \
    KalmanFilter.o Updater.o "$N2P2_DEVEL/lib/libnnp.a"
./e2e_single_structure_test "$N2P2_DEVEL/temp/H2O_2G/input.data" "$N2P2_DEVEL/temp/H2O_2G/input.nn"

echo "Done"
