#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:10:00        # format: HH:MM:SS
#SBATCH -N 1                   # 1 node
#SBATCH --ntasks-per-node=1    # 1 task
#SBATCH --gres=gpu:1           # 1 gpu (step 2 test needs it; step 1 doesn't)
#SBATCH --mem=8GB
#SBATCH --job-name=soa-batch-test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load cuda/12.2
# No GSL/BLAS/MPI modules needed -- same as gpu/smoke/dump_real_neighbors.cpp,
# libnnp.a has zero undefined gsl_* symbols for this ElementMap/Structure-only
# usage (confirmed via nm -D when that test was built).

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
cd "$N2P2_DEVEL/gpu/soa"

# -std=c++14 throughout (matches nvcc's host dialect below) so the plain
# g++-compiled AtomBatch.o and nvcc-compiled objects link cleanly together.
echo "Compiling AtomBatch (shared host-side object)..."
g++ -O3 -std=c++14 -c -I"$N2P2_DEVEL/include" -I"$EIGEN" -DEIGEN_DONT_PARALLELIZE \
    -o AtomBatch.o AtomBatch.cpp

echo "=== Step 1: AtomBatch vs. Structure/Atom::neighbors (host-only) ==="
g++ -O3 -std=c++14 -I"$N2P2_DEVEL/include" -I"$EIGEN" -DEIGEN_DONT_PARALLELIZE \
    -o build_batch_test AtomBatch.o build_batch_test.cpp \
    "$N2P2_DEVEL/lib/libnnp.a"
./build_batch_test "$N2P2_DEVEL/temp/H2O_2G/input.data"

echo ""
echo "=== Step 2: ExpRad CUDA kernel consuming AtomBatch directly ==="
nvcc -O3 -std=c++14 -arch=sm_80 -I"$N2P2_DEVEL/include" -I"$EIGEN" \
    -DEIGEN_DONT_PARALLELIZE \
    -o symfnc_exprad_soa_test AtomBatch.o symfnc_exprad_soa_test.cu \
    "$N2P2_DEVEL/lib/libnnp.a"
./symfnc_exprad_soa_test "$N2P2_DEVEL/temp/H2O_2G/input.data"

echo "Done"
