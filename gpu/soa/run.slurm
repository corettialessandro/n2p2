#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:10:00        # format: HH:MM:SS
#SBATCH -N 1                   # 1 node
#SBATCH --ntasks-per-node=1    # 1 task (single-process host-side validation)
#SBATCH --mem=8GB
#SBATCH --job-name=soa-batch-test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# No --gres=gpu here: this step (Phase 1 SoA/CSR layout) is pure host-side
# C++, no CUDA involved yet -- it only needs to prove the new data layout is
# a faithful re-encoding of Structure/Atom before any kernel consumes it.

module purge
# No GSL/BLAS/MPI modules needed -- same as gpu/smoke/dump_real_neighbors.cpp,
# libnnp.a has zero undefined gsl_* symbols for this ElementMap/Structure-only
# usage (confirmed via nm -D when that test was built).

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
cd "$N2P2_DEVEL/gpu/soa"

echo "Compiling AtomBatch + validation test (links against libnnp.a)..."
g++ -O3 -std=c++11 -I"$N2P2_DEVEL/include" -I"$EIGEN" -DEIGEN_DONT_PARALLELIZE \
    -o build_batch_test AtomBatch.cpp build_batch_test.cpp \
    "$N2P2_DEVEL/lib/libnnp.a"

echo "Running validation against real H2O_2G structure..."
./build_batch_test "$N2P2_DEVEL/temp/H2O_2G/input.data"

echo "Done"
