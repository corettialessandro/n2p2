#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:10:00        # format: HH:MM:SS
#SBATCH -N 1                   # 1 node
#SBATCH --ntasks-per-node=1    # 1 task
#SBATCH --gres=gpu:1           # 1 gpu
#SBATCH --mem=8GB
#SBATCH --job-name=libnnpgpu-forces-test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load cuda/12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
TEMP="$N2P2_DEVEL/temp/H2O_2G"

# Rebuild libnnp.a fresh on THIS node (see gpu/README.md's Phase 6 section
# and run_libnnpgpu_dfdc.slurm for why).
(cd "$N2P2_DEVEL/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" libnnp)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h

cd "$N2P2_DEVEL/gpu/gemm"

echo "Compiling src/libnnpgpu/GpuForces.cu..."
nvcc -O3 -std=c++14 -arch=sm_80 -I"$N2P2_DEVEL/include" \
    -c -o GpuForces.o "$N2P2_DEVEL/src/libnnpgpu/GpuForces.cu"

echo "Compiling libnnpgpu_forces_test.cu..."
nvcc -O3 -std=c++14 -arch=sm_80 -I"$N2P2_DEVEL/include" -I"$EIGEN" -I"$N2P2_DEVEL/src/libnnpgpu" \
    -DEIGEN_DONT_PARALLELIZE \
    -o libnnpgpu_forces_test GpuForces.o libnnpgpu_forces_test.cu \
    "$N2P2_DEVEL/lib/libnnp.a"

echo ""
echo "=== Setting up a real H2O_2G prediction directory (input.nn/scaling.data/weights) ==="
rm -rf run-forces-test && mkdir run-forces-test && cd run-forces-test
cp "$TEMP/input.nn" "$TEMP/scaling.data" "$TEMP/input.data" .
# Prediction expects "weights.%03zu.data" -- reuse this session's actual
# trained epoch-1 weights (real, not random) so the test exercises a
# genuine, physically meaningful dEdG/force, not all-zero/garbage weights.
cp "$TEMP/weights.001.000001.out" weights.001.data
cp "$TEMP/weights.008.000001.out" weights.008.data

echo ""
echo "=== Validating gpuCalculateForces() (Mode::calculateForces() GPU port) ==="
../libnnpgpu_forces_test

echo "Done"
