#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:10:00        # format: HH:MM:SS
#SBATCH -N 1                   # 1 node
#SBATCH --ntasks-per-node=1    # 1 task
#SBATCH --gres=gpu:1           # 1 gpu
#SBATCH --mem=8GB
#SBATCH --job-name=libnnpgpu-test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load cuda/12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0

# Rebuild libnnp.a fresh on THIS node -- a stale/missing libnnp.a (this
# repo's lib/ is a shared build artifact, left in whatever state the last
# job's `make clean`/build cycle happened to leave it) has caused real
# link failures before (see gpu/README.md's Phase 6 section), so every
# validation job rebuilds it locally rather than trusting whatever is
# already sitting in lib/.
(cd "$N2P2_DEVEL/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" libnnp)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h

cd "$N2P2_DEVEL/gpu/gemm"

echo "Compiling src/libnnpgpu/GpuNeuralNetwork.cu..."
nvcc -O3 -std=c++14 -arch=sm_80 -I"$N2P2_DEVEL/include" \
    -c -o GpuNeuralNetwork.o "$N2P2_DEVEL/src/libnnpgpu/GpuNeuralNetwork.cu"

echo "=== Validating src/libnnpgpu/GpuNeuralNetwork.cu (Phase 6's new library) ==="
nvcc -O3 -std=c++14 -arch=sm_80 -I"$N2P2_DEVEL/include" -I"$EIGEN" -I"$N2P2_DEVEL/src/libnnpgpu" \
    -DEIGEN_DONT_PARALLELIZE \
    -o libnnpgpu_test GpuNeuralNetwork.o libnnpgpu_test.cu \
    "$N2P2_DEVEL/lib/libnnp.a" -lcublas
./libnnpgpu_test

echo "Done"
