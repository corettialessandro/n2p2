#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --job-name=elecforces-test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load cuda/12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
FIXTURE="$N2P2_DEVEL/temp/4g_profile/elecforces_fixture"

cd "$N2P2_DEVEL/gpu/gemm"

echo "Compiling elecforces_test.cu..."
nvcc -O3 -std=c++14 -arch=sm_80 -o elecforces_test elecforces_test.cu

echo ""
echo "=== Validating GpuElecForces kernel design against real fElec from H2O_4G ==="
./elecforces_test "$FIXTURE"

echo "Done"
