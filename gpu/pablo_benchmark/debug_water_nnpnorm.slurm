#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-water-nnpnorm
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Corroborating test requested directly: does a genuine, water-specific
# nnp-norm calibration (derived from REFERENCE/DFT energy+force
# statistics, not an untrained network's own force predictions like
# normalize_data_set=force uses) also fix the CPU-vs-GPU divergence?
# This avoids the earlier test's weakness of borrowing H2O_2G's exact
# numbers from an unrelated chemical system.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_CPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water
N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force/nnpnorm_test

echo "=== Build nnp-norm ==="
(cd "$N2P2_CPU/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" nnp-norm > "$DBG/../build-nnpnorm.log" 2>&1)
git -C "$N2P2_CPU" checkout -- src/libnnp/version.h

echo "=== Run nnp-norm on water's actual dataset ==="
cd "$DBG"
mpirun -np 4 "$N2P2_CPU/bin/nnp-norm"

echo ""
echo "=== nnp-norm-computed normalization (water-specific, from reference data) ==="
grep "mean_energy\|conv_energy\|conv_length" input.nn

echo ""
echo "=== For comparison: water's own normalize_data_set=force calibration ==="
grep "mean_energy\|conv_energy\|conv_length" "$DBG/../tanh_control/input.nn"
echo "=== H2O_2G's fixed calibration ==="
grep "mean_energy\|conv_energy\|conv_length" /leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/temp/H2O_2G/input.nn

echo "Done"
