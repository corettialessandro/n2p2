#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-water-gpu-normalize
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Does a GPU-linked nnp-train compute a DIFFERENT normalize_data_set
# force calibration (mean_energy/conv_energy/conv_length) than the
# CPU-linked build did, on the identical fresh input.nn/random_seed?
# If normalize_data_set's calibration uses GPU-dispatched force
# computation internally, and that computation is the same buggy path
# already isolated in GpuForces.cu, the resulting global scaling
# constant could differ -- which would then get applied to every
# subsequent force prediction, exactly matching the observed pattern
# (energy always exact, forces wrong by a roughly consistent large
# factor across all structures).

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force/tanh_control_gpu

echo "=== Build nnp-train in the GPU worktree (only nnp-dataset was built there before) ==="
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
(cd "$N2P2_GPU/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-train > "$DBG/../build-gpu-train.log" 2>&1)
git -C "$N2P2_GPU" checkout -- src/libnnp/version.h

echo "=== 1-epoch GPU-linked nnp-train (tanh, same random_seed) ==="
cd "$DBG"
mpirun -np 4 "$N2P2_GPU/bin/nnp-train"

echo ""
echo "=== GPU-computed normalization (written into input.nn) ==="
grep "mean_energy\|conv_energy\|conv_length" input.nn

echo ""
echo "=== CPU-computed normalization (reference, from the earlier run) ==="
grep "mean_energy\|conv_energy\|conv_length" "$DBG/../tanh_control/input.nn"

echo "Done"
