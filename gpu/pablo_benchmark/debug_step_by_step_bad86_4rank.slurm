#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-step-bad86-4rank
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Structure 86 traced clean at 1 rank -- but the original bad-structure
# identification (nnpnorm_test) used 4 ranks. Rerun the identical dump
# trace at 4 ranks (matching the original) to see if rank/GPU-sharing
# context is what actually determines which structures get corrupted.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force
STEP="$DBG/step_by_step"

mkdir -p "$STEP/water_bad86_4rank_gpu"
cp "$DBG/nnpnorm_test/input.nn" "$DBG/nnpnorm_test/scaling.data" \
   "$DBG/nnpnorm_test/weights.001.data" "$DBG/nnpnorm_test/weights.008.data" \
   "$DBG/full_test.data" "$STEP/water_bad86_4rank_gpu/"

cd "$STEP/water_bad86_4rank_gpu"
export N2P2_DUMP_STRUCT_FILE="$STEP/water_bad86_4rank_gpu/dump_struct86.txt"
export N2P2_DUMP_STRUCT_INDEX=86
mpirun -np 4 "$N2P2_GPU/bin/nnp-dataset" 0 full_test.data
unset N2P2_DUMP_STRUCT_FILE N2P2_DUMP_STRUCT_INDEX

echo "Done"
