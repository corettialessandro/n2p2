#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-step-bad86
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Same trace as debug_step_by_step.slurm, but for water structure 86 --
# identified as the worst-disagreeing structure in the nnp-norm-header
# comparison. Binary already built with the dump instrumentation.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_CPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water
N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force
STEP="$DBG/step_by_step"

mkdir -p "$STEP/water_bad86_gpu"
cp "$DBG/nnpnorm_test/input.nn" "$DBG/nnpnorm_test/scaling.data" \
   "$DBG/nnpnorm_test/weights.001.data" "$DBG/nnpnorm_test/weights.008.data" \
   "$DBG/full_test.data" "$STEP/water_bad86_gpu/"

cd "$STEP/water_bad86_gpu"
export N2P2_DUMP_STRUCT_FILE="$STEP/water_bad86_gpu/dump_struct86.txt"
export N2P2_DUMP_STRUCT_INDEX=86
mpirun -np 1 "$N2P2_GPU/bin/nnp-dataset" 0 full_test.data
unset N2P2_DUMP_STRUCT_FILE N2P2_DUMP_STRUCT_INDEX

echo "--- water structure 86: trusted CPU force (physical + internal units) ---"
awk '$1==86' "$DBG/nnpnorm_test/cpu_run/forces.comp" | head -6

echo "Done"
