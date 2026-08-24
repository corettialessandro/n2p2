#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-kf-small-plain
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Disambiguates: did the 200-structure subset avoid the countUpdates
# corruption on its own merits, or did compute-sanitizer's
# serialization mask a genuine timing-dependent race again (as it did
# for GpuForces.cu earlier)? Same 200-structure dataset, no sanitizer.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force
KF="$DBG/kf_trace"

mkdir -p "$KF/small_plain_run"
cp "$KF/input.nn" "$KF/scaling.data" "$KF/small_plain_run/"
cp "$KF/input_small.data" "$KF/small_plain_run/input.data"
cd "$KF/small_plain_run"

mpirun -np 4 "$N2P2_GPU/bin/nnp-train"

echo "--- train-log.out, first force update ---"
awk '$1=="F"' train-log.out | head -6

echo "Done"
