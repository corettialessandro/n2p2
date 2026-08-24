#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8GB
#SBATCH --job-name=debug-header-sens
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Isolate the amplification mechanism with ONE binary (CPU nnp-dataset,
# no GPU involved at all): does swapping in the GPU build's own
# force-calibrated conv_energy/conv_length header (differing from
# CPU's only at the ~15th significant digit) alone reproduce the large
# force disagreement, using the exact same CPU binary/weights/data
# otherwise? If yes, this confirms the instability is a pre-existing
# numerical sensitivity to conv_length in water's config -- not a GPU
# kernel defect -- merely triggered by any tiny (even CPU-vs-CPU
# reproducibility-scale) calibration difference.

set -e
module purge
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

TEST=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force/kf_trace/nnpdataset_epoch0_test
CPU_BIN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water/bin/nnp-dataset

echo "=== CPU nnp-dataset with GPU's header (last-digit-different conv_energy/conv_length) ==="
cd "$TEST/cpu_with_gpu_header"
"$CPU_BIN" 0 > run.out 2>&1
tail -5 run.out

echo "Done"
