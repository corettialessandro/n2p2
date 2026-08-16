#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time 1-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --mem=200GB
#SBATCH --job-name=lammps-hdnnp-prod-gpu-booster32
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Full 200000-step H2O_2G production run, GPU: Booster, 32 ranks / 4
# A100s (8 ranks/GPU via a single shared MPS daemon + rank_gpu_wrapper.sh),
# normal QOS -- the fastest of the 4 configs measured in the short
# (2000-step) scan (scan_gpu_ranks.slurm, jobs 52494963/52494969/52495167):
# 4r/1g 186.4s, 8r/2g 99.5s, 16r/4g 55.6s, 32r/4g 34.6s loop time.
# Companion to run_production_cpu_dcgp.slurm. Reuses the already-built,
# already-validated GPU lmp_mpi from build_gpu.slurm (job 52482335).

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
LAMMPS_GPU_BIN=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025/src/lmp_mpi
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"
CHK="$N2P2_DEVEL/gpu/e2e_train_check"
RUNDIR="$DYN/production_gpu_booster32"

rm -rf "$RUNDIR" && mkdir -p "$RUNDIR"
cp "$DYN/init_conf_minimized.data" "$DYN/md.lmp" "$RUNDIR/"
ln -s "$DYN/hdnnp-data" "$RUNDIR/hdnnp-data"
cd "$RUNDIR"

echo "=== Starting a single MPS control daemon serving all 4 GPUs ==="
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-pipe-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
sleep 2

echo ""
echo "=== Full 200000-step H2O_2G run, GPU (Booster, 32 ranks / 4 GPUs, 8/GPU) ==="
export RANKS_PER_GPU=8
mpirun -np 32 "$CHK/rank_gpu_wrapper.sh" "$LAMMPS_GPU_BIN" -in md.lmp -log log.production_gpu_32

echo ""
echo "=== Stopping MPS control daemon ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2

echo ""
echo "=== Pair time summary ==="
grep -A2 "^Pair " log.production_gpu_32
grep "^Loop time" log.production_gpu_32

echo "Done"
