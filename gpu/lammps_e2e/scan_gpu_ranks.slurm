#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --mem=64GB
#SBATCH --job-name=lammps-hdnnp-gpu-scan
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Short (2000-step) scan across MPI-rank / GPU-count configurations for
# the H2O_2G (630 atoms) case, to find the fastest Booster setup for the
# real 200000-step production run -- rather than assuming the 32-rank/
# 4-GPU/8-per-GPU config already validated for nnp-train (gpu/README.md's
# MPS section) is automatically best for MD's very different workload
# shape (small, fixed-size spatial domain decomposition every timestep,
# vs per-structure training batches). NP and RANKS_PER_GPU are passed in
# via `sbatch --export=NP=..,RANKS_PER_GPU=.. --ntasks-per-node=$NP
# --gres=gpu:$((NP/RANKS_PER_GPU))`.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
LAMMPS_GPU_BIN=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025/src/lmp_mpi
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"
CHK="$N2P2_DEVEL/gpu/e2e_train_check"

NGPU=$(( NP / RANKS_PER_GPU ))
TAG="${NP}r_${NGPU}g"
# Own subdirectory per config -- these scan jobs run concurrently, and
# md_smoketest.lmp writes fixed filenames (dump.lammpstrj, final_conf.data)
# that would otherwise collide.
SCANDIR="$DYN/scan_$TAG"
rm -rf "$SCANDIR" && mkdir -p "$SCANDIR"
cp "$DYN/init_conf_minimized.data" "$DYN/md_smoketest.lmp" "$SCANDIR/"
ln -s "$DYN/hdnnp-data" "$SCANDIR/hdnnp-data"
cd "$SCANDIR"
echo "=== Scanning NP=$NP ranks / $NGPU GPUs (RANKS_PER_GPU=$RANKS_PER_GPU) ==="

if [ "$NP" -gt 1 ]; then
  export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-pipe-$SLURM_JOB_ID"
  export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$SLURM_JOB_ID"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d
  sleep 2
  export RANKS_PER_GPU
  mpirun -np "$NP" "$CHK/rank_gpu_wrapper.sh" "$LAMMPS_GPU_BIN" -in md_smoketest.lmp -log "log.scan_$TAG"
  echo quit | nvidia-cuda-mps-control || true
  sleep 2
else
  mpirun -np "$NP" "$LAMMPS_GPU_BIN" -in md_smoketest.lmp -log "log.scan_$TAG"
fi

echo ""
echo "=== Pair time summary (NP=$NP, NGPU=$NGPU) ==="
grep -A2 "^Pair " "log.scan_$TAG"
grep "^Loop time" "log.scan_$TAG"

echo "Done"
