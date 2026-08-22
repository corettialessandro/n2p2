#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --mem=64GB
#SBATCH --job-name=longrun-cpu-glue-profile
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Candidate next-phase approach #1 (gpu/README.md): does the CPU-glue
# (CSR build + scatter-back) to GPU-call time ratio grow over the
# course of a long run? Windowed Stopwatch instrumentation (temporary,
# see Mode.cpp), printed every 2000 timesteps across the full
# 200000-step H2O_2G production config, 32 ranks/4 GPUs (the same
# best-known config used for the production comparison). GPU-only run
# -- no CPU comparison needed, this is purely about the internal
# phase breakdown of the GPU path over time.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
LAMMPS_GPU=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"
WRAPPER="$N2P2_DEVEL/gpu/e2e_train_check/rank_gpu_wrapper.sh"

echo "=== [0/2] Rebuild GPU tree with windowed profiling instrumentation ==="
(cd "$N2P2_DEVEL/src/libnnpgpu" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpgpu" && make COMP=gnu MODE=static GPU=1)
(cd "$N2P2_DEVEL/src" && make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 GPU_SF=1 PROJECT_EIGEN="$EIGEN" \
    INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnp libnnpgpu libnnpif)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
cd "$LAMMPS_GPU/src"
make yes-ml-hdnnp
make lib-hdnnp args="-p $N2P2_DEVEL"
make mpi -j"$SLURM_NTASKS_PER_NODE"

echo ""
echo "=== [1/2] Starting MPS control daemon serving all 4 GPUs ==="
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-pipe-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
sleep 2
export RANKS_PER_GPU=8

echo ""
echo "=== [2/2] Full 200000-step H2O_2G run, GPU, 32 ranks/4 GPUs, windowed profiling ==="
RUNDIR="$DYN/longrun_cpu_glue_profile"
rm -rf "$RUNDIR" && mkdir -p "$RUNDIR"
cp "$DYN/init_conf_minimized.data" "$DYN/md.lmp" "$RUNDIR/"
ln -s "$DYN/hdnnp-data" "$RUNDIR/hdnnp-data"
cd "$RUNDIR"
mpirun -np 32 "$WRAPPER" "$LAMMPS_GPU/src/lmp_mpi" -in md.lmp -log log.longrun_cpu_glue_profile

echo ""
echo "=== Stopping MPS control daemon ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2

echo ""
echo "=== Pair time summary ==="
grep -A2 "^Pair " log.longrun_cpu_glue_profile
grep "^Loop time" log.longrun_cpu_glue_profile
echo "Done"
