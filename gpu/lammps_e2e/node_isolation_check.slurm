#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --nodelist=lrdn1600
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --mem=64GB
#SBATCH --job-name=node-isolation-check
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Controlled test: is the H2O_2G GPU-SF slowdown a property of run
# LENGTH (as configured in the LAMMPS input script) or of which
# physical NODE the job lands on? job 52576480 (2000-step run, node
# lrdn3271) was fast (Pair avg 23.768s/2000 = 0.01188s/step incl. NN);
# job 52603120 (200000-step run, node lrdn1600) was slow from its very
# first 2000 steps (SF-only already 0.0186s/step). This job forces the
# SAME short 2000-step config onto the SAME node (lrdn1600) that
# produced the slow long run, to see whether short+lrdn1600 is fast
# (pointing at run length/config) or slow (pointing at node identity).
# Reuses the already-built, already-current lmp_mpi -- no rebuild.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
LAMMPS_GPU=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"
WRAPPER="$N2P2_DEVEL/gpu/e2e_train_check/rank_gpu_wrapper.sh"

echo "=== Node check ==="
hostname

echo ""
echo "=== Starting MPS control daemon ==="
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-pipe-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
sleep 2
export RANKS_PER_GPU=8

echo ""
echo "=== H2O_2G, GPU SF, 32 ranks/4 GPUs, 2000 steps, forced onto lrdn1600 ==="
RUNDIR="$DYN/node_isolation_check"
rm -rf "$RUNDIR" && mkdir -p "$RUNDIR"
cp "$DYN/init_conf_minimized.data" "$DYN/md_smoketest.lmp" "$RUNDIR/"
ln -s "$DYN/hdnnp-data" "$RUNDIR/hdnnp-data"
cd "$RUNDIR"
mpirun -np 32 "$WRAPPER" "$LAMMPS_GPU/src/lmp_mpi" -in md_smoketest.lmp -log log.node_isolation_check

echo ""
echo "=== Stopping MPS control daemon ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2

echo ""
echo "=== Pair time summary ==="
grep -A1 "^Pair " log.node_isolation_check
grep "^Loop time" log.node_isolation_check
echo "Done"
