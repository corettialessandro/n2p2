#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --mem=64GB
#SBATCH --job-name=h2o2g-warp-32rank-check
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Check the warp-per-atom kernel fix at H2O_2G's most extreme failure
# case: 32 ranks, ~20 atoms/rank (630/32). This was ~30x slower than CPU
# with the old one-thread-per-atom kernel (job 52567137: GPU 748.19s vs
# CPU 25.31s Pair, 2000 steps). Reuses the already-built, already-current
# GPU tree from job 52576146 (warp-per-atom kernels, GPU=1 GPU_SF=1) --
# no rebuild needed.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
LAMMPS_GPU=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025
LAMMPS_CPU=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_cpu_booster/lammps-22Jul2025
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"
WRAPPER="$N2P2_DEVEL/gpu/e2e_train_check/rank_gpu_wrapper.sh"

echo "=== Starting MPS control daemon ==="
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-pipe-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
sleep 2
export RANKS_PER_GPU=8

echo ""
echo "=== [1/2] H2O_2G, GPU SF (warp-per-atom), 32 ranks/4 GPUs, 2000 steps ==="
RUNDIR="$DYN/h2o2g_warp_32rank_gpu"
rm -rf "$RUNDIR" && mkdir -p "$RUNDIR"
cp "$DYN/init_conf_minimized.data" "$DYN/md_smoketest.lmp" "$RUNDIR/"
ln -s "$DYN/hdnnp-data" "$RUNDIR/hdnnp-data"
cd "$RUNDIR"
mpirun -np 32 "$WRAPPER" "$LAMMPS_GPU/src/lmp_mpi" -in md_smoketest.lmp -log log.h2o2g_warp_32rank_gpu
echo "--- GPU result ---"
grep -A1 "^Pair " log.h2o2g_warp_32rank_gpu
grep "^Loop time" log.h2o2g_warp_32rank_gpu

echo ""
echo "=== Stopping MPS control daemon ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2

echo ""
echo "=== [2/2] H2O_2G, CPU-only, 32 ranks, 2000 steps (for direct comparison) ==="
RUNDIR="$DYN/h2o2g_warp_32rank_cpu"
rm -rf "$RUNDIR" && mkdir -p "$RUNDIR"
cp "$DYN/init_conf_minimized.data" "$DYN/md_smoketest.lmp" "$RUNDIR/"
ln -s "$DYN/hdnnp-data" "$RUNDIR/hdnnp-data"
cd "$RUNDIR"
mpirun -np 32 "$LAMMPS_CPU/src/lmp_mpi" -in md_smoketest.lmp -log log.h2o2g_warp_32rank_cpu
echo "--- CPU result ---"
grep -A1 "^Pair " log.h2o2g_warp_32rank_cpu
grep "^Loop time" log.h2o2g_warp_32rank_cpu

echo ""
echo "Done"
