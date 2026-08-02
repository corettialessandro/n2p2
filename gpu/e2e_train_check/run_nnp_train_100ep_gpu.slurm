#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time 1-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --mem=200GB
#SBATCH --job-name=nnp-train-100ep-gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

# Builds in an ISOLATED git worktree, not the shared, main n2p2_devel
# checkout -- see run_nnp_train_100ep_cpu.slurm's comment for why (a real
# build race between concurrent `make clean && make ...` runs in a
# shared src/lib/bin tree was directly observed as a very confusing,
# non-reproducible "heisenbug" while validating nnp-dataset). Each
# concurrent build gets its own git worktree (see
# `git worktree add --detach n2p2_devel_100ep_gpu gpu-portability`).
# temp/H2O_2G (read-only input data), this script's own output
# directory, and rank_gpu_wrapper.sh all still point at the original
# checkout.
N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_100ep_gpu
N2P2_ORIG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
TEMP="$N2P2_ORIG/temp/H2O_2G"
CHK="$N2P2_ORIG/gpu/e2e_train_check"
cd "$CHK"

# Real, 100-epoch, GPU-accelerated H2O_2G training run -- companion to
# run_nnp_train_100ep_cpu.slurm. Uses the configuration this session's
# investigation found actually works: 32 ranks spread 8-per-GPU across
# this node's 4 A100s via a single shared MPS daemon
# (rank_gpu_wrapper.sh) -- NOT 32 ranks on 1 GPU, which the persistent
# per-structure force-topology cache (GpuForces.cu) would run out of
# memory on over a run this long (it only survived 1-epoch tests).
# NOT cleaned up afterward -- weights.*.out/test.data/train.data are
# needed later for nnp-dataset energy/force comparisons.

mkdir -p results
rm -rf results/run-100ep-gpu && mkdir results/run-100ep-gpu && cd results/run-100ep-gpu
cp "$TEMP/input.data" "$TEMP/scaling.data" .
sed 's/^epochs.*/epochs                          100            # Number of training epochs./' \
    "$TEMP/input.nn" > input.nn

echo "=== Clean build: GPU-enabled nnp-train (GPU=1) (isolated worktree: $N2P2_DEVEL) ==="
(cd "$N2P2_DEVEL/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-train > build-gpu.log 2>&1)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
cp "$N2P2_DEVEL/bin/nnp-train" ./nnp-train-gpu

echo ""
echo "=== Starting a single MPS control daemon serving all 4 GPUs ==="
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-pipe-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
sleep 2
echo "--- control.log ---"; cat "$CUDA_MPS_LOG_DIRECTORY/control.log" 2>&1

echo ""
echo "=== Running GPU build, 100 epochs, 32 ranks / 4 GPUs (8/GPU) ==="
export RANKS_PER_GPU=8
mpirun -np 32 "$CHK/rank_gpu_wrapper.sh" ./nnp-train-gpu

echo ""
echo "=== Stopping MPS control daemon ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2

echo ""
echo "=== timing.out (all epochs) ==="
cat timing.out | grep -v "^#"

echo "Done"
