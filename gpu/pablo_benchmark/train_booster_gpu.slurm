#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time 1-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --mem=200GB
#SBATCH --job-name=train-booster-gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Parameterized by SYS (water/magnetite/feldspar) and EPOCHS (10 or 100),
# exported via --export at submission time. Booster, 32 ranks spread
# 8-per-GPU across 4 A100s via a single shared MPS daemon
# (rank_gpu_wrapper.sh) -- the configuration gpu/e2e_train_check's H2O_2G
# work found actually works (32 ranks on 1 GPU runs out of memory on the
# persistent per-structure force-topology cache over a long run).
# NN-forward GPU dispatch only (GPU=1, not GPU_SF=1 -- that flag is
# LAMMPS-MD-specific and not validated/used for nnp-train).

set -e
: "${SYS:?Set SYS=water|magnetite|feldspar via --export}"
: "${EPOCHS:?Set EPOCHS=10|100 via --export}"

module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_${SYS}
N2P2_ORIG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
DATA="$N2P2_ORIG/gpu/pablo_benchmark/data/$SYS"
OUT="$N2P2_ORIG/gpu/pablo_benchmark/results/${SYS}_booster-gpu_${EPOCHS}ep"
WRAPPER="$N2P2_ORIG/gpu/e2e_train_check/rank_gpu_wrapper.sh"

mkdir -p "$N2P2_ORIG/gpu/pablo_benchmark/results"
rm -rf "$OUT" && mkdir -p "$OUT" && cd "$OUT"
cp "$DATA/input.data" "$DATA/scaling.data" .
sed "s/^epochs.*/epochs                          ${EPOCHS}            # Number of training epochs./" \
    "$DATA/input.nn" > input.nn

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

echo ""
echo "=== Running GPU build (Booster, 32 ranks / 4 GPUs, 8/GPU), $SYS, $EPOCHS epochs ==="
export RANKS_PER_GPU=8
mpirun -np 32 "$WRAPPER" ./nnp-train-gpu

echo ""
echo "=== Stopping MPS control daemon ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2

echo ""
echo "=== timing.out (all epochs) ==="
cat timing.out | grep -v "^#"

echo "Done"
