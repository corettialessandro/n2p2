#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time 1-00:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --mem=200GB
#SBATCH --job-name=train-booster-gpu-mn
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Multi-node variant of train_booster_gpu.slurm. GpuForces.cu's
# persistent per-structure device cache is unbounded (uploaded once,
# never evicted -- deliberately deferred back when only H2O_2G, ~1254
# structures, existed). Per-GPU cached-structure count works out to
# (total dataset structures) / (total GPUs used), independent of how
# many MPI ranks share each GPU -- so the only lever that actually
# reduces per-GPU load, without code changes, is more GPUs, i.e. more
# nodes. Magnetite's single-node run OOM'd at ~6700 structures/GPU
# (26789/4); this spreads across --nodes x 4 GPUs instead.
#
# Every allocated node needs its OWN MPS control daemon (MPS is
# per-node/per-GPU-set) -- started/stopped via srun --ntasks-per-node=1
# across the whole allocation, unlike the single-node script where one
# direct call sufficed.

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
RESULTS_BASE=/leonardo_work/L-AUT_Giane_26/acoretti/NEURALCPM/gpu_porting/pablo_benchmark
OUT="$RESULTS_BASE/${SYS}_booster-gpu-mn${SLURM_JOB_NUM_NODES}_${EPOCHS}ep"
WRAPPER="$N2P2_ORIG/gpu/e2e_train_check/rank_gpu_wrapper.sh"

mkdir -p "$RESULTS_BASE"
rm -rf "$OUT" && mkdir -p "$OUT" && cd "$OUT"
cp "$DATA/input.data" "$DATA/scaling.data" .
sed -e "s/^epochs.*/epochs                          ${EPOCHS}            # Number of training epochs./" \
    -e "s/^write_trainpoints.*/write_trainpoints               0/" \
    -e "s/^write_trainforces.*/write_trainforces               0/" \
    "$DATA/input.nn" > input.nn

echo "=== Clean build: GPU-enabled nnp-train (GPU=1) (isolated worktree: $N2P2_DEVEL) ==="
(cd "$N2P2_DEVEL/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-train > build-gpu.log 2>&1)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
cp "$N2P2_DEVEL/bin/nnp-train" ./nnp-train-gpu

echo ""
echo "=== Starting one MPS control daemon per node ($SLURM_JOB_NUM_NODES nodes) ==="
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-pipe-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$SLURM_JOB_ID"
srun --ntasks-per-node=1 --ntasks="$SLURM_JOB_NUM_NODES" bash -c \
    'mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" && nvidia-cuda-mps-control -d'
sleep 2

echo ""
echo "=== Running GPU build ($SLURM_JOB_NUM_NODES nodes x 32 ranks / 4 GPUs, 8/GPU), $SYS, $EPOCHS epochs ==="
export RANKS_PER_GPU=8
mpirun -np "$SLURM_NTASKS" "$WRAPPER" ./nnp-train-gpu

echo ""
echo "=== Stopping MPS control daemons ==="
srun --ntasks-per-node=1 --ntasks="$SLURM_JOB_NUM_NODES" bash -c \
    'echo quit | nvidia-cuda-mps-control' || true
sleep 2

echo ""
echo "=== timing.out (all epochs) ==="
cat timing.out | grep -v "^#"

echo "Done"
