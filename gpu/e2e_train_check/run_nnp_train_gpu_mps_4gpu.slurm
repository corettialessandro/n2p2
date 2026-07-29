#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --mem=200GB
#SBATCH --job-name=nnp-train-gpu-mps-4gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
TEMP="$N2P2_DEVEL/temp/H2O_2G"
CHK="$N2P2_DEVEL/gpu/e2e_train_check"
cd "$CHK"

# Follow-up to gpu/README.md's rank0-profiling finding: 32 ranks sharing
# ONE physical GPU spend most of F_err's time blocked in
# cudaDeviceSynchronize/cudaFree (contention for a shared MPS queue), not
# computing -- kernels themselves are already fast. This node has 4 A100s
# (sinfo: gpu:a100:4); this run spreads the same 32 ranks 8-per-GPU across
# all 4 via a SINGLE shared MPS daemon (no CUDA_VISIBLE_DEVICES
# restriction on the daemon itself -- it can see and serve all 4 GPUs;
# only each CLIENT rank restricts itself to one GPU, in
# rank_gpu_wrapper.sh), to see whether quartering the ranks-per-GPU count
# reduces F_err roughly proportionally, as the contention hypothesis
# predicts. An earlier attempt with 4 SEPARATE per-GPU daemons hit
# intermittent cublasCreate() failures on some ranks (see
# rank_gpu_wrapper.sh's comment) -- this single-daemon setup is simpler
# and is the pattern NVIDIA's own MPS docs describe for multi-GPU nodes.

rm -rf run-mps-4gpu && mkdir run-mps-4gpu && cd run-mps-4gpu
cp "$TEMP/input.data" "$TEMP/scaling.data" .
sed 's/^epochs.*/epochs                          1              # Number of training epochs./' \
    "$TEMP/input.nn" > input.nn

echo "=== Clean build: GPU-enabled nnp-train (GPU=1) ==="
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
echo "=== Running GPU build WITH shared MPS (32 ranks, 8 ranks/GPU across 4 GPUs) ==="
export RANKS_PER_GPU=8
mpirun -np 32 "$CHK/rank_gpu_wrapper.sh" ./nnp-train-gpu

echo ""
echo "=== Stopping MPS control daemon ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2

echo ""
echo "=== timing.out (this run: 32 ranks / 4 GPUs, 8 ranks/GPU) ==="
cat timing.out | grep -v "^#"

echo "Done"
