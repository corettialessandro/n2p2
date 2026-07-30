#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --mem=200GB
#SBATCH --job-name=nnp-train-gpu-forces-4gpu
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

# gpuCalculateForces()'s persistent per-structure topology cache (see
# GpuForces.h/.cu) ran OUT OF GPU MEMORY under 32 ranks sharing one GPU:
# each structure's edge list is ~54MB (H2O_2G, ~1.68M edges), each rank
# caches ~35-40 distinct structures over an epoch (~1129 train structures
# / 32 ranks) = ~2GB/rank, and 32 ranks x ~2GB far exceeds what one A100
# can hold alongside everything else. Spreading the same 32 ranks 8/GPU
# across this node's 4 A100s (reusing rank_gpu_wrapper.sh + the single
# shared MPS daemon pattern from the earlier MPS-contention follow-up)
# cuts the aggregate cached memory per GPU to ~8 ranks x ~2GB = ~16GB,
# comfortably within one A100's capacity.

rm -rf run-gpu-forces-4gpu && mkdir run-gpu-forces-4gpu && cd run-gpu-forces-4gpu
cp "$TEMP/input.data" "$TEMP/scaling.data" .
sed 's/^epochs.*/epochs                          1              # Number of training epochs./' \
    "$TEMP/input.nn" > input.nn

echo "=== Clean build: GPU-enabled nnp-train (GPU=1), with cached-topology gpuCalculateForces() ==="
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
echo "=== timing.out ==="
cat timing.out | grep -v "^#"

echo ""
echo "=== learning-curve.out (sanity: forces GPU port shouldn't change training results) ==="
cat learning-curve.out | grep -v "^#"

echo "Done"
