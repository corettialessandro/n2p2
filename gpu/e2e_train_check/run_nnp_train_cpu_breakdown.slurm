#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#SBATCH --job-name=nnp-train-cpu-breakdown
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

# TEMPORARY instrumentation run (gpu/README.md CPU-side-cost follow-up):
# Training.cpp was temporarily edited to add two extra stopwatches,
# force_collect (the CPU-only collectDGdxia loop) and force_gpu (the
# gpuNnForceDFdcSum dispatch call itself), printed as a DEBUG line at the
# end of each epoch. Reproduces the exact 32-ranks/1-GPU/MPS scenario
# whose F_err = 107.9s this is meant to decompose.

rm -rf run-cpu-breakdown && mkdir run-cpu-breakdown && cd run-cpu-breakdown
cp "$TEMP/input.data" "$TEMP/scaling.data" .
sed 's/^epochs.*/epochs                          1              # Number of training epochs./' \
    "$TEMP/input.nn" > input.nn

echo "=== Clean build: GPU-enabled nnp-train (GPU=1), with temporary force_collect/force_gpu instrumentation ==="
(cd "$N2P2_DEVEL/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-train > build-gpu.log 2>&1)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
cp "$N2P2_DEVEL/bin/nnp-train" ./nnp-train-gpu

echo ""
echo "=== Starting MPS control daemon ==="
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-pipe-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
sleep 2
echo "--- control.log ---"; cat "$CUDA_MPS_LOG_DIRECTORY/control.log" 2>&1

echo ""
echo "=== Running GPU build WITH MPS (32 ranks, 1 GPU) ==="
mpirun -np 32 ./nnp-train-gpu

echo ""
echo "=== Stopping MPS control daemon ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2

echo ""
echo "=== timing.out ==="
cat timing.out | grep -v "^#"

echo ""
echo "=== DEBUG force_collect/force_gpu/force_calcforces (rank 0, nnp-train.log.0000) ==="
grep "^DEBUG" nnp-train.log.0000

echo "Done"
