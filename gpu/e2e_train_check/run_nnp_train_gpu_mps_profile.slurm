#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#SBATCH --job-name=nnp-train-gpu-mps-profile
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

# Purpose: confirm/deny whether the ~150-185ms/call implied by F_err in
# the real 32-rank MPS run (gpu/README.md's persistent-state follow-up)
# is caused by 32 ranks contending for one MPS-shared GPU, something the
# single-process gpu/gemm/profile_dfdc.cu harness structurally cannot
# reproduce (it measured ~4.6ms/call with NO contention). Runs the real
# training job exactly like run_nnp_train_gpu_mps.slurm's "with MPS"
# case, but traces ONLY rank 0 with nsys (see profile_rank0_wrapper.sh
# for why nsys, not ncu, and why only one rank).

rm -rf run-mps-profile && mkdir run-mps-profile && cd run-mps-profile
cp "$TEMP/input.data" "$TEMP/scaling.data" .
sed 's/^epochs.*/epochs                          1              # Number of training epochs./' \
    "$TEMP/input.nn" > input.nn

echo "=== Clean build: GPU-enabled nnp-train (GPU=1) ==="
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
echo "=== Running GPU build WITH MPS (32 ranks), tracing rank 0 with nsys ==="
export NSYS_RANK0_OUT="$PWD/nsys_rank0"
mpirun -np 32 "$CHK/profile_rank0_wrapper.sh" ./nnp-train-gpu

echo ""
echo "=== Stopping MPS control daemon ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2

echo ""
echo "=== timing.out (this run) ==="
cat timing.out | grep -v "^#"

echo ""
ls -la nsys_rank0.nsys-rep
echo "Done"
