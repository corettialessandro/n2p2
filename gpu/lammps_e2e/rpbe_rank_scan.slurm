#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128GB
#SBATCH --job-name=lammps-hdnnp-rpbe-rank-scan
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Phase 5/6 multi-rank scaling check at H2O_RPBE-D3 scale (8640 atoms).
# Single-rank comparison (jobs 52552607/52553563) found GPU SF dispatch
# ~2.4x faster than CPU. But real production runs use more than 1 rank,
# and per-rank atom count shrinks as rank count grows -- at H2O_2G's 630
# atoms (which failed badly, ~10x slower on GPU) the failure mode was
# GPU underutilization from too few atoms/call. Need to check whether
# the RPBE-D3 win survives at realistic rank counts, or erodes/reverses
# as atoms/rank shrinks back toward H2O_2G's regime. Scans 4/8/16/32
# ranks for both CPU and GPU, all on Booster (isolates GPU-vs-CPU from
# DCGP-vs-Booster differences, per Phase 3b), same 40-step config.
#
# GPU side rebuilt clean first (the correctness check, job 52555908,
# reused a binary that still had the since-reverted profiling
# instrumentation compiled in -- harmless for correctness, but these
# timing numbers should reflect the actual code being committed).
# CPU side reuses the already-current binary from job 52553563 --
# Mode.cpp's revert was entirely inside #ifdef N2P2_GPU, so the CPU
# build was never affected.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
LAMMPS_GPU=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025
LAMMPS_CPU=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_cpu_booster/lammps-22Jul2025
EX="$N2P2_DEVEL/examples/interface-LAMMPS/H2O_RPBE-D3"

echo "=== [0/2] Clean rebuild of GPU tree (reverted Mode.cpp, no profiling) ==="
(cd "$N2P2_DEVEL/src/libnnpgpu" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpgpu" && make COMP=gnu MODE=static GPU=1)
(cd "$N2P2_DEVEL/src/libnnp" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpif" && make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" \
    INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnp libnnpif)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
(cd "$LAMMPS_GPU/src" && touch main.cpp && make mpi -j"$SLURM_NTASKS_PER_NODE")

echo ""
echo "=== Starting a single MPS control daemon serving all 4 GPUs ==="
echo "(matches Phase 3's established multi-rank-per-GPU pattern --"
echo " gpu/e2e_train_check/rank_gpu_wrapper.sh assigns contiguous"
echo " blocks of RANKS_PER_GPU local ranks to each GPU via its own"
echo " CUDA_VISIBLE_DEVICES, all served by one MPS daemon)"
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-pipe-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
sleep 2
WRAPPER="$N2P2_DEVEL/gpu/e2e_train_check/rank_gpu_wrapper.sh"

echo ""
echo "=== [1/2] GPU scan: 4/8/16/32 ranks spread across up to 4 GPUs, 40 steps each ==="
for R in 4 8 16 32; do
    RPG=$(( R < 4 ? 1 : R / 4 ))
    echo "--- GPU, $R ranks, RANKS_PER_GPU=$RPG ---"
    PROFDIR="$EX/scan_gpu_r${R}"
    rm -rf "$PROFDIR" && mkdir -p "$PROFDIR"
    sed 's/variable numSteps.*equal.*5.*/variable numSteps equal 40 # rank scan/' \
        "$EX/md.lmp" > "$PROFDIR/md_scan.lmp"
    cp "$EX/h2o_8640_liquid_NpT_RPBE-D3.data" "$PROFDIR/"
    ln -s "$EX/hdnnp-data" "$PROFDIR/hdnnp-data"
    cd "$PROFDIR"
    export RANKS_PER_GPU=$RPG
    mpirun -np "$R" "$WRAPPER" "$LAMMPS_GPU/src/lmp_mpi" -in md_scan.lmp -log log.scan_gpu_r${R}
    grep -A1 "^Pair " log.scan_gpu_r${R} || true
    grep "^Loop time" log.scan_gpu_r${R} || true
done

echo ""
echo "=== Stopping MPS control daemon ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2

echo ""
echo "=== [2/2] CPU scan: 4/8/16/32 ranks, 40 steps each ==="
for R in 4 8 16 32; do
    echo "--- CPU, $R ranks ---"
    PROFDIR="$EX/scan_cpu_r${R}"
    rm -rf "$PROFDIR" && mkdir -p "$PROFDIR"
    sed 's/variable numSteps.*equal.*5.*/variable numSteps equal 40 # rank scan/' \
        "$EX/md.lmp" > "$PROFDIR/md_scan.lmp"
    cp "$EX/h2o_8640_liquid_NpT_RPBE-D3.data" "$PROFDIR/"
    ln -s "$EX/hdnnp-data" "$PROFDIR/hdnnp-data"
    cd "$PROFDIR"
    mpirun -np "$R" "$LAMMPS_CPU/src/lmp_mpi" -in md_scan.lmp -log log.scan_cpu_r${R}
    grep -A1 "^Pair " log.scan_cpu_r${R} || true
    grep "^Loop time" log.scan_cpu_r${R} || true
done

echo ""
echo "=== Summary ==="
for R in 4 8 16 32; do
    echo "--- $R ranks ---"
    echo "GPU: $(grep '^Pair ' $EX/scan_gpu_r${R}/log.scan_gpu_r${R} 2>/dev/null | head -1)"
    echo "CPU: $(grep '^Pair ' $EX/scan_cpu_r${R}/log.scan_cpu_r${R} 2>/dev/null | head -1)"
done
