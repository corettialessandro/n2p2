#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --mem=64GB
#SBATCH --job-name=h2o2g-gpu-rank-scan
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Find GPU's best rank count for H2O_2G with the warp-per-atom kernel
# (the RPBE-D3 scan, job 52576146, already covers this system and found
# 32 ranks/4 GPUs winning throughout -- this fills the matching gap for
# H2O_2G, the smaller/more extreme system, where only a single 32-rank
# data point exists post-fix so far). Always 4 GPUs (established
# 8-ranks/GPU-style split via MPS + rank_gpu_wrapper.sh), scanning total
# rank count 4/8/16/32. CPU side intentionally not rescanned -- 32 ranks
# (full node) is already established as CPU's best single-node Booster
# config from prior runs this session. Short run (300 steps) for fast,
# stable per-step Pair-time comparison; reuses the already-current,
# already-validated GPU lmp_mpi (no rebuild).

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
LAMMPS_GPU=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"
WRAPPER="$N2P2_DEVEL/gpu/e2e_train_check/rank_gpu_wrapper.sh"

echo "=== Starting MPS control daemon serving all 4 GPUs ==="
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-pipe-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
sleep 2

sed 's/variable numSteps.*equal.*2000.*/variable numSteps equal 300 # rank scan/' \
    "$DYN/md_smoketest.lmp" > "$DYN/md_scan_h2o2g.lmp"

for R in 4 8 16 32; do
    RPG=$(( R / 4 ))
    echo ""
    echo "=== GPU, $R ranks / 4 GPUs (RANKS_PER_GPU=$RPG), 300 steps ==="
    RUNDIR="$DYN/scan_gpu_h2o2g_r${R}"
    rm -rf "$RUNDIR" && mkdir -p "$RUNDIR"
    cp "$DYN/init_conf_minimized.data" "$DYN/md_scan_h2o2g.lmp" "$RUNDIR/"
    ln -s "$DYN/hdnnp-data" "$RUNDIR/hdnnp-data"
    cd "$RUNDIR"
    export RANKS_PER_GPU=$RPG
    mpirun -np "$R" "$WRAPPER" "$LAMMPS_GPU/src/lmp_mpi" -in md_scan_h2o2g.lmp -log log.scan_gpu_r${R}
    grep -A1 "^Pair " log.scan_gpu_r${R} || true
    grep "^Loop time" log.scan_gpu_r${R} || true
done

echo ""
echo "=== Stopping MPS control daemon ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2

echo ""
echo "=== Summary ==="
for R in 4 8 16 32; do
    echo "--- $R ranks ---"
    grep '^Pair ' "$DYN/scan_gpu_h2o2g_r${R}/log.scan_gpu_r${R}" 2>/dev/null
done
echo "Done"
