#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time 18:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128GB
#SBATCH --job-name=rpbe-prod-200k-sf
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Real production-length (200000-step) validation of the GPU_SF=1 win on
# H2O_RPBE-D3, GPU_SF=1 half. See rpbe_production_200k_nnonly.slurm's
# comment for why this length matters (H2O_2G's own 2000-step GPU_SF win
# reversed at 200000 steps -- the 1.56x number for H2O_RPBE-D3 is only
# trustworthy once checked at comparable length).
#
# This job pins its OWN exact copy of the Mode.cpp state to test
# (gpu/lammps_e2e/mode_variants/Mode_step1_arena_PINNED.cpp -- updated
# after submission to the best validated state found: step 1's
# single-pass CSR, the shared-buffer arena, and the fused radial+angular
# GPU dispatch, 9 kernel launches/timestep down to 4, a validated +10.2%
# win at 2000 steps) and swaps it into src/libnnp/Mode.cpp for the
# duration of its own build only, snapshotting + restoring whatever the
# live working tree looks like at the moment THIS JOB runs (not at
# submission time) via a trap. This job may sit queued for a long time on
# a congested cluster while further, unrelated Mode.cpp work continues
# interactively in the meantime -- this makes the two fully independent:
# this job always tests exactly the pinned code, and always leaves the
# live tree exactly as it found it, regardless of what changed on it
# between submission and actually running.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
LAMMPS_SRC=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025
EX="$N2P2_DEVEL/examples/interface-LAMMPS/H2O_RPBE-D3"
WRAPPER="$N2P2_DEVEL/gpu/e2e_train_check/rank_gpu_wrapper.sh"
PINNED="$N2P2_DEVEL/gpu/lammps_e2e/mode_variants/Mode_step1_arena_PINNED.cpp"
MODE_LIVE="$N2P2_DEVEL/src/libnnp/Mode.cpp"
LIVE_SNAPSHOT="$N2P2_DEVEL/gpu/lammps_e2e/mode_variants/Mode_LIVE_snapshot_${SLURM_JOB_ID}.cpp"

restore_mode() {
    if [ -f "$LIVE_SNAPSHOT" ]; then
        cp "$LIVE_SNAPSHOT" "$MODE_LIVE"
        rm -f "$LIVE_SNAPSHOT"
        echo "=== [restore trap] Mode.cpp restored to whatever the live working tree held when this job started running ==="
    fi
}
trap restore_mode EXIT

echo "=== [0/2] Snapshot live Mode.cpp, swap in the pinned step1+arena state, build ==="
cp "$MODE_LIVE" "$LIVE_SNAPSHOT"
cp "$PINNED" "$MODE_LIVE"
(cd "$N2P2_DEVEL/src/libnnpgpu" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpgpu" && make COMP=gnu MODE=static GPU=1)
(cd "$N2P2_DEVEL/src/libnnp" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpif" && make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 GPU_SF=1 PROJECT_EIGEN="$EIGEN" \
    INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnp libnnpif)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
(cd "$LAMMPS_SRC/src" && touch main.cpp && make mpi -j"$SLURM_NTASKS_PER_NODE")

echo ""
echo "=== Starting a single MPS control daemon serving all 4 GPUs ==="
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-pipe-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
sleep 2
export RANKS_PER_GPU=8

echo ""
echo "=== [1/2] H2O_RPBE-D3 (8640 atoms), GPU SF (step1+arena), 32 ranks/4 GPUs, 200000 steps ==="
RUNDIR="$EX/production_200k_sf"
rm -rf "$RUNDIR" && mkdir -p "$RUNDIR"
sed -e 's/variable numSteps.*equal.*5.*/variable numSteps equal 200000 # full production-length validation/' \
    -e 's/^thermo 1$/thermo 1000/' \
    -e 's/^dump 1 all atom 1 traj.dump/dump 1 all atom 10000 traj.dump/' \
    -e 's/resetew no maxew 100 /resetew yes maxew 1000000 /' \
    "$EX/md.lmp" > "$RUNDIR/md_200k.lmp"
cp "$EX/h2o_8640_liquid_NpT_RPBE-D3.data" "$RUNDIR/"
ln -s "$EX/hdnnp-data" "$RUNDIR/hdnnp-data"
cd "$RUNDIR"
mpirun -np 32 "$WRAPPER" "$LAMMPS_SRC/src/lmp_mpi" -in md_200k.lmp -log log.production_200k_sf
echo "--- H2O_RPBE-D3 GPU SF (step1+arena), 200000-step result ---"
grep -A1 "^Pair " log.production_200k_sf
grep "^Loop time" log.production_200k_sf

echo ""
echo "=== Stopping MPS control daemon ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2

echo "=== [2/2] Done ==="
