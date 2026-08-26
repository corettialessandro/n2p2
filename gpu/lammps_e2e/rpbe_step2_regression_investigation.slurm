#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128GB
#SBATCH --job-name=rpbe-step2-regression
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Investigates step 2's (persistent scratch buffers) apparent small
# regression vs step 1 (single-pass CSR) alone on H2O_RPBE-D3, found
# across two independent normal/dbg-QOS job submissions (149.90s/150.79s
# vs step 1's 147.03s). Rather than compare across separate job
# submissions (this cluster has a documented history of run-to-run
# variance from causes outside the code, per gpu/README.md), builds and
# runs BOTH variants back-to-back on the SAME allocated node in ONE job,
# with temporary phase-level Stopwatch profiling on both (CSR-build /
# GPU-call / scatter, plus an extra MapLookup phase on the step1+step2
# variant isolating the one genuinely new operation it introduces) to
# see exactly where any real difference lives, not just whether one.
#
# Mode.cpp is swapped in place sequentially (build variant A, run, swap
# to variant B, build, run) rather than building in an isolated tree,
# since LAMMPS's pre-generated lib/hdnnp/Makefile.lammps links against
# this project's fixed include/lib paths. A trap guarantees Mode.cpp is
# restored to the real (uninstrumented, step1+step2) working-tree state
# on exit regardless of success/failure -- verified against a snapshot
# taken immediately before this job was submitted.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
LAMMPS_GPU=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025
EX="$N2P2_DEVEL/examples/interface-LAMMPS/H2O_RPBE-D3"
WRAPPER="$N2P2_DEVEL/gpu/e2e_train_check/rank_gpu_wrapper.sh"
VARIANTS="$N2P2_DEVEL/gpu/lammps_e2e/mode_variants"
MODE_LIVE="$N2P2_DEVEL/src/libnnp/Mode.cpp"

restore_mode() {
    cp "$VARIANTS/Mode_step1and2_CLEAN_restore.cpp" "$MODE_LIVE"
    echo "=== [restore trap] Mode.cpp restored to clean step1+step2 working-tree state ==="
}
trap restore_mode EXIT

build_variant() {
    local variant_file="$1"
    cp "$variant_file" "$MODE_LIVE"
    (cd "$N2P2_DEVEL/src/libnnpgpu" && make clean > /dev/null 2>&1 || true)
    (cd "$N2P2_DEVEL/src/libnnpgpu" && make COMP=gnu MODE=static GPU=1)
    (cd "$N2P2_DEVEL/src/libnnp" && make clean > /dev/null 2>&1 || true)
    (cd "$N2P2_DEVEL/src/libnnpif" && make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true)
    (cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 GPU_SF=1 PROJECT_EIGEN="$EIGEN" \
        INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnp libnnpif)
    git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
    (cd "$LAMMPS_GPU/src" && touch main.cpp && make mpi -j"$SLURM_NTASKS_PER_NODE")
}

run_1rank_profile() {
    local tag="$1" binary="$2"
    local PROFDIR="$EX/regression_1rank_${tag}"
    rm -rf "$PROFDIR" && mkdir -p "$PROFDIR"
    sed -e 's/variable numSteps.*equal.*5.*/variable numSteps equal 300 # step1-vs-step1and2 regression probe/' \
        -e 's/^dump 1 all atom 1 traj.dump/dump 1 all atom 100 traj.dump/' -e 's/resetew no maxew 100 /resetew yes maxew 1000000 /' \
        "$EX/md.lmp" > "$PROFDIR/md_probe.lmp"
    cp "$EX/h2o_8640_liquid_NpT_RPBE-D3.data" "$PROFDIR/"
    ln -s "$EX/hdnnp-data" "$PROFDIR/hdnnp-data"
    cd "$PROFDIR"
    mpirun -np 1 "$binary" -in md_probe.lmp -log "log.${tag}"
    echo "--- ${tag}: 1-rank, 300 steps ---"
    grep -A1 "^Pair " "log.${tag}"
    grep "^Loop time" "log.${tag}"
}

run_32rank_perf() {
    local tag="$1" binary="$2"
    local RUNDIR="$EX/regression_32rank_${tag}"
    rm -rf "$RUNDIR" && mkdir -p "$RUNDIR"
    sed -e 's/variable numSteps.*equal.*5.*/variable numSteps equal 2000 # step1-vs-step1and2 regression check/' \
        -e 's/^dump 1 all atom 1 traj.dump/dump 1 all atom 500 traj.dump/' -e 's/resetew no maxew 100 /resetew yes maxew 1000000 /' \
        "$EX/md.lmp" > "$RUNDIR/md_2k.lmp"
    cp "$EX/h2o_8640_liquid_NpT_RPBE-D3.data" "$RUNDIR/"
    ln -s "$EX/hdnnp-data" "$RUNDIR/hdnnp-data"
    cd "$RUNDIR"
    mpirun -np 32 "$WRAPPER" "$binary" -in md_2k.lmp -log "log.${tag}"
    echo "--- ${tag}: 32 ranks/4 GPUs, 2000 steps ---"
    grep -A1 "^Pair " "log.${tag}"
    grep "^Loop time" "log.${tag}"
}

echo "=== [1/4] Build + run variant A: step1-only ==="
build_variant "$VARIANTS/Mode_step1only.cpp"
cp "$LAMMPS_GPU/src/lmp_mpi" "$LAMMPS_GPU/src/lmp_mpi_step1only"
run_1rank_profile "step1only" "$LAMMPS_GPU/src/lmp_mpi_step1only"

echo ""
echo "=== [2/4] Build + run variant B: step1and2 (instrumented) ==="
build_variant "$VARIANTS/Mode_step1and2.cpp"
cp "$LAMMPS_GPU/src/lmp_mpi" "$LAMMPS_GPU/src/lmp_mpi_step1and2"
run_1rank_profile "step1and2" "$LAMMPS_GPU/src/lmp_mpi_step1and2"

echo ""
echo "=== [3/4] Starting a single MPS control daemon serving all 4 GPUs ==="
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-pipe-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
sleep 2
export RANKS_PER_GPU=8

echo ""
echo "=== [4/4] 32-rank/2000-step perf, both binaries back-to-back on this node ==="
run_32rank_perf "step1only" "$LAMMPS_GPU/src/lmp_mpi_step1only"
run_32rank_perf "step1and2" "$LAMMPS_GPU/src/lmp_mpi_step1and2"

echo ""
echo "=== Stopping MPS control daemon ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2

echo ""
echo "=== Phase-profile lines (both variants) ==="
grep "SF-GPU-PROFILE" "$SLURM_SUBMIT_DIR/rpbe-step2-regression_${SLURM_JOB_ID}.err"

echo "=== Done ==="
