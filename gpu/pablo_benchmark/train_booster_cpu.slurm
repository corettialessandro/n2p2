#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time 1-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=200GB
#SBATCH --job-name=train-booster-cpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Parameterized by SYS (water/magnetite/feldspar, exported via
# --export=SYS=...,EPOCHS=...) and EPOCHS (10 or 100). Booster, 32 CPU
# ranks, no GPU -- one leg of the CPU-vs-GPU-vs-DCGP comparison requested
# for Pablo's three new systems. Same isolated-worktree-per-config
# pattern as gpu/e2e_train_check's H2O_2G scripts (avoids the
# concurrent-build-race hazard documented there).

set -e
: "${SYS:?Set SYS=water|magnetite|feldspar via --export}"
: "${EPOCHS:?Set EPOCHS=10|100 via --export}"

module purge
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_${SYS}
N2P2_ORIG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
DATA="$N2P2_ORIG/gpu/pablo_benchmark/data/$SYS"
# Results go straight to the work filesystem, not home -- home hit its
# 50GB cap during the 10-epoch preliminary runs (feldspar/magnetite's
# per-epoch trainforces.*.out files alone ran to hundreds of MB each).
RESULTS_BASE=/leonardo_work/L-AUT_Giane_26/acoretti/NEURALCPM/gpu_porting/pablo_benchmark
OUT="$RESULTS_BASE/${SYS}_booster-cpu_${EPOCHS}ep"

mkdir -p "$RESULTS_BASE"
rm -rf "$OUT" && mkdir -p "$OUT" && cd "$OUT"
cp "$DATA/input.data" "$DATA/scaling.data" .
# write_trainpoints/write_trainforces=0: only need final-epoch weights
# for the energy/force error check, not a per-epoch comparison dump
# (feldspar/water had these left at 1 in Pablo's input.nn -- at 100
# epochs that's ~42GB for feldspar alone; matches H2O_2G's/magnetite's
# already-sensible convention of 0).
sed -e "s/^epochs.*/epochs                          ${EPOCHS}            # Number of training epochs./" \
    -e "s/^write_trainpoints.*/write_trainpoints               0/" \
    -e "s/^write_trainforces.*/write_trainforces               0/" \
    "$DATA/input.nn" > input.nn

echo "=== Clean build: CPU-only nnp-train (isolated worktree: $N2P2_DEVEL) ==="
(cd "$N2P2_DEVEL/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" nnp-train > build-cpu.log 2>&1)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
cp "$N2P2_DEVEL/bin/nnp-train" ./nnp-train-cpu

echo ""
echo "=== Running CPU build (Booster, 32 ranks), $SYS, $EPOCHS epochs ==="
mpirun -np 32 ./nnp-train-cpu

echo ""
echo "=== timing.out (all epochs) ==="
cat timing.out | grep -v "^#"

echo "Done"
