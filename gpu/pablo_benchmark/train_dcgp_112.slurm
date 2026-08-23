#!/bin/bash
#SBATCH -A L-AUT_DCGP2026
#SBATCH -p dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --time 1-00:00:00
#SBATCH --job-name=train-dcgp-112
#SBATCH --nodes=1
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Parameterized by SYS (water/magnetite/feldspar) and EPOCHS (10 or 100),
# exported via --export at submission time. DCGP, 112 CPU ranks (full
# node) -- the strongest CPU configuration this project has tested (see
# gpu/README.md's H2O_2G DCGP-vs-Booster follow-up).

set -e
: "${SYS:?Set SYS=water|magnetite|feldspar via --export}"
: "${EPOCHS:?Set EPOCHS=10|100 via --export}"

module purge
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_dcgp_${SYS}
N2P2_ORIG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
DATA="$N2P2_ORIG/gpu/pablo_benchmark/data/$SYS"
RESULTS_BASE=/leonardo_work/L-AUT_Giane_26/acoretti/NEURALCPM/gpu_porting/pablo_benchmark
OUT="$RESULTS_BASE/${SYS}_dcgp-112_${EPOCHS}ep"

mkdir -p "$RESULTS_BASE"
rm -rf "$OUT" && mkdir -p "$OUT" && cd "$OUT"
cp "$DATA/input.data" "$DATA/scaling.data" .
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
echo "=== Running CPU build (DCGP, 112 ranks, full node), $SYS, $EPOCHS epochs ==="
mpirun -np 112 ./nnp-train-cpu

echo ""
echo "=== timing.out (all epochs) ==="
cat timing.out | grep -v "^#"

echo "Done"
