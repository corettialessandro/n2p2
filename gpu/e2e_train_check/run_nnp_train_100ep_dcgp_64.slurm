#!/bin/bash
#SBATCH -A L-AUT_DCGP2026
#SBATCH -p dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --time 1-00:00:00        # format: D-HH:MM:SS
#SBATCH --job-name=train-dcgp-64
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

# Real-CPU comparison, requested to rule out the possibility that
# Booster's Ice Lake CPUs (used for every CPU baseline in gpu/README.md
# so far) understate real CPU performance relative to DCGP's CPUs --
# DCGP is this cluster's partition built for CPU workloads (112 cores/
# node, vs. Booster's 32). Builds in an isolated git worktree (not the
# shared main checkout) -- this runs in parallel with the 64- and
# 112-core sibling jobs, and concurrent `make clean && make ...` in a
# shared src/lib/bin tree is a real, previously-hit build-race hazard
# (see gpu/README.md's nnp-dataset follow-up).
N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_dcgp64
N2P2_ORIG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
TEMP="$N2P2_ORIG/temp/H2O_2G"
CHK="$N2P2_ORIG/gpu/e2e_train_check"
cd "$CHK"

mkdir -p results
rm -rf results/run-100ep-dcgp-64 && mkdir results/run-100ep-dcgp-64 && cd results/run-100ep-dcgp-64
cp "$TEMP/input.data" "$TEMP/scaling.data" .
sed 's/^epochs.*/epochs                          100            # Number of training epochs./' \
    "$TEMP/input.nn" > input.nn

echo "=== Clean build: CPU-only nnp-train (isolated worktree: $N2P2_DEVEL) ==="
(cd "$N2P2_DEVEL/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" nnp-train > build-cpu.log 2>&1)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
cp "$N2P2_DEVEL/bin/nnp-train" ./nnp-train-cpu

echo ""
echo "=== Running CPU build (DCGP, 64 ranks), 100 epochs ==="
mpirun -np 64 ./nnp-train-cpu

echo ""
echo "=== timing.out (all epochs) ==="
cat timing.out | grep -v "^#"

echo "Done"
