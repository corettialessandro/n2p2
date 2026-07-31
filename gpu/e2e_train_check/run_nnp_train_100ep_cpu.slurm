#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time 1-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=200GB
#SBATCH --job-name=nnp-train-100ep-cpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

# Builds in an ISOLATED git worktree, not the shared, main n2p2_devel
# checkout -- running this alongside the companion GPU 100-epoch job (or
# any other job doing `make clean && make ...` in the shared src/lib/bin
# tree) risks a build race: one job's `make clean` deleting .o files
# while another job's `make` is mid-compile/link, silently producing a
# corrupted binary. This was directly observed as an extremely confusing,
# non-reproducible "heisenbug" (nnp-dataset segfaulting inconsistently,
# fixed by unrelated timing changes) while validating nnp-dataset earlier
# -- see that investigation for the full trail. Each concurrent
# build now gets its own git worktree (see
# `git worktree add --detach n2p2_devel_100ep_cpu gpu-portability`), so
# there is no shared src/lib/bin state to race on. temp/H2O_2G (read-only
# input data) and this script's own output directory are NOT duplicated
# -- both still point at the original checkout.
N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_100ep_cpu
N2P2_ORIG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
TEMP="$N2P2_ORIG/temp/H2O_2G"
CHK="$N2P2_ORIG/gpu/e2e_train_check"
cd "$CHK"

# Real, 100-epoch, CPU-only H2O_2G training run -- a genuine multi-epoch
# timing/learning-curve baseline to compare against the GPU-accelerated
# run (run_nnp_train_100ep_gpu.slurm). NOT a throwaway check: this
# directory is deliberately NOT cleaned up afterward, since
# weights.*.out/test.data/train.data are needed later for nnp-dataset
# energy/force comparisons on the held-out test set.

rm -rf run-100ep-cpu && mkdir run-100ep-cpu && cd run-100ep-cpu
cp "$TEMP/input.data" "$TEMP/scaling.data" .
sed 's/^epochs.*/epochs                          100            # Number of training epochs./' \
    "$TEMP/input.nn" > input.nn

echo "=== Clean build: CPU-only nnp-train (isolated worktree: $N2P2_DEVEL) ==="
(cd "$N2P2_DEVEL/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" nnp-train > build-cpu.log 2>&1)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
cp "$N2P2_DEVEL/bin/nnp-train" ./nnp-train-cpu

echo ""
echo "=== Running CPU build, 100 epochs, 32 ranks ==="
mpirun -np 32 ./nnp-train-cpu

echo ""
echo "=== timing.out (all epochs) ==="
cat timing.out | grep -v "^#"

echo "Done"
