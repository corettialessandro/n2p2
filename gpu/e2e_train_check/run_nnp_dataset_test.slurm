#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=32GB
#SBATCH --job-name=nnp-dataset-test
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

# Validates nnp-dataset's optional <data_file> CLI argument (default
# behaviour must stay identical, and the new argument must actually make
# it read the given file). Run alone -- do NOT run this concurrently with
# another job's `make clean`/`make` in this shared src/lib/bin tree; a
# real build race between two such jobs was the root cause of a very
# confusing, non-reproducible nnp-dataset segfault while this feature was
# first being validated (nothing wrong with the source, the binary was
# just corrupted by two overlapping builds) -- see the isolated git
# worktrees run_nnp_train_100ep_{cpu,gpu}.slurm use for a real example of
# how to avoid this when concurrency is actually needed.

rm -f "$N2P2_DEVEL/src/application/nnp-dataset" "$N2P2_DEVEL/src/application/nnp-dataset.o" \
      "$N2P2_DEVEL/bin/nnp-dataset"
(cd "$N2P2_DEVEL/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" nnp-dataset)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h

rm -rf /tmp/nnp_dataset_test_$SLURM_JOB_ID
mkdir -p /tmp/nnp_dataset_test_$SLURM_JOB_ID
cd /tmp/nnp_dataset_test_$SLURM_JOB_ID
cp "$TEMP/input.nn" "$TEMP/scaling.data" "$TEMP/input.data" .
cp "$TEMP/weights.001.000002.out" weights.001.data
cp "$TEMP/weights.008.000002.out" weights.008.data

echo ""
echo "=== Default 2-arg call (backward compat: reads input.data) ==="
mpirun -np 2 "$N2P2_DEVEL/bin/nnp-dataset" 0
echo "energy.comp structure count (2-arg): $(grep -vc '^#' energy.comp)"
mv energy.comp energy.comp.2arg
mv forces.comp forces.comp.2arg
rm -f input.data

echo ""
echo "=== New 3-arg call, distinctly-named file: nnp-dataset 0 mydata.data ==="
cp "$TEMP/input.data" mydata.data
mpirun -np 2 "$N2P2_DEVEL/bin/nnp-dataset" 0 mydata.data
echo "energy.comp structure count (3-arg): $(grep -vc '^#' energy.comp)"

echo ""
echo "=== Diff: 2-arg output vs 3-arg output (should be identical) ==="
diff energy.comp.2arg energy.comp && echo "energy.comp: IDENTICAL"
diff forces.comp.2arg forces.comp && echo "forces.comp: IDENTICAL"

echo "Done"
