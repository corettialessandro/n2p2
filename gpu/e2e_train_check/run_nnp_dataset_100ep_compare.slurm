#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32GB
#SBATCH --job-name=nnp-dataset-100ep-compare
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
CHK="$N2P2_DEVEL/gpu/e2e_train_check"

echo "=== Clean build: nnp-dataset (CPU-only, plain evaluation, no GPU needed) ==="
rm -f "$N2P2_DEVEL/src/application/nnp-dataset" "$N2P2_DEVEL/src/application/nnp-dataset.o" \
      "$N2P2_DEVEL/bin/nnp-dataset"
(cd "$N2P2_DEVEL/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" nnp-dataset)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h

for run in run-100ep-cpu run-100ep-gpu; do
    echo ""
    echo "=== nnp-dataset on $run's held-out test.data (final epoch-100 weights) ==="
    cd "$CHK/results/$run"
    cp weights.001.000100.out weights.001.data
    cp weights.008.000100.out weights.008.data
    mpirun -np 4 "$N2P2_DEVEL/bin/nnp-dataset" 0 test.data
    echo "$run: energy.comp structures = $(grep -vc '^#' energy.comp), forces.comp lines = $(grep -vc '^#' forces.comp)"
done

echo "Done"
