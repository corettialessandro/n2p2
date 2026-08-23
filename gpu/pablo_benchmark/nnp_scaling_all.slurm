#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=200GB
#SBATCH --job-name=nnp-scaling-pablo
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Generates scaling.data for Pablo's three new systems (water, magnetite,
# feldspar) -- required by nnp-train, not shipped with the input files
# Pablo provided (only *-input.data/*-input.nn). Isolated worktree, same
# race-avoidance reasoning as gpu/e2e_train_check's training scripts.

set -e
module purge
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_scaling
N2P2_ORIG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
DATA="$N2P2_ORIG/gpu/pablo_benchmark/data"

echo "=== Clean build: nnp-scaling (CPU-only, isolated worktree: $N2P2_DEVEL) ==="
(cd "$N2P2_DEVEL/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" nnp-scaling)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h

for sys in water magnetite feldspar; do
    echo ""
    echo "=== nnp-scaling: $sys ==="
    cd "$DATA/$sys"
    rm -f scaling.data function.data neighbors.histo sf.*.histo nnp-scaling.log.*
    mpirun -np "$SLURM_NTASKS" "$N2P2_DEVEL/bin/nnp-scaling" 100
    ls -la scaling.data
done

echo "Done"
