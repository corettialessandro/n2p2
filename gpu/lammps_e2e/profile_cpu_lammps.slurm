#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --job-name=lammps-hdnnp-profile
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Phase 5 step 1 (soft-percolating-jellyfish.md): real perf profile of a
# CPU LAMMPS H2O_2G run, single rank (matching GPU_PORTING_PLAN.md's
# Phase 0 methodology -- profiling one rank is representative since all
# ranks run the same per-atom code on different atoms), to get the
# actual, non-amortized split between symmetry-function evaluation
# (calculateSymmetryFunctionGroups), NN evaluation
# (calculateAtomicNeuralNetworks), and force assembly (getForces) inside
# LAMMPS's "Pair" bucket -- confirming or correcting the hypothesis that
# SF evaluation, not NN evaluation, dominates per-timestep MD cost
# (unlike nnp-train, where memorize_symfunc_results amortizes it away).

set -e
module purge
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
LAMMPS_CPU_BIN=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_cpu_booster/lammps-22Jul2025/src/lmp_mpi
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"
PROFDIR="$DYN/profile_cpu_1rank"

rm -rf "$PROFDIR" && mkdir -p "$PROFDIR"
cp "$DYN/init_conf_minimized.data" "$PROFDIR/"
ln -s "$DYN/hdnnp-data" "$PROFDIR/hdnnp-data"
sed 's/^variable numSteps.*equal.*/variable numSteps equal 500 # profiling run/' "$DYN/md.lmp" > "$PROFDIR/md_profile.lmp"
cd "$PROFDIR"

echo "=== Recording perf profile: single-rank, 500-step H2O_2G run ==="
perf record -g --call-graph=dwarf,65528 -o perf.data -- \
    "$LAMMPS_CPU_BIN" -in md_profile.lmp -log log.profile

echo ""
echo "=== Flat self-time report (top 60 by self overhead) ==="
perf report -i perf.data --stdio --no-children --sort=overhead,symbol 2>/dev/null | head -80

echo ""
echo "=== Filtered: symmetry-function / NN / force-assembly related symbols ==="
perf report -i perf.data --stdio --no-children --sort=overhead,symbol 2>/dev/null \
    | grep -iE "symgrp|symfnc|calculatesymmetryfunction|calculateatomicneuralnetwork|propagatelayer|neuralnetwork|getforces|calculatepairforce|calculateforces"

echo ""
echo "Done"
