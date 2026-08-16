#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --job-name=lammps-hdnnp-profile-dbg
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Retry of profile_cpu_lammps.slurm (job 52505205): that run's perf
# report couldn't resolve almost any n2p2 function name (hundreds of raw
# hex addresses) because libnnp/libnnpif were built with -O3 and no -g
# (LAMMPS's own files have -g by default, n2p2's don't). Rebuild n2p2
# with -g added (keeping -O3/-march=native unchanged so the code being
# profiled still behaves like the real optimized build), relink the
# existing lammps_hdnnp_cpu_booster tree, then re-profile.

set -e
module purge
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
LAMMPS_SRC=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_cpu_booster/lammps-22Jul2025
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"

echo "=== [1/3] Rebuild n2p2_devel CPU with -g added ==="
cd "$N2P2_DEVEL/src"
make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true
make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" INTERFACES=LAMMPS \
     PROJECT_CFLAGS="-O3 -march=native -std=c++11 -g" \
     -j"$SLURM_NTASKS_PER_NODE" libnnpif
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h

echo ""
echo "=== [2/3] Relink lmp_mpi against the debug-symbol libnnp/libnnpif ==="
cd "$LAMMPS_SRC/src"
touch main.cpp  # force relink even if object files look unchanged
make mpi -j"$SLURM_NTASKS_PER_NODE"

echo ""
echo "=== [3/3] Re-profile: single-rank, 500-step H2O_2G run ==="
PROFDIR="$DYN/profile_cpu_1rank_dbg"
rm -rf "$PROFDIR" && mkdir -p "$PROFDIR"
cp "$DYN/init_conf_minimized.data" "$PROFDIR/"
ln -s "$DYN/hdnnp-data" "$PROFDIR/hdnnp-data"
sed 's/^variable numSteps.*equal.*/variable numSteps equal 500 # profiling run/' "$DYN/md.lmp" > "$PROFDIR/md_profile.lmp"
cd "$PROFDIR"

perf record -g --call-graph=dwarf,65528 -o perf.data -- \
    "$LAMMPS_SRC/src/lmp_mpi" -in md_profile.lmp -log log.profile

echo ""
echo "=== Flat self-time report (top 80 by self overhead) ==="
perf report -i perf.data --stdio --no-children --sort=overhead,symbol 2>/dev/null | head -100

echo ""
echo "=== Filtered: symmetry-function / NN / force-assembly related symbols ==="
perf report -i perf.data --stdio --no-children --sort=overhead,symbol 2>/dev/null \
    | grep -iE "symgrp|symfnc|calculatesymmetryfunction|calculateatomicneuralnetwork|propagatelayer|neuralnetwork|getforces|calculatepairforce|calculateforces"

echo ""
echo "Done"
