#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --job-name=lammps-hdnnp-profile-sw
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Phase 5 step 1 (soft-percolating-jellyfish.md), attempt 2: perf's
# symbol resolution didn't work on this cluster (profile_cpu_lammps*.
# slurm), so this uses manual Stopwatch instrumentation instead
# (InterfaceLammps.{h,cpp}, temporary -- see the comments there),
# mirroring Training.cpp's _err/_com/_upd pattern. Single rank, matching
# the same profiling methodology as the perf attempts.

set -e
module purge
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
LAMMPS_SRC=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_cpu_booster/lammps-22Jul2025
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"

echo "=== [1/3] Rebuild n2p2_devel CPU with Stopwatch profiling instrumentation ==="
cd "$N2P2_DEVEL/src"
make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true
make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" INTERFACES=LAMMPS \
     -j"$SLURM_NTASKS_PER_NODE" libnnpif
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h

echo ""
echo "=== [2/3] Relink lmp_mpi ==="
cd "$LAMMPS_SRC/src"
touch main.cpp
make mpi -j"$SLURM_NTASKS_PER_NODE"

echo ""
echo "=== [3/3] Run: single-rank, 2000-step H2O_2G run ==="
PROFDIR="$DYN/profile_sw_1rank"
rm -rf "$PROFDIR" && mkdir -p "$PROFDIR"
cp "$DYN/init_conf_minimized.data" "$DYN/md_smoketest.lmp" "$PROFDIR/"
ln -s "$DYN/hdnnp-data" "$PROFDIR/hdnnp-data"
cd "$PROFDIR"

"$LAMMPS_SRC/src/lmp_mpi" -in md_smoketest.lmp -log log.profile_sw

echo ""
echo "=== SF/NN/Forces profile lines (from stderr) ==="
echo "(see .err file for the [SF/NN/Forces profile] lines printed every 500 calls)"

echo ""
echo "Done"
