#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time 01:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=64GB
#SBATCH --job-name=report-cpu-2k
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# User-requested report: CPU-only timings for 2000-step MD runs on both
# H2O_2G (630 atoms) and H2O_RPBE-D3 (8640 atoms), Booster partition,
# 32 ranks (max cores on a Booster node), no GPU. Fresh rebuild for full
# current-state accuracy.

set -e
module purge
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
LAMMPS_SRC=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_cpu_booster/lammps-22Jul2025
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"
EX="$N2P2_DEVEL/examples/interface-LAMMPS/H2O_RPBE-D3"

echo "=== [0/3] Build CPU-only (no GPU=1) ==="
(cd "$N2P2_DEVEL/src" && make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" \
    INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnpif)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
(cd "$LAMMPS_SRC/src" && touch main.cpp && make mpi -j"$SLURM_NTASKS_PER_NODE")

echo ""
echo "=== [1/3] H2O_2G (630 atoms), CPU-only, 32 ranks, 2000 steps ==="
RUNDIR="$DYN/report_cpu_2k_h2o2g"
rm -rf "$RUNDIR" && mkdir -p "$RUNDIR"
sed 's/^dump TRAJ.*/dump TRAJ all custom 500 dump.lammpstrj id type element x y z fx fy fz/' \
    "$DYN/md_smoketest.lmp" > "$RUNDIR/md_smoketest.lmp"
cp "$DYN/init_conf_minimized.data" "$RUNDIR/"
ln -s "$DYN/hdnnp-data" "$RUNDIR/hdnnp-data"
cd "$RUNDIR"
mpirun -np 32 "$LAMMPS_SRC/src/lmp_mpi" -in md_smoketest.lmp -log log.report_cpu_2k_h2o2g
echo "--- H2O_2G CPU-only result ---"
grep -A1 "^Pair " log.report_cpu_2k_h2o2g
grep "^Loop time" log.report_cpu_2k_h2o2g

echo ""
echo "=== [2/3] H2O_RPBE-D3 (8640 atoms), CPU-only, 32 ranks, 2000 steps ==="
RUNDIR="$EX/report_cpu_2k_rpbe"
rm -rf "$RUNDIR" && mkdir -p "$RUNDIR"
sed -e 's/variable numSteps.*equal.*5.*/variable numSteps equal 2000 # 2k report/' \
    -e 's/^dump 1 all atom 1 traj.dump/dump 1 all atom 500 traj.dump/' -e 's/resetew no maxew 100 /resetew yes maxew 1000000 /' \
    "$EX/md.lmp" > "$RUNDIR/md_2k.lmp"
cp "$EX/h2o_8640_liquid_NpT_RPBE-D3.data" "$RUNDIR/"
ln -s "$EX/hdnnp-data" "$RUNDIR/hdnnp-data"
cd "$RUNDIR"
mpirun -np 32 "$LAMMPS_SRC/src/lmp_mpi" -in md_2k.lmp -log log.report_cpu_2k_rpbe
echo "--- H2O_RPBE-D3 CPU-only result ---"
grep -A1 "^Pair " log.report_cpu_2k_rpbe
grep "^Loop time" log.report_cpu_2k_rpbe

echo ""
echo "=== [3/3] Done ==="
