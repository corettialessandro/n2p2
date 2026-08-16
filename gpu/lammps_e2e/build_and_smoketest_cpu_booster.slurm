#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=64GB
#SBATCH --job-name=lammps-hdnnp-cpu-booster-build
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Same-hardware CPU baseline: a CPU-only lmp_mpi built and run ON
# BOOSTER (not DCGP), so the GPU-vs-CPU comparison isolates the GPU
# speedup itself rather than mixing in DCGP-vs-Booster CPU differences.
# -march=native means the DCGP-built CPU binary (build_and_smoketest_cpu.
# slurm) is NOT safe to reuse here -- a fresh build is required, in its
# own scratch tree (lammps_hdnnp_cpu_booster) so it doesn't collide with
# the GPU build/run in lammps_hdnnp_gpu. No --gres requested: this job
# uses no GPU at all.

set -e
module purge
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
LAMMPS_SRC=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_cpu_booster/lammps-22Jul2025
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"

echo "=== [1/3] Build n2p2_devel CPU on Booster hardware (libnnp + libnnpif) ==="
cd "$N2P2_DEVEL/src"
make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true
make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnpif
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h

echo ""
echo "=== [2/3] Enable ML-HDNNP package in LAMMPS, link against Booster-built n2p2_devel ==="
cd "$LAMMPS_SRC/src"
make yes-ml-hdnnp
make lib-hdnnp args="-p $N2P2_DEVEL"

echo ""
echo "=== [3/3] Build lmp_mpi, then smoke test (2000 steps, 32 ranks, no GPU) ==="
make mpi -j"$SLURM_NTASKS_PER_NODE"
ls -la "$LAMMPS_SRC/src/lmp_mpi"

SCANDIR="$DYN/scan_cpu_booster32"
rm -rf "$SCANDIR" && mkdir -p "$SCANDIR"
cp "$DYN/init_conf_minimized.data" "$DYN/md_smoketest.lmp" "$SCANDIR/"
ln -s "$DYN/hdnnp-data" "$SCANDIR/hdnnp-data"
cd "$SCANDIR"
mpirun -np 32 "$LAMMPS_SRC/src/lmp_mpi" -in md_smoketest.lmp -log log.scan_cpu_booster32

echo ""
echo "=== Pair time summary (CPU, Booster, 32 ranks, 2000-step smoke test) ==="
grep -A2 "^Pair " log.scan_cpu_booster32
grep "^Loop time" log.scan_cpu_booster32

echo ""
echo "Done"
