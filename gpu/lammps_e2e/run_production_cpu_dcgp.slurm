#!/bin/bash
#SBATCH -A L-AUT_DCGP2026
#SBATCH -p dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --time 1-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=112
#SBATCH --mem=200GB
#SBATCH --job-name=lammps-hdnnp-prod-cpu-dcgp112
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Full 200000-step H2O_2G production run, CPU baseline: DCGP, 112 ranks
# (full node), normal QOS. Companion to run_production_gpu_booster.slurm.
# Reuses the already-built, already-validated CPU lmp_mpi from
# build_and_smoketest_cpu.slurm (job 52481389) -- no rebuild needed,
# MODE=static means the binary is self-contained.

set -e
module purge
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
LAMMPS_CPU_BIN=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_cpu/lammps-22Jul2025/src/lmp_mpi
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"
RUNDIR="$DYN/production_cpu_dcgp112"

rm -rf "$RUNDIR" && mkdir -p "$RUNDIR"
cp "$DYN/init_conf_minimized.data" "$DYN/md.lmp" "$RUNDIR/"
ln -s "$DYN/hdnnp-data" "$RUNDIR/hdnnp-data"
cd "$RUNDIR"

echo "=== Full 200000-step H2O_2G run, CPU (DCGP, 112 ranks) ==="
mpirun -np 112 "$LAMMPS_CPU_BIN" -in md.lmp -log log.production_cpu_112

echo ""
echo "=== Pair time summary ==="
grep -A2 "^Pair " log.production_cpu_112
grep "^Loop time" log.production_cpu_112

echo "Done"
