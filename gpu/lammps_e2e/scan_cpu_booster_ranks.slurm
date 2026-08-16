#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --mem=64GB
#SBATCH --job-name=lammps-hdnnp-cpu-booster-scan
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Matched-rank-count companion to scan_cpu_dcgp_ranks.slurm, to compare
# DCGP vs Booster CPU at IDENTICAL rank counts (isolating hardware/
# topology from rank-count effects). NP via sbatch --export.

set -e
module purge
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
LAMMPS_CPU_BIN=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_cpu_booster/lammps-22Jul2025/src/lmp_mpi
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"
SCANDIR="$DYN/scan_cpu_booster_${NP}r"

rm -rf "$SCANDIR" && mkdir -p "$SCANDIR"
cp "$DYN/init_conf_minimized.data" "$DYN/md_smoketest.lmp" "$SCANDIR/"
ln -s "$DYN/hdnnp-data" "$SCANDIR/hdnnp-data"
cd "$SCANDIR"

echo "=== Booster CPU scan: NP=$NP ranks, 2000-step H2O_2G ==="
mpirun -np "$NP" "$LAMMPS_CPU_BIN" -in md_smoketest.lmp -log "log.scan_${NP}r"

echo ""
echo "=== Pair/Comm time summary (NP=$NP) ==="
grep -A3 "^Pair " "log.scan_${NP}r"
grep "^Loop time" "log.scan_${NP}r"

echo "Done"
