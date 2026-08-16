#!/bin/bash
#SBATCH -A L-AUT_DCGP2026
#SBATCH -p dcgp_usr_prod
#SBATCH --qos=dcgp_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --mem=64GB
#SBATCH --job-name=lammps-hdnnp-cpu-dcgp-scan
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Core-count scaling scan on DCGP for the H2O_2G (630 atoms) case --
# we already have 4 ranks (492.6s Pair/2000 steps, job 52481389) and 112
# ranks (production run, job 52496567: 2231s Pair/200000 steps -> 22.3s
# per 2000-step equivalent, but Comm at 48% of wall-clock -- badly
# over-decomposed). This scan fills in the intermediate points to find
# where DCGP's optimum actually is. NP passed via `sbatch
# --export=ALL,NP=.. --ntasks-per-node=$NP`.

set -e
module purge
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
LAMMPS_CPU_BIN=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_cpu/lammps-22Jul2025/src/lmp_mpi
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"
SCANDIR="$DYN/scan_cpu_dcgp_${NP}r"

rm -rf "$SCANDIR" && mkdir -p "$SCANDIR"
cp "$DYN/init_conf_minimized.data" "$DYN/md_smoketest.lmp" "$SCANDIR/"
ln -s "$DYN/hdnnp-data" "$SCANDIR/hdnnp-data"
cd "$SCANDIR"

echo "=== DCGP scan: NP=$NP ranks, 2000-step H2O_2G ==="
mpirun -np "$NP" "$LAMMPS_CPU_BIN" -in md_smoketest.lmp -log "log.scan_${NP}r"

echo ""
echo "=== Pair/Comm time summary (NP=$NP) ==="
grep -A3 "^Pair " "log.scan_${NP}r"
grep "^Loop time" "log.scan_${NP}r"

echo "Done"
