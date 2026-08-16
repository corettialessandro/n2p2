#!/bin/bash
#SBATCH -A L-AUT_DCGP2026
#SBATCH -p dcgp_usr_prod
#SBATCH --qos=dcgp_qos_dbg
#SBATCH --time 00:05:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=check-hw-dcgp
#SBATCH --output=%x_%j.out
lscpu
echo "=== /proc/cpuinfo model name / MHz ==="
grep -m1 "model name" /proc/cpuinfo
grep "MHz" /proc/cpuinfo | head -3
echo "=== module list ==="
module purge
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
which mpicxx
mpicxx --version
which gcc
gcc --version
ompi_info --version 2>&1 | head -3
