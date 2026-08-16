#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --job-name=lammps-hdnnp-gpu-build
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Phase 2 of the LAMMPS/ML-HDNNP GPU port: build only (no MD run here --
# kept in a separate short job so a slow from-scratch LAMMPS compile on
# Booster's 32 cores doesn't risk running past the 30-minute debug QOS
# together with an MD smoke test). Links the stock LAMMPS ML-HDNNP
# package against n2p2_devel built with GPU=1 -- libnnp.a then has
# unresolved gpuNnForwardDEdG() references that only get resolved here,
# at LAMMPS's own final link of lmp_mpi (see the GPU= additions in
# src/libnnpif/makefile's lammps-mf rule and src/makefile's libnnpif
# rule, committed alongside this script).

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
LAMMPS_SRC=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025

echo "=== [1/3] Build n2p2_devel GPU (libnnp + libnnpgpu + libnnpif, LAMMPS interface only) ==="
cd "$N2P2_DEVEL/src"
# -march=native means .o files from a prior build on a different partition
# (e.g. DCGP) are not safe to reuse here -- always clean first.
make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true
make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" INTERFACES=LAMMPS GPU=1 -j"$SLURM_NTASKS_PER_NODE" libnnp libnnpgpu libnnpif
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
echo "--- generated lib/Makefile.lammps-extra ---"
cat "$N2P2_DEVEL/lib/Makefile.lammps-extra"

echo ""
echo "=== [2/3] Enable ML-HDNNP package in LAMMPS, link against GPU-built n2p2_devel ==="
cd "$LAMMPS_SRC/src"
make yes-ml-hdnnp
make lib-hdnnp args="-p $N2P2_DEVEL"

echo ""
echo "=== [3/3] Build lmp_mpi ==="
make mpi -j"$SLURM_NTASKS_PER_NODE"
ls -la "$LAMMPS_SRC/src/lmp_mpi"

echo ""
echo "Done"
