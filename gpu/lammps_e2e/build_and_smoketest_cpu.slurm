#!/bin/bash
#SBATCH -A L-AUT_DCGP2026
#SBATCH -p dcgp_usr_prod
#SBATCH --qos=dcgp_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=112
#SBATCH --mem=200GB
#SBATCH --job-name=lammps-hdnnp-cpu-build
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Phase 1 of the LAMMPS/ML-HDNNP GPU port (soft-percolating-jellyfish.md
# plan): CPU baseline. Builds n2p2_devel's libnnp+libnnpif (LAMMPS
# interface only, no CabanaMD -- Kokkos isn't available on this cluster),
# links them into a stock LAMMPS stable tarball's official ML-HDNNP
# package (unmodified -- it's already upstream, no need to copy any of
# our src/interface/LAMMPS/src/ML-HDNNP/ files), then smoke-tests with
# both LAMMPS's own bundled ML-HDNNP example and our own H2O_2G case
# (630 atoms, using the converged 100-epoch weights from
# gpu/e2e_train_check/results/run-100ep-cpu/, NOT the 2-epoch smoke-test
# weights that were sitting directly in temp/H2O_2G/).

set -e
module purge
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
LAMMPS_SRC=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_cpu/lammps-22Jul2025

echo "=== [1/5] Build n2p2_devel CPU (libnnp + libnnpif, LAMMPS interface only) ==="
cd "$N2P2_DEVEL/src"
# -march=native means .o files from a prior build on a different partition
# (e.g. Booster) are not safe to reuse here -- always clean first.
make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true
make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnpif
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
ls -la "$N2P2_DEVEL/lib/"

echo ""
echo "=== [2/5] Enable ML-HDNNP package in LAMMPS, link against n2p2_devel ==="
cd "$LAMMPS_SRC/src"
make yes-ml-hdnnp
make lib-hdnnp args="-p $N2P2_DEVEL"

echo ""
echo "=== [3/5] Build lmp_mpi ==="
make mpi -j"$SLURM_NTASKS_PER_NODE"
ls -la "$LAMMPS_SRC/src/lmp_mpi"

echo ""
echo "=== [4/5] Smoke test: LAMMPS's own bundled ML-HDNNP example (360 water molecules, 10 steps) ==="
cd "$LAMMPS_SRC/examples/PACKAGES/hdnnp"
mpirun -np 4 "$LAMMPS_SRC/src/lmp_mpi" -in in.hdnnp

echo ""
echo "=== [5/5] Smoke test: our own H2O_2G/dynamics case (630 atoms, reduced step count) ==="
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"
TRAIN100="$N2P2_DEVEL/gpu/e2e_train_check/results/run-100ep-cpu"
mkdir -p "$DYN/hdnnp-data"
cp "$TRAIN100/input.nn" "$TRAIN100/scaling.data" "$DYN/hdnnp-data/"
cp "$TRAIN100/weights.001.000100.out" "$DYN/hdnnp-data/weights.001.data"
cp "$TRAIN100/weights.008.000100.out" "$DYN/hdnnp-data/weights.008.data"

cd "$DYN"
sed 's/^variable numSteps.*equal.*/variable numSteps equal 2000 # smoke test override/' md.lmp > md_smoketest.lmp
mpirun -np 4 "$LAMMPS_SRC/src/lmp_mpi" -in md_smoketest.lmp -log log.smoketest_cpu

echo ""
echo "Done"
