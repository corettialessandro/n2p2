#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=64GB
#SBATCH --job-name=rebuild-lammps-cpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
SCRATCH=/leonardo_scratch/large/userexternal/acoretti

echo "=== [1/3] Download + extract LAMMPS 22Jul2025 ==="
mkdir -p "$SCRATCH/lammps_hdnnp_cpu_booster"
cd "$SCRATCH/lammps_hdnnp_cpu_booster"
tar -xzf lammps.tar.gz
rm lammps.tar.gz
LAMMPS_CPU="$SCRATCH/lammps_hdnnp_cpu_booster/lammps-22Jul2025"

echo ""
echo "=== [2/3] Build n2p2_devel CPU-only ==="
(cd "$N2P2_DEVEL/src" && make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" \
    INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnpif)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h

echo ""
echo "=== [3/3] Enable ML-HDNNP, link, build lmp_mpi ==="
cd "$LAMMPS_CPU/src"
make yes-ml-hdnnp
make lib-hdnnp args="-p $N2P2_DEVEL"
make mpi -j"$SLURM_NTASKS_PER_NODE"
ls -la "$LAMMPS_CPU/src/lmp_mpi"
echo "Done"
