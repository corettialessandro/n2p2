#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --job-name=rebuild-lammps-gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
SCRATCH=/leonardo_scratch/large/userexternal/acoretti

echo "=== [1/3] Download + extract LAMMPS 22Jul2025 ==="
mkdir -p "$SCRATCH/lammps_hdnnp_gpu"
cd "$SCRATCH/lammps_hdnnp_gpu"
tar -xzf lammps.tar.gz
rm lammps.tar.gz
LAMMPS_GPU="$SCRATCH/lammps_hdnnp_gpu/lammps-22Jul2025"

echo ""
echo "=== [2/3] Build n2p2_devel GPU (GPU=1 GPU_SF=1) ==="
(cd "$N2P2_DEVEL/src/libnnpgpu" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpgpu" && make COMP=gnu MODE=static GPU=1)
(cd "$N2P2_DEVEL/src" && make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 GPU_SF=1 PROJECT_EIGEN="$EIGEN" \
    INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnp libnnpgpu libnnpif)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h

echo ""
echo "=== [3/3] Enable ML-HDNNP, link, build lmp_mpi ==="
cd "$LAMMPS_GPU/src"
make yes-ml-hdnnp
make lib-hdnnp args="-p $N2P2_DEVEL"
make mpi -j"$SLURM_NTASKS_PER_NODE"
ls -la "$LAMMPS_GPU/src/lmp_mpi"
echo "Done"
