#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time 01:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --job-name=rebuild-lammps-trees
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# The two LAMMPS+ML-HDNNP build trees (lammps_hdnnp_gpu, lammps_hdnnp_cpu_booster)
# were deleted during the scratch cleanup as "regenerable build artifacts" --
# rebuilding both here from a fresh LAMMPS 22Jul2025 stable download, so the
# GPU_SF=1 200K-step H2O_2G production comparison (run_production_gpu_booster.slurm
# / run_production_cpu_booster.slurm, unchanged) can be rerun to check
# reproducibility of the "slightly weird" result (GPU 28.6% slower than CPU,
# job 52588528 vs 52588529).

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
SCRATCH=/leonardo_scratch/large/userexternal/acoretti

echo "=== [1/5] Download + extract LAMMPS 22Jul2025 once ==="
mkdir -p "$SCRATCH/lammps_hdnnp_gpu"
cd "$SCRATCH/lammps_hdnnp_gpu"
curl -sL -o lammps.tar.gz https://download.lammps.org/tars/releases/stable/lammps-20250722.tar.gz
tar -xzf lammps.tar.gz
rm lammps.tar.gz
ls -d lammps-22Jul2025
LAMMPS_GPU="$SCRATCH/lammps_hdnnp_gpu/lammps-22Jul2025"

echo ""
echo "=== [2/5] Copy the extracted tree for the CPU build (avoid a second download) ==="
mkdir -p "$SCRATCH/lammps_hdnnp_cpu_booster"
cp -r "$LAMMPS_GPU" "$SCRATCH/lammps_hdnnp_cpu_booster/"
LAMMPS_CPU="$SCRATCH/lammps_hdnnp_cpu_booster/lammps-22Jul2025"

echo ""
echo "=== [3/5] Build n2p2_devel GPU (GPU=1 GPU_SF=1) + LAMMPS ML-HDNNP + lmp_mpi ==="
(cd "$N2P2_DEVEL/src/libnnpgpu" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpgpu" && make COMP=gnu MODE=static GPU=1)
(cd "$N2P2_DEVEL/src" && make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 GPU_SF=1 PROJECT_EIGEN="$EIGEN" \
    INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnp libnnpgpu libnnpif)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
cd "$LAMMPS_GPU/src"
make yes-ml-hdnnp
make lib-hdnnp args="-p $N2P2_DEVEL"
make mpi -j"$SLURM_NTASKS_PER_NODE"
ls -la "$LAMMPS_GPU/src/lmp_mpi"

echo ""
echo "=== [4/5] Build n2p2_devel CPU-only + LAMMPS ML-HDNNP + lmp_mpi ==="
(cd "$N2P2_DEVEL/src" && make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" \
    INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnpif)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
cd "$LAMMPS_CPU/src"
make yes-ml-hdnnp
make lib-hdnnp args="-p $N2P2_DEVEL"
make mpi -j"$SLURM_NTASKS_PER_NODE"
ls -la "$LAMMPS_CPU/src/lmp_mpi"

echo ""
echo "=== [5/5] Done -- both trees ready ==="
