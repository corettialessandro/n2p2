#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time 18:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128GB
#SBATCH --job-name=rpbe-prod-200k-nnonly
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Real production-length (200000-step) validation of the GPU_SF=1 win on
# H2O_RPBE-D3, GPU=1 (NN-only) baseline half. Everything measured so far
# (the 1.56x GPU_SF=1-vs-GPU=1 win, and today's host-side CSR/buffer
# tuning) is at 2000 steps -- this project has already been burned once
# by a short-run number (H2O_2G's 2000-step GPU_SF win reversing at
# 200000 steps), so the 1.56x claim needs checking at real length before
# it's trusted. Companion job: rpbe_production_200k_sf.slurm (GPU_SF=1,
# current code state -- step 1 + the shared-buffer arena).
#
# 32 ranks/4 GPUs (8/GPU via MPS), matching every prior comparison in
# this investigation.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
LAMMPS_SRC=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu_nnonly/lammps-22Jul2025
EX="$N2P2_DEVEL/examples/interface-LAMMPS/H2O_RPBE-D3"
WRAPPER="$N2P2_DEVEL/gpu/e2e_train_check/rank_gpu_wrapper.sh"

if [ ! -d "$LAMMPS_SRC" ]; then
    echo "=== Setting up fresh lammps_hdnnp_gpu_nnonly tree (copy from lammps_hdnnp_gpu) ==="
    mkdir -p "$(dirname "$LAMMPS_SRC")"
    cp -r /leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025 "$LAMMPS_SRC"
fi

echo "=== [0/2] Build libnnpgpu.a + libnnp.a + libnnpif.a (GPU=1, no GPU_SF) ==="
(cd "$N2P2_DEVEL/src/libnnpgpu" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpgpu" && make COMP=gnu MODE=static GPU=1)
(cd "$N2P2_DEVEL/src/libnnp" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpif" && make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" \
    INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnp libnnpif)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
(cd "$LAMMPS_SRC/src" && touch main.cpp && make mpi -j"$SLURM_NTASKS_PER_NODE")

echo ""
echo "=== Starting a single MPS control daemon serving all 4 GPUs ==="
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-pipe-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
sleep 2
export RANKS_PER_GPU=8

echo ""
echo "=== [1/2] H2O_RPBE-D3 (8640 atoms), GPU NN-only, 32 ranks/4 GPUs, 200000 steps ==="
RUNDIR="$EX/production_200k_nnonly"
rm -rf "$RUNDIR" && mkdir -p "$RUNDIR"
sed -e 's/variable numSteps.*equal.*5.*/variable numSteps equal 200000 # full production-length validation/' \
    -e 's/^thermo 1$/thermo 1000/' \
    -e 's/^dump 1 all atom 1 traj.dump/dump 1 all atom 10000 traj.dump/' \
    -e 's/resetew no maxew 100 /resetew yes maxew 1000000 /' \
    "$EX/md.lmp" > "$RUNDIR/md_200k.lmp"
cp "$EX/h2o_8640_liquid_NpT_RPBE-D3.data" "$RUNDIR/"
ln -s "$EX/hdnnp-data" "$RUNDIR/hdnnp-data"
cd "$RUNDIR"
mpirun -np 32 "$WRAPPER" "$LAMMPS_SRC/src/lmp_mpi" -in md_200k.lmp -log log.production_200k_nnonly
echo "--- H2O_RPBE-D3 GPU NN-only, 200000-step result ---"
grep -A1 "^Pair " log.production_200k_nnonly
grep "^Loop time" log.production_200k_nnonly

echo ""
echo "=== Stopping MPS control daemon ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2

echo "=== [2/2] Done ==="
