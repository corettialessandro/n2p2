#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=verify-gpu-sf-flag-split
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Phase 5/6 wrap-up: SF-GPU dispatch was just moved behind a separate
# opt-in flag (GPU_SF=1, distinct from GPU=1) so plain GPU=1 builds keep
# only the still-valid NN-only win and don't silently inherit SF-GPU's
# rank-count-dependent regression. Pure compile verification (the
# underlying dispatch code/logic is unchanged, only which macro gates it
# -- already validated for correctness and performance in prior jobs) --
# confirms both configurations build cleanly:
#   (a) GPU=1 alone -> N2P2_GPU defined, N2P2_GPU_SF NOT defined
#   (b) GPU=1 GPU_SF=1 -> both defined
# and great a quick grep-based sanity check that the compiled object
# code actually differs (SF dispatch symbols present only in case b).

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0

echo "=== [1/3] Build libnnpgpu.a (GPU=1, unaffected by the flag split) ==="
(cd "$N2P2_DEVEL/src/libnnpgpu" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpgpu" && make COMP=gnu MODE=static GPU=1)

echo ""
echo "=== [2/3] Build (a): GPU=1 alone (SF-GPU should NOT be compiled in) ==="
(cd "$N2P2_DEVEL/src/libnnp" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpif" && make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" \
    INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnp libnnpif)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
echo "--- symbol check (expect: gpuNnForwardDEdG present, gpuSfExpRadGroup ABSENT) ---"
nm "$N2P2_DEVEL/lib/libnnp.a" 2>/dev/null | grep -c "gpuNnForwardDEdG" || echo "0 gpuNnForwardDEdG refs"
nm "$N2P2_DEVEL/lib/libnnp.a" 2>/dev/null | grep -c "gpuSfExpRadGroup" || echo "0 gpuSfExpRadGroup refs (expected)"

echo ""
echo "=== [3/3] Build (b): GPU=1 GPU_SF=1 (SF-GPU SHOULD be compiled in) ==="
(cd "$N2P2_DEVEL/src/libnnp" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpif" && make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 GPU_SF=1 PROJECT_EIGEN="$EIGEN" \
    INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnp libnnpif)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
echo "--- symbol check (expect: BOTH gpuNnForwardDEdG and gpuSfExpRadGroup present) ---"
nm "$N2P2_DEVEL/lib/libnnp.a" 2>/dev/null | grep -c "gpuNnForwardDEdG" || echo "0 gpuNnForwardDEdG refs"
nm "$N2P2_DEVEL/lib/libnnp.a" 2>/dev/null | grep -c "gpuSfExpRadGroup" || echo "0 gpuSfExpRadGroup refs"

echo ""
echo "Done -- both configurations built successfully."
