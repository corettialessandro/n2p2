#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=lammps-hdnnp-gpu-sf-probe
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Phase 5 step 8 overhead investigation, round 2: round 1 (job 52549120)
# found the GPU SF dispatch was ~10x slower than CPU-only, and isolated
# two compounding causes -- (a) 4 ranks contending for one GPU with no
# MPS (~3.7x penalty), and (b) even single-rank/zero-contention was
# still ~2.85x slower than the old CPU baseline, traced to ~16
# *synchronous* CUDA calls per (element,group) pair per timestep, each
# individually blocking the host thread. Fixed cause (b) in
# GpuSymmetryFunction.cu: pinned host staging buffers + cudaMemcpyAsync/
# cudaMemsetAsync on one dedicated stream + a single cudaStreamSynchronize
# per call, collapsing ~16 blocking round-trips into 1 (re-validated
# correctness first via gpu/e2e_symfnc_check -- still ALL PASS, same
# ~1e-15/1e-16/1e-17 tolerances). This script rebuilds everything fresh
# (unlike round 1's probe, which reused an existing binary) and reruns
# the identical 1-rank/4-rank 200-step comparison to measure the effect.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
LAMMPS_SRC=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"

echo "=== [0/3] Rebuild libnnpgpu.a + libnnp.a + libnnpif.a (GPU=1) and relink lmp_mpi ==="
(cd "$N2P2_DEVEL/src/libnnpgpu" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpgpu" && make COMP=gnu MODE=static GPU=1)
(cd "$N2P2_DEVEL/src/libnnp" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpif" && make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" \
    INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnp libnnpif)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
(cd "$LAMMPS_SRC/src" && touch main.cpp && make mpi -j"$SLURM_NTASKS_PER_NODE")

echo ""
echo "=== [1/3] Single-rank (1 rank/1 GPU), 200 steps ==="
PROFDIR="$DYN/probe_1rank"
rm -rf "$PROFDIR" && mkdir -p "$PROFDIR"
cp "$DYN/init_conf_minimized.data" "$DYN/md_probe_200.lmp" "$PROFDIR/"
ln -s "$DYN/hdnnp-data" "$PROFDIR/hdnnp-data"
cd "$PROFDIR"
mpirun -np 1 "$LAMMPS_SRC/src/lmp_mpi" -in md_probe_200.lmp -log log.probe_1rank

echo ""
echo "=== [2/3] Multi-rank (4 ranks/1 GPU, no MPS), 200 steps ==="
PROFDIR="$DYN/probe_4rank"
rm -rf "$PROFDIR" && mkdir -p "$PROFDIR"
cp "$DYN/init_conf_minimized.data" "$DYN/md_probe_200.lmp" "$PROFDIR/"
ln -s "$DYN/hdnnp-data" "$PROFDIR/hdnnp-data"
cd "$PROFDIR"
mpirun -np 4 "$LAMMPS_SRC/src/lmp_mpi" -in md_probe_200.lmp -log log.probe_4rank

echo ""
echo "=== Summary: Pair time / Loop time from both logs ==="
for f in "$DYN/probe_1rank/log.probe_1rank" "$DYN/probe_4rank/log.probe_4rank"; do
    echo "--- $f ---"
    grep -E "Loop time|^Pair " "$f"
done
