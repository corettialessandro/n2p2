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

# Phase 5 step 8 overhead investigation: the GPU SF dispatch (job 52523099)
# is numerically correct but ~10x SLOWER than CPU-only. Isolate whether
# this is (a) inherent per-call GPU-dispatch overhead (many small
# cudaMemcpy + blocking cudaDeviceSynchronize() calls per timestep -- 9
# element/group pairs per rank per timestep, each doing ~16 CUDA API
# calls), or (b) GPU-context contention from 4 MPI ranks sharing one GPU
# with no MPS (this project already knows from nnp-train that MPS matters
# for multi-rank-per-GPU sharing -- see soft-percolating-jellyfish.md
# Phase 3's "8/GPU, matching the nnp-train MPS precedent"). Reuses the
# already-built, already-validated lmp_mpi from job 52523099 -- no
# rebuild needed, this is a pure timing probe.
#
# Runs the same 200-step config at 1 rank and at 4 ranks (both GPU-SF),
# back to back, same node/GPU. If per-step time is roughly the same at
# 1 rank as at 4 ranks (i.e. 4-rank total isn't ~4x worse), contention
# isn't the story. If 1 rank is dramatically faster per-step, contention
# (no MPS) is confirmed as the dominant cost.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
LAMMPS_SRC=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"

echo "=== [1/2] Single-rank (1 rank/1 GPU), 200 steps ==="
PROFDIR="$DYN/probe_1rank"
rm -rf "$PROFDIR" && mkdir -p "$PROFDIR"
cp "$DYN/init_conf_minimized.data" "$DYN/md_probe_200.lmp" "$PROFDIR/"
ln -s "$DYN/hdnnp-data" "$PROFDIR/hdnnp-data"
cd "$PROFDIR"
mpirun -np 1 "$LAMMPS_SRC/src/lmp_mpi" -in md_probe_200.lmp -log log.probe_1rank

echo ""
echo "=== [2/2] Multi-rank (4 ranks/1 GPU, no MPS), 200 steps ==="
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
