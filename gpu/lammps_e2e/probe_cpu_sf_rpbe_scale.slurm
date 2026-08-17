#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --job-name=lammps-hdnnp-cpu-sf-rpbe-scale
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Phase 5 step 8 overhead investigation, round 5: direct CPU-only
# counterpart to probe_gpu_sf_rpbe_scale.slurm (job 52552607, which got
# 48.3s/40 steps = 1.208s/step Pair time on GPU, at H2O_RPBE-D3's 8640
# atoms, 1 rank -- notably, the GPU SF dispatch's own cost per step was
# nearly IDENTICAL to its cost at H2O_2G's 630-atom scale, ~0.64-0.67s
# either way, consistent with the GPU being underutilized at 630 atoms
# and now doing 14x more useful work "for free" at 8640). This job
# measures the CPU-only baseline at the exact same config (1 rank,
# H2O_RPBE-D3, 40 steps) on Booster hardware (not DCGP -- isolates the
# GPU-vs-CPU comparison from the DCGP-vs-Booster NUMA/topology
# differences already characterized in Phase 3b) to see whether the GPU
# SF dispatch is now actually a net win at this larger system size.
# Fresh rebuild (own scratch tree, lammps_hdnnp_cpu_booster) since
# -march=native forbids reusing a binary from a different node/partition.

set -e
module purge
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
LAMMPS_SRC=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_cpu_booster/lammps-22Jul2025
EX="$N2P2_DEVEL/examples/interface-LAMMPS/H2O_RPBE-D3"

echo "=== [1/3] Build n2p2_devel CPU-only (libnnp + libnnpif, no GPU=1) on Booster ==="
(cd "$N2P2_DEVEL/src" && make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" \
    INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnpif)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h

echo ""
echo "=== [2/3] Relink lmp_mpi (CPU-only LAMMPS tree, Booster-built) ==="
cd "$LAMMPS_SRC/src"
touch main.cpp
make mpi -j"$SLURM_NTASKS_PER_NODE"

echo ""
echo "=== [3/3] Single-rank (1 rank, no GPU), H2O_RPBE-D3 (8640 atoms), 40 steps ==="
PROFDIR="$EX/probe_rpbe_scale_cpu"
rm -rf "$PROFDIR" && mkdir -p "$PROFDIR"
sed 's/variable numSteps.*equal.*5.*/variable numSteps equal 40 # GPU-SF scale probe (CPU counterpart)/' \
    "$EX/md.lmp" > "$PROFDIR/md_probe.lmp"
cp "$EX/h2o_8640_liquid_NpT_RPBE-D3.data" "$PROFDIR/"
ln -s "$EX/hdnnp-data" "$PROFDIR/hdnnp-data"
cd "$PROFDIR"
grep numSteps md_probe.lmp
mpirun -np 1 "$LAMMPS_SRC/src/lmp_mpi" -in md_probe.lmp -log log.probe_rpbe_scale_cpu

echo ""
echo "=== Pair time summary (CPU, Booster, 1 rank, 40-step H2O_RPBE-D3) ==="
grep -A1 "^Pair " log.probe_rpbe_scale_cpu
grep "^Loop time" log.probe_rpbe_scale_cpu
