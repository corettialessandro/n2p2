#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=lammps-hdnnp-gpu-sf-rpbe-scale
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Phase 5 step 8 overhead investigation, round 4: rounds 1-3 found the GPU
# SF kernels are underutilized at H2O_2G's scale (630 atoms -> only 2-4
# CUDA thread blocks per call on a 108-SM GPU, ~0.66s/step "gpu" bucket,
# ~93% of total Pair time). Before attempting a real kernel-parallelization
# redesign, check whether the SAME unmodified kernel already does much
# better at the 8640-atom H2O_RPBE-D3 example (Phase 6, previously
# deferred) -- 14x more atoms/call could mean ~30-50 thread blocks
# instead of 2-4, possibly closing most of the gap for free. Confirmed
# GPU-compatible first: hdnnp-data/input.nn uses only symfunction types
# 2/3 with cutoff_type 2 (CT_TANHU), same as H2O_2G.
#
# Single rank (isolates kernel behavior from the no-MPS multi-rank
# contention already characterized separately), short run (profiling
# print interval lowered to every 20 calls for fast feedback at this
# larger, slower-per-step scale).

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
LAMMPS_SRC=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025
EX="$N2P2_DEVEL/examples/interface-LAMMPS/H2O_RPBE-D3"

echo "=== [0/2] Rebuild libnnpgpu.a + libnnp.a + libnnpif.a (GPU=1) and relink lmp_mpi ==="
(cd "$N2P2_DEVEL/src/libnnpgpu" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpgpu" && make COMP=gnu MODE=static GPU=1)
(cd "$N2P2_DEVEL/src/libnnp" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpif" && make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" \
    INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnp libnnpif)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
(cd "$LAMMPS_SRC/src" && touch main.cpp && make mpi -j4)

echo ""
echo "=== [1/2] Single-rank (1 rank/1 GPU), H2O_RPBE-D3 (8640 atoms), 40 steps ==="
PROFDIR="$EX/probe_rpbe_scale"
rm -rf "$PROFDIR" && mkdir -p "$PROFDIR"
sed 's/variable numSteps.*equal.*5.*/variable numSteps equal 40 # GPU-SF scale probe/' \
    "$EX/md.lmp" > "$PROFDIR/md_probe.lmp"
cp "$EX/h2o_8640_liquid_NpT_RPBE-D3.data" "$PROFDIR/"
ln -s "$EX/hdnnp-data" "$PROFDIR/hdnnp-data"
cd "$PROFDIR"
grep numSteps md_probe.lmp
mpirun -np 1 "$LAMMPS_SRC/src/lmp_mpi" -in md_probe.lmp -log log.probe_rpbe_scale

echo ""
echo "=== [2/2] Profile lines from stderr ==="
grep "SF-GPU-PROFILE" "$SLURM_SUBMIT_DIR/lammps-hdnnp-gpu-sf-rpbe-scale_${SLURM_JOB_ID}.err" || true
