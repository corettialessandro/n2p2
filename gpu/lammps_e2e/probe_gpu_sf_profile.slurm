#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --job-name=lammps-hdnnp-gpu-sf-profile2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Phase 5 step 8 overhead investigation, round 3: rounds 1-2 ruled out
# GPU-call blocking-sync overhead (pinned+async made zero measurable
# difference). Add real Stopwatch instrumentation (temporary, matching
# Phase 5 step 1's methodology) around the three real phases inside
# Mode::calculateSymmetryFunctionGroups()'s GPU dispatch -- CSR build,
# GPU call, scatter-back -- to find out where the ~0.7s/step (1 rank,
# zero contention) actually goes. Single rank, 300 steps (enough for 3
# profile prints at the 100-call interval).

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
LAMMPS_SRC=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"

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
echo "=== [1/2] Single-rank (1 rank/1 GPU), 300 steps, profiling on ==="
PROFDIR="$DYN/probe_profile"
rm -rf "$PROFDIR" && mkdir -p "$PROFDIR"
sed 's/variable numSteps equal 200.*/variable numSteps equal 300 # profiling probe/' \
    "$DYN/md_probe_200.lmp" > "$PROFDIR/md_probe_300.lmp"
cp "$DYN/init_conf_minimized.data" "$PROFDIR/"
ln -s "$DYN/hdnnp-data" "$PROFDIR/hdnnp-data"
cd "$PROFDIR"
mpirun -np 1 "$LAMMPS_SRC/src/lmp_mpi" -in md_probe_300.lmp -log log.probe_profile

echo ""
echo "=== [2/2] Profile lines from stderr ==="
grep "SF-GPU-PROFILE" "$SLURM_SUBMIT_DIR/lammps-hdnnp-gpu-sf-profile2_${SLURM_JOB_ID}.err" || true
