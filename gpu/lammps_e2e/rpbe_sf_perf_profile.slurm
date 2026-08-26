#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=rpbe-sf-perf-profile
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Deeper profiling of GPU_SF=1's host-side scatter-back cost (gpu/README.md's
# matching follow-up). Two prior hypothesis-driven attempts (persistent
# scratch buffers, hoisting per-member lookups out of the scatter loop)
# both found no net win at production scale -- rather than guess a third
# fix blind, use `perf record`/`perf annotate` (hardware cache-miss/cycle
# counters, user-space events only -- confirmed available via
# perf_event_paranoid=2) on a real 1-rank H2O_RPBE-D3 run to see, at the
# source-line level, whether the cost is genuinely memory-access-pattern
# bound (e.g. pointer-chasing through Atom::Neighbor::dGdr, a plausible
# structural cause) or something else. Rebuilds libnnp/libnnpif with -g
# added alongside -O3 (profiles the real optimized binary, just with
# line-level debug info) -- CUDA kernel itself untouched, this profiles
# host-side code only.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
LAMMPS_GPU=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025
EX="$N2P2_DEVEL/examples/interface-LAMMPS/H2O_RPBE-D3"

echo "=== [0/4] Rebuild libnnpgpu.a (GPU=1, unchanged) ==="
(cd "$N2P2_DEVEL/src/libnnpgpu" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpgpu" && make COMP=gnu MODE=static GPU=1)

echo ""
echo "=== [1/4] Rebuild libnnp.a + libnnpif.a (GPU=1 GPU_SF=1, WITH -g for perf annotate) ==="
(cd "$N2P2_DEVEL/src/libnnp" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpif" && make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 GPU_SF=1 PROJECT_EIGEN="$EIGEN" \
    PROJECT_CFLAGS="-O3 -march=native -std=c++11 -g" \
    INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnp libnnpif)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
(cd "$LAMMPS_GPU/src" && touch main.cpp && make mpi -j4)

echo ""
echo "=== [2/4] perf record: 1 rank, 300 steps, H2O_RPBE-D3 ==="
PROFDIR="$EX/perf_profile"
rm -rf "$PROFDIR" && mkdir -p "$PROFDIR"
sed 's/variable numSteps.*equal.*5.*/variable numSteps equal 300 # perf profiling/' \
    "$EX/md.lmp" > "$PROFDIR/md_probe.lmp"
cp "$EX/h2o_8640_liquid_NpT_RPBE-D3.data" "$PROFDIR/"
ln -s "$EX/hdnnp-data" "$PROFDIR/hdnnp-data"
cd "$PROFDIR"
perf record --call-graph dwarf -o perf.data -- \
    mpirun -np 1 "$LAMMPS_GPU/src/lmp_mpi" -in md_probe.lmp -log log.perf_profile
echo "--- Loop/Pair time for context ---"
grep -A1 "^Pair " log.perf_profile
grep "^Loop time" log.perf_profile

echo ""
echo "=== [3/4] perf report -- top symbols by self time ==="
perf report -i perf.data --stdio --sort=overhead,symbol -g none 2>/dev/null | head -60

echo ""
echo "=== [4/4] perf annotate -- source-line breakdown, full dump ==="
perf annotate -i perf.data --stdio > perf_annotate_full.txt 2>&1
echo "Full annotate output: $(wc -l < perf_annotate_full.txt) lines, saved to perf_annotate_full.txt"
echo ""
echo "--- grep for calculateSymmetryFunctionGroups block ---"
awk '/calculateSymmetryFunctionGroups/,/^$/' perf_annotate_full.txt | head -400

echo ""
echo "=== Done ==="
