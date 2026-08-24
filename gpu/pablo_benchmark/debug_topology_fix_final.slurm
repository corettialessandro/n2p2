#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-topo-final
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Final verification of the REAL fix (Mode::resetForceTopologyCache(),
# called from Training::dataSetNormalization() after its cutoff
# rescale) -- not the earlier N2P2_FORCE_TOPOLOGY_REUPLOAD debug env
# var, which is now removed. Runs the ORIGINAL reproducing config
# (4 ranks, selection_mode=2 i.e. water's real input.nn unmodified,
# full input_small.data) to confirm the actual training-loop NaN
# explosion is gone, matching CPU.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_CPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water
N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
KF=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force/kf_trace
TEST="$KF/topofix_final"

echo "=== Clean rebuild: CPU nnp-train ==="
(cd "$N2P2_CPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_CPU/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" nnp-train > "$TEST.build-cpu.log" 2>&1) || \
    (echo "--- CPU build failed ---" && tail -60 "$TEST.build-cpu.log" && exit 1)
git -C "$N2P2_CPU" checkout -- src/libnnp/version.h

echo "=== Clean rebuild: GPU nnp-train (real fix, no debug env var) ==="
(cd "$N2P2_GPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_GPU/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-train > "$TEST.build-gpu.log" 2>&1) || \
    (echo "--- GPU build failed ---" && tail -60 "$TEST.build-gpu.log" && exit 1)
git -C "$N2P2_GPU" checkout -- src/libnnp/version.h
grep -c "resetForceTopologyCache" "$N2P2_GPU/src/libnnp/Mode.cpp" "$N2P2_GPU/src/libnnptrain/Training.cpp"

mkdir -p "$TEST/cpu_orig" "$TEST/gpu_orig" "$TEST/cpu_1r" "$TEST/gpu_1r"
# Original reproducing config: 4 ranks, UNMODIFIED input.nn (selection_mode=2, real water config)
for d in cpu_orig gpu_orig; do
  cp "$KF/input.nn" "$KF/scaling.data" "$TEST/$d/"
  cp "$KF/input_small.data" "$TEST/$d/input.data"
done
# 1-rank config matching the earlier isolation tests (selection_mode=0)
for d in cpu_1r gpu_1r; do
  cp "$KF/input.nn" "$KF/scaling.data" "$TEST/$d/"
  cp "$KF/input_small.data" "$TEST/$d/input.data"
  sed -i 's/^selection_mode\s\+2/selection_mode                   0/' "$TEST/$d/input.nn"
done

echo "=== CPU, 4 ranks, ORIGINAL config ==="
cd "$TEST/cpu_orig"
mpirun -np 4 "$N2P2_CPU/bin/nnp-train" > run.out 2>&1 || true
grep -E "^ENERGY|^FORCE" run.out

echo "=== GPU, 4 ranks, ORIGINAL config (real fix applied) ==="
cd "$TEST/gpu_orig"
mpirun -np 4 "$N2P2_GPU/bin/nnp-train" > run.out 2>&1 || true
grep -E "^ENERGY|^FORCE" run.out

echo "=== CPU, 1 rank, selection_mode=0 ==="
cd "$TEST/cpu_1r"
mpirun -np 1 "$N2P2_CPU/bin/nnp-train" > run.out 2>&1 || true
grep -E "^ENERGY|^FORCE" run.out

echo "=== GPU, 1 rank, selection_mode=0 (real fix applied) ==="
cd "$TEST/gpu_1r"
mpirun -np 1 "$N2P2_GPU/bin/nnp-train" > run.out 2>&1 || true
grep -E "^ENERGY|^FORCE" run.out

echo "Done"
