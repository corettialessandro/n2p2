#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-kf-trace-sm0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Under selection_mode=0 (pure round-robin), water's GPU training STILL
# explodes -- ruling out the candidate-selection feedback loop as the
# cause (see debug_kf_selection_mode.slurm). Since round-robin selection
# is 100% deterministic and identical between CPU/GPU builds (each rank
# walks its own pre-shuffled list with the same seed, no live GPU-
# computed value feeds the decision), whichever candidate each rank
# picks must now be IDENTICAL on CPU and GPU. So any divergence found
# in this trace is a genuine numerical difference in the actual
# Jacobian computation (gpuNnEnergyDEdcSum/gpuNnForceDFdcSum) or the
# Kalman update math (GpuKalmanFilter) for a known-identical candidate
# -- not a selection artifact. Dumps pu.error/pu.jacobian/weights
# before+after each of the first several updates on both CPU and GPU
# builds (both with the ts-fix'd Dataset.cpp), for direct comparison.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_CPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water
N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force
KF="$DBG/kf_trace"
TRACE="$KF/trace_sm0"

echo "=== Clean rebuild: CPU nnp-train (ts-fixed, instrumented) ==="
(cd "$N2P2_CPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_CPU/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" nnp-train > "$TRACE.build-cpu.log" 2>&1) || \
    (echo "--- CPU build failed ---" && tail -60 "$TRACE.build-cpu.log" && exit 1)
git -C "$N2P2_CPU" checkout -- src/libnnp/version.h

echo "=== Clean rebuild: GPU nnp-train (ts-fixed, instrumented, no ASan) ==="
(cd "$N2P2_GPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_GPU/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-train > "$TRACE.build-gpu.log" 2>&1) || \
    (echo "--- GPU build failed ---" && tail -60 "$TRACE.build-gpu.log" && exit 1)
git -C "$N2P2_GPU" checkout -- src/libnnp/version.h

mkdir -p "$TRACE/cpu_run" "$TRACE/gpu_run"
cp "$KF/input.nn" "$KF/scaling.data" "$TRACE/cpu_run/"
cp "$KF/input_small.data" "$TRACE/cpu_run/input.data"
cp "$KF/input.nn" "$KF/scaling.data" "$TRACE/gpu_run/"
cp "$KF/input_small.data" "$TRACE/gpu_run/input.data"

sed -i 's/^selection_mode\s\+2/selection_mode                   0/' "$TRACE/cpu_run/input.nn" "$TRACE/gpu_run/input.nn"
grep -n "^selection_mode" "$TRACE/cpu_run/input.nn" "$TRACE/gpu_run/input.nn"

echo "=== CPU run (selection_mode=0, dumping first 10 updates of each property) ==="
cd "$TRACE/cpu_run"
export N2P2_DUMP_UPDATE_FILE="$TRACE/cpu_run/dump"
export N2P2_DUMP_UPDATE_MAXCALLS=10
mpirun -np 4 "$N2P2_CPU/bin/nnp-train" > run.out 2>&1 || true
unset N2P2_DUMP_UPDATE_FILE N2P2_DUMP_UPDATE_MAXCALLS
echo "--- ENERGY/FORCE (cpu, sm0) ---"
grep -E "^ENERGY|^FORCE" run.out

echo "=== GPU run (selection_mode=0, dumping first 10 updates of each property) ==="
cd "$TRACE/gpu_run"
export N2P2_DUMP_UPDATE_FILE="$TRACE/gpu_run/dump"
export N2P2_DUMP_UPDATE_MAXCALLS=10
mpirun -np 4 "$N2P2_GPU/bin/nnp-train" > run.out 2>&1 || true
unset N2P2_DUMP_UPDATE_FILE N2P2_DUMP_UPDATE_MAXCALLS
echo "--- ENERGY/FORCE (gpu, sm0) ---"
grep -E "^ENERGY|^FORCE" run.out

echo "Done"
