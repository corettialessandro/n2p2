#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-kf-trace
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Water's real nnp-train GPU run diverges to NaN during training, but
# GpuForces.cu (force prediction) is now confirmed correct. The actual
# weight UPDATE during Kalman-filter training uses two different GPU
# kernels never examined before: gpuNnEnergyDEdcSum/gpuNnForceDFdcSum
# (Jacobians w.r.t. weights) and GpuKalmanFilter (the Kalman gain/
# covariance update itself). Dumps the Jacobian/error/weights
# immediately before and after each of the first few weight updates,
# for both CPU and GPU builds, on identical water config/weights/seed.

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

echo "=== Clean rebuild: CPU nnp-train (instrumented) ==="
(cd "$N2P2_CPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_CPU/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" nnp-train > "$KF/build-cpu.log" 2>&1)
git -C "$N2P2_CPU" checkout -- src/libnnp/version.h

echo "=== Clean rebuild: GPU nnp-train (instrumented) ==="
(cd "$N2P2_GPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_GPU/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-train > "$KF/build-gpu.log" 2>&1)
git -C "$N2P2_GPU" checkout -- src/libnnp/version.h

grep -c "N2P2_DUMP_UPDATE_FILE" "$N2P2_CPU/src/libnnptrain/Training.cpp" "$N2P2_GPU/src/libnnptrain/Training.cpp"
echo "(sanity check: both should be >0)"

mkdir -p "$KF/cpu_run" "$KF/gpu_run"
cp "$KF/input.nn" "$KF/input.data" "$KF/scaling.data" "$KF/cpu_run/"
cp "$KF/input.nn" "$KF/input.data" "$KF/scaling.data" "$KF/gpu_run/"

echo "=== CPU run (dumping first 5 updates of each property) ==="
cd "$KF/cpu_run"
export N2P2_DUMP_UPDATE_FILE="$KF/cpu_run/dump"
export N2P2_DUMP_UPDATE_MAXCALLS=5
mpirun -np 4 "$N2P2_CPU/bin/nnp-train"
unset N2P2_DUMP_UPDATE_FILE N2P2_DUMP_UPDATE_MAXCALLS

echo "=== GPU run (dumping first 5 updates of each property) ==="
cd "$KF/gpu_run"
export N2P2_DUMP_UPDATE_FILE="$KF/gpu_run/dump"
export N2P2_DUMP_UPDATE_MAXCALLS=5
mpirun -np 4 "$N2P2_GPU/bin/nnp-train"
unset N2P2_DUMP_UPDATE_FILE N2P2_DUMP_UPDATE_MAXCALLS

echo ""
echo "=== ENERGY/FORCE RMSE trajectory: CPU ==="
grep -E "^ENERGY|^FORCE" "$KF/cpu_run"/*.out 2>/dev/null || grep -E "^ENERGY|^FORCE" "$KF/cpu_run/train-log.out" 2>/dev/null || true
echo "=== ENERGY/FORCE RMSE trajectory: GPU ==="
grep -E "^ENERGY|^FORCE" "$KF/gpu_run"/*.out 2>/dev/null || grep -E "^ENERGY|^FORCE" "$KF/gpu_run/train-log.out" 2>/dev/null || true

echo "Done"
