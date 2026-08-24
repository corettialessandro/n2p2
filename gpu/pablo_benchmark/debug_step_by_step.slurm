#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --job-name=debug-step-by-step
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Step-by-step trace: for structure 0 of water and of H2O_2G (both
# using their own genuine nnp-norm normalization header), dump the
# exact topology/dEdG fed into GpuForces.cu and the GPU's raw output
# force, plus the trusted CPU force for the same structure. Lets us
# manually recompute the expected force (in Python, double precision,
# same formula as selfForceKernel/pairForceKernel) from the dumped
# inputs and see precisely where CPU-trusted, "recomputed from dumped
# inputs", and "actual GPU output" start to disagree, for each system.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_CPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water
N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force
STEP="$DBG/step_by_step"
mkdir -p "$STEP"

echo "=== Rebuild instrumented GPU nnp-dataset (clean, to force picking up the Mode.cpp edit) ==="
(cd "$N2P2_GPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_GPU/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-dataset > "$DBG/build-step-gpu.log" 2>&1)
git -C "$N2P2_GPU" checkout -- src/libnnp/version.h
grep -c "N2P2_DUMP_STRUCT_FILE" "$N2P2_GPU/src/libnnp/Mode.cpp"
echo "(sanity check: should be >0, confirming the instrumented source is what got built)"

# ---------- WATER (nnp-norm header, own calibration) ----------
mkdir -p "$STEP/water_cpu" "$STEP/water_gpu"
cp "$DBG/nnpnorm_test/input.nn" "$DBG/nnpnorm_test/scaling.data" \
   "$DBG/nnpnorm_test/weights.001.data" "$DBG/nnpnorm_test/weights.008.data" \
   "$DBG/full_test.data" "$STEP/water_cpu/"
cp "$DBG/nnpnorm_test/input.nn" "$DBG/nnpnorm_test/scaling.data" \
   "$DBG/nnpnorm_test/weights.001.data" "$DBG/nnpnorm_test/weights.008.data" \
   "$DBG/full_test.data" "$STEP/water_gpu/"

echo "=== CPU nnp-dataset: water, structure 0 (trusted force) ==="
cd "$STEP/water_cpu"
mpirun -np 1 "$N2P2_CPU/bin/nnp-dataset" 0 full_test.data

echo "=== GPU nnp-dataset (instrumented dump): water, structure 0 ==="
cd "$STEP/water_gpu"
export N2P2_DUMP_STRUCT_FILE="$STEP/water_gpu/dump_struct0.txt"
export N2P2_DUMP_STRUCT_INDEX=0
mpirun -np 1 "$N2P2_GPU/bin/nnp-dataset" 0 full_test.data
unset N2P2_DUMP_STRUCT_FILE N2P2_DUMP_STRUCT_INDEX

echo "--- water structure 0: trusted CPU force ---"
awk '$1==0' "$STEP/water_cpu/forces.comp" || true

# ---------- H2O_2G (nnp-norm header, own calibration) ----------
mkdir -p "$STEP/h2o2g_cpu" "$STEP/h2o2g_gpu"
H2O2G=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/temp/H2O_2G
cp "$H2O2G/input.nn" "$H2O2G/scaling.data" "$H2O2G/weights.001.000000.out" \
   "$H2O2G/weights.008.000000.out" "$H2O2G/test.data" "$STEP/h2o2g_cpu/"
mv "$STEP/h2o2g_cpu/weights.001.000000.out" "$STEP/h2o2g_cpu/weights.001.data"
mv "$STEP/h2o2g_cpu/weights.008.000000.out" "$STEP/h2o2g_cpu/weights.008.data"
cp "$STEP/h2o2g_cpu"/{input.nn,scaling.data,weights.001.data,weights.008.data,test.data} "$STEP/h2o2g_gpu/"

echo "=== CPU nnp-dataset: H2O_2G, structure 0 (trusted force) ==="
cd "$STEP/h2o2g_cpu"
mpirun -np 1 "$N2P2_CPU/bin/nnp-dataset" 0 test.data

echo "=== GPU nnp-dataset (instrumented dump): H2O_2G, structure 0 ==="
cd "$STEP/h2o2g_gpu"
export N2P2_DUMP_STRUCT_FILE="$STEP/h2o2g_gpu/dump_struct0.txt"
export N2P2_DUMP_STRUCT_INDEX=0
mpirun -np 1 "$N2P2_GPU/bin/nnp-dataset" 0 test.data
unset N2P2_DUMP_STRUCT_FILE N2P2_DUMP_STRUCT_INDEX

echo "--- H2O_2G structure 0: trusted CPU force ---"
awk '$1==0' "$STEP/h2o2g_cpu/forces.comp" || true

echo "Done"
