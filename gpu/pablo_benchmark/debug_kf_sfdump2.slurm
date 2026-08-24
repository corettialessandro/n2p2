#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --job-name=debug-kf-sfdump2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Structure 0's G values matched between CPU/GPU (clean structure).
# Target structure 164, atom 110 instead -- one of the WORST-offending
# rows in the earlier row-aligned per-atom force comparison
# (Fnnp CPU -4.008 vs GPU -27.03). Dump G (pre-NN), dEdG and final
# force (post-force) for this exact atom, on both CPU and GPU builds,
# to see whether the divergence originates in the symmetry function
# values (G) themselves, in the NN backprop (dEdG), or only appears
# after calculateForces combines dEdG with dGdr.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_CPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water
N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
KF=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force/kf_trace
TEST="$KF/sfdump2_test"

echo "=== Clean rebuild: CPU nnp-train (plain, SF-dump v2 instrumented) ==="
(cd "$N2P2_CPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_CPU/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" nnp-train > "$TEST.build-cpu.log" 2>&1) || \
    (echo "--- CPU build failed ---" && tail -60 "$TEST.build-cpu.log" && exit 1)
git -C "$N2P2_CPU" checkout -- src/libnnp/version.h

echo "=== Clean rebuild: GPU nnp-train (plain, SF-dump v2 instrumented) ==="
(cd "$N2P2_GPU/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_GPU/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-train > "$TEST.build-gpu.log" 2>&1) || \
    (echo "--- GPU build failed ---" && tail -60 "$TEST.build-gpu.log" && exit 1)
git -C "$N2P2_GPU" checkout -- src/libnnp/version.h

mkdir -p "$TEST/cpu" "$TEST/gpu"
cp "$KF/input.nn" "$KF/scaling.data" "$TEST/cpu/"
cp "$KF/input_small.data" "$TEST/cpu/input.data"
cp "$KF/input.nn" "$KF/scaling.data" "$TEST/gpu/"
cp "$KF/input_small.data" "$TEST/gpu/input.data"
sed -i 's/^selection_mode\s\+2/selection_mode                   0/' "$TEST/cpu/input.nn" "$TEST/gpu/input.nn"

echo "=== CPU nnp-train, 1 rank, SF dump struct=164 atom=110 ==="
cd "$TEST/cpu"
export N2P2_DUMP_SF_FILE="$TEST/cpu/sfdump"
export N2P2_DUMP_SF_STRUCT_INDEX=164
export N2P2_DUMP_SF_ATOM_INDEX=110
mpirun -np 1 "$N2P2_CPU/bin/nnp-train" > run.out 2>&1 || true
unset N2P2_DUMP_SF_FILE N2P2_DUMP_SF_STRUCT_INDEX N2P2_DUMP_SF_ATOM_INDEX
echo "--- SF dump pre (CPU) ---"
cat "$TEST/cpu/sfdump.pre"
echo "--- SF dump post (CPU) ---"
cat "$TEST/cpu/sfdump.post"

echo "=== GPU nnp-train, 1 rank, SF dump struct=164 atom=110 ==="
cd "$TEST/gpu"
export N2P2_DUMP_SF_FILE="$TEST/gpu/sfdump"
export N2P2_DUMP_SF_STRUCT_INDEX=164
export N2P2_DUMP_SF_ATOM_INDEX=110
mpirun -np 1 "$N2P2_GPU/bin/nnp-train" > run.out 2>&1 || true
unset N2P2_DUMP_SF_FILE N2P2_DUMP_SF_STRUCT_INDEX N2P2_DUMP_SF_ATOM_INDEX
echo "--- SF dump pre (GPU) ---"
cat "$TEST/gpu/sfdump.pre"
echo "--- SF dump post (GPU) ---"
cat "$TEST/gpu/sfdump.post"

echo "Done"
