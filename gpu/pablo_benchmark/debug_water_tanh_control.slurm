#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-water-tanh-control
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Control test: is the GpuForces divergence specific to softplus, or
# does it reproduce identically with tanh (same architecture: 2 hidden
# layers, 25/25 nodes, same H/O elements, same dataset)? GpuForces.cu
# only consumes already-computed dEdG -- if the bug is really isolated
# there (as the investigation so far suggests: energy always correct,
# sync fix didn't help, bounds-check pending), it should be completely
# activation-independent.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_CPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water
N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force
TANH="$DBG/tanh_control"

echo "=== Build nnp-train in the CPU worktree (only nnp-dataset was built there before) ==="
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
(cd "$N2P2_CPU/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" nnp-train > "$DBG/build-cpu-train.log" 2>&1)
git -C "$N2P2_CPU" checkout -- src/libnnp/version.h

echo "=== 1-epoch CPU run (tanh) to get epoch-0 initial weights ==="
cd "$TANH"
mpirun -np 4 "$N2P2_CPU/bin/nnp-train" > train.log 2>&1 || tail -50 train.log
ls weights.001.000000.out weights.008.000000.out

cp "$DBG/gpu_run_1r_sanitize/full_test.data" .
cp weights.001.000000.out weights.001.data
cp weights.008.000000.out weights.008.data

mkdir -p cpu_run gpu_run
cp input.nn scaling.data weights.001.data weights.008.data full_test.data cpu_run/
cp input.nn scaling.data weights.001.data weights.008.data full_test.data gpu_run/

echo ""
echo "=== CPU nnp-dataset (tanh, epoch-0 weights, full 140 structures) ==="
cd "$TANH/cpu_run"
mpirun -np 1 "$N2P2_CPU/bin/nnp-dataset" 0 full_test.data

echo ""
echo "=== GPU nnp-dataset (tanh, epoch-0 weights, 4 ranks/1 GPU, no wrapper, full 140 structures) ==="
cd "$TANH/gpu_run"
mpirun -np 4 "$N2P2_GPU/bin/nnp-dataset" 0 full_test.data

echo ""
echo "=== energy.comp diff ==="
diff <(grep -v '^#' "$TANH/cpu_run/energy.comp" | sort) <(grep -v '^#' "$TANH/gpu_run/energy.comp" | sort) | head -10
echo "(no output above = identical)"

echo ""
echo "=== forces.comp: distinct structure indices with a disagreement (>1e-6) ==="
join -j 1 \
  <(grep -v '^#' "$TANH/cpu_run/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) \
  <(grep -v '^#' "$TANH/gpu_run/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) | \
awk '{d=$2-$3; if (d<0) d=-d; if (d>1e-6) {split($1,a,"_"); print a[1]}}' | sort -n -u | wc -l
echo "(count of disagreeing structures out of 140, with TANH instead of softplus)"

echo "Done"
