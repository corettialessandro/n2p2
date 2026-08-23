#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-h2o2g-check
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Does the GpuForces divergence found in water/magnetite/feldspar also
# affect H2O_2G -- the original reference system this whole GPU port
# was validated against (100-epoch, "17.8x, no drift")? That earlier
# validation only checked aggregate test-set RMSE (0.08% CPU-vs-GPU
# difference); this does the same direct per-structure check used to
# catch the bug in the other three systems. Reuses the already-built
# CPU/GPU nnp-dataset binaries from the water worktrees -- H2O_2G uses
# the same architecture family (tanh, 2x25 hidden nodes), no rebuild
# needed.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_CPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water
N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force/h2o2g_check

echo "=== CPU nnp-dataset (H2O_2G, epoch-0 weights, 118 test structures) ==="
cd "$DBG/cpu_run"
mpirun -np 4 "$N2P2_CPU/bin/nnp-dataset" 0 test.data

echo ""
echo "=== GPU nnp-dataset (H2O_2G, epoch-0 weights, 4 ranks/1 GPU, no wrapper) ==="
cd "$DBG/gpu_run"
mpirun -np 4 "$N2P2_GPU/bin/nnp-dataset" 0 test.data

echo ""
echo "=== energy.comp diff ==="
diff <(grep -v '^#' "$DBG/cpu_run/energy.comp" | sort) <(grep -v '^#' "$DBG/gpu_run/energy.comp" | sort) | head -10
echo "(no output above = identical)"

echo ""
echo "=== forces.comp: distinct structure indices with a disagreement (>1e-6), out of 118 ==="
join -j 1 \
  <(grep -v '^#' "$DBG/cpu_run/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) \
  <(grep -v '^#' "$DBG/gpu_run/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) | \
awk '{d=$2-$3; if (d<0) d=-d; if (d>1e-6) {split($1,a,"_"); print a[1]}}' | sort -n -u > "$DBG/bad_indices.txt"
wc -l "$DBG/bad_indices.txt"

echo ""
echo "=== magnitude of force disagreements ==="
join -j 1 \
  <(grep -v '^#' "$DBG/cpu_run/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) \
  <(grep -v '^#' "$DBG/gpu_run/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) | \
awk '{d=$2-$3; if (d<0) d=-d; print d}' | sort -g | awk '{a[NR]=$1} END{print "count="NR, "median="a[int(NR/2)], "max="a[NR]}'

echo "Done"
