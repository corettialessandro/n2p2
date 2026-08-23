#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --mem=32GB
#SBATCH --job-name=debug-water-force-full
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Follow-up to debug_water_force.slurm: that 20-structure subset matched
# CPU/GPU exactly, so this reruns on the FULL 140-structure test.data at
# the real 32-rank count, to reproduce and localize whichever structure
# triggers the divergence seen in the real training run.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_CPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water
N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force

echo "=== Running CPU nnp-dataset on full_test.data (140 structures, 32 ranks) ==="
cd "$DBG/cpu_run_full"
mpirun -np 32 "$N2P2_CPU/bin/nnp-dataset" 0 full_test.data

echo ""
echo "=== Running GPU nnp-dataset on full_test.data (140 structures, 32 ranks) ==="
cd "$DBG/gpu_run_full"
mpirun -np 32 "$N2P2_GPU/bin/nnp-dataset" 0 full_test.data

echo ""
echo "=== energy.comp diff ==="
diff <(grep -v '^#' "$DBG/cpu_run_full/energy.comp" | sort) <(grep -v '^#' "$DBG/gpu_run_full/energy.comp" | sort) | head -20
echo "(no output above = identical)"

echo ""
echo "=== forces.comp: (index_s, index_a) rows where predicted force (col 4) disagrees CPU vs GPU by > 1e-6 ==="
join -j 1 \
  <(grep -v '^#' "$DBG/cpu_run_full/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) \
  <(grep -v '^#' "$DBG/gpu_run_full/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) | \
awk '{d=$2-$3; if (d<0) d=-d; if (d>1e-6) print}' | sort -t_ -k1,1n | head -40
echo "--- distinct structure indices with a disagreement ---"
join -j 1 \
  <(grep -v '^#' "$DBG/cpu_run_full/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) \
  <(grep -v '^#' "$DBG/gpu_run_full/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) | \
awk '{d=$2-$3; if (d<0) d=-d; if (d>1e-6) {split($1,a,"_"); print a[1]}}' | sort -n -u

echo "Done"
