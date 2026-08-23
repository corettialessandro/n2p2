#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-water-4r1g-full
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Last isolation step: exact original working config (4 ranks, 1 GPU,
# no wrapper/MPS -- identical to debug_water_force.slurm which matched
# CPU/GPU exactly), but with the FULL 140-structure test.data instead of
# the 20-structure subset. Pure test of dataset size as the variable,
# holding rank count / GPU count / wrapper presence fixed at the known
# good configuration.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_CPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water
N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force

echo "=== CPU nnp-dataset on full_test.data (140 structures, 4 ranks) ==="
cd "$DBG/cpu_run_4r1g_full"
mpirun -np 4 "$N2P2_CPU/bin/nnp-dataset" 0 full_test.data

echo ""
echo "=== GPU nnp-dataset on full_test.data (140 structures, 4 ranks, 1 GPU, no wrapper) ==="
cd "$DBG/gpu_run_4r1g_full"
mpirun -np 4 "$N2P2_GPU/bin/nnp-dataset" 0 full_test.data

echo ""
echo "=== energy.comp diff ==="
diff <(grep -v '^#' "$DBG/cpu_run_4r1g_full/energy.comp" | sort) <(grep -v '^#' "$DBG/gpu_run_4r1g_full/energy.comp" | sort) | head -20
echo "(no output above = identical)"

echo ""
echo "=== forces.comp: distinct structure indices with a disagreement (>1e-6) ==="
join -j 1 \
  <(grep -v '^#' "$DBG/cpu_run_4r1g_full/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) \
  <(grep -v '^#' "$DBG/gpu_run_4r1g_full/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) | \
awk '{d=$2-$3; if (d<0) d=-d; if (d>1e-6) {split($1,a,"_"); print a[1]}}' | sort -n -u > bad_indices_4r1g.txt
wc -l bad_indices_4r1g.txt
head -20 bad_indices_4r1g.txt
echo "(list of distinct structures with disagreement, out of 140)"

echo "Done"
