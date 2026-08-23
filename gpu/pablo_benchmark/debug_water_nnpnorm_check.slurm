#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-water-nnpnorm-check
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Does water's OWN, genuinely-computed nnp-norm calibration (from
# reference/DFT statistics, conv_length=51.4/conv_energy=61.6 -- not
# borrowed from H2O_2G) also fix the CPU-vs-GPU force divergence?
# Same epoch-0 weights, same data, only the normalization source
# differs from the earlier (borrowed-values) test.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_CPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water
N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force/nnpnorm_test

echo "=== CPU nnp-dataset (water, nnp-norm's own calibration) ==="
cd "$DBG/cpu_run"
mpirun -np 4 "$N2P2_CPU/bin/nnp-dataset" 0 full_test.data

echo ""
echo "=== GPU nnp-dataset (water, nnp-norm's own calibration) ==="
cd "$DBG/gpu_run"
mpirun -np 4 "$N2P2_GPU/bin/nnp-dataset" 0 full_test.data

echo ""
echo "=== energy.comp diff ==="
diff <(grep -v '^#' "$DBG/cpu_run/energy.comp" | sort) <(grep -v '^#' "$DBG/gpu_run/energy.comp" | sort) | head -10
echo "(no output above = identical)"

echo ""
echo "=== forces.comp: distinct structure indices with a disagreement (>1e-6), out of 140 ==="
join -j 1 \
  <(grep -v '^#' "$DBG/cpu_run/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) \
  <(grep -v '^#' "$DBG/gpu_run/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) | \
awk '{d=$2-$3; if (d<0) d=-d; if (d>1e-6) {split($1,a,"_"); print a[1]}}' | sort -n -u | wc -l

echo ""
echo "=== magnitude of force disagreements ==="
join -j 1 \
  <(grep -v '^#' "$DBG/cpu_run/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) \
  <(grep -v '^#' "$DBG/gpu_run/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) | \
awk '{d=$2-$3; if (d<0) d=-d; print d}' | sort -g | awk '{a[NR]=$1} END{if(NR>0) print "count="NR, "median="a[int(NR/2)], "max="a[NR]; else print "count=0"}'

echo "Done"
