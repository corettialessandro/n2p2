#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-water-sanitizer
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Single rank, full 140-structure dataset -- simplest possible
# reproduction of the GpuForces divergence (no MPI/rank-count
# complexity). Run first plain (confirm repro), then under
# compute-sanitizer to catch the actual memory error directly instead
# of reasoning blind from code inspection.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force

echo "=== Plain single-rank GPU run (confirm repro) ==="
cd "$DBG/gpu_run_1r_full"
mpirun -np 1 "$N2P2_GPU/bin/nnp-dataset" 0 full_test.data
echo "--- compare vs. the known-correct CPU forces.comp (4-rank run, same weights/data) ---"
join -j 1 \
  <(grep -v '^#' "$DBG/cpu_run_4r1g_full/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) \
  <(grep -v '^#' forces.comp | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) | \
awk '{d=$2-$3; if (d<0) d=-d; if (d>1e-6) {split($1,a,"_"); print a[1]}}' | sort -n -u | wc -l
echo "(count of structures disagreeing at 1 rank, out of 140 -- 0 means bug does NOT reproduce at 1 rank)"

echo ""
echo "=== compute-sanitizer availability ==="
which compute-sanitizer || echo "not found"

echo ""
echo "=== Running under compute-sanitizer (memcheck) ==="
mkdir -p "$DBG/sanitizer_run" && cd "$DBG/sanitizer_run"
cp "$DBG/gpu_run_1r_full"/{input.nn,scaling.data,weights.001.data,weights.008.data,full_test.data} .
compute-sanitizer --tool memcheck --error-exitcode 0 "$N2P2_GPU/bin/nnp-dataset" 0 full_test.data > sanitizer.log 2>&1 || true
echo "--- sanitizer.log summary ---"
grep -E "ERROR SUMMARY|Invalid|out-of-bounds|Malloc|Free" sanitizer.log | sort | uniq -c | sort -rn | head -30
echo "--- first error block ---"
grep -n -A20 "^========= Invalid\|^========= Program hit" sanitizer.log | head -60

echo "Done"
