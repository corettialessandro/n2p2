#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-water-racecheck
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# racecheck/synccheck are the sanitizer tools actually built for
# concurrency bugs (memcheck came back clean but the divergence
# disappeared under it -- classic race signature, but memcheck itself
# doesn't detect races, only invalid/OOB memory access).

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_CPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_water
N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force

echo "=== Fresh CPU baseline (debug dir was cleaned up earlier) ==="
mkdir -p "$DBG/cpu_run_4r1g_full" && cd "$DBG/cpu_run_4r1g_full"
cp "$DBG/gpu_run_1r_sanitize"/{input.nn,scaling.data,weights.001.data,weights.008.data,full_test.data} .
mpirun -np 1 "$N2P2_CPU/bin/nnp-dataset" 0 full_test.data

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader
export OMPI_MCA_osc=^ucx
export UCX_TLS=^cuda,gdr_copy,cuda_copy,cuda_ipc

for tool in racecheck synccheck; do
    echo ""
    echo "=== Running under compute-sanitizer --tool $tool ==="
    mkdir -p "$DBG/${tool}_run" && cd "$DBG/${tool}_run"
    cp "$DBG/gpu_run_1r_sanitize"/{input.nn,scaling.data,weights.001.data,weights.008.data,full_test.data} .
    mpirun -np 1 compute-sanitizer --tool "$tool" --error-exitcode 0 --log-file "sanitizer_%p.log" \
        "$N2P2_GPU/bin/nnp-dataset" 0 full_test.data > run.out 2>&1 || true
    echo "--- $tool: ERROR SUMMARY ---"
    grep "ERROR SUMMARY" sanitizer_*.log || echo "(no ERROR SUMMARY line found)"
    echo "--- $tool: first reported issue (if any) ---"
    grep -n -B2 -A20 "^========= Race reported\|^========= Barrier error\|^========= Program hit\|^========= Error" sanitizer_*.log | head -40
    echo "--- $tool: compare forces vs known-good CPU baseline ---"
    join -j 1 \
      <(grep -v '^#' "$DBG/cpu_run_4r1g_full/forces.comp" | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) \
      <(grep -v '^#' forces.comp | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) | \
    awk '{d=$2-$3; if (d<0) d=-d; if (d>1e-6) {split($1,a,"_"); print a[1]}}' | sort -n -u | wc -l
    echo "(count of disagreeing structures out of 140, under $tool)"
done

echo "Done"
