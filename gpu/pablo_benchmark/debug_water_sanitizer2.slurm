#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=debug-water-sanitizer2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_water
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force

mkdir -p "$DBG/sanitizer_run2" && cd "$DBG/sanitizer_run2"
cp "$DBG/gpu_run_1r_full"/{input.nn,scaling.data,weights.001.data,weights.008.data,full_test.data} .

echo "=== Running under compute-sanitizer (memcheck), launched via mpirun ==="
# UCX's CUDA transport probe (mca_pml_ucx) hits CUDA_ERROR_INVALID_CONTEXT
# under compute-sanitizer's driver-API interception, unrelated to our
# own code -- force the plain TCP/shared-memory transport so sanitizer
# only sees nnp-dataset's own CUDA calls.
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader
export OMPI_MCA_osc=^ucx
export UCX_TLS=^cuda,gdr_copy,cuda_copy,cuda_ipc
mpirun -np 1 compute-sanitizer --tool memcheck --error-exitcode 0 --log-file sanitizer_%p.log \
    "$N2P2_GPU/bin/nnp-dataset" 0 full_test.data || true

echo "--- log files ---"
ls sanitizer_*.log
for f in sanitizer_*.log; do
  echo "=== $f: ERROR SUMMARY ==="
  grep "ERROR SUMMARY" "$f"
done

echo ""
echo "--- first error block (if any) ---"
grep -n -B2 -A25 "^========= Invalid\|^========= Program hit\|^========= Race reported" sanitizer_*.log | head -80

echo "Done"
