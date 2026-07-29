#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:25:00        # format: HH:MM:SS
#SBATCH -N 1                   # 1 node
#SBATCH --ntasks-per-node=1    # 1 task
#SBATCH --gres=gpu:1           # 1 gpu
#SBATCH --mem=8GB
#SBATCH --job-name=profile-dfdc
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load cuda/12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0

# Rebuild libnnp.a fresh on THIS node (see gpu/README.md's Phase 6 section
# and run_libnnpgpu_dfdc.slurm for why).
(cd "$N2P2_DEVEL/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" libnnp)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h

cd "$N2P2_DEVEL/gpu/gemm"

echo "Compiling src/libnnpgpu/GpuNeuralNetwork.cu (with -lineinfo for ncu source correlation)..."
nvcc -O3 -lineinfo -std=c++14 -arch=sm_80 -I"$N2P2_DEVEL/include" \
    -c -o GpuNeuralNetwork_prof.o "$N2P2_DEVEL/src/libnnpgpu/GpuNeuralNetwork.cu"

echo "Compiling profile_dfdc.cu..."
nvcc -O3 -lineinfo -std=c++14 -arch=sm_80 -I"$N2P2_DEVEL/include" -I"$EIGEN" -I"$N2P2_DEVEL/src/libnnpgpu" \
    -DEIGEN_DONT_PARALLELIZE \
    -o profile_dfdc GpuNeuralNetwork_prof.o profile_dfdc.cu \
    "$N2P2_DEVEL/lib/libnnp.a" -lcublas

echo "=== Sanity run (no profiler) ==="
./profile_dfdc

echo
echo "=== nsys: full timeline of the single bracketed call ==="
nsys profile --force-overwrite=true \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o nsys_dfdc ./profile_dfdc

echo
echo "--- nsys stats: CUDA API summary ---"
nsys stats --report cuda_api_sum nsys_dfdc.nsys-rep

echo
echo "--- nsys stats: GPU kernel summary (time, count, avg per launch) ---"
nsys stats --report cuda_gpu_kern_sum nsys_dfdc.nsys-rep

echo
echo "--- nsys stats: memcpy summary ---"
nsys stats --report cuda_gpu_mem_time_sum nsys_dfdc.nsys-rep

echo
echo "=== ncu: lightweight per-kernel-instance timeline of the same call ==="
echo "(single metric, one pass -- avoids multi-pass replay blowing up runtime"
echo " across the ~hundreds of kernel launches in one gpuNnForceDFdcSum() call;"
echo " --print-summary per-kernel aggregates repeated kernel names for us)"
ncu --profile-from-start off --clock-control none \
    --metrics gpu__time_duration.sum \
    --print-summary per-kernel \
    -o ncu_dfdc -f \
    ./profile_dfdc > ncu_dfdc_report.txt
cat ncu_dfdc_report.txt

echo
echo "Done"
