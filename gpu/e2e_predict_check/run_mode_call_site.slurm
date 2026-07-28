#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --job-name=mode-gpu-call-site-test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load cuda/12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
CHK="$N2P2_DEVEL/gpu/e2e_predict_check"
TEMP="$N2P2_DEVEL/temp/H2O_2G"
cd "$CHK"

# input.nn/scaling.data/weights.*.data must sit in Prediction::setup()'s
# working directory (it doesn't take a path); these are small (~100KB total)
# so copying them is fine. input.data is NOT copied -- it's 117 MB (the full
# 1254-structure training set, this project's usual temp/H2O_2G/input.data)
# and Structure::readFromFile() only ever reads the first structure from it
# anyway, so mode_gpu_call_site_test.cpp takes its directory as an argument
# instead and reads it in place.
cp "$TEMP/input.nn" "$TEMP/scaling.data" .
cp "$TEMP/weights.001.000002.out" weights.001.data
cp "$TEMP/weights.008.000002.out" weights.008.data

echo "=== Building libnnp.a WITHOUT GPU (CPU baseline) ==="
rm -f "$N2P2_DEVEL/src/libnnp/Mode.o"
(cd "$N2P2_DEVEL/src/libnnp" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" > /dev/null)
cp "$N2P2_DEVEL/lib/libnnp.a" libnnp-cpu.a

echo "=== Building libnnp.a + libnnpgpu.a WITH GPU=1 ==="
rm -f "$N2P2_DEVEL/src/libnnp/Mode.o"
(cd "$N2P2_DEVEL/src/libnnpgpu" && make COMP=gnu MODE=static GPU=1 > /dev/null)
(cd "$N2P2_DEVEL/src/libnnp" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" > /dev/null)
cp "$N2P2_DEVEL/lib/libnnp.a" libnnp-gpu.a
cp "$N2P2_DEVEL/lib/libnnpgpu.a" .

echo "=== Compiling mode_gpu_call_site_test.cpp (CPU build) ==="
g++ -O3 -std=c++11 -I"$N2P2_DEVEL/include" -I"$EIGEN" -DEIGEN_DONT_PARALLELIZE \
    -o mode_test_cpu.o -c mode_gpu_call_site_test.cpp
g++ -O3 -std=c++11 -o mode_test_cpu mode_test_cpu.o libnnp-cpu.a

echo "=== Compiling mode_gpu_call_site_test.cpp (GPU build) ==="
CUDA_LIB="$CUDA_HOME/lib64"
g++ -O3 -std=c++11 -I"$N2P2_DEVEL/include" -I"$EIGEN" -DEIGEN_DONT_PARALLELIZE \
    -o mode_test_gpu.o -c mode_gpu_call_site_test.cpp
g++ -O3 -std=c++11 -o mode_test_gpu mode_test_gpu.o libnnp-gpu.a libnnpgpu.a \
    -L"$CUDA_LIB" -lcudart -lcublas

echo ""
echo "=== Running CPU build ==="
./mode_test_cpu "$TEMP" > out_cpu.txt

echo ""
echo "=== Running GPU build ==="
./mode_test_gpu "$TEMP" > out_gpu.txt

echo ""
echo "=== Comparing (numeric tolerance -- text diff alone would fail on the"
echo "compile-timestamp banner line and on ~1e-13 floating-point summation-"
echo "order noise between cuBLAS and the sequential CPU loop, same as every"
echo "other GPU-vs-CPU comparison in this project) ==="
paste <(grep "^atom " out_cpu.txt) <(grep "^atom " out_gpu.txt) | awk '
{
    split($3, e1, "="); split($8, e2, "=");
    split($4, d1a, "="); split($9, d2a, "=");
    split($5, d1b, "="); split($10, d2b, "=");
    de = e1[2]-e2[2]; if (de<0) de=-de;
    dda = d1a[2]-d2a[2]; if (dda<0) dda=-dda;
    ddb = d1b[2]-d2b[2]; if (ddb<0) ddb=-ddb;
    if (de>maxE) maxE=de;
    if (dda>maxD) maxD=dda;
    if (ddb>maxD) maxD=ddb;
}
END {
    printf "max|E_cpu-E_gpu|=%.3E  max|dEdG_cpu-dEdG_gpu|=%.3E\n", maxE, maxD;
    pass = (maxE < 1e-9 && maxD < 1e-9);
    print pass ? "ALL PASS" : "SOME FAILED";
    exit !pass;
}'

echo "Done"
