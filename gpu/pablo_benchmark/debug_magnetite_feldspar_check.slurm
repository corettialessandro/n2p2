#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --job-name=debug-mag-feld-check
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Water's GpuForces divergence turned out to be activation-independent
# (tanh reproduces it identically to softplus) -- which raises real
# doubt about whether magnetite/feldspar's GPU results, only checked at
# the aggregate epoch-RMSE level so far, are actually clean or just
# diluted (a similar fraction of corrupted structures would barely move
# an RMSE averaged over 12M+/2.9M+ force components, unlike water's
# 140-structure test set where 140/140 failing was impossible to miss).
# Direct per-structure force check, same methodology as water's.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
RESULTS=/leonardo_work/L-AUT_Giane_26/acoretti/NEURALCPM/gpu_porting/pablo_benchmark
DBG=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel/gpu/pablo_benchmark/debug_water_force
mkdir -p "$DBG/mag_feld_check" && cd "$DBG/mag_feld_check"

for sys in magnetite feldspar; do
    echo ""
    echo "############################################################"
    echo "### $sys"
    echo "############################################################"
    N2P2_CPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_cpu_${sys}
    N2P2_GPU=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel_pablo_booster_gpu_${sys}
    RUN10EP="$RESULTS/${sys}_booster-gpu-mn2_10ep"

    echo "=== Build nnp-dataset (CPU worktree already has nnp-train) ==="
    (cd "$N2P2_CPU/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" nnp-dataset > "$DBG/build-cpu-dataset-${sys}.log" 2>&1)
    git -C "$N2P2_CPU" checkout -- src/libnnp/version.h

    echo "=== Build nnp-dataset (GPU worktree) ==="
    (cd "$N2P2_GPU/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-dataset > "$DBG/build-gpu-dataset-${sys}.log" 2>&1)
    git -C "$N2P2_GPU" checkout -- src/libnnp/version.h

    mkdir -p "$sys" && cd "$sys"
    cp "$RUN10EP/input.nn" "$RUN10EP/scaling.data" .
    cp "$RUN10EP"/weights.*.000000.out .
    for f in weights.*.000000.out; do
        newname=$(echo "$f" | sed -E 's/(weights\.[0-9]+)\.000000\.out/\1.data/')
        cp "$f" "$newname"
    done
    awk '/^begin/{c++} c<=150' "$RUN10EP/test.data" > small_test.data
    grep -c "^begin" small_test.data

    mkdir -p cpu_run gpu_run
    cp input.nn scaling.data weights.*.data small_test.data cpu_run/
    cp input.nn scaling.data weights.*.data small_test.data gpu_run/

    echo "=== CPU nnp-dataset ($sys, epoch-0 weights, 150-structure subset) ==="
    cd cpu_run
    mpirun -np 4 "$N2P2_CPU/bin/nnp-dataset" 0 small_test.data
    cd ..

    echo "=== GPU nnp-dataset ($sys, epoch-0 weights, 4 ranks/1 GPU, no wrapper) ==="
    cd gpu_run
    mpirun -np 4 "$N2P2_GPU/bin/nnp-dataset" 0 small_test.data
    cd ..

    echo "=== $sys: energy.comp diff ==="
    diff <(grep -v '^#' cpu_run/energy.comp | sort) <(grep -v '^#' gpu_run/energy.comp | sort) | head -10
    echo "(no output above = identical)"

    echo "=== $sys: forces.comp disagreement count (>1e-6), out of 150 structures ==="
    join -j 1 \
      <(grep -v '^#' cpu_run/forces.comp | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) \
      <(grep -v '^#' gpu_run/forces.comp | awk '{printf "%s_%s %s\n",$1,$2,$4}' | sort -k1,1) | \
    awk '{d=$2-$3; if (d<0) d=-d; if (d>1e-6) {split($1,a,"_"); print a[1]}}' | sort -n -u | wc -l

    cd "$DBG/mag_feld_check"
done

echo "Done"
