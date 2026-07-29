#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#SBATCH --job-name=nnp-train-gpu-vs-cpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
TEMP="$N2P2_DEVEL/temp/H2O_2G"
CHK="$N2P2_DEVEL/gpu/e2e_train_check"
cd "$CHK"

rm -rf run-e2e && mkdir run-e2e && cd run-e2e

# Real production H2O_2G dataset/architecture, but a single epoch is
# enough to check numerical correctness of the energy-Jacobian GPU path
# (not a performance run -- Training::update()'s energy branch is a tiny
# fraction of total epoch time anyway, see timing.out in ../../temp/H2O_2G,
# dominated by the (still CPU-only) force-Jacobian update).
mkdir run-cpu run-gpu
for d in run-cpu run-gpu; do
    cp "$TEMP/input.data" "$TEMP/scaling.data" "$d/"
    sed 's/^epochs.*/epochs                          1              # Number of training epochs./' \
        "$TEMP/input.nn" > "$d/input.nn"
done

echo "=== Clean build: CPU-only nnp-train ==="
(cd "$N2P2_DEVEL/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" nnp-train > "$CHK/run-e2e/build-cpu.log" 2>&1)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
cp "$N2P2_DEVEL/bin/nnp-train" ./nnp-train-cpu

echo "=== Clean build: GPU-enabled nnp-train (GPU=1) ==="
(cd "$N2P2_DEVEL/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-train > "$CHK/run-e2e/build-gpu.log" 2>&1)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
cp "$N2P2_DEVEL/bin/nnp-train" ./nnp-train-gpu

echo ""
echo "=== Running CPU build (32 ranks) ==="
(cd run-cpu && mpirun -np 32 ../nnp-train-cpu)

echo ""
echo "=== Running GPU build (32 ranks) ==="
(cd run-gpu && mpirun -np 32 ../nnp-train-gpu)

echo ""
echo "=== Comparing learning-curve.out / weights.*.000001.out (numeric tolerance) ==="
python3 - <<'PYEOF'
import glob

def load_numeric_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            vals = []
            ok = True
            for p in parts:
                try:
                    vals.append(float(p))
                except ValueError:
                    ok = False
                    break
            if ok:
                rows.append(vals)
    return rows

allOk = True

a = load_numeric_rows("run-cpu/learning-curve.out")
b = load_numeric_rows("run-gpu/learning-curve.out")
if len(a) != len(b):
    print(f"learning-curve.out: ROW COUNT MISMATCH {len(a)} vs {len(b)}")
    allOk = False
else:
    maxdiff = 0.0
    for ra, rb in zip(a, b):
        for va, vb in zip(ra, rb):
            maxdiff = max(maxdiff, abs(va - vb))
    print(f"learning-curve.out: {len(a)} rows, max abs diff = {maxdiff:.3E}")
    if maxdiff > 1e-6:
        allOk = False

for wfile in sorted(glob.glob("run-cpu/weights.*.000001.out")):
    import os
    base = os.path.basename(wfile)
    a = load_numeric_rows(wfile)
    b = load_numeric_rows(f"run-gpu/{base}")
    if len(a) != len(b):
        print(f"{base}: ROW COUNT MISMATCH {len(a)} vs {len(b)}")
        allOk = False
        continue
    maxdiff = 0.0
    for ra, rb in zip(a, b):
        for va, vb in zip(ra, rb):
            maxdiff = max(maxdiff, abs(va - vb))
    print(f"{base}: {len(a)} rows, max abs diff = {maxdiff:.3E}")
    if maxdiff > 1e-6:
        allOk = False

print("ALL PASS" if allOk else "SOME FAILED")
PYEOF

echo "Done"
