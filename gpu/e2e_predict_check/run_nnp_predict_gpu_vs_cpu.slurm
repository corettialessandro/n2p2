#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --job-name=nnp-predict-gpu-vs-cpu-final
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load cuda/12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
TEMP="$N2P2_DEVEL/temp/H2O_2G"
CHK="$N2P2_DEVEL/gpu/e2e_predict_check"
cd "$CHK"

rm -rf run-e2e && mkdir run-e2e && cd run-e2e
cp "$TEMP/input.nn" "$TEMP/input.data" "$TEMP/scaling.data" .
cp "$TEMP/weights.001.000002.out" weights.001.data
cp "$TEMP/weights.008.000002.out" weights.008.data

echo "=== Clean build: CPU-only nnp-predict ==="
(cd "$N2P2_DEVEL/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" nnp-predict > /dev/null)
cp "$N2P2_DEVEL/bin/nnp-predict" ./nnp-predict-cpu

echo "=== Clean build: GPU-enabled nnp-predict (GPU=1) ==="
(cd "$N2P2_DEVEL/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-predict > /dev/null)
cp "$N2P2_DEVEL/bin/nnp-predict" ./nnp-predict-gpu

echo ""
echo "=== Running CPU build ==="
mkdir run-cpu && cp input.nn input.data scaling.data weights.001.data weights.008.data run-cpu/
(cd run-cpu && ../nnp-predict-cpu 0)

echo ""
echo "=== Running GPU build ==="
mkdir run-gpu && cp input.nn input.data scaling.data weights.001.data weights.008.data run-gpu/
(cd run-gpu && ../nnp-predict-gpu 0)

echo ""
echo "=== Comparing energy.out / nnatoms.out / nnforces.out (numeric tolerance) ==="
python3 - <<'PYEOF'
import re

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
for fname in ["energy.out", "nnatoms.out", "nnforces.out"]:
    a = load_numeric_rows(f"run-cpu/{fname}")
    b = load_numeric_rows(f"run-gpu/{fname}")
    if len(a) != len(b):
        print(f"{fname}: ROW COUNT MISMATCH {len(a)} vs {len(b)}")
        allOk = False
        continue
    maxdiff = 0.0
    for ra, rb in zip(a, b):
        for va, vb in zip(ra, rb):
            maxdiff = max(maxdiff, abs(va - vb))
    print(f"{fname}: {len(a)} rows, max abs diff = {maxdiff:.3E}")
    if maxdiff > 1e-8:
        allOk = False

print("ALL PASS" if allOk else "SOME FAILED")
PYEOF

echo "Done"
