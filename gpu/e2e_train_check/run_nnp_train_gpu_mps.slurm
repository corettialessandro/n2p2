#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#SBATCH --job-name=nnp-train-gpu-mps
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

rm -rf run-mps && mkdir run-mps && cd run-mps

mkdir run-nomps run-withmps
for d in run-nomps run-withmps; do
    cp "$TEMP/input.data" "$TEMP/scaling.data" "$d/"
    sed 's/^epochs.*/epochs                          1              # Number of training epochs./' \
        "$TEMP/input.nn" > "$d/input.nn"
done

echo "=== Clean build: GPU-enabled nnp-train (GPU=1) ==="
(cd "$N2P2_DEVEL/src" && make clean > /dev/null 2>&1)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" nnp-train > "$CHK/run-mps/build-gpu.log" 2>&1)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
cp "$N2P2_DEVEL/bin/nnp-train" ./nnp-train-gpu

echo ""
echo "=== Running GPU build WITHOUT MPS (32 ranks, baseline reproduction) ==="
(cd run-nomps && mpirun -np 32 ../nnp-train-gpu)

echo ""
echo "=== Starting MPS control daemon ==="
# MPS's control pipe is a UNIX domain socket -- it needs to live on a
# node-local filesystem (/tmp), not the NFS/parallel-filesystem-backed
# home/project storage under $CHK. A first attempt using a directory
# under $CHK failed silently (empty control.log/server.log, exit 1)
# almost certainly for this reason.
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-pipe-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
sleep 2
echo "get_server_list" | nvidia-cuda-mps-control || true
echo "--- control.log ---"; cat "$CUDA_MPS_LOG_DIRECTORY/control.log" 2>&1
echo "--- server.log ---"; cat "$CUDA_MPS_LOG_DIRECTORY/server.log" 2>&1

echo ""
echo "=== Running GPU build WITH MPS (32 ranks) ==="
(cd run-withmps && mpirun -np 32 ../nnp-train-gpu)

echo ""
echo "=== Stopping MPS control daemon ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2

echo ""
echo "=== Comparing timing.out: no-MPS vs with-MPS ==="
echo "--- no-MPS ---"
cat run-nomps/timing.out | grep -v "^#"
echo "--- with-MPS ---"
cat run-withmps/timing.out | grep -v "^#"

echo ""
echo "=== Comparing learning-curve.out: no-MPS vs with-MPS (sanity: MPS shouldn't change results) ==="
python3 - <<'PYEOF'
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

a = load_numeric_rows("run-nomps/learning-curve.out")
b = load_numeric_rows("run-withmps/learning-curve.out")
if len(a) != len(b):
    print(f"ROW COUNT MISMATCH {len(a)} vs {len(b)}")
else:
    maxdiff = 0.0
    for ra, rb in zip(a, b):
        for va, vb in zip(ra, rb):
            maxdiff = max(maxdiff, abs(va - vb))
    print(f"learning-curve.out: {len(a)} rows, max abs diff = {maxdiff:.3E}")
PYEOF

echo "Done"
