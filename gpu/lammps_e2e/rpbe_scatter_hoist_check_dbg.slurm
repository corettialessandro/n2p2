#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128GB
#SBATCH --job-name=rpbe-scatter-hoist-check-dbg
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Validates + times step 1 of the GPU_SF=1 CSR-build optimization for
# H2O_RPBE-D3 (single-pass CSR construction in Mode.cpp, replacing the
# original two-pass count-then-fill loop -- see gpu/README.md's matching
# follow-up). Fresh rebuild of BOTH CPU and GPU trees (CPU is unaffected
# by this change -- its calculateSymmetryFunctionGroups() path never
# touches the N2P2_GPU_SF-gated block -- but rebuilt anyway to get a
# fresh ground-truth reference, matching this project's standing
# "always compare against a real freshly-built CPU run" practice).
#
# [1/3] correctness: 40-step, 1-rank CPU vs GPU_SF force/thermo
# comparison (reuses rpbe_correctness_check.slurm's methodology).
# [2/3] performance: 2000-step, 32-rank GPU_SF=1 timing (reuses
# report_gpu_sf_2k.slurm's H2O_RPBE-D3 config) to compare against the
# pre-optimization baseline (149.41s Loop time, job 53995440).

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
LAMMPS_GPU=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025
LAMMPS_CPU=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_cpu_booster/lammps-22Jul2025
EX="$N2P2_DEVEL/examples/interface-LAMMPS/H2O_RPBE-D3"

echo "=== [0a/3] Rebuild GPU tree (GPU=1 GPU_SF=1, with the scatter-loop member-lookup hoist) ==="
(cd "$N2P2_DEVEL/src/libnnpgpu" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpgpu" && make COMP=gnu MODE=static GPU=1)
(cd "$N2P2_DEVEL/src/libnnp" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpif" && make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 GPU_SF=1 PROJECT_EIGEN="$EIGEN" \
    INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnp libnnpif)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
(cd "$LAMMPS_GPU/src" && touch main.cpp && make mpi -j"$SLURM_NTASKS_PER_NODE")

echo ""
echo "=== [0b/3] Rebuild CPU tree (no GPU=1, fresh ground-truth reference) ==="
(cd "$N2P2_DEVEL/src/libnnp" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpif" && make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static PROJECT_EIGEN="$EIGEN" \
    INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnp libnnpif)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h
(cd "$LAMMPS_CPU/src" && touch main.cpp && make mpi -j"$SLURM_NTASKS_PER_NODE")

echo ""
echo "=== [1/3] Correctness: 40-step, 1-rank CPU vs GPU_SF force/thermo comparison ==="
echo "--- GPU run ---"
PROFDIR="$EX/correctness_gpu_hoist"
rm -rf "$PROFDIR" && mkdir -p "$PROFDIR"
cp "$EX/md_correctness_check.lmp" "$EX/h2o_8640_liquid_NpT_RPBE-D3.data" "$PROFDIR/"
ln -s "$EX/hdnnp-data" "$PROFDIR/hdnnp-data"
cd "$PROFDIR"
mpirun -np 1 "$LAMMPS_GPU/src/lmp_mpi" -in md_correctness_check.lmp -log log.correctness_gpu

echo "--- CPU run ---"
PROFDIR="$EX/correctness_cpu_hoist"
rm -rf "$PROFDIR" && mkdir -p "$PROFDIR"
cp "$EX/md_correctness_check.lmp" "$EX/h2o_8640_liquid_NpT_RPBE-D3.data" "$PROFDIR/"
ln -s "$EX/hdnnp-data" "$PROFDIR/hdnnp-data"
cd "$PROFDIR"
mpirun -np 1 "$LAMMPS_CPU/src/lmp_mpi" -in md_correctness_check.lmp -log log.correctness_cpu

echo ""
echo "=== Comparing thermo trajectory and forces ==="
cd "$EX"
python3 - <<'PYEOF'
import re

def parse_thermo(path):
    rows = []
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("Step") and "Temp" in line:
            for l2 in lines[i+1:]:
                parts = l2.split()
                if len(parts) < 2 or not parts[0].lstrip('-').isdigit():
                    break
                rows.append([float(x) for x in parts])
    return rows

def parse_dump_forces(path):
    frames = {}
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            ts = int(lines[i+1])
            n = int(lines[i+3])
            atoms = {}
            for j in range(n):
                parts = lines[i+9+j].split()
                aid = int(parts[0])
                fx, fy, fz = float(parts[5]), float(parts[6]), float(parts[7])
                atoms[aid] = (fx, fy, fz)
            frames[ts] = atoms
            i += 9 + n
        else:
            i += 1
    return frames

cpu_thermo = parse_thermo("correctness_cpu_hoist/log.correctness_cpu")
gpu_thermo = parse_thermo("correctness_gpu_hoist/log.correctness_gpu")
print(f"CPU thermo rows: {len(cpu_thermo)}, GPU thermo rows: {len(gpu_thermo)}")
n = min(len(cpu_thermo), len(gpu_thermo))
max_temp_diff = max(abs(cpu_thermo[i][1] - gpu_thermo[i][1]) for i in range(n))
max_epair_diff = max(abs(cpu_thermo[i][2] - gpu_thermo[i][2]) for i in range(n))
print(f"Max |Temp_cpu - Temp_gpu| = {max_temp_diff:.6E}")
print(f"Max |E_pair_cpu - E_pair_gpu| = {max_epair_diff:.6E}")

cpu_f = parse_dump_forces("correctness_cpu_hoist/traj_forces.dump")
gpu_f = parse_dump_forces("correctness_gpu_hoist/traj_forces.dump")
common_ts = sorted(set(cpu_f.keys()) & set(gpu_f.keys()))
print(f"Common dumped timesteps: {common_ts}")
max_force_diff = 0.0
for ts in common_ts:
    for aid, (fx1, fy1, fz1) in cpu_f[ts].items():
        fx2, fy2, fz2 = gpu_f[ts][aid]
        d = max(abs(fx1-fx2), abs(fy1-fy2), abs(fz1-fz2))
        max_force_diff = max(max_force_diff, d)
print(f"Max |force_cpu - force_gpu| over all dumped atoms/timesteps = {max_force_diff:.6E}")

if max_temp_diff < 1e-4 and max_epair_diff < 1e-3 and max_force_diff < 1e-5:
    print("ALL PASS")
else:
    print("MISMATCH -- investigate")
PYEOF

echo ""
echo "=== [2/3] Starting a single MPS control daemon serving all 4 GPUs ==="
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-pipe-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
sleep 2
export RANKS_PER_GPU=8
WRAPPER="$N2P2_DEVEL/gpu/e2e_train_check/rank_gpu_wrapper.sh"

echo ""
echo "=== [3/3] Performance: H2O_RPBE-D3, GPU SF (scatter-loop hoist), 32 ranks/4 GPUs, 2000 steps ==="
RUNDIR="$EX/hoist_2k_rpbe"
rm -rf "$RUNDIR" && mkdir -p "$RUNDIR"
sed -e 's/variable numSteps.*equal.*5.*/variable numSteps equal 2000 # SF-GPU single-pass CSR check/' \
    -e 's/^dump 1 all atom 1 traj.dump/dump 1 all atom 500 traj.dump/' -e 's/resetew no maxew 100 /resetew yes maxew 1000000 /' \
    "$EX/md.lmp" > "$RUNDIR/md_2k.lmp"
cp "$EX/h2o_8640_liquid_NpT_RPBE-D3.data" "$RUNDIR/"
ln -s "$EX/hdnnp-data" "$RUNDIR/hdnnp-data"
cd "$RUNDIR"
mpirun -np 32 "$WRAPPER" "$LAMMPS_GPU/src/lmp_mpi" -in md_2k.lmp -log log.hoist_2k_rpbe
echo "--- H2O_RPBE-D3 GPU SF result (single-pass CSR, 32 ranks) ---"
grep -A1 "^Pair " log.hoist_2k_rpbe
grep "^Loop time" log.hoist_2k_rpbe

echo ""
echo "=== Stopping MPS control daemon ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2

echo "=== Done ==="
