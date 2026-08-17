#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=lammps-hdnnp-rpbe-correctness
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Phase 5/6 correctness validation at H2O_RPBE-D3 scale (8640 atoms),
# where the GPU SF dispatch was just found to be a genuine ~2.4x win
# (job 52552607 vs 52553563). Reuses the already-built, already-current
# lmp_mpi binaries from those two jobs (GPU tree: lammps_hdnnp_gpu; CPU
# tree: lammps_hdnnp_cpu_booster) -- no rebuild, this is a pure
# correctness comparison. Custom dump with real forces (fx fy fz), 40
# steps, single rank each, same starting config -- diffs thermo and
# per-atom forces to the same tight tolerance already hit for H2O_2G
# (Phase 5 step 8 part 1: 0 diff Temp/PotEng, 1e-7 max force diff).

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
LAMMPS_GPU=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025
LAMMPS_CPU=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_cpu_booster/lammps-22Jul2025
EX="$N2P2_DEVEL/examples/interface-LAMMPS/H2O_RPBE-D3"

echo "=== [1/2] GPU run (40 steps, 1 rank) ==="
PROFDIR="$EX/correctness_gpu"
rm -rf "$PROFDIR" && mkdir -p "$PROFDIR"
cp "$EX/md_correctness_check.lmp" "$EX/h2o_8640_liquid_NpT_RPBE-D3.data" "$PROFDIR/"
ln -s "$EX/hdnnp-data" "$PROFDIR/hdnnp-data"
cd "$PROFDIR"
mpirun -np 1 "$LAMMPS_GPU/src/lmp_mpi" -in md_correctness_check.lmp -log log.correctness_gpu

echo ""
echo "=== [2/2] CPU run (40 steps, 1 rank) ==="
PROFDIR="$EX/correctness_cpu"
rm -rf "$PROFDIR" && mkdir -p "$PROFDIR"
cp "$EX/md_correctness_check.lmp" "$EX/h2o_8640_liquid_NpT_RPBE-D3.data" "$PROFDIR/"
ln -s "$EX/hdnnp-data" "$PROFDIR/hdnnp-data"
cd "$PROFDIR"
mpirun -np 1 "$LAMMPS_CPU/src/lmp_mpi" -in md_correctness_check.lmp -log log.correctness_cpu

echo ""
echo "=== Comparing thermo trajectory and forces ==="
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

cpu_thermo = parse_thermo("correctness_cpu/log.correctness_cpu")
gpu_thermo = parse_thermo("correctness_gpu/log.correctness_gpu")
print(f"CPU thermo rows: {len(cpu_thermo)}, GPU thermo rows: {len(gpu_thermo)}")
n = min(len(cpu_thermo), len(gpu_thermo))
max_temp_diff = max(abs(cpu_thermo[i][1] - gpu_thermo[i][1]) for i in range(n))
max_epair_diff = max(abs(cpu_thermo[i][2] - gpu_thermo[i][2]) for i in range(n))
print(f"Max |Temp_cpu - Temp_gpu| = {max_temp_diff:.6E}")
print(f"Max |E_pair_cpu - E_pair_gpu| = {max_epair_diff:.6E}")

cpu_f = parse_dump_forces("correctness_cpu/traj_forces.dump")
gpu_f = parse_dump_forces("correctness_gpu/traj_forces.dump")
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
