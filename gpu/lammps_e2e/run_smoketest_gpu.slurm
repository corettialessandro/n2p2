#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=lammps-hdnnp-gpu-smoketest
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Phase 2 (continued): run the same 2000-step H2O_2G smoke test used for
# the DCGP CPU baseline (job 52481389), but with the GPU-linked lmp_mpi
# built in build_gpu.slurm, then compare thermo trajectory and per-atom
# forces/energy against the CPU log/dump -- correctness before any
# performance claim, matching this project's standing rule.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
LAMMPS_GPU_BIN=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025/src/lmp_mpi
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"

cd "$DYN"
echo "=== Running GPU build, same 2000-step smoke test as the CPU baseline ==="
sed 's/dump\.lammpstrj/dump_gpu.lammpstrj/' md_smoketest.lmp > md_smoketest_gpu.lmp
mpirun -np "$SLURM_NTASKS_PER_NODE" "$LAMMPS_GPU_BIN" -in md_smoketest_gpu.lmp -log log.smoketest_gpu

echo ""
echo "=== Comparing CPU (log.smoketest_cpu / dump.lammpstrj) vs GPU (log.smoketest_gpu / dump_gpu.lammpstrj) ==="
python3 - <<'PYEOF'
import re

def parse_thermo(path):
    rows = []
    header = None
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("Step") and "Temp" in line:
            header = line.split()
            continue
        if header is not None:
            parts = line.split()
            if len(parts) == len(header):
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    header = None
    return rows

def parse_dump_forces(path):
    # Returns {timestep: {atom_id: (fx, fy, fz)}}
    frames = {}
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            ts = int(lines[i+1])
            i += 2
            assert lines[i].startswith("ITEM: NUMBER OF ATOMS")
            natoms = int(lines[i+1])
            i += 2
            # BOX BOUNDS (3 lines) + header line
            while not lines[i].startswith("ITEM: ATOMS"):
                i += 1
            cols = lines[i].split()[2:]
            i += 1
            fx_i, fy_i, fz_i, id_i = cols.index("fx"), cols.index("fy"), cols.index("fz"), cols.index("id")
            atoms = {}
            for _ in range(natoms):
                parts = lines[i].split()
                atoms[int(parts[id_i])] = (float(parts[fx_i]), float(parts[fy_i]), float(parts[fz_i]))
                i += 1
            frames[ts] = atoms
        else:
            i += 1
    return frames

cpu_thermo = parse_thermo("log.smoketest_cpu")
gpu_thermo = parse_thermo("log.smoketest_gpu")
print(f"CPU thermo rows: {len(cpu_thermo)}, GPU thermo rows: {len(gpu_thermo)}")
allOk = True
if len(cpu_thermo) != len(gpu_thermo):
    print("ROW COUNT MISMATCH")
    allOk = False
else:
    maxdiff_T = max(abs(a[1]-b[1]) for a, b in zip(cpu_thermo, gpu_thermo))
    maxdiff_E = max(abs(a[6]-b[6]) for a, b in zip(cpu_thermo, gpu_thermo))
    print(f"Max |Temp_cpu - Temp_gpu| = {maxdiff_T:.6E}")
    print(f"Max |PotEng_cpu - PotEng_gpu| = {maxdiff_E:.6E}")

cpu_frames = parse_dump_forces("dump.lammpstrj")
gpu_frames = parse_dump_forces("dump_gpu.lammpstrj")
common_ts = sorted(set(cpu_frames) & set(gpu_frames))
print(f"Common dumped timesteps: {common_ts}")
maxfdiff = 0.0
for ts in common_ts:
    a, b = cpu_frames[ts], gpu_frames[ts]
    for aid in a:
        fa, fb = a[aid], b[aid]
        maxfdiff = max(maxfdiff, max(abs(x-y) for x, y in zip(fa, fb)))
print(f"Max |force_cpu - force_gpu| over all dumped atoms/timesteps = {maxfdiff:.6E}")

if maxfdiff > 1e-6 or (len(cpu_thermo) == len(gpu_thermo) and maxdiff_T > 1e-3):
    allOk = False

print("ALL PASS" if allOk else "CHECK TOLERANCES ABOVE")
PYEOF

echo ""
echo "Done"
