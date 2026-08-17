#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=lammps-hdnnp-gpu-sf-smoketest
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Phase 5 step 8 (soft-percolating-jellyfish.md), part 1: LAMMPS-level
# end-to-end validation of the symmetry-function GPU dispatch just wired
# into Mode::calculateSymmetryFunctionGroups(). Rebuilds libnnpgpu.a +
# libnnp.a (GPU=1, now auto-carries -DN2P2_FULL_SFD_MEMORY) and relinks
# the existing GPU lmp_mpi (lammps_hdnnp_gpu tree, already built/
# validated for the NN-only dispatch in Phase 2), then reruns the same
# 2000-step H2O_2G smoke test and diffs against the CPU baseline log
# (temp/H2O_2G/dynamics/log.smoketest_cpu, from Phase 1) -- short run
# first, full 200000-step production comparison only after this passes.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
EIGEN=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/eigen-3.4.0
LAMMPS_SRC=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"

echo "=== [1/4] Rebuild libnnpgpu.a (GPU=1) ==="
(cd "$N2P2_DEVEL/src/libnnpgpu" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpgpu" && make COMP=gnu MODE=static GPU=1)

echo ""
echo "=== [2/4] Rebuild libnnp.a + libnnpif.a (GPU=1, now auto-carries"
echo "    N2P2_FULL_SFD_MEMORY) -- libnnpif regenerates lib/Makefile.lammps-extra,"
echo "    which lib/hdnnp/Makefile.lammps (in the LAMMPS tree) symlinks to; forgetting"
echo "    this step is exactly what broke the first attempt at this script. Full"
echo "    clean (both libnnp AND libnnpif -- -march=native means any stale .o left"
echo "    from a build on a different partition is not safe to reuse, same failure"
echo "    mode already root-caused once before, commit 1477941) ==="
(cd "$N2P2_DEVEL/src/libnnp" && make clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src/libnnpif" && make INTERFACES=LAMMPS clean > /dev/null 2>&1 || true)
(cd "$N2P2_DEVEL/src" && make COMP=gnu MODE=static GPU=1 PROJECT_EIGEN="$EIGEN" \
    INTERFACES=LAMMPS -j"$SLURM_NTASKS_PER_NODE" libnnp libnnpif)
git -C "$N2P2_DEVEL" checkout -- src/libnnp/version.h

echo ""
echo "=== [3/4] Relink lmp_mpi (GPU-linked LAMMPS tree) ==="
cd "$LAMMPS_SRC/src"
touch main.cpp
make mpi -j"$SLURM_NTASKS_PER_NODE"

echo ""
echo "=== [4/4] Run: same 2000-step H2O_2G smoke test as Phase 1's CPU baseline ==="
PROFDIR="$DYN/smoketest_gpu_sf"
rm -rf "$PROFDIR" && mkdir -p "$PROFDIR"
cp "$DYN/init_conf_minimized.data" "$DYN/md_smoketest.lmp" "$PROFDIR/"
ln -s "$DYN/hdnnp-data" "$PROFDIR/hdnnp-data"
cd "$PROFDIR"

mpirun -np "$SLURM_NTASKS_PER_NODE" "$LAMMPS_SRC/src/lmp_mpi" -in md_smoketest.lmp -log log.smoketest_gpu_sf

echo ""
echo "=== Comparing thermo trajectory and forces against the Phase 1 CPU baseline ==="
python3 - <<'PYEOF'
import re

def parse_thermo(path):
    rows = []
    header = None
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        if line.strip().startswith("Step") and "Temp" in line:
            header = line.split(); continue
        if header is not None:
            parts = line.split()
            if len(parts) == len(header):
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    header = None
    return rows

def parse_dump_forces(path):
    frames = {}
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            ts = int(lines[i+1]); i += 2
            natoms = int(lines[i+1]); i += 2
            while not lines[i].startswith("ITEM: ATOMS"):
                i += 1
            cols = lines[i].split()[2:]; i += 1
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

cpu_thermo = parse_thermo("../log.smoketest_cpu")
gpu_thermo = parse_thermo("log.smoketest_gpu_sf")
print(f"CPU thermo rows: {len(cpu_thermo)}, GPU(SF) thermo rows: {len(gpu_thermo)}")
allOk = True
if len(cpu_thermo) != len(gpu_thermo):
    print("ROW COUNT MISMATCH"); allOk = False
else:
    maxdiff_T = max(abs(a[1]-b[1]) for a, b in zip(cpu_thermo, gpu_thermo))
    maxdiff_E = max(abs(a[6]-b[6]) for a, b in zip(cpu_thermo, gpu_thermo))
    print(f"Max |Temp_cpu - Temp_gpu_sf| = {maxdiff_T:.6E}")
    print(f"Max |PotEng_cpu - PotEng_gpu_sf| = {maxdiff_E:.6E}")
    if maxdiff_T > 1e-3: allOk = False

cpu_frames = parse_dump_forces("../dump.lammpstrj")
gpu_frames = parse_dump_forces("dump.lammpstrj")
common_ts = sorted(set(cpu_frames) & set(gpu_frames))
print(f"Common dumped timesteps: {common_ts}")
maxfdiff = 0.0
for ts in common_ts:
    a, b = cpu_frames[ts], gpu_frames[ts]
    for aid in a:
        fa, fb = a[aid], b[aid]
        maxfdiff = max(maxfdiff, max(abs(x-y) for x, y in zip(fa, fb)))
print(f"Max |force_cpu - force_gpu_sf| over all dumped atoms/timesteps = {maxfdiff:.6E}")
if maxfdiff > 1e-6: allOk = False

print("ALL PASS" if allOk else "CHECK TOLERANCES ABOVE")
PYEOF

echo ""
echo "Done"
