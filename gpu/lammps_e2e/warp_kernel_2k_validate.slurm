#!/bin/bash
#SBATCH -A L-AUT_Giane_26
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:4
#SBATCH --mem=64GB
#SBATCH --job-name=warp-kernel-2k-validate
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Two-tier validation of the warp-per-atom SF-GPU kernel rewrite, per
# user request: (1) short-window tight numerical match (bug-catching --
# the first ~100 steps of a 2000-step run, before chaos has time to
# amplify anything), (2) full 2000-step statistical match (physics-
# catching -- thermo means/character, not step-by-step exact agreement,
# since ExpAngn's atomicAdd neighbor-side accumulation makes GPU results
# NOT bit-reproducible run-to-run, and chaotic MD dynamics will amplify
# that ~1e-15 nondeterminism into visible trajectory divergence over a
# long run -- expected behavior, not a bug), and (3) a GPU-vs-GPU repeat
# run (same input, same binary, run twice) to distinguish "expected
# chaos-driven divergence" from "a real systematic GPU-vs-CPU bug": if
# GPU1-vs-GPU2 diverges at a similar rate to GPU-vs-CPU, that confirms
# it's inherent atomic nondeterminism, not incorrectness.
#
# H2O_2G, 32 ranks/4 GPUs (the most extreme case tested so far -- ~20
# atoms/rank, smallest batch, most atomic contention). Reuses the
# already-current, already-validated GPU lmp_mpi (warp-per-atom kernels,
# bug-fixed, built in job 52576146) -- no rebuild needed since no code
# has changed since then.

set -e
module purge
module load cuda/12.2
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

N2P2_DEVEL=/leonardo/home/userexternal/acoretti/Projects/NEURALCPM/Code/n2p2_devel
LAMMPS_GPU=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_gpu/lammps-22Jul2025
LAMMPS_CPU=/leonardo_scratch/large/userexternal/acoretti/lammps_hdnnp_cpu_booster/lammps-22Jul2025
DYN="$N2P2_DEVEL/temp/H2O_2G/dynamics"
WRAPPER="$N2P2_DEVEL/gpu/e2e_train_check/rank_gpu_wrapper.sh"

echo "=== Starting MPS control daemon ==="
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps-pipe-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
sleep 2
export RANKS_PER_GPU=8

echo ""
echo "=== [1/3] GPU run #1, 32 ranks/4 GPUs, 2000 steps ==="
RUNDIR="$DYN/validate_gpu_run1"
rm -rf "$RUNDIR" && mkdir -p "$RUNDIR"
cp "$DYN/init_conf_minimized.data" "$DYN/md_validate_2k.lmp" "$RUNDIR/"
ln -s "$DYN/hdnnp-data" "$RUNDIR/hdnnp-data"
cd "$RUNDIR"
mpirun -np 32 "$WRAPPER" "$LAMMPS_GPU/src/lmp_mpi" -in md_validate_2k.lmp -log log.validate_gpu_run1

echo ""
echo "=== [2/3] GPU run #2 (identical input, repeat run), 32 ranks/4 GPUs, 2000 steps ==="
RUNDIR="$DYN/validate_gpu_run2"
rm -rf "$RUNDIR" && mkdir -p "$RUNDIR"
cp "$DYN/init_conf_minimized.data" "$DYN/md_validate_2k.lmp" "$RUNDIR/"
ln -s "$DYN/hdnnp-data" "$RUNDIR/hdnnp-data"
cd "$RUNDIR"
mpirun -np 32 "$WRAPPER" "$LAMMPS_GPU/src/lmp_mpi" -in md_validate_2k.lmp -log log.validate_gpu_run2

echo ""
echo "=== Stopping MPS control daemon ==="
echo quit | nvidia-cuda-mps-control || true
sleep 2

echo ""
echo "=== [3/3] CPU run, 32 ranks, 2000 steps ==="
RUNDIR="$DYN/validate_cpu_run"
rm -rf "$RUNDIR" && mkdir -p "$RUNDIR"
cp "$DYN/init_conf_minimized.data" "$DYN/md_validate_2k.lmp" "$RUNDIR/"
ln -s "$DYN/hdnnp-data" "$RUNDIR/hdnnp-data"
cd "$RUNDIR"
mpirun -np 32 "$LAMMPS_CPU/src/lmp_mpi" -in md_validate_2k.lmp -log log.validate_cpu_run

echo ""
echo "Done"
