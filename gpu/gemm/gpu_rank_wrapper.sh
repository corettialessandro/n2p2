#!/bin/bash
# mpirun wrapper for spreading N MPI ranks across multiple physical GPUs
# on one node, all served by a SINGLE nvidia-cuda-mps-control daemon
# (started by the calling slurm script with no CUDA_VISIBLE_DEVICES
# restriction of its own, so it can see and serve all GPUs on the node --
# per NVIDIA's MPS docs, one control daemon forks its own internal MPS
# server process per GPU that actually receives client work). Ranks are
# assigned to GPUs in contiguous blocks of RANKS_PER_GPU (local rank
# 0..RANKS_PER_GPU-1 -> GPU 0, etc.) purely via each CLIENT's own
# CUDA_VISIBLE_DEVICES.
#
# Copied verbatim (comment included) from gpu/e2e_train_check's
# rank_gpu_wrapper.sh, the pattern already validated for the 2G port's
# multi-GPU MPS runs on this exact cluster -- an earlier attempt in
# THIS (4G/GpuQeqSolver) porting effort independently ran 4 SEPARATE
# per-GPU control daemons and hit exactly the failure this comment
# already documents (every rank bound to GPU 1/2/3 got "no CUDA-capable
# device is detected", confirmed via isolated/serialized testing to not
# be a startup race but a hard failure) -- this single-daemon setup
# fixes it, matching what the 2G work already found.
#
# Requires RANKS_PER_GPU exported by the caller before mpirun (the shared
# CUDA_MPS_PIPE_DIRECTORY/CUDA_MPS_LOG_DIRECTORY are inherited unchanged
# from the caller's environment -- no per-rank override needed here).
gpu_id=$(( OMPI_COMM_WORLD_LOCAL_RANK / RANKS_PER_GPU ))
export CUDA_VISIBLE_DEVICES=$gpu_id
exec "$@"
