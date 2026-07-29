#!/bin/bash
# mpirun wrapper for spreading N MPI ranks across multiple physical GPUs
# on one node, all served by a SINGLE nvidia-cuda-mps-control daemon
# (started by the calling slurm script with no CUDA_VISIBLE_DEVICES
# restriction of its own, so it can see and serve all GPUs on the node --
# per NVIDIA's MPS docs, one control daemon forks its own internal MPS
# server process per GPU that actually receives client work). Ranks are
# assigned to GPUs in contiguous blocks of RANKS_PER_GPU (local rank
# 0..RANKS_PER_GPU-1 -> GPU 0, etc.) purely via each CLIENT's own
# CUDA_VISIBLE_DEVICES -- so that fewer ranks contend for each GPU's
# queue, the follow-up experiment to gpu/README.md's finding that 32
# ranks sharing ONE GPU spend most of F_err's time blocked in
# cudaDeviceSynchronize/cudaFree, not computing.
#
# An earlier attempt ran 4 SEPARATE control daemons (one per GPU, each
# restricted via its own CUDA_VISIBLE_DEVICES) and hit intermittent
# cublasCreate() CUBLAS_STATUS_NOT_INITIALIZED failures on some ranks,
# most likely a startup race from spawning 4 daemons back-to-back with
# no per-daemon settle time. This single-daemon setup sidesteps that
# entirely -- there is only one daemon to race with, and it's the pattern
# NVIDIA's own docs describe as the default multi-GPU case.
#
# Requires RANKS_PER_GPU exported by the caller before mpirun (the shared
# CUDA_MPS_PIPE_DIRECTORY/CUDA_MPS_LOG_DIRECTORY are inherited unchanged
# from the caller's environment -- no per-rank override needed here).
gpu_id=$(( OMPI_COMM_WORLD_LOCAL_RANK / RANKS_PER_GPU ))
export CUDA_VISIBLE_DEVICES=$gpu_id
exec "$@"
