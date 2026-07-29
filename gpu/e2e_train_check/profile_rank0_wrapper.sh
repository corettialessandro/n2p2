#!/bin/bash
# mpirun wrapper: only rank 0 runs under `nsys profile`, the other 31
# ranks run the plain binary. nsys does a single-pass trace (unlike ncu's
# multi-pass kernel replay), so it does not re-execute or reorder any
# kernel and does not desync this rank from the other 31 at any MPI
# collective -- it only adds recording overhead to rank 0's own launches.
# ncu was deliberately NOT used here for this reason: replaying kernels
# on one rank while the other 31 do not wait would desync the run.
if [ "$OMPI_COMM_WORLD_RANK" = "0" ]; then
    exec nsys profile --force-overwrite=true --trace=cuda \
        -o "$NSYS_RANK0_OUT" "$@"
else
    exec "$@"
fi
