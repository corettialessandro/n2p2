#!/bin/make -f

###############################################################################
# EXTERNAL LIBRARY PATHS
###############################################################################
# Enter here paths to GSL or EIGEN if they are not in your standard include
# path. DO NOT completely remove the entry, leave at least "./".
PROJECT_GSL=./
#PROJECT_GSL=${HOME}/local/src/gsl-2.8/build/include
PROJECT_EIGEN=/usr/include/eigen3/
#PROJECT_EIGEN=${HOME}/local/src/eigen/
PROJECT_BOOST=${BOOST_ROOT}

###############################################################################
# COMPILERS AND FLAGS
###############################################################################
PROJECT_CC=g++
PROJECT_MPICC=mpic++
# OpenMP parallelization is disabled by default, add flag "-fopenmp" to enable.
PROJECT_CFLAGS=-O3 -march=native -std=c++11 #-fopenmp
PROJECT_CFLAGS_MPI=-Wno-long-long
PROJECT_DEBUG=-g -pedantic-errors -Wall -Wextra
PROJECT_TEST=--coverage -fno-default-inline -fno-inline -fno-inline-small-functions -fno-elide-constructors
PROJECT_AR=ar
PROJECT_ARFLAGS=-rcsv
PROJECT_CFLAGS_BLAS=
PROJECT_LDFLAGS_BLAS=-lopenblas -lgsl -lgslcblas

# GPU support (src/libnnpgpu, gated by `make GPU=1` -- see the GPU section
# further down and GPU_PORTING_PLAN.md's Phase 6). Not used at all unless
# GPU=1 is passed on the make command line, so a plain `make` build is
# completely unaffected. PROJECT_CUDA_HOME defaults to $(CUDA_HOME), set by
# `module load cuda/...` on clusters using environment modules -- override
# on the command line (or here) if your CUDA install isn't found that way.
PROJECT_NVCC=nvcc
PROJECT_CUDA_ARCH=sm_80
PROJECT_CUDA_HOME=${CUDA_HOME}
PROJECT_LDFLAGS_GPU=-L$(PROJECT_CUDA_HOME)/lib64 -lcudart -lcublas

###############################################################################
# COMPILE-TIME OPTIONS
###############################################################################

# Do not use symmetry function groups.
#PROJECT_OPTIONS+= -DN2P2_NO_SF_GROUPS

# Do not use symmetry function cache.
#PROJECT_OPTIONS+= -DN2P2_NO_SF_CACHE

# Disable asymmetric polynomial symmetry functions.
#PROJECT_OPTIONS+= -DN2P2_NO_ASYM_POLY

# Build with dummy Stopwatch class.
#PROJECT_OPTIONS+= -DN2P2_NO_TIME

# Disable check for low number of neighbors.
#PROJECT_OPTIONS+= -DN2P2_NO_NEIGH_CHECK

# Use alternative (older) memory layout for symmetry function derivatives.
#PROJECT_OPTIONS+= -DN2P2_FULL_SFD_MEMORY

# Compile without MPI support.
#PROJECT_OPTIONS+= -DN2P2_NO_MPI

# Use BLAS together with Eigen.
#PROJECT_OPTIONS+= -DEIGEN_USE_BLAS

# Disable all C++ asserts (also Eigen debugging).
#PROJECT_OPTIONS+= -DNDEBUG

# Use Intel MKL together with Eigen.
#PROJECT_OPTIONS+= -DEIGEN_USE_MKL_ALL

# Disable Eigen multi threading.
PROJECT_OPTIONS+= -DEIGEN_DONT_PARALLELIZE

# GPU acceleration (src/libnnpgpu) is NOT toggled here like the flags above --
# it needs an extra nvcc-compiled library and extra link flags, not just a
# -D define, so it's controlled by `make GPU=1` instead (see src/libnnp/
# makefile, src/libnnpgpu/makefile, src/application/makefile).
