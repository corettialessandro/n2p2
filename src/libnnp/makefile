#!/bin/make -f

###############################################################################
# PROJECT PATHS
###############################################################################
PROJECT_DIR=../..
PROJECT_BIN=$(PROJECT_DIR)/bin
PROJECT_INCLUDE=$(PROJECT_DIR)/include
PROJECT_LIB=$(PROJECT_DIR)/lib


###############################################################################
# GENERAL SETTINGS
###############################################################################
# Default compiler, may be overridden by master makefile or command line.
COMP=gnu

# Default build mode, may be overridden by master makefile or command line.
# Possible modes are "static", "shared" and "test".
MODE=static

# GPU acceleration (src/libnnpgpu), off by default -- set GPU=1 on the make
# command line (or via the master makefile) to enable. See
# GPU_PORTING_PLAN.md's Phase 6 and src/libnnpgpu/makefile.
GPU=

# Include global (project-wide) settings.
include $(PROJECT_DIR)/src/makefile.$(COMP)


###############################################################################
# LIBRARY NAME
###############################################################################
NAME=nnp
LIB=lib$(NAME)


###############################################################################
# PROJECT COMPILER AND FLAGS
###############################################################################
CC=$(PROJECT_CC)
MPICC=$(PROJECT_MPICC)
CFLAGS=$(PROJECT_CFLAGS)
OPTIONS+=$(PROJECT_OPTIONS)
DEBUG=$(PROJECT_DEBUG)
ifeq ($(MODE), test)
DEBUG+= $(PROJECT_TEST)
endif
AR=$(PROJECT_AR)
ARFLAGS=$(PROJECT_ARFLAGS)

# Extra include paths for compiling.
INCLUDES=-I./ -I$(PROJECT_EIGEN)

# GPU acceleration: Mode.cpp's #ifdef N2P2_GPU branch needs both the define
# and to find GpuNeuralNetwork.h/GpuSymmetryFunction.h (src/libnnpgpu is
# not otherwise on the include path since it's a separate, nvcc-built
# library). This is the NN-forward dispatch only (a genuine, if modest,
# win -- see gpu/README.md) and is safe to enable on its own.
ifeq ($(GPU), 1)
OPTIONS+= -DN2P2_GPU
INCLUDES+= -I$(PROJECT_DIR)/src/libnnpgpu
endif

# Symmetry-function GPU dispatch: a SEPARATE opt-in flag from GPU=1 above,
# deliberately not folded into it. Validated correct (bit-identical to
# CPU at short/medium trajectory length, statistically consistent at
# full 200000-step production length). Performance history is not a
# simple story -- an initial one-thread-per-atom kernel design lost
# badly at real production rank counts (up to 3.1x slower at 32
# ranks/270 atoms-per-rank) due to GPU occupancy starvation (only 2-4
# CUDA thread blocks launched per call on a 108-SM GPU); a warp-per-atom
# redesign fixed that specific problem completely -- every short/medium
# measurement after the fix (rank scans, a 2000-step trajectory) showed
# GPU consistently AHEAD of CPU at every rank count tested, including
# this project's actual 32-rank production configuration. But the one
# measurement taken at real 200000-step production LENGTH reversed that
# again: GPU 28.6% SLOWER than CPU overall (Pair time 1.6x higher), for
# reasons not yet confirmed (sustained-load GPU thermal/clock behavior
# is the leading hypothesis, unverified). That full-length number is the
# one to trust, since it's the only one measured at real use-case scale
# -- see gpu/README.md for the complete measurement history. Folding
# this into plain GPU=1 would silently regress anyone building for the
# (still-valid) NN-only win -- hence the separate flag, off by default
# even when GPU=1 is set.
# Requires GPU=1 to also be set (Mode.cpp checks N2P2_GPU && N2P2_GPU_SF
# together); the include path is added again here defensively in case
# GPU_SF=1 is ever passed on its own.
# -DN2P2_FULL_SFD_MEMORY is required by this path specifically (it writes
# Atom::Neighbor::dGdr indexed by global symmetry-function index
# directly, only correct under the full/uncompacted memory layout) --
# NOT needed by the NN-only path, so it stays out of the GPU=1 block too.
ifeq ($(GPU_SF), 1)
OPTIONS+= -DN2P2_GPU_SF -DN2P2_FULL_SFD_MEMORY
INCLUDES+= -I$(PROJECT_DIR)/src/libnnpgpu
endif


###############################################################################
# DEFINE SOURCES AND HEADERS
###############################################################################
SRC=$(wildcard *.cpp)
HEADERS=$(wildcard *.h)
OBJ=$(SRC:.cpp=.o)


###############################################################################
# RULES
###############################################################################
.PHONY: all headers version clean rebuild

ifeq ($(MODE), shared)
all: headers $(LIB).so
else
all: headers $(LIB).a
endif

headers: version
	cp $(HEADERS) $(PROJECT_INCLUDE)

version:
	@$(eval GIT_VERSION = $(shell git describe --tags --always))
	@sed -i.bak -E "s?(N2P2_GIT_VERSION) .*?\1 \"$(GIT_VERSION)\"?" version.h
	@$(eval GIT_REV = $(shell git rev-parse HEAD))
	@sed -i.bak -E "s?(N2P2_GIT_REV) .*?\1 \"$(GIT_REV)\"?" version.h
	@$(eval GIT_BRANCH = $(shell git rev-parse --abbrev-ref HEAD))
	@sed -i.bak -E "s?(N2P2_GIT_BRANCH) .*?\1 \"$(GIT_BRANCH)\"?" version.h
	@rm version.h.bak

$(LIB).so: CFLAGS+= -fPIC
$(LIB).so: $(OBJ)
	$(CC) $(CFLAGS) $(DEBUG) $(OPTIONS) -o $@ $+ -shared
	cp $@ $(PROJECT_LIB)

$(LIB).a: $(OBJ)
	$(AR) $(ARFLAGS) $@ $+
	cp $@ $(PROJECT_LIB)

%.o: %.cpp
	$(CC) $(CFLAGS) $(DEBUG) $(INCLUDES) $(OPTIONS) -o $@ -c $<

clean: clean-version clean-headers
	$(RM) $(OBJ)
	$(RM) *.gcno *.gcda $(LIB).so $(LIB).a
	$(RM) $(PROJECT_LIB)/$(LIB).so $(PROJECT_LIB)/$(LIB).a

clean-version:
	@sed -i.bak -E "s?(N2P2_GIT_VERSION) .*?\1 \"\"?" version.h
	@sed -i.bak -E "s?(N2P2_GIT_REV) .*?\1 \"\"?" version.h
	@sed -i.bak -E "s?(N2P2_GIT_BRANCH) .*?\1 \"\"?" version.h
	@rm version.h.bak

clean-headers:
	cd $(PROJECT_INCLUDE) && $(RM) $(HEADERS)

rebuild: clean all
