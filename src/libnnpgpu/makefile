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
MODE=static

# GPU acceleration is off by default -- this whole library is a no-op unless
# GPU=1 is passed (see the `all` rule below), so a plain `make` build never
# needs nvcc or a CUDA install at all.
GPU=

# Include global (project-wide) settings.
include $(PROJECT_DIR)/src/makefile.$(COMP)


###############################################################################
# LIBRARY NAME
###############################################################################
NAME=nnpgpu
LIB=lib$(NAME)


###############################################################################
# PROJECT COMPILER AND FLAGS
###############################################################################
NVCC=$(PROJECT_NVCC)
CUDA_ARCH=$(PROJECT_CUDA_ARCH)
AR=$(PROJECT_AR)
ARFLAGS=$(PROJECT_ARFLAGS)

# Extra include paths for compiling.
INCLUDES=-I./ -I$(PROJECT_INCLUDE)


###############################################################################
# DEFINE SOURCES AND HEADERS
###############################################################################
SRC=$(wildcard *.cu)
HEADERS=$(wildcard *.h)
OBJ=$(SRC:.cu=.o)


###############################################################################
# RULES
###############################################################################
.PHONY: all headers clean rebuild

ifeq ($(GPU), 1)
all: headers $(LIB).a
else
all:
	@echo "libnnpgpu: GPU=1 not set, nothing to build (this is expected for a normal CPU-only build)."
endif

headers:
	cp $(HEADERS) $(PROJECT_INCLUDE)

$(LIB).a: $(OBJ)
	$(AR) $(ARFLAGS) $@ $+
	cp $@ $(PROJECT_LIB)

%.o: %.cu
	$(NVCC) -O3 -std=c++14 -arch=$(CUDA_ARCH) $(INCLUDES) -o $@ -c $<

clean: clean-headers
	$(RM) $(OBJ)
	$(RM) $(LIB).a
	$(RM) $(PROJECT_LIB)/$(LIB).a

clean-headers:
	cd $(PROJECT_INCLUDE) && $(RM) $(HEADERS)

rebuild: clean all
