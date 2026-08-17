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
# command line. Needed here (not just in libnnp/application) so that when
# this library is consumed by an external build (e.g. LAMMPS's ML-HDNNP
# package, via the generated lib/Makefile.lammps-extra below) that final
# link also gets libnnpgpu.a + the CUDA link flags -- libnnp.a built with
# GPU=1 has unresolved references to gpuNnForwardDEdG() etc. that are only
# resolved at whichever binary's final link step (see application/makefile
# for the same pattern applied to our own nnp-* binaries).
GPU=

# Include global (project-wide) settings.
include $(PROJECT_DIR)/src/makefile.$(COMP)


###############################################################################
# INTERFACE LIBRARY OPTIONS
###############################################################################

# Default interfaces compiled into libnnpif, may be overridden by master
# makefile or command line. Provide a space-separated list of the following
# available options:
#    * LAMMPS
#    * CabanaMD
INTERFACES=LAMMPS CabanaMD


###############################################################################
# LIBRARY NAME
###############################################################################
NAME=nnpif
LIB=lib$(NAME)


###############################################################################
# PROJECT COMPILER AND FLAGS
###############################################################################
CC=$(PROJECT_CC)
ifneq (,$(findstring N2P2_NO_MPI,${PROJECT_OPTIONS}))
MPICC=${PROJECT_CC}
CFLAGS=${PROJECT_CFLAGS}
else
MPICC=${PROJECT_MPICC}
CFLAGS=${PROJECT_CFLAGS} ${PROJECT_CFLAGS_MPI}
endif
OPTIONS+=$(PROJECT_OPTIONS)
# GPU=1 must add the exact same defines here as libnnp/makefile does --
# not just for symbol availability, but because -DN2P2_FULL_SFD_MEMORY
# changes Atom's binary layout (Atom.h has #ifdef N2P2_FULL_SFD_MEMORY
# guarding actual member declarations, not just runtime behavior).
# InterfaceLammps.cpp (this library) creates/manipulates Atom/Structure
# objects that Mode::calculateSymmetryFunctionGroups() (libnnp, built
# with GPU=1's defines) later operates on -- if the two libraries
# disagree on this define, that's a real ODR violation/ABI mismatch: one
# side reads/writes members at offsets the other side didn't allocate,
# which manifests as corrupted-looking data appearing well after the
# point of actual mismatch (not a crash at the mismatch site itself,
# making it easy to misdiagnose as a logic bug elsewhere -- confirmed
# by hitting exactly this while validating the symmetry-function GPU
# dispatch, soft-percolating-jellyfish.md Phase 5 step 8).
ifeq ($(GPU), 1)
OPTIONS+= -DN2P2_GPU -DN2P2_FULL_SFD_MEMORY
endif
DEBUG=$(PROJECT_DEBUG)
ifeq ($(MODE), test)
DEBUG+= $(PROJECT_TEST)
endif
AR=$(PROJECT_AR)
ARFLAGS=$(PROJECT_ARFLAGS)

# Extra include paths for compiling.
INCLUDES=-I./ -I$(PROJECT_INCLUDE)/ -I$(PROJECT_EIGEN)


###############################################################################
# DEFINE SOURCES AND HEADERS
###############################################################################

$(info The following interfaces were selected: $(INTERFACES))
IF=LAMMPS
ifneq ($(filter $(IF),$(INTERFACES)),)
SRC:=$(SRC) $(wildcard $(IF)/*.cpp)
HEADERS:=$(HEADERS) $(wildcard $(IF)/*.h)
endif

IF=CabanaMD
ifneq ($(filter $(IF),$(INTERFACES)),)
SRC:=$(SRC) $(wildcard $(IF)/*.cpp)
HEADERS:=$(HEADERS) $(wildcard $(IF)/*.h)
endif

OBJ=$(SRC:.cpp=.o)


###############################################################################
# RULES
###############################################################################
.PHONY: all \
	headers \
	lammps-prep \
	lammps-mf \
	lammps-cmake \
	clean rebuild

ifeq ($(MODE), shared)
all: headers lammps-prep $(LIB).so
else
all: headers lammps-prep $(LIB).a
endif

headers:
	cp $(HEADERS) $(PROJECT_INCLUDE)

LAMMPS_OPTIONS := $(filter -DN2P2_%,$(OPTIONS))
lammps-prep: lammps-mf lammps-cmake

lammps-mf:
	cp LAMMPS/Makefile.lammps $(PROJECT_LIB)/Makefile.lammps-extra
	@sed -i.bak -E "s,(hdnnp_SYSINC =).*,\1 $(LAMMPS_OPTIONS) -I$(PROJECT_EIGEN)," $(PROJECT_LIB)/Makefile.lammps-extra
ifeq ($(GPU), 1)
	@sed -i.bak -E "s,(hdnnp_SYSLIB =.*),\1 $(PROJECT_LDFLAGS_BLAS) -lnnpgpu $(PROJECT_LDFLAGS_GPU)," $(PROJECT_LIB)/Makefile.lammps-extra
else
	@sed -i.bak -E "s,(hdnnp_SYSLIB =.*),\1 $(PROJECT_LDFLAGS_BLAS)," $(PROJECT_LIB)/Makefile.lammps-extra
endif
	@rm $(PROJECT_LIB)/Makefile.lammps-extra.bak

lammps-cmake:
	$(RM) -f $(PROJECT_LIB)/lammps-extra.cmake
	touch $(PROJECT_LIB)/lammps-extra.cmake
	for flag in $(LAMMPS_OPTIONS); do \
		echo "target_compile_definitions(lammps PRIVATE $${flag})" >> $(PROJECT_LIB)/lammps-extra.cmake; \
	done

$(LIB).so: CFLAGS+= -fPIC
$(LIB).so: $(OBJ)
	$(MPICC) $(CFLAGS) $(DEBUG) $(OPTIONS) -o $@ $+ -shared
	cp $@ $(PROJECT_LIB)

$(LIB).a: $(OBJ)
	$(AR) $(ARFLAGS) $@ $+
	cp $@ $(PROJECT_LIB)

%.o: %.cpp
	$(MPICC) $(CFLAGS) $(DEBUG) $(INCLUDES) $(OPTIONS) -o $@ -c $<

clean: clean-headers clean-lammps-prep
	$(RM) $(OBJ)
	$(RM) $(OBJ:.o=.gcno) $(OBJ:.o=.gcda) $(LIB).so $(LIB).a
	$(RM) $(PROJECT_LIB)/$(LIB).so $(PROJECT_LIB)/$(LIB).a

clean-headers:
	cd $(PROJECT_INCLUDE) && $(RM) $(notdir $(HEADERS))

clean-lammps-prep:
	$(RM) $(PROJECT_LIB)/Makefile.lammps-extra
	$(RM) $(PROJECT_LIB)/lammps-extra.cmake

rebuild: clean all
