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
# Only a static library is built, in all modes. Its objects are position
# independent, so it can also be linked into shared libraries.
MODE=static

# Include global (project-wide) settings.
include $(PROJECT_DIR)/src/makefile.$(COMP)


###############################################################################
# PROJECT COMPILER AND FLAGS
###############################################################################
NVCC=$(PROJECT_NVCC)
NVCCFLAGS=$(PROJECT_NVCCFLAGS) -Xcompiler -fPIC
AR=$(PROJECT_AR)
ARFLAGS=$(PROJECT_ARFLAGS)

# Extra include paths for compiling.
INCLUDES=-I./ -I$(PROJECT_DIR)/src/libnnp


###############################################################################
# DEFINE SOURCES AND HEADERS
###############################################################################
LIB=libnnpgpu
SRC=$(wildcard *.cu)
HEADERS=$(wildcard *.h)
OBJ=$(SRC:.cu=.o)


###############################################################################
# RULES
###############################################################################
.PHONY: all headers clean rebuild

all: headers $(LIB).a

headers:
	cp $(HEADERS) $(PROJECT_INCLUDE)

$(LIB).a: $(OBJ)
	$(AR) $(ARFLAGS) $@ $+
	cp $@ $(PROJECT_LIB)

%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<

clean: clean-headers
	$(RM) $(OBJ) $(LIB).a
	$(RM) $(PROJECT_LIB)/$(LIB).a

clean-headers:
	cd $(PROJECT_INCLUDE) && $(RM) $(HEADERS)

rebuild: clean all
