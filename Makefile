#############################################################################
#
# Author:  wyh
# Date:    2018.5.16
#
# Description:
# ------------
# This is an easily customizable makefile template. The purpose is to
# provide an instant building environment for C/C++ programs.
# may or may not work.
#
# Make Target:
# ------------
# The Makefile provides the following targets to make:
#   $ make           compile and link
#   $ make NODEP=yes compile and link without generating dependencies
#   $ make objs      compile only (no linking)
#   $ make tags      create tags for Emacs editor
#   $ make ctags     create ctags for VI editor
#   $ make clean     clean objects and the executable file
#   $ make distclean clean objects, the executable and dependencies
#   $ make help      get the usage of the makefile
#
#===========================================================================

## Customizable Section: adapt those variables to suit your program.
##==========================================================================
#PROGRAM_REVISION
PROGRAM_REVISION= "V1.0.0"  

#CUDA
CUDA = /usr/local/cuda
CUDA_INCLUDE = $(CUDA)/include
CUDA_LIBRARY = $(CUDA)/lib64


#include
INCLUDEDIR= -I $(CUDA_INCLUDE) 
 
# The directories in which source files reside.
# # If not specified, only the current directory will be serached.
SRCDIRS   = ./  

# The pre-processor and compiler options.
# MY_CFLAGS =
MY_CFLAGS = 
#-D PRINTDEBUG

# The linker options.
MY_LIBS   =  -lpthread -lrt -lm -L$(CUDA_LIBRARY) -lcudart  -ldl -lcublas
#-crs 

# The pre-processor options used by the cpp (man cpp for more).
CPPFLAGS  = $(INCLUDEDIR) 

# The options used in linking as well as in any direct use of ld.
LDFLAGS   = 


# The pre-processor and compiler options.
# Users can override those variables from the command line.
CFLAGS  = -O0 -m64 
#CXXFLAGS= -g -std=c++0x 
CXXFLAGS= -O0 -m64 

# The executable file name.
# If not specified, current directory name or `a.out' will be used.
PROGRAM   = sha3Test

## Implicit Section: change the following only when necessary.
##==========================================================================
GENCODE_SM30    :=  -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_SM60    :=  -gencode arch=compute_60,code=\"sm_60,compute_60\"   -gencode arch=compute_30,code=sm_30

# The source file types (headers excluded).
# .c indicates C source files, and others C++ ones.
SRCEXTS = .c .C .cc .cpp .CPP .c++ .cxx .cp .s .S .cu

# The header file types.
HDREXTS = .h .H .hh .hpp .HPP .h++ .hxx .hp

# The C program compiler.
CC     = g++

# The C++ program compiler.
CXX    = g++

#The CUDA program compiler.
NVCC = $(CUDA)/bin/nvcc

# Un-comment the following line to compile C programs as C++ ones.
#CC     = $(CXX)

# The command used to delete file.
#RM     = rm -f

ETAGS = etags
ETAGSFLAGS =

CTAGS = ctags
CTAGSFLAGS =

## Stable Section: usually no need to be changed. But you can add more.
##==========================================================================
SHELL   = /bin/sh
EMPTY   =
SPACE   = $(EMPTY) $(EMPTY)
ifeq ($(PROGRAM),)
  CUR_PATH_NAMES = $(subst /,$(SPACE),$(subst $(SPACE),_,$(CURDIR)))
  PROGRAM = $(word $(words $(CUR_PATH_NAMES)),$(CUR_PATH_NAMES))
  ifeq ($(PROGRAM),)
    PROGRAM = a.out
  endif
endif
ifeq ($(SRCDIRS),)
  SRCDIRS = .
endif

ifeq ($(PROGRAM_REVISION),)
  PROGRAM_REVISION = V3.0.0
endif

SOURCES = $(foreach d,$(SRCDIRS),$(wildcard $(addprefix $(d)/*,$(SRCEXTS))))
HEADERS = $(foreach d,$(SRCDIRS),$(wildcard $(addprefix $(d)/*,$(HDREXTS))))
SRC_CXX = $(filter-out %.c,$(SOURCES))
OBJS    = $(addsuffix .o, $(basename $(SOURCES)))

DEPS    = $(OBJS:.o=.d)

## Define some useful variables.
DEP_OPT = $(shell if `$(CC) --version | grep "GCC" >/dev/null`; then \
                  echo "-MM -MP"; else echo "-M"; fi )
DEPEND      = $(CC)  $(DEP_OPT)  $(MY_CFLAGS) $(CFLAGS) $(CPPFLAGS)
DEPEND.d    = $(subst -g ,,$(DEPEND))
COMPILE.c   = $(CC)  $(MY_CFLAGS) $(CFLAGS)   $(CPPFLAGS)  -DPROGRAM_REVISION=$(PROGRAM_REVISION) -c
COMPILE.cxx = $(CXX) $(MY_CFLAGS) $(CXXFLAGS) $(CPPFLAGS) -DPROGRAM_REVISION=$(PROGRAM_REVISION) -c 
COMPILE.cu  = $(NVCC) $(MY_CFLAGS) $(CXXFLAGS) $(CPPFLAGS) -DPROGRAM_REVISION=$(PROGRAM_REVISION) --ptxas-options="-v" $(GENCODE_SM30) -c 
LINK.c      = $(CC)  $(MY_CFLAGS) $(CFLAGS)   $(CPPFLAGS) $(LDFLAGS) -DPROGRAM_REVISION=$(PROGRAM_REVISION)
LINK.cxx    = $(CXX) $(MY_CFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) -DPROGRAM_REVISION=$(PROGRAM_REVISION)

.PHONY: all objs tags ctags clean distclean help show

# Delete the default suffixes
.SUFFIXES:

all: $(PROGRAM)

# Rules for generating object files (.o).
#----------------------------------------
objs:$(OBJS)

%.o:%.c
	$(COMPILE.c) $< -o $@

%.o:%.C
	$(COMPILE.cxx) $< -o $@

%.o:%.cc
	$(COMPILE.cxx) $< -o $@

%.o:%.cpp
	$(COMPILE.cxx) $< -o $@

%.o:%.CPP
	$(COMPILE.cxx) $< -o $@

%.o:%.c++
	$(COMPILE.cxx) $< -o $@

%.o:%.cp
	$(COMPILE.cxx) $< -o $@

%.o:%.cxx
	$(COMPILE.cxx) $< -o $@

%.o:%.s
	$(COMPILE.c) $< -o $@

%.o:%.S
	$(COMPILE.c) $< -o $@

%.o:%.cu
	$(COMPILE.cu) $< -o $@


# Rules for generating the tags.
#-------------------------------------
#tags: $(HEADERS) $(SOURCES)
#	$(ETAGS) $(ETAGSFLAGS) $(HEADERS) $(SOURCES)

#ctags: $(HEADERS) $(SOURCES)
#	$(CTAGS) $(CTAGSFLAGS) $(HEADERS) $(SOURCES)

# Rules for generating the executable.
#-------------------------------------
$(PROGRAM):$(OBJS)
ifeq ($(SRC_CXX),)              # C program
ifdef IS_LIB
	cd ./0mq && ar x ./lib0mq.a 
	ar -o $@.a $(MY_LIBS) $(OBJS) ./0mq/*.o 
	cd ./0mq && rm -rf ./*.o
	@echo include ./$@.a to enjoy your program.
else
	$(LINK.c)   $(OBJS) $(MY_LIBS) -o $@
	@echo Type ./$@ to execute the program.
endif
else                            # C++ program
ifdef IS_LIB
	cd ./0mq && ar x ./lib0mq.a 
	ar $(MY_LIBS) $(OBJS) ./0mq/*.o -o $@.a 
	cd ./0mq && rm -rf ./*.o
	@echo include ./$@.a to enjoy your program.
else
	$(LINK.cxx) $(OBJS) $(MY_LIBS) -o $@
	@echo Type ./$@ to execute the program.
endif
endif

ifndef NODEP
ifneq ($(DEPS),)
  sinclude $(DEPS)
endif
endif

clean:
	$(RM) $(OBJS) 
	#$(PROGRAM) $(PROGRAM).exe

distclean: clean
	$(RM) $(DEPS) TAGS

# Show help.
help:
	@echo 'xq5 hash'
	@echo 'Usage: make [TARGET]'
	@echo 'TARGETS:'
	@echo '  IS_LIB=1  make release lib.'
	@echo '  all       (=make) compile and link.'
	@echo '  NODEP=yes make without generating dependencies.'
	@echo '  objs      compile only (no linking).'
	@echo '  tags      create tags for Emacs editor.'
	@echo '  ctags     create ctags for VI editor.'
	@echo '  clean     clean objects and the executable file.'
	@echo '  distclean clean objects, the executable and dependencies.'
	@echo '  show      show variables (for debug use only).'
	@echo '  help      print this message.'
	@echo

# Show variables (for debug use only.)
show:
	@echo 'PROGRAM     :' $(PROGRAM)
	@echo 'SRCDIRS     :' $(SRCDIRS)
	@echo 'HEADERS     :' $(HEADERS)
	@echo 'SOURCES     :' $(SOURCES)
	@echo 'SRC_CXX     :' $(SRC_CXX)
	@echo 'OBJS        :' $(OBJS)
	@echo 'DEPS        :' $(DEPS)
	@echo 'DEPEND      :' $(DEPEND)
	@echo 'COMPILE.c   :' $(COMPILE.c)
	@echo 'COMPILE.cxx :' $(COMPILE.cxx)
	@echo 'link.c      :' $(LINK.c)
	@echo 'link.cxx    :' $(LINK.cxx)

## End of the Makefile ##  Suggestions are welcome  ## All rights reserved ##
#############################################################################
