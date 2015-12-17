
# LATTICE
NX = 256
NY = 32
NZ = 32

PROG_DESC = 
INTERACTIONS = 

# SIMULATION PARAMETERS
# change parameters with make -e PROG_DESC=
# ifdef $(PROG_DESC)
# 	ifeq ($(PROG_DESC), unitary)
# 		INTERACTIONS = 0
# 	endif
# 	ifeq ($(PROG_DESC), nonunitary)
# 		INTERACTIONS = 1
# 	endif
# 	ifeq ($(PROG_DESC), bec)
# 		INTERACTIONS = -1
# 	endif
# else 
# 	# unitary gas by default
# 	PROG_DESC = unitary
# 	INTERACTIONS = 0
# endif


# GPU COMPUTE CAPABILITY
GPU_ARCH = -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=sm_37 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52

# c compiler
CC 	  = gcc
CC_FLAGS  = -Wall -Wundef -m64 -march=native -O3 -msse4 -ftree-vectorizer-verbose=1 -fopenmp -fPIC
CC_INC    = -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I.
CC_DEFS   = -DNX=$(NX) -DNY=$(NY) -DNZ=$(NZ) -DINTERACTIONS=$(INTERACTIONS)
CC_LIBS   = -lcudart -lgomp -lpthread -lm

# c++ compiler
CXX	  = g++
CXX_FLAGS = -Wall -Wundef -m64 -march=native -O3 -msse4 -ftree-vectorizer-verbose=1 -fopenmp -fPIC
CXX_INC   = -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I.
CXX_DEFS  = -DNX=$(NX) -DNY=$(NY) -DNZ=$(NZ) -DINTERACTIONS=$(INTERACTIONS)
CXX_LIBS  = -lcudart -lgomp -lpthread -lm

# cuda compiler
NVCC      = /usr/local/cuda-7.0/bin/nvcc
NVCC_FLAGS= -m64 -O3 -arch $(GPU_ARCH) -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used" -Xcompiler -fPIC
NVCC_INC  = -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I.
NVCC_DEFS = -DNX=$(NX) -DNY=$(NY) -DNZ=$(NZ) -DINTERACTIONS=$(INTERACTIONS)
NVCC_LIBS = -lcudart -lcufft -lm

# linker
LD        = $(NVCC)
LD_FLAGS  = $(GPU_ARCH)


# LIBRARY NAME

CUGPELIB  = ../lib/libcugpe_$(PROG_DESC)_$(NX)x$(NY)x$(NZ).a


# FILES
SRC       = gpe_engine.cu reductions.cu
OBJS      = ../lib/gpe_engine.o ../lib/reductions.o

# TODO:
# think one more time about this and read carefully:
# http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#using-separate-compilation-in-cuda
$(CUGPELIB) :	
		$(NVCC) -dc $(SRC) $(NVCC_FLAGS) $(NVCC_INC) $(NVCC_DEFS)
		mv *.o ../lib/
		$(LD) -o ../lib/libcugpe.o -dlink $(OBJS) $(LD_FLAGS) $(NVCC_LIBS) $(CXX_LIBS)
		$(LD) --lib -o $@ ../lib/libcugpe.o $(OBJS) $(LD_FLAGS) $(NVCC_LIBS) $(CXX_LIBS)
#		mv $@ ../lib/libcugpe_$(NX)x$(NY)x$(NZ).a
		make clean

# all : 
# 	$(CUGPELIB)

.SUFFIXES:

%.o:	%.cu
	$(NVCC) -o ../lib/$@ -c $< $(NVCC_FLAGS) $(NVCC_INC) $(NVCC_DEFS)

%.o:	%.c
	$(CC) -o ../lib/$@ -c $< $(CC_FLAGS) $(CC_INC) $(CC_DEFS)

/%.o:	%.cpp
	$(CXX) -o ../lib/$@ -c $< $(CXX_FLAGS) $(CXX_INC) $(CXX_DEFS)

clean: 
	rm -f ../lib/*.o

purge:
	rm -f ../lib/*.o ../lib/*.linkinfo ../lib/*.a