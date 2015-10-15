
# LATTICE
NX = 256
NY = 32
NZ = 32

# GPU COMPUTE CAPABILITY
GPU_ARCH = sm_52

# c compiler
CC 	  = gcc
CC_FLAGS  = -Wall -Wundef -m64 -march=native -O3 -msse4 -ftree-vectorizer-verbose=1 -fopenmp -fPIC
CC_INC    = -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I.
CC_DEFS   = -DNX=$(NX) -DNY=$(NY) -DNZ=$(NZ)
CC_LIBS   = -lcudart -lgomp -lpthread -lm

# c++ compiler
CXX	  = g++
CXX_FLAGS = -Wall -Wundef -m64 -march=native -O3 -msse4 -ftree-vectorizer-verbose=1 -fopenmp -fPIC
CXX_INC   = -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I.
CXX_DEFS  = -DNX=$(NX) -DNY=$(NY) -DNZ=$(NZ)
CXX_LIBS  = -lcudart -lgomp -lpthread -lm

# cuda compiler
NVCC      = /usr/local/cuda-7.0/bin/nvcc
NVCC_FLAGS= -m64 -O3 -arch $(GPU_ARCH) -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used" -Xcompiler -fPIC
NVCC_INC  = -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I.
NVCC_DEFS = -DNX=$(NX) -DNY=$(NY) -DNZ=$(NZ)
NVCC_LIBS = -lcudart -lcufft -lm

# linker
LD        = $(NVCC)
LD_FLAGS  = -arch $(GPU_ARCH)


# LIBRARY NAME
CUGPELIB  = ../lib/libcugpe.a

# FILES
SRC       = gpe_engine.cu reductions.cu
OBJS      = ../lib/gpe_engine.o ../lib/reductions.o

all : $(CUGPELIB)

# TODO:
# think one more time about this
# and read carefully:
# http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#using-separate-compilation-in-cuda
$(CUGPELIB) :	#$(OBJS)
		$(NVCC) -dc $(SRC) $(NVCC_FLAGS) $(NVCC_INC) $(NVCC_DEFS)
		mv *.o ../lib/
		$(LD) -o ../lib/libcugpe.o -dlink $(OBJS) $(LD_FLAGS) $(NVCC_LIBS) $(CXX_LIBS) $(CC_LIBS)
		$(LD) --lib -o $@  $(OBJS) $(LD_FLAGS) $(NVCC_LIBS) $(CXX_LIBS) $(CC_LIBS)

.SUFFIXES:

%.o:	%.cu
	$(NVCC) -o ../lib/$@ -c $< $(NVCC_FLAGS) $(NVCC_INC) $(NVCC_DEFS)

%.o:	%.c
	$(CC) -o ../lib/$@ -c $< $(CC_FLAGS) $(CC_INC) $(CC_DEFS)

/%.o:	%.cpp
	$(CXX) -o ../lib/$@ -c $< $(CXX_FLAGS) $(CXX_INC) $(CXX_DEFS)

clean: 
	rm -f ../lib/*.o ../lib/*.linkinfo ../lib/*.a