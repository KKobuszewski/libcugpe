# LIBRARY PATH
CUGPE_HOME = /home/konrad/CProjects/libcugpe


# LATTICE
NX = 256
NY = 32
NZ = 32



# SIMULATION TYPE PARAMETERS

# CHOOSE WHICH TYPE OF PROGRAM TO COMPILE
PROG_DESC = fer_unitary
INTERACTIONS = 0
# PROG_DESC = fer_bec
# INTERACTIONS = 1


# ifdef $(PROG_DESC)
# 	ifeq ($(PROG_DESC), unitary)
# 		INTERACTIONS = 0
# 	endif
# 	ifeq ($(PROG_DESC), nonunitary)
# 		INTERACTIONS = 1
# 	endif
# 	# GIVE MORE OPTIONS
# 	# TODO: GPE FOR BOSONS WITH DEFINITION OF GPE_FOR PARTICLES
# else 
# 	# unitary gas by default
# 	PROG_DESC = unitary
# 	INTERACTIONS = 0
# endif


# GPU COMPUTE CAPABILITY
GPU_ARCH = sm_52

# c compiler
CC 	  = gcc
CC_FLAGS  = -Wall -Wundef -m64 -march=native -O3 -msse4 -ftree-vectorizer-verbose=1 -fopenmp -fPIC
CC_INC    = -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I. -I$(CUGPE_HOME)/dev/
CC_DEFS   = -DNX=$(NX) -DNY=$(NY) -DNZ=$(NZ) -DINTERACTIONS=$(INTERACTIONS)
CC_LIBS   = -lcudart -lgomp -lpthread -lm

# c++ compiler
CXX	  = g++
CXX_FLAGS = -Wall -Wundef -m64 -march=native -O3 -msse4 -ftree-vectorizer-verbose=1 -fopenmp -fPIC
CXX_INC   = -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I. -I$(CUGPE_HOME)/dev/
CXX_DEFS  = -DNX=$(NX) -DNY=$(NY) -DNZ=$(NZ) -DINTERACTIONS=$(INTERACTIONS)
CXX_PATH  = -L$(CUGPE_HOME)/lib/
CXX_LIBS  = -lcudart -lgomp -lpthread -lm

# cuda compiler
NVCC      = /usr/local/cuda-7.0/bin/nvcc
NVCC_FLAGS= -m64 -O3 -arch $(GPU_ARCH) -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used" -Xcompiler -fPIC
NVCC_INC  = -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I. -I$(CUGPE_HOME)/dev/
NVCC_DEFS = -DNX=$(NX) -DNY=$(NY) -DNZ=$(NZ) -DINTERACTIONS=$(INTERACTIONS)
NVCC_LIBS = -lcudart -lcufft -lm -lcugpe_$(PROG_DESC)_$(NX)x$(NY)x$(NZ)

# linker
LD        = $(NVCC)
LD_FLAGS  = -arch $(GPU_ARCH)


# PRGORAMS' NAMES
PROG_ITE  = ite_vortex_$(PROG_DESC)_$(NX)x$(NY)x$(NZ).exe
PROG_RTE  = rte_vortex_$(PROG_DESC)_$(NX)x$(NY)x$(NZ).exe

# FILES
SRC_ITE   = ite_vortex.c
SRC_RTE   = rte_vortex.c
OBJS_ITE  = ite_vortex.o
OBJS_RTE  = rte_vortex.o




# OPTIONS
all : $(PROG_ITE) $(PROG_RTE)

ite : $(PROG_ITE)

rte : $(PROG_RTE)


# COMPILATION AND LINKING COMANDS
$(PROG_ITE) :	
		@echo ""
		@echo "compiling for $(PROG_DESC), constant $(INTERACTIONS)"
		$(CUGPE_HOME)/check_lib.sh $(NX) $(NY) $(NZ) $(PROG_DESC) $(INTERACTIONS)
		$(CXX) -c $(SRC_ITE) $(CXX_FLAGS) $(CXX_INC) $(CXX_DEFS)
		$(LD) -o $@  $(OBJS_ITE) $(LD_FLAGS) $(CXX_PATH) $(NVCC_LIBS) $(CXX_LIBS)
		make clean
		@echo ""

$(PROG_RTE) :	
		@echo ""
		@echo "compiling for $(PROG_DESC), constant $(INTERACTIONS)"
		$(CUGPE_HOME)/check_lib.sh $(NX) $(NY) $(NZ) $(PROG_DESC) $(INTERACTIONS)
		$(CXX) -c $(SRC_RTE) $(CXX_FLAGS) $(CXX_INC) $(CXX_DEFS)
		$(LD) -o $@  $(OBJS_RTE) $(LD_FLAGS) $(CXX_PATH) $(NVCC_LIBS) $(CXX_LIBS)
		make clean
		@echo ""

# .SUFFIXES:
# 
# %.o:	%.cu
# 	$(NVCC) -o $@ -c $< $(NVCC_FLAGS) $(NVCC_INC) $(NVCC_DEFS)
# 
# %.o:	%.c
# 	$(CC) -o $@ -c $< $(CC_FLAGS) $(CC_INC) $(CC_DEFS)
# 
# /%.o:	%.cpp
# 	$(CXX) -o $@ -c $< $(CXX_FLAGS) $(CXX_INC) $(CXX_DEFS)

clean: 
	rm -f *.o *.linkinfo