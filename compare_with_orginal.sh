#!/bin/bash

unset NX
unset NY
unset NZ


# CHANGE LATTICE HERE
NX=256
NY=32
NZ=32

# specify architecture of gpu
GPU_ARCH=sm_52



# TESTED VERSION OF LIBRARY
PROG_REAL_OLD=`echo "test_compare/test_real_old"$NX"x"$NY"x"$NZ`
PROG_IMAG_OLD=`echo "test_compare/test_imag_old"$NX"x"$NY"x"$NZ`
PROG_FRIC_OLD=`echo "test_compare/test_fric_old"$NX"x"$NY"x"$NZ`

# NEW VERSION OF LIBRARY
PROG_REAL_DEV=`echo "test_compare/test_real_dev"$NX"x"$NY"x"$NZ`
PROG_IMAG_DEV=`echo "test_compare/test_imag_dev"$NX"x"$NY"x"$NZ`
PROG_FRIC_DEV=`echo "test_compare/test_fric_dev"$NX"x"$NY"x"$NZ`


# COMPILE LIBRARY
echo "compiling library"

# ORGINAL VERSION OF LIBRARY
if [ ! -f ./orginal/gpe_engine.o ]; then
nvcc -c ./orginal/gpe_engine.cu -o ./orginal/gpe_engine.o -m64 -O3 -arch $GPU_ARCH -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used" -Xcompiler -fPIC -DNX=$NX -DNY=$NY -DNZ=$NZ
fi  # else there is no necessity to recompile it!

# NEW VERSION OF LIBRARY
cd dev
make clean
make
cd ..


# COMPILE PROGRAMS
echo "compiling: " $PROG_IMAG_OLD

g++ -c test_compare/gpe_imag.c -o test_compare/gpe_imag.o -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I./orginal/ -O3 -march=native -msse4 -m64 -fopenmp -DNX=$NX -DNY=$NY -DNZ=$NZ
# nvcc -c test_compare/gpe_imag.c -o test_compare/gpe_imag.o -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I./orginal/ -Xcompiler -O3 -Xcompiler -march=native -Xcompiler -msse4 -Xcompiler -m64 -Xcompiler -fopenmp -DNX=$NX -DNY=$NY -DNZ=$NZ
nvcc test_compare/gpe_imag.o orginal/gpe_engine.o -o $PROG_IMAG_OLD -lcudart -lcufft -lm
 
rm test_compare/gpe_imag.o

echo "compiling: " $PROG_REAL_OLD
g++ -c test_compare/gpe_real.c -o test_compare/gpe_real.o -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I./orginal/ -O3 -march=native -msse4 -m64 -fopenmp -DNX=$NX -DNY=$NY -DNZ=$NZ
# nvcc -c test_compare/gpe_real.c -o test_compare/gpe_real.o -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I./orginal/ -Xcompiler -O3 -Xcompiler -march=native -Xcompiler -msse4 -Xcompiler -m64 -Xcompiler -fopenmp -DNX=$NX -DNY=$NY -DNZ=$NZ
nvcc test_compare/gpe_real.o orginal/gpe_engine.o -o $PROG_REAL_OLD -lcudart -lcufft -lm -lgomp -Xcompiler -fopenmp

rm test_compare/gpe_real.o

# echo "compiling: " $PROG_FRIC_OLD
# nvcc -c gpe_engine.cu -o gpe_engine.o -O3 -m64 -arch $GPU_ARCH -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used" -DNX=$NX -DNY=$NY -DNZ=$NZ
# g++ -c test_compare/gpe_real.cu -o test_compare/gpe_real.o -I /usr/local/cuda-7.0/include -I /usr/local/cuda/samples/common/inc -O3 -msse4 -m64 -fopenmp -DNX=$NX -DNY=$NY -DNZ=$NZ
# nvcc test_compare/gpe_real.o gpe_engine.o -o $PROG_REAL_OLD -lcudart -lcufft -lm -lgomp -Xcompiler -fopenmp
# 
# rm *.o

echo "compiling: " $PROG_IMAG_DEV

g++ -c test_compare/gpe_imag.c -o test_compare/gpe_imag.o -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I./dev/ -O3 -march=native -msse4 -m64 -fopenmp -DNX=$NX -DNY=$NY -DNZ=$NZ -DDEV
# nvcc -c test_compare/gpe_imag.c -o test_compare/gpe_imag.o -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I./dev/ -Xcompiler -O3 -Xcompiler -march=native -Xcompiler -msse4 -Xcompiler -m64 -Xcompiler -fopenmp -DNX=$NX -DNY=$NY -DNZ=$NZ -DDEV
nvcc test_compare/gpe_imag.o lib/libcugpe.a -o $PROG_IMAG_DEV -arch $GPU_ARCH -lcudart -lcufft -lm

rm test_compare/gpe_imag.o

echo "compiling: " $PROG_REAL_DEV
g++ -c test_compare/gpe_real.c -o test_compare/gpe_real.o -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I./dev/ -O3 -march=native -msse4 -m64 -fopenmp -DNX=$NX -DNY=$NY -DNZ=$NZ -DDEV
# nvcc -c test_compare/gpe_real.c -o test_compare/gpe_real.o -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I./dev/ -Xcompiler -O3 -Xcompiler -march=native -Xcompiler -msse4 -Xcompiler -m64 -Xcompiler -fopenmp -DNX=$NX -DNY=$NY -DNZ=$NZ -DDEV
nvcc test_compare/gpe_real.o lib/libcugpe.a -o $PROG_REAL_DEV -arch $GPU_ARCH -lcudart -lcufft -lm -lgomp -Xcompiler -fopenmp

rm test_compare/gpe_real.o

# echo "compiling: " $PROG_FRIC_OLD
# nvcc -c gpe_engine.cu -o libcugpe.a -O3 -m64 -arch $GPU_ARCH -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used" -DNX=$NX -DNY=$NY -DNZ=$NZ
# g++ -c test_compare/gpe_real.cu -o test_compare/gpe_real.o -I /usr/local/cuda-7.0/include -I /usr/local/cuda/samples/common/inc -O3 -msse4 -m64 -fopenmp -DNX=$NX -DNY=$NY -DNZ=$NZ
# nvcc test_compare/gpe_real.o libcugpe.a -o $PROG_REAL_OLD -lcudart -lcufft -lm -lgomp -Xcompiler -fopenmp
# 
# rm *.o



# RUN PROGRAMS AND GENERATE DATA
echo "running: " $PROG_IMAG_OLD
./$PROG_IMAG_OLD

echo "running: " $PROG_REAL_OLD
./$PROG_REAL_OLD

echo "running: " $PROG_IMAG_DEV
./$PROG_IMAG_DEV

echo "running: " $PROG_REAL_DEV
./$PROG_REAL_DEV



# COMPARE RESULTS
echo "comparing results ..."

python3 test_compare/compare.py

# REMOVE LIBRARY
#rm *.o