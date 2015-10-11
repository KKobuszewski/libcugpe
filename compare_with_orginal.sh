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
PROG_REAL_OLD=`echo "gpe_real_old"$NX"x"$NY"x"$NZ`
PROG_IMAG_OLD=`echo "gpe_imag_old"$NX"x"$NY"x"$NZ`
PROG_FRIC_OLD=`echo "gpe_fric_old"$NX"x"$NY"x"$NZ`

# NEW VERSION OF LIBRARY
PROG_REAL_DEV=`echo "gpe_real_dev"$NX"x"$NY"x"$NZ`
PROG_IMAG_DEV=`echo "gpe_imag_dev"$NX"x"$NY"x"$NZ`
PROG_FRIC_DEV=`echo "gpe_imag_dev"$NX"x"$NY"x"$NZ`

# COMPILE LIBRARY
echo "compiling library"
if [ ! -f ./orginal/gpe_engine.o ]; then
nvcc -c ./orginal/gpe_engine.cu -o ./orginal/gpe_engine.o -m64 -O3 -arch $GPU_ARCH -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used" -Xcompiler -fPIC -DNX=$NX -DNY=$NY -DNZ=$NZ
fi  # else there is no necessity to recompile it!

# nvcc -c ./dev/gpe_engine.cu -o ./dev/gpe_engine.o -m64 -O3 -arch $GPU_ARCH -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used" -Xcompiler -fPIC -DNX=$NX -DNY=$NY -DNZ=$NZ
cd dev
make
cd ..

# COMPILE PROGRAMS
echo "compiling: " $PROG_IMAG_OLD

g++ -c gpe_imag.c -o gpe_imag.o -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I./orginal/ -O3 -march=native -msse4 -m64 -fopenmp -DNX=$NX -DNY=$NY -DNZ=$NZ
nvcc gpe_imag.o orginal/gpe_engine.o -o $PROG_IMAG_OLD -lcudart -lcufft -lm
 
rm gpe_imag.o

echo "compiling: " $PROG_REAL_OLD
g++ -c gpe_real.c -o gpe_real.o -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I./orginal/ -O3 -march=native -msse4 -m64 -fopenmp -DNX=$NX -DNY=$NY -DNZ=$NZ
nvcc gpe_real.o orginal/gpe_engine.o -o $PROG_REAL_OLD -lcudart -lcufft -lm -lgomp -Xcompiler -fopenmp

rm gpe_real.o

# echo "compiling: " $PROG_FRIC_OLD
# nvcc -c gpe_engine.cu -o gpe_engine.o -O3 -m64 -arch $GPU_ARCH -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used" -DNX=$NX -DNY=$NY -DNZ=$NZ
# g++ -c gpe_real.cu -o gpe_real.o -I /usr/local/cuda-7.0/include -I /usr/local/cuda/samples/common/inc -O3 -msse4 -m64 -fopenmp -DNX=$NX -DNY=$NY -DNZ=$NZ
# nvcc gpe_real.o gpe_engine.o -o $PROG_REAL_OLD -lcudart -lcufft -lm -lgomp -Xcompiler -fopenmp
# 
# rm *.o

echo "compiling: " $PROG_IMAG_DEV

g++ -c gpe_imag.c -o gpe_imag.o -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I./dev/ -O3 -march=native -msse4 -m64 -fopenmp -DNX=$NX -DNY=$NY -DNZ=$NZ -DDEV
nvcc gpe_imag.o dev/libcugpe.a -o $PROG_IMAG_DEV -arch $GPU_ARCH -lcudart -lcufft -lm

rm gpe_imag.o

echo "compiling: " $PROG_REAL_DEV
g++ -c gpe_real.c -o gpe_real.o -I/usr/local/cuda-7.0/include -I/usr/local/cuda/samples/common/inc -I./dev/ -O3 -march=native -msse4 -m64 -fopenmp -DNX=$NX -DNY=$NY -DNZ=$NZ -DDEV
nvcc gpe_real.o dev/libcugpe.a -o $PROG_REAL_DEV -arch $GPU_ARCH -lcudart -lcufft -lm -lgomp -Xcompiler -fopenmp

rm gpe_real.o

# echo "compiling: " $PROG_FRIC_OLD
# nvcc -c gpe_engine.cu -o libcugpe.a -O3 -m64 -arch $GPU_ARCH -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used" -DNX=$NX -DNY=$NY -DNZ=$NZ
# g++ -c gpe_real.cu -o gpe_real.o -I /usr/local/cuda-7.0/include -I /usr/local/cuda/samples/common/inc -O3 -msse4 -m64 -fopenmp -DNX=$NX -DNY=$NY -DNZ=$NZ
# nvcc gpe_real.o libcugpe.a -o $PROG_REAL_OLD -lcudart -lcufft -lm -lgomp -Xcompiler -fopenmp
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

python3 compare.py

# REMOVE LIBRARY
#rm *.o