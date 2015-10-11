/***************************************************************************
 *   Copyright (C) 2015 by                                                 *
 *   WARSAW UNIVERSITY OF TECHNOLOGY                                       *
 *   FACULTY OF PHYSICS                                                    *
 *   NUCLEAR THEORY GROUP                                                  *
 *   See also AUTHORS file                                                 *
 *                                                                         *
 *   This file is a part of GPE for GPU project.                           *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "reductions.cuh"

////////////////////////////////////////////////////////////////////////////////
// LOCAL REDUCTIONS
////////////////////////////////////////////////////////////////////////////////
int opt_threads(int new_blocks,int threads, int current_size)
{
    int new_threads;
    if(new_blocks==1)
    {
        new_threads=2;
        while(new_threads<threads)
        {
            if(new_threads>=current_size) break;
            new_threads*=2;
        }
    }
    else new_threads=threads;
    return new_threads;
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile double *sdata, unsigned int tid)
{
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid +  8];
    if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
    if (blockSize >=  4) sdata[tid] += sdata[tid +  2];
    if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
}

template <unsigned int blockSize>
__global__ void __reduce_kernel__(double *g_idata, double *g_odata, int n)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int ishift=i+blockDim.x;

    if(ishift<n) sdata[tid] = g_idata[i] + g_idata[ishift];
    else if(i<n) sdata[tid] = g_idata[i];
    else         sdata[tid] = 0.0;
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >=  512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >=  256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >=  128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void call_reduction_kernel(int dimGrid, int dimBlock, int size, double *d_idata, double *d_odata, cudaStream_t stream)
{
    int smemSize=dimBlock*sizeof(double);
    switch (dimBlock)
    {
        case 1024:
            __reduce_kernel__<1024><<< dimGrid, dimBlock, smemSize , stream >>>(d_idata, d_odata, size); break;
        case 512:
            __reduce_kernel__< 512><<< dimGrid, dimBlock, smemSize , stream >>>(d_idata, d_odata, size); break;
        case 256:
            __reduce_kernel__< 256><<< dimGrid, dimBlock, smemSize , stream >>>(d_idata, d_odata, size); break;
        case 128:
            __reduce_kernel__< 128><<< dimGrid, dimBlock, smemSize , stream >>>(d_idata, d_odata, size); break;
        case 64:
            __reduce_kernel__<  64><<< dimGrid, dimBlock, smemSize , stream >>>(d_idata, d_odata, size); break;
    }
}

/**
 * Function does fast reduction (sum of elements) of array.
 * Result is located in partial_sums[0] element
 * If partial_sums==array then array will be destroyed
 * */
int local_reduction(double *array, int size, double *partial_sums, int threads, cudaStream_t stream)
{
    int blocks=(int)ceil((float)size/threads);
    unsigned int lthreads=threads/2; // Threads is always power of 2
    if(lthreads<64) lthreads=64; // at least 2*warp_size
    unsigned int new_blocks, current_size;
    
    // First reduction of the array
    call_reduction_kernel(blocks, lthreads, size, array, partial_sums,stream);
    
    // Do iteratively reduction of partial_sums
    current_size=blocks;
    while(current_size>1)
    {
        new_blocks=(int)ceil((float)current_size/threads);
        lthreads=opt_threads(new_blocks,threads, current_size)/2;
        if(lthreads<64) lthreads=64; // at least 2*warp_size
        call_reduction_kernel(new_blocks, lthreads, current_size, partial_sums, partial_sums,stream);
        current_size=new_blocks;
    }
    
    return 0;
}
