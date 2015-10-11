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

/**
 * @file
 * @brief cuGPE library - reductions module
 * */

#ifndef __REDUCTIONS_CUH__
#define __REDUCTIONS_CUH__

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

//void call_reduction_kernel(int dimGrid, int dimBlock, int size, double *d_idata, double *d_odata, cudaStream_t stream);

/**
 * Function does fast reduction (sum of elements) of array.
 * Result is located in partial_sums[0] element
 * If partial_sums==array then array will be destroyed
 * @param array - 
 * @param size
 * @param partial_sums
 * @param threads
 * @param stream - stream which the reduction will be performed on
 * */
int local_reduction(double *array, int size, double *partial_sums, int threads, cudaStream_t stream);

#endif