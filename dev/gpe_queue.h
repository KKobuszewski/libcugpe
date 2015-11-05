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


#ifndef __GPE_QUEUE_H__
#define __GPE_QUEUE_H__


#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdint.h>

#include <queue>

#include <complex.h>

#include "gpe_engine.h"

uint8_t flag_stop = 0;

typedef struct
{
    std::queue<cplx*> q;
    uint16_t counter;
    uint16_t max_size;
    cplx** data;
} gpe_queue;

extern gpe_queue wf_queue;

/*
 * Use it in this way:
 * cplx* psi = NULL;
 * void* ret = create_gpe_queue(uint64_t nxyz);
 * if (!ret) psi = (cplx*) ret;
 */
static inline void* create_gpe_queue(const uint64_t nxyz) 
{
    cudaError err;
    
    std::queue<cplx*> queue;
    wf_queue.q = queue;
    
    wf_queue.counter = 0;
    
    uint16_t queue_max_size = 3;
//     uint16_t queue_max_size = ( sysconf( _SC_PHYS_PAGES ) * sysconf( _SC_PAGE_SIZE ) )/( nxyz * sizeof(cplx) );
//     queue_max_size = lround( ceil( 0.5*((double) queue_max_size) ) );
//     if (queue_max_size < 1) { printf("Error: Wavefunction is to big to allocate in RAM!\n"); exit(EXIT_FAILURE); }
    printf("Maximal sizeof queue: %u\n", queue_max_size);
    wf_queue.max_size = queue_max_size;
    
    wf_queue.data = (cplx**) malloc( queue_max_size*sizeof(cplx*) );
    for (uint16_t ii = 0; ii < queue_max_size; ii++) 
    {
        err=cudaHostAlloc( &(wf_queue.data[ii]) , sizeof(cplx)*nxyz, cudaHostAllocDefault );
        if (err != cudaSuccess) 
        {
            printf("Error: Cannot allocate memory for gpe_queue!\n");
            exit(EXIT_FAILURE);
        }
    }
    
    if (queue_max_size == 1)
        return (void*) wf_queue.data[0];
    else 
        return NULL;
}

static inline cplx* get_gpe_queue_buffer()
{
    return wf_queue.data[ wf_queue.counter % wf_queue.max_size ]; // returns poiter to one of allocated elements
}

static inline void add_to_gpe_queue(cplx* psi) 
{
     // checks if it is possible to add next element to
    while(1) { if (wf_queue.q.size() < wf_queue.max_size) {wf_queue.q.push(psi); break;} }
    wf_queue.counter++;
}

static inline cplx* take_off_gpe_queue() 
{
    cplx* psi = wf_queue.q.front();
    wf_queue.q.pop();
    return psi;
}

static inline void destroy_gpe_queue() 
{
    cudaError err;
    for (uint64_t ii = 0; ii < wf_queue.max_size; ii++) 
    {
        err=cudaFreeHost( wf_queue.data[ii] );
        if (err != cudaSuccess) 
        {
            printf("Error: Cannot allocate memory for initial wavefunction!\n");
            exit(EXIT_FAILURE);
        }
    }
    free(wf_queue.data);
}

#endif