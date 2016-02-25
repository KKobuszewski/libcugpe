/* *********************************************************************** *
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
 *                                                                         *
 * *********************************************************************** */ 
 
#ifndef __GPE_VORTEX_DETECT_H__
#define __GPE_VORTEX_DETECT_H__

#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include <omp.h>

#include "gpe_engine.h"

#ifndef NEI_SIZE
#define NEI_SIZE 0.05
#endif


/***************************************************************************/ 
/*************************** VORTEX DETECTION ******************************/
/***************************************************************************/

#define VORTEX_CORE 1
#define NO_VORTEX_CORE 0
/*
 * returns if point with neighbourhood given by array *idx is a vortexcore
 * 
 * @Author: Gabriel Wlazłowski
 */
static inline int check_if_vortex_core(int *idx, double complex *psi)
{
    double tmp2;
    int tmpi=-1;
    int i, j;
    
    
    // from the smallest
    tmp2=99.;
    for(i=0; i<8; i++) if(carg(psi[idx[i]])<tmp2) {tmp2=carg(psi[idx[i]]); tmpi=i; }
    j=0;
    for(i=0; i<7; i++) if(carg(psi[idx[(i+tmpi)%8]])<carg(psi[idx[(i+tmpi+1)%8]])) j++;
    if(j==7 && ( carg(psi[idx[(7+tmpi)%8]]) - carg(psi[idx[tmpi]]) )>1.0*M_PI ) return VORTEX_CORE;
    
    // from the biggest
    tmp2=-99.;
    for(i=0; i<8; i++) if(carg(psi[idx[i]])>tmp2) {tmp2=carg(psi[idx[i]]); tmpi=i; }
    j=0;
    for(i=0; i<7; i++) if(carg(psi[idx[(i+tmpi)%8]])>carg(psi[idx[(i+tmpi+1)%8]])) j++;
    if(j==7 && ( carg(psi[idx[tmpi]]) - carg(psi[idx[(7+tmpi)%8]]) )>1.5*M_PI ) return VORTEX_CORE;
    
    return NO_VORTEX_CORE;
}


/*
 * This function takes double complex array representing wavefunction at particular moment of time.
 * Returns vortex x-index and y-index stored in CUDA's double2 struct.
 * @param iz     - z-index of x-y plane to be searched
 * @param psi    - pointer to array represanting wavefunction on the lattice
 */
static inline double2 find_vortex_in_plane(const int iz, 
                                           double complex *psi, 
                                           double2 vortex_ixiy_prev, 
                                           const double density_eps = 1e-10)
{
    register double2 vortex_ixiy;  // for storing position index in 
    vortex_ixiy.x = NAN;
    vortex_ixiy.y = NAN;
    
    register int nx, ny, nz;
    gpe_get_lattice(&nx, &ny, &nz);
    
    register int idx[8];                             // for indices of current points' neighbourhood
    register int c_x[8]={ 1, 1, 1, 0,-1,-1,-1, 0 };
    register int c_y[8]={ 1, 0,-1,-1,-1, 0, 1, 1 };
    
    register int ix, iy, i;
    register double nei_size = NEI_SIZE;
    
    // neighbourhood size
    if ( isnan(vortex_ixiy_prev.x) )  nei_size *= 3;
    
    // Detect vortex position
    for ( ix = 0; ix < nx; ix++ )
    {
        for ( iy = 0; iy < ny; iy++ )
        {
            if ( fabs( ((double) ix - vortex_ixiy_prev.x )) < ((double) nx)*nei_size  &&
                 fabs( ((double) iy - vortex_ixiy_prev.y )) < ((double) ny)*nei_size  &&
                 (ix < nx-1) && (ix > 0) && (iy < ny-1) && (iy > 0) )
            {
                // create array with neighbourhood indices
                for(i=0; i<8; i++) ixiyiz2ixyz(idx[i], (ix+c_x[i]), (iy+c_y[i]), iz, ny, nz);
                
                // check if norm is not too small
                uint8_t density_criterion = 1;
                for(i=0; i<8; i++) if ( cabs(psi[idx[i]])*cabs(psi[idx[i]]) < density_eps ) density_criterion = 0.;
				if ( density_criterion ) 
                {
					// check rotation of phase
					if (check_if_vortex_core(idx, psi)) // VORTEX_CORE == 1 - only one possible, so break loop
					{
						vortex_ixiy.x = (double) ix;
						vortex_ixiy.y = (double) iy;
						//printf("Vortex detected! (%d,%d)\n",ix,iy);
						return vortex_ixiy;
					}
				}
            }
        }
    }
    
    return vortex_ixiy;
}


/*
 * @param psi - wavefunction
 * @param arr_vortex_ixiy - array with last vortex posiotns in different planes
 * @param planes_num num
 */
int find_vortex_in_volume(double complex* psi, 
                          double2 *arr_vortex_ixiy, 
                          const int planes_num = 1, 
                          const double density_eps = 1e-10)
{
    double2 tmp_vortex_ixiy;
    int nx, ny, nz;
    gpe_get_lattice(&nx, &ny, &nz);
    
    if ( planes_num > nz ) {printf("Error! To many planes given to "); return -1;} // error!
    
    #pragma omp parallel for shared(nz, nx, ny, psi, arr_vortex_ixiy)
    for (int ii = 0; ii < planes_num; ii++)
    {
        int iz = nz/2 - planes_num/2 + ii;                                                    // z-index of a x-y plane, where looking for a vortex - using only indices in centre of a cloud
        tmp_vortex_ixiy = find_vortex_in_plane(iz, psi, arr_vortex_ixiy[ii], density_eps);
        
        arr_vortex_ixiy[ii] = tmp_vortex_ixiy;
    }
    
    return 0;
}



#endif
