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
 
#ifndef __GPE_CLOUD_PARAMS_H__
#define __GPE_CLOUD_PARAMS_H__

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <complex.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include <omp.h>

#include "gpe_engine.h"


extern int nx, ny, nz, nxyz; // TODO: check if define nx,ny,nz in a library are not used <- or maybe should use this?

/***************************************************************************/ 
/*************************** CLOUD PARAMETERS ******************************/
/***************************************************************************/


static inline double cabs_pow2(double complex z)                              { return creal(z)*creal(z) + cimag(z)*cimag(z); }
static inline bool compare_cplx_density(double complex z1, double complex z2) { return cabs_pow2(z1)<cabs_pow2(z2); }

/**
 * Finds density of wavefunction in lattice point [nx/2,ny/2,nz/2] (centre of a lattice).
 * Assuming wavefunction is stored in array copied to host.
 */
static inline double find_n0(double complex* psi)
{
	int index = 0;
	ixiyiz2ixyz(index,nx/2,ny/2,nz/2,ny,nz);
	double complex psi0 = psi[index];
	return GAMMA*cabs_pow2(psi0);
}

/**
 * Finds maximal density of wavefunction.
 * Assuming wavefunction is stored in array copied to host.
 */
static inline double find_max_density(double complex* psi)
{
	// TODO: Change it to OpenMP. http://stackoverflow.com/questions/11242439/finding-minimal-element-in-array-and-its-index
	return GAMMA*cabs_pow2(*std::max_element(psi,psi+nxyz,compare_cplx_density));
}


static inline double cloud_range_ix(double complex* psi, const double epsilon = 1e-08)
{
	int index = 0;
	double density;
	for ( int ix = nx/2; ix > 0; ix-- )
	{
		ixiyiz2ixyz(index,ix,ny/2,nz/2,ny,nz);
		density = GAMMA*cabs_pow2(psi[index]);
		
		if (density < epsilon) return (nx/2 - ix);
	}
	return NAN;
}

static inline double cloud_range_iy(double complex* psi, const double epsilon = 1e-08)
{
	int index = 0;
	double density;
	for ( int iy = ny/2; iy > 0; iy-- )
	{
		ixiyiz2ixyz(index,nx/2,iy,nz/2,ny,nz);
		density = GAMMA*cabs_pow2(psi[index]);
		
		if (density < epsilon) return (ny/2 - iy);
	}
	return NAN;
}

inline double cloud_range_iz(double complex* psi, const double epsilon = 1e-08)
{
	int index = 0;
	double density;
	for ( int iz = nz/2; iz > 0; iz-- )
	{
		ixiyiz2ixyz(index,nx/2,ny/2,iz,ny,nz);
		density = GAMMA*cabs_pow2(psi[index]);
		
		if (density < epsilon) return (nz/2 - iz);
	}
	return NAN;
}


#endif
