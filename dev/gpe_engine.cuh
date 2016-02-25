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
 * @brief cuGPE library
 * */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <complex.h>      // not std::complex!
#include <cuComplex.h>
#include <cufft.h>


#ifndef __GPE_ENGINE_CUH__
#define __GPE_ENGINE_CUH__

#include "gpe_engine.h"

#ifndef GPE_FOR
#define GAMMA -9999
#endif
#if GPE_FOR == PARTICLES
#define GAMMA 1.0
#elif GPE_FOR == DIMERS
#define GAMMA 2.0
#endif

// TODO: think if this could be in constant memory!
#define nx NX
#define ny NY
#define nz NZ
#define nxyz (NX*NY*NZ)

#define GPE_QF_EPSILON 1.0e-12



static inline void check_particle_type()
{
	if (GPE_FOR == PARTICLES) printf("# GPE FOR PARTICLES\n");
	if (GPE_FOR == DIMERS   ) printf("# GPE FOR DIMERS\n");
	
	printf("GAMMA: %.2f\n\n",GAMMA);
}


#endif
