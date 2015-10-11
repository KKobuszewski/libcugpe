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
#include <cufft.h>
#include <math.h>
#include <complex.h>

#ifndef DEV
typedef cuDoubleComplex cuCplx;
#endif

/***************************************************************************/ 
/**************************** GPE HEADERS **********************************/
/***************************************************************************/
#include "gpe_engine.h"
// Only to do timing
#include "gpe_timing.h" 
// #include "gpe_user_defined.h"

/***************************************************************************/ 
/************************* MAIN FUNCTION  **********************************/
/***************************************************************************/
int main( int argc , char ** argv ) 
{
    // SETTINGS
    double alpha=0.0;
    double beta=1.0;
    double dt=0.025;
    double npart=1000.0;
    int device=0;
    
    int ierr;
    cudaError_t err;
    
    err=cudaSetDevice( device );
    if(err != cudaSuccess) 
    {
        printf("Error: Cannot cudaSetDevice(%d)!\n", device);
        return 1;
    }
    
    int nx, ny, nz;
    gpe_get_lattice(&nx, &ny, &nz);
    printf("# GPE engine compiled for lattice: %d x %d x %d\n", nx, ny, nz);
    
    uint nxyz=nx*ny*nz;
    uint ix, iy, iz, ixyz;
    
    double ekin, eint, eext, etot, etot_prev;
    double time;
    double rt;
  
    // ***************** Imaginary time projection **************************
    printf("# IMAGINARY TIME PROJECTION\n");
    
    // CPU memory for wave function - pinned for fast transfers
    cuCplx *psi; // Complex type defined in gpe_engine.h - structure with two doubles x and y for real and imaginary parts
    err=cudaHostAlloc((void**) &psi , sizeof(cuCplx)*nxyz, cudaHostAllocDefault );
    if(err != cudaSuccess) 
    {
        printf("Error: Cannot allocate memory!\n");
        return 1;
    }
    
    // Set initial wave function - constant
    for(ixyz=0; ixyz<nxyz; ixyz++) { psi[ixyz].x = 1.0; psi[ixyz].y = 0.0; } // Fill with initial values
    
    // Create engine
    gpe_exec( gpe_create_engine(alpha, beta, dt, npart), ierr );
    
    // Prepare user defined parameters
    double params[3];
    params[0] = 0.01; // omega_x
    params[1] = 0.10; // omega_y
    params[2] = 0.11; // omega_z
    
    // Copy parameters to engine
    gpe_exec( gpe_set_user_params(3, params), ierr );
    
    // Copy wave function to GPU and normalize
    gpe_exec( gpe_set_psi(0.0, psi), ierr ) ;
    gpe_exec( gpe_normalize_psi(), ierr );
    
    
#ifdef DEV
    FILE * fout = fopen("energy_ite_dev.txt", "w");
#else
    FILE * fout = fopen("energy_ite_orginal.txt", "w");
#endif
    
    // For nice printing
    printf("#%7s %12s %12s %12s %12s %12s %14s %12s\n", "time", "etot", "ekin", "eint", "eext", "(eint+eext)", "diff", "comp.time");
    fprintf(fout,"#%7s %12s %12s %12s %12s %12s %14s %12s\n", "time", "etot", "ekin", "eint", "eext", "(eint+eext)", "diff", "comp.time");
    
    // initial energy
    gpe_exec( gpe_energy(&time, &ekin, &eint, &eext), ierr );
    etot = ekin + eint + eext;
    printf("%8.2f %12.8f %12.8f %12.8f %12.8f %12.8f\n",time, etot/npart, ekin/npart, eint/npart, eext/npart, (eint+eext)/npart);  
    fprintf(fout,"%8.2f %12.8f %12.8f %12.8f %12.8f %12.8f\n",time, etot/npart, ekin/npart, eint/npart, eext/npart, (eint+eext)/npart);
    
    // evolve in imaginary time until convergence is achieved
    while(1)
    {
        b_t(); // reset timer
        
        // Evolve 100 steps forward
        gpe_exec( gpe_evolve_qf(100), ierr ); 
        
        // Compute energy 
        gpe_exec( gpe_energy(&time, &ekin, &eint, &eext), ierr );
        
        rt = e_t(0); // get time
        
        etot_prev=etot;
        etot = ekin + eint + eext;
        double diff=(etot_prev-etot)/npart; // diference in energy per particle
        printf("%8.2f %12.8f %12.8f %12.8f %12.8f %12.8f %12.6g %12.4f\n",time, etot/npart, ekin/npart, eint/npart, eext/npart, (eint+eext)/npart, diff, rt); 
        fprintf(fout,"%8.2f %12.8f %12.8f %12.8f %12.8f %12.8f %12.6g %12.4f\n",time, etot/npart, ekin/npart, eint/npart, eext/npart, (eint+eext)/npart, diff, rt);
        
        if(fabs(diff)<1.0e-5) break;
    }    
    
    fclose(fout);
    
    
    // Get wave function and save it to file
    gpe_exec( gpe_get_psi(&time, psi), ierr ) ;    
            
    // write to binary file
    printf("# Writing psi to file\n");
#ifdef DEV
    FILE *psiFile = fopen ("psi_dev.bin", "wb");
#else
    FILE *psiFile = fopen ("psi_orginal.bin", "wb");
#endif
    fwrite (psi , sizeof(cuCplx)*nxyz, 1, psiFile);
    fclose (psiFile);
    
    // Destroy engine
    gpe_exec( gpe_destroy_engine(), ierr) ;
    
    // Clear memory
    cudaFreeHost(psi);
    
    printf("%15.15lf + %15.15lfj\n",creal(cexp(1+2*I)),cimag(cexp(1+2*I)));
    
    return 0;
}