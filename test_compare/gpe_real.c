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
#ifdef DEV
    printf("# DEVELOPMENT VERSION\n");
#else
    printf("# ORGINAL VERSION\n");
#endif
    
    // SETTINGS
    double alpha=1.0;
    double beta=0.0;
    double dt=0.025;
    double npart=1000.0;
    int device=0;
    
    int ierr;
    cudaError err;
    
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
    
    double ekin, eint, eext, etot;
    double time;
    double rt;
  
    // ***************** Imaginary time projection **************************
    printf("# REAL TIME EVOLUTION\n");
    
    // CPU memory for wave function - pinned for fast transfers
    cuCplx *psi; // Complex type defined in gpe_engine.h - structure with two doubles x and y for real and imaginary parts
    err=cudaHostAlloc((void**) &psi , sizeof(cuCplx)*nxyz, cudaHostAllocDefault );
    if(err != cudaSuccess) 
    {
        printf("Error: Cannot allocate memory!\n");
        return 1;
    }
    
    // Set initial wave function - read it from file psi.dat (see gpe_imag.cu)
    printf("# Reading psi from file\n");
    FILE * psiFile;
#ifdef DEV
    psiFile = fopen ("psi_dev.bin", "rb");
#else
    psiFile = fopen ("psi_orginal.bin", "rb");
#endif
    size_t readok = fread (psi , sizeof(cuCplx)*nxyz, 1, psiFile);
    if (readok != 1)
    {
        printf("Reading error\n");
        return 1;
    }
    fclose (psiFile);  
    
    // Create engine
    gpe_exec( gpe_create_engine(alpha, beta, dt, npart), ierr );
    
    // Prepare user defined parameters
    double params[3];
    params[0] = 0.01; // omega_x
    params[1] = 0.10; // omega_y
    params[2] = 0.11; // omega_z
    
    // Copy parameters to engine
    gpe_exec( gpe_set_user_params(3, params), ierr );
    
    // Copy wave function to GPU
    gpe_exec( gpe_set_psi(0.0, psi), ierr ) ;
    
    
#ifdef DEV
    FILE * fout = fopen("energy_rte_dev.txt", "w");
#else
    FILE * fout = fopen("energy_rte_orginal.txt", "w");
#endif
    
    // For nice printing
    printf("#%7s %12s %12s %12s %12s %12s %12s\n", "time", "etot", "ekin", "eint", "eext", "(eint+eext)", "comp.time");
    fprintf(fout,"#%7s %12s %12s %12s %12s %12s %14s %12s\n", "time", "etot", "ekin", "eint", "eext", "(eint+eext)", "diff", "comp.time");
    
    // initial energy
    gpe_exec( gpe_energy(&time, &ekin, &eint, &eext), ierr );
    etot = ekin + eint + eext;
    printf("%8.2f %12.8f %12.8f %12.8f %12.8f %12.8f\n",time, etot/npart, ekin/npart, eint/npart, eext/npart, (eint+eext)/npart);  
    fprintf(fout,"%8.2f %12.8f %12.8f %12.8f %12.8f %12.8f %12.4f\n",time, etot/npart, ekin/npart, eint/npart, eext/npart, (eint+eext)/npart, 0.);
    
    // evolve in real time
    while(1)
    {
        b_t(); // reset timer
        
        // Evolve 100 steps forward
        gpe_exec( gpe_evolve_qf(100), ierr ); 
        
        // Compute energy 
        gpe_exec( gpe_energy(&time, &ekin, &eint, &eext), ierr );
        
        rt = e_t(0); // get time
        
        etot = ekin + eint + eext;

        printf("%8.2f %12.8f %12.8f %12.8f %12.8f %12.8f %12.4f\n",time, etot/npart, ekin/npart, eint/npart, eext/npart, (eint+eext)/npart, rt);
        fprintf(fout,"%8.2f %12.8f %12.8f %12.8f %12.8f %12.8f %12.4f\n",time, etot/npart, ekin/npart, eint/npart, eext/npart, (eint+eext)/npart, rt);  
        
        if(time>10.) break;
    }  
    
    fclose(fout);  
    
    // Get wave function and save it to file
    gpe_exec( gpe_get_psi(&time, psi), ierr ) ;    
            
    // write to binary file
    printf("# Writing psi to file\n");
    FILE* psiFileOut;
#ifdef DEV
    psiFileOut = fopen ("psi_dev.dat", "wb");
#else
    psiFileOut = fopen ("psi_orginal.dat", "wb");
#endif
    fwrite (psi , sizeof(cuCplx)*nxyz, 1, psiFileOut);
    fclose (psiFileOut);
    
    
    // Destroy engine
    gpe_exec( gpe_destroy_engine(), ierr) ;
    
    // Clear memory
    cudaFreeHost(psi);
    
    return 0;
}