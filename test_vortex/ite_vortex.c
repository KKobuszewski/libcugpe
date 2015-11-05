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

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdint.h>

#include <ctime>
#include <queue>

#include <omp.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <complex.h>

/***************************************************************************/ 
/**************************** GPE HEADERS **********************************/
/***************************************************************************/
#include "gpe_engine.h"

/*
 * run with:
 * ./ ite_vortex.exe <
 * 
 */
int main( int argc , char ** argv ) 
{
    // SETTINGS
    double alpha=0.0;
    double beta=1.0;
    double dt=0.025;
    double npart=1000.0;
    double aspect_ratio=0.0; // to be set as cmd line parameter
    double scat_lenght=0.0;
    double r0=0.0; // distance between cetre of vortex and cetre of cloud in xy plane
    int device=0;
    
    if (argc < 3)
    {
        printf("Error: To less parameters given to program!\n");
        printf("param1: <aspect_ratio> - harmonic trap shape\n");
        printf("param2: <scat_lenght>  - scattering length of atoms in condensate\n");
        printf("param3: <r0>           - position of vortex\n");
        return EXIT_FAILURE;
    }
    else
    {
        sscanf(argv[1],"%lf",&aspect_ratio);
        sscanf(argv[2],"%lf",&scat_lenght );
        sscanf(argv[3],"%lf",&r0          );
        // TODO: Set device via cmd line!
    }
    
    int ierr;
    cudaError_t err;
    
    err=cudaSetDevice( device );
    if(err != cudaSuccess) 
        {printf("Error: Cannot cudaSetDevice(%d)!\n", device); return EXIT_FAILURE;}
    
    int nx, ny, nz;
    gpe_get_lattice(&nx, &ny, &nz);
    printf("# GPE engine compiled for lattice: %d x %d x %d\n", nx, ny, nz);
    
    uint nxyz=nx*ny*nz;
    uint /*ix, iy, iz,*/ ixyz;
    
    double ekin, eint, eext, etot, etot_prev;
    double simtime;
    double rt;
    
    
    //get current date
    char datetime[20];
    time_t t;
    time(&t);
    strftime(datetime, sizeof(datetime), "%F_%T", localtime(&t));
    
    // create directory for files
    char dirname[256];
    sprintf( dirname,"AspectRatio_%2.1lf_%dx%dx%d/ITE__a_%e__r0_%.1lf__%s",aspect_ratio,nx,ny,nz,scat_lenght,r0,datetime );
    struct stat st = {0};
    if (stat(dirname, &st) == -1) { mkdir(dirname, 0777); }
    
    
    // ***************** Imaginary time projection **************************
    printf("# IMAGINARY TIME PROJECTION\n");
    printf("\n");
    printf("# PARAMETERS OF SIMULATION:\n");
    printf("Aspect ratio:      %2.2lf\n",aspect_ratio);
    printf("Scattering lenght: %lf\n"   ,scat_lenght );
    printf("\n");
    
    // CPU memory for wave function - pinned for fast transfers
    cuCplx *psi; // Complex type defined in gpe_engine.h - structure with two doubles x and y for real and imaginary parts
    err=cudaHostAlloc((void**) &psi , sizeof(cuCplx)*nxyz, cudaHostAllocDefault );
    if(err != cudaSuccess) 
    {
        printf("Error: Cannot allocate memory!\n");
        return 1;
    }
    
    // Create engine
    gpe_exec( gpe_create_engine(alpha, beta, dt, npart), ierr );
    
    
    // Set parameters
    // TODO: change params on device to use only aspect ratio and omega_x for counting external potential
    // TODO: Corelate omega_x with 'thiner' range of cloud
    // Prepare user defined parameters
    double omega_x = 0.01;
    
    double params[4];
    params[OMEGA_X] = omega_x;                                // omega_x
    params[OMEGA_Y] = aspect_ratio*aspect_ratio*omega_x;      // omega_y
    params[OMEGA_Z] = aspect_ratio*aspect_ratio*omega_x;      // omega_z
    params[A_SCAT] = scat_lenght;
    
    // Copy parameters to engine
    gpe_exec( gpe_set_user_params(4, params), ierr );
    
    
    
/* *********************************************************************************************************************************** *
 *                                                                                                                                     *
 *                                                        FINDING GROUND STATE                                                         *
 *                                                                                                                                     *
 * *********************************************************************************************************************************** */
    
    printf("# Ground state\n");
    
    // Set initial wave function - constant
    for(ixyz=0; ixyz<nxyz; ixyz++) { psi[ixyz].x = 1.0; psi[ixyz].y = 0.0; } // Fill with initial values
    
    // Copy wave function to GPU and normalize
    gpe_exec( gpe_set_psi(0.0, psi), ierr ); // TODO: Give function as a parameter
    gpe_exec( gpe_normalize_psi(), ierr );
    
    char filename_energy[256];
    sprintf( filename_energy,"%s/energy_gs.txt",dirname );
    FILE* file_energy = fopen(filename_energy, "w");
    if (!file_energy) printf("Error: Cannot open file %s\n",filename_energy);
    
    
    // For nice printing
    printf("#%7s %12s %12s %12s %12s %12s %14s %12s\n", "time", "etot", "ekin", "eint", "eext", "(eint+eext)", "diff", "comp.time");
    fprintf(file_energy,"#%7s %16s %16s %16s %16s %16s %14s %12s\n", "time", "etot", "ekin", "eint", "eext", "(eint+eext)", "diff", "comp.time");
    
    // initial energy
    gpe_exec( gpe_energy(&simtime, &ekin, &eint, &eext), ierr );
    etot = ekin + eint + eext;
    printf("%8.2f %12.8f %12.8f %12.8f %12.8f %12.8f\n",simtime, etot/npart, ekin/npart, eint/npart, eext/npart, (eint+eext)/npart);  
    fprintf(file_energy,"%8.2f %12.15lf %12.15lf %12.15lf %12.15lf %12.15lf %12.6g %12.4f\n",simtime, etot/npart, ekin/npart, eint/npart, eext/npart, (eint+eext)/npart, 0., 0.);
    
    // evolve in imaginary time until convergence is achieved
    while(1)
    {
        b_t(); // reset timer
        
        // Evolve 100 steps forward
        gpe_exec( gpe_evolve_qf(100), ierr ); 
        
        // Compute energy 
        gpe_exec( gpe_energy(&simtime, &ekin, &eint, &eext), ierr );
        
        rt = e_t(0); // get time
        
        etot_prev=etot;
        etot = ekin + eint + eext;
        double diff=(etot_prev-etot)/npart; // diference in energy per particle
        printf("%8.2f %12.8f %12.8f %12.8f %12.8f %12.8f %12.6g %12.4f\n",simtime, etot/npart, ekin/npart, eint/npart, eext/npart, (eint+eext)/npart, diff, rt); 
        fprintf(file_energy,"%8.2f %12.15lf %12.15lf %12.15lf %12.15lf %12.15lf %12.6g %12.4f\n",
                        simtime, etot/npart, ekin/npart, eint/npart, eext/npart, (eint+eext)/npart, diff, rt);
        
        if(fabs(diff)<1.0e-14) break;
    }
    
    fclose(file_energy);
    
    // write to binary file
    printf("# Writing psi to file\n");
    
    // Get wave function and save it to file
    gpe_exec( gpe_get_psi(&simtime, psi), ierr );
    
    // Open file
    char filename_psi_gs[256];
    sprintf( filename_psi_gs,"%s/psi_gs.dftc",dirname );
    FILE* file_psi_gs = fopen(filename_psi_gs, "wb");
    if (!file_psi_gs) printf("Error: Cannot open file %s\n",filename_psi_gs);
    
    // Save
    fwrite (psi , sizeof(cuCplx)*nxyz, 1, file_psi_gs);
    fclose (file_psi_gs);
    
    // Create dftc.info file
    char filename_info_gs[256];
    sprintf( filename_info_gs,"%s.info",filename_psi_gs );
    FILE* file_info_gs = fopen(filename_info_gs,"w");
    fprintf(file_info_gs,"%d # nx\n",nx);
    fprintf(file_info_gs,"%d # ny\n",ny);
    fprintf(file_info_gs,"%d # nz\n",nz);
    fprintf(file_info_gs,"%lf # dt\n",dt);
    fprintf(file_info_gs,"1 # nom\n");
    fclose(file_info_gs);
    
    printf("\n");
    
    
/* *********************************************************************************************************************************** *
 *                                                                                                                                     *
 *                                                        IMPRINTING VORTEX                                                            *
 *                                                                                                                                     *
 * *********************************************************************************************************************************** */
    
    printf("# Imprinting vortex in ITE\n");
    printf("Vortex r0: %2.2lf",r0);
    
    // set vortex
    double vortex_x0 = 0.;
    double vortex_y0 = r0;
    gpe_exec( gpe_set_vortex(vortex_x0, vortex_y0), ierr );
    
    
    char filename_energy_vortex[256];
    sprintf( filename_energy_vortex,"%s/energy_vortex.txt",dirname );
    FILE* file_energy_vortex = fopen(filename_energy_vortex, "w");
    if (!file_energy_vortex) printf("Error: Cannot open file %s\n",filename_energy_vortex);
    
    // For nice printing
    printf("#%7s %12s %12s %12s %12s %12s %14s %12s\n", "time", "etot", "ekin", "eint", "eext", "(eint+eext)", "diff", "comp.time");
    fprintf(file_energy_vortex ,"#%7s %12s %12s %12s %12s %12s %14s %12s\n", "time", "etot", "ekin", "eint", "eext", "(eint+eext)", "diff", "comp.time");
    
    // initial energy
    gpe_exec( gpe_energy(&simtime, &ekin, &eint, &eext), ierr );
    etot = ekin + eint + eext;
    printf("%8.2f %12.8f %12.8f %12.8f %12.8f %12.8f\n",simtime, etot/npart, ekin/npart, eint/npart, eext/npart, (eint+eext)/npart);  
    fprintf(file_energy_vortex ,"%8.2f %12.15lf %12.15lf %12.15lf %12.15lf %12.15lf %12.6g %12.4f\n",simtime, etot/npart, ekin/npart, eint/npart, eext/npart, (eint+eext)/npart, 0., 0.);
    
    // evolve in imaginary time until convergence is achieved
    while(1)
    {
        b_t(); // reset timer
        
        // Evolve 100 steps forward
        gpe_exec( gpe_evolve_vortex(100), ierr ); 
        
        // Compute energy 
        gpe_exec( gpe_energy(&simtime, &ekin, &eint, &eext), ierr );
        
        rt = e_t(0); // get time
        
        etot_prev=etot;
        etot = ekin + eint + eext;
        double diff=(etot_prev-etot)/npart; // diference in energy per particle
        printf("%8.2f %12.8f %12.8f %12.8f %12.8f %12.8f %12.6g %12.4f\n",simtime, etot/npart, ekin/npart, eint/npart, eext/npart, (eint+eext)/npart, diff, rt); 
        fprintf(file_energy_vortex ,"%8.2f %12.15lf %12.15lf %12.15lf %12.15lf %12.15lf %12.8f %12.6g %12.4f\n",
                                simtime, etot/npart, ekin/npart, eint/npart, eext/npart, (eint+eext)/npart, diff, rt);
        
        if(fabs(diff)<1.0e-14) break;
    }
    
    fclose(file_energy_vortex);
    
    // write to binary file
    printf("# Writing psi to file\n");
    
    // Get wave function and save it to file
    gpe_exec( gpe_get_psi(&simtime, psi), ierr ) ; 
    
    
    // Open file
    char filename_vortex[256];
    sprintf( filename_vortex,"%s/psi_vortex.dftc",dirname );
    FILE* file_vortex = fopen(filename_vortex, "wb");
    if (!file_vortex) printf("Error: Cannot open file %s\n",filename_vortex);
    
    fwrite (psi , sizeof(cuCplx)*nxyz, 1, file_vortex);
    fclose (file_vortex);
    
    // Create dftc.info file
    char filename_info_vortex[256];
    sprintf( filename_info_vortex,"%s.info",filename_vortex );
    FILE* file_info_vortex = fopen(filename_info_vortex,"w");
    fprintf(file_info_vortex,"%d # nx\n",nx);
    fprintf(file_info_vortex,"%d # ny\n",ny);
    fprintf(file_info_vortex,"%d # nz\n",nz);
    fprintf(file_info_vortex,"%lf # dt\n",dt);
    fprintf(file_info_vortex,"%d # nom\n",nom);
    fprintf(file_info_vortex,"%e # a_scattering\n",scat_lenght);
    fprintf(file_info_vortex,"%lf # \n");
    fprintf(file_info_vortex,"1 # nom\n");
    fclose(file_info_vortex);
    
    // Destroy engine
    gpe_exec( gpe_destroy_engine(), ierr) ;
    
    // Clear memory
    cudaFreeHost(psi);
    
    return 0;
}

static inline void save_info(char* dftc_filename,
                      const int nx,
                      const int ny,
                      const int nz,
                      const double dt,
                      const int nom,
                      const double scat_lenght,
                      const double 