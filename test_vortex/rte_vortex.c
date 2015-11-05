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
#include <cstring>
#include <iostream>

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
#include "gpe_queue.h"


/***************************************************************************/ 
/****************************** GPE QUEUE **********************************/
/***************************************************************************/

gpe_queue wf_queue;

#pragma GCC push_options
#pragma GCC optimize ("O0")
inline void process_gpe_queue(FILE* psiFramesFile, const uint64_t nxyz) 
{
    volatile register uint64_t counter = 0;
    cplx* psi_buffer;
    
    // wait for elements in queue or end
    while(1) {
            if (wf_queue.q.empty()) 
        {
            counter++;
            if (wf_queue.q.empty() && flag_stop) break; // stop condition for processing data
        }
        else
        {
            // there are some data to process
            psi_buffer = take_off_gpe_queue();
            // write to binary file
            printf("# Writing psi to file\n");
            //fwrite (psi_buffer, sizeof(cplx)*nxyz, 1, psiFile);
            // TODO: better performance for wrintig file
            //       http://beej.us/blog/data/parallel-programming-openmp/
            fwrite (psi_buffer, sizeof(cplx)*nxyz, 1, psiFramesFile);
        }
    }
    printf("# Ending second loop (%lu empty iterations)\n",counter);
}
#pragma GCC pop_options


/***************************************************************************/ 
/********************* READING TXT FILES  **********************************/
/***************************************************************************/



char *getword(FILE *fp)
{
    char word[100];
    int ch; 
    size_t idx ;

    for (idx=0; idx < sizeof word -1; ) {
        ch = fgetc(fp);
        if (ch == EOF) break;
        if (!isalpha(ch)) {
           if (!idx) continue; // Nothing read yet; skip this character
           else break; // we are beyond the current word
           }
        word[idx++] = tolower(ch);
        }
    if (!idx) return NULL; // No characters were successfully read
    word[idx] = '\0';
    return strdup(word);
}



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
    double aspect_ratio=0.0; // to be set as cmd line parameter
    double scat_lenght=0.0;
    char path[256];
    int device=0;
    
    int ierr;
    cudaError err;
    
    // =============== read cmd args =======================================
    if (argc < 3)
    {
        printf("Error: To less parameters given to program!\n");
        printf("param1: <aspect_ratio> - harmonic trap shape\n");
        printf("param2: <scat_lenght>  - scattering length of atoms in condensate\n");
        printf("param3: <path>         - path to file *.dftc.\n");
        return EXIT_FAILURE;
    }
    else
    {
        sscanf(argv[1],"%lf",&aspect_ratio);
        sscanf(argv[2],"%lf",&scat_lenght );
        sscanf(argv[3],"%s", &path[0]     );
        // TODO: Set device via cmd line!
    }
    
    // ============= check path =========================
    struct stat st0 = {0};
    if (stat(path, &st0) == -1) {printf("Error: Cannot find directory %s!\n",path); return EXIT_FAILURE;}
    
    
    // ============= read information from *.dftc.info =======================
    char filename_init_info
    
    
    // ============= set device ==============================================
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
//     uint ix, iy, iz, ixyz;
    
    double ekin, eint, eext, etot;
    double sim_time;
    double rt;
    
    //get current date
    char datetime[20];
    time_t t;
    time(&t);
    strftime(datetime, sizeof(datetime), "%F_%T", localtime(&t));
    
    // create directory for files
    char dirname[256];
    sprintf( dirname,"AspectRatio_%2.1lf_%dx%dx%d/RTE__a_%e__r0_%.1lf__%s",aspect_ratio,nx,ny,nz,scat_lenght,r0,datetime );;
    struct stat st = {0};
    if (stat(dirname, &st) == -1) { mkdir(dirname, 0777); }
    
    // create files for saving data
    char filename_psi[256];
    sprintf( filename_psi,"%s/psi_frames.bin",dirname );
    FILE* psiFramesFile = fopen(filename_psi,"wb");
    if (!psiFramesFile) printf("Error: Cannot open file %s\n",filename_psi);
    
    char filename_energy[256];
    sprintf( filename_energy,"%s/energy.txt",dirname );
    FILE* energyFile = fopen(filename_energy,"w");
    if (!energyFile) printf("Error: Cannot open file %s\n",filename_psi);
    
    
    printf("# REAL TIME EVOLUTION\n");
    printf("\n");
    printf("# PARAMETERS OF SIMULATION:\n");
    printf("Aspect ratio:      %2.2lf\n",aspect_ratio);
    printf("Scattering lenght: %lf\n"   ,scat_lenght );
    printf("\n");
    
    // ***************** Loading initial wavefunction **************************
    
    printf("# LOADING WAVEFUNTION\n");
    printf("\tprocessing file %s\n",path);
    
    
    
    // CPU memory for wave function - pinned for fast transfers
    cplx *psi_init; // Complex type defined in gpe_complex.cuh
    err=cudaHostAlloc( &psi_init , sizeof(cplx)*nxyz, cudaHostAllocDefault );
    if (err != cudaSuccess) 
    {
        printf("Error: Cannot allocate memory for initial wavefunction!\n");
        return 1;
    }
    
    // Set initial wave function - read it from file psi.dat (see gpe_imag.cu)
    printf("# Reading psi from file\n");
    FILE * psiInitFile;
    psiInitFile = fopen (path, "rb");
    size_t readok = fread (psi_init , sizeof(cplx)*nxyz, 1, psiInitFile);
    if (readok != 1)
    {
        printf("Reading error\n");
        return 1;
    }
    fclose (psiInitFile);  
    
    // Create engine
    printf("# Creating engine\n");
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
    
    
    // Copy wave function to GPU
    gpe_exec( gpe_set_psi(0.0, (cuCplx*) psi_init), ierr );
    
    err = cudaFreeHost(psi_init);
    if (err != cudaSuccess) 
    {
        printf("Error: Cannot free memory for initial wavefunction!\n");
        return 1;
    }
    
    
    // create queue and allocate memory for processing wavefunction copied from device
    printf("# Creating queue\n");
    cplx* psi = NULL;
    void* ret = create_gpe_queue(nxyz);
    if (!ret) psi = (cplx*) ret; // if psi is not NULL then the queue exists (there is enough memory to create few buffers for copying psi from device)
    
    
    // For nice printing
    printf("#%7s %12s %12s %12s %12s %12s %12s\n", "time", "etot", "ekin", "eint", "eext", "(eint+eext)", "comp.time");
    
    // initial energy
    gpe_exec( gpe_energy(&sim_time, &ekin, &eint, &eext), ierr );
    etot = ekin + eint + eext;
    printf("%8.2f %12.8f %12.8f %12.8f %12.8f %12.8f\n",sim_time, etot/npart, ekin/npart, eint/npart, eext/npart, (eint+eext)/npart);  
    
    // set OpenMP parameters
    //omp_set_num_threads(2);
    omp_set_nested(1);
    
    #pragma omp parallel num_threads(2) shared(wf_queue, psi)
    {
        int gtid = omp_get_thread_num();
        #pragma omp sections 
        {
            // 1st section - creating data
            #pragma omp section
            {
                // evolve in real time
                //while(1)
                for (int ii = 0; ii < 10; ii++)
                {
                    // allocate memory for frame of psi
//                     err=cudaHostAlloc( &psi_init , sizeof(cplx)*nxyz, cudaHostAllocDefault );
//                     if (err != cudaSuccess) 
//                     {
//                         printf("Error: Cannot allocate memory for initial wavefunction!\n");
//                         return 1;
//                     }
                    
                    // allocating memory when creating queue
                    // take care of queue_max_size
                    psi = get_gpe_queue_buffer();
                    
                    b_t(); // reset timer
                    
                    // Evolve 100 steps forward
                    gpe_exec( gpe_evolve_qf(100), ierr ); 
                    
                    // Compute energy 
                    gpe_exec( gpe_energy(&sim_time, &ekin, &eint, &eext), ierr );
                    
                    rt = e_t(); // get time
                    
                    etot = ekin + eint + eext;
                    
                    printf("%8.2f %12.8f %12.8f %12.8f %12.8f %12.8f %12.4f\n",sim_time, etot/npart, ekin/npart, eint/npart, eext/npart, (eint+eext)/npart, rt);  
                    
                    // Copy wave function from device and place in buffer
                    gpe_exec( gpe_get_psi(&sim_time, (cuCplx*) psi), ierr ) ;
                    
                    add_to_gpe_queue(psi);
                    //printf("Queue size: %lu)\n",ii,wf_queue.size());
                    
                    //if(sim_time>1000.) break;
                }    
                
                flag_stop = 1;
            }
            
            // 2nd section - processing data
            #pragma omp section
            {
                
                // here we process wavefunctions saved in queue
                // currently only saving frames
                process_gpe_queue(psiFramesFile, nxyz);
                 
                fclose(psiFramesFile);
            }
            
        } //end sections
    } // end parallel
    
        
    // Destroy engine
    gpe_exec( gpe_destroy_engine(), ierr) ;
    
    // Clear memory
    destroy_gpe_queue();
    //cudaFreeHost(psi);
    
    return 0;
}