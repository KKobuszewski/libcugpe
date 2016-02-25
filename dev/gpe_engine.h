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


#ifndef __GPE_ENGINE__
#define __GPE_ENGINE__


#include "gpe_complex.cuh"
#include "gpe_timing.h"
#include "gpe_user_defined.h"



/* ************************************************************************************ *
 *                                                                                      *
 *                                  MACRO CONSTANTS                                     *
 *                                                                                      *
 * ************************************************************************************ */




/* ************************************************************************************ *
 *                                                                                      *
 *                                     TYPEDEFS                                         *
 *                                                                                      *
 * ************************************************************************************ */

typedef int gpe_result_t;
typedef unsigned int uint;

typedef struct
{
    double *kkx;
    double *kky;
    double *kkz;
    cufftHandle plan; // cufft plan
    cuCplx * d_wrk; // workspace for cufft - device
    cuCplx * d_wrk2; // additional work space - device
    cuCplx * d_psi; // wave function - device
    cuCplx * d_psi2; // copy of wave function - device
    double * d_phase; // additional array for storing phase of psi to be 
    double * d_wrk2R; // d_wrk2R = (double *) d_wrk2; - for convinience
    double * d_wrk3R; // for quantum friction computation
    cuCplx * d_wrk3C; // for quantum friction computation
    
    // other variables
    uint it; // number of steps performed from last call gpe_set_psi
    double dt;
    double t0;
    double alpha;
    double beta;
    double npart;
    double qfcoeff;
    int threads;
    int blocks;
    
} gpe_mem_t;

typedef struct
{
    uint8_t vortex_set;
    uint8_t phase_set;
} gpe_flags_t;


/* ************************************************************************************ *
 *                                                                                      *
 *                                  MACRO FUNCTIONS                                     *
 *                                                                                      *
 * ************************************************************************************ */


#define ixyz2ixiyiz(ixyz,_ix,_iy,_iz,i)     \
    i=ixyz;                                 \
    _ix=i/(ny*nz);                          \
    i=i-_ix * ny * nz;                      \
    _iy=i/nz;                               \
    _iz=i-_iy * nz;



#define ixiyiz2ixyz(_ixyz,_ix,_iy,_iz, ny, nz)      \
    _ixyz = _iz + nz*_iy + nz*ny*_ix

#define cuErrCheck(err)                                                                                 \
    {                                                                                                   \
        if(err != cudaSuccess)                                                                          \
        {                                                                                               \
            fprintf( stderr, "ERROR: file=`%s`, line=%d\n", __FILE__, __LINE__ ) ;                      \
            fprintf( stderr, "CUDA ERROR %d: %s\n", err, cudaGetErrorString((cudaError_t)(err)));       \
            return (gpe_result_t)(err);                                                                 \
        }                                                                                               \
    } 


/* 
 * This macro enables simple handling of cufftResult (status of cufft-library operation)
 */
#define CHECK_CUFFT( cufft_res ) {                                                                      \
    if (cufft_res != CUFFT_SUCCESS) {                                                                   \
        printf( "CUFFT error in %s at line %d\n", __FILE__, __LINE__ );                                 \
        return cufft_res;                                                                               \
    }                                                                                                   \
} 

// Macro allocates memory and check final status
#define gpemalloc(pointer,size,type)                                            \
    if ( ( pointer = (type *) malloc( (size) * sizeof( type ) ) ) == NULL )     \
    {                                                                           \
        fprintf( stderr , "error: cannot malloc()! Exiting!\n") ;               \
        fprintf( stderr , "error: file=`%s`, line=%d\n", __FILE__, __LINE__ ) ; \
        exit(1) ;                                                               \
    }

// Macro checks corectness of execution of gpe interface functions
#define gpe_exec( cmd, ierr )                                                   \
    { ierr=cmd;                                                                 \
    if(ierr)                                                                    \
    {                                                                           \
        fprintf( stderr , "error: cannot execute: %s\n", #cmd) ;                \
        fprintf( stderr , "file=`%s`, line=%d\n" ,__FILE__,__LINE__) ;          \
        fprintf( stderr , "Error=%d\nExiting!\n" ,ierr) ;                       \
        exit(1) ;                                                               \
    } }



/* ************************************************************************************ *
 *                                                                                      *
 *                               GPE ERRORS DEFINITION                                  *
 *                                                                                      *
 * ************************************************************************************ */
// TODO: Define all exit/error codes for gpe (including errors of CUFFT and CUBLAS)!
#define GPE_SUCCESS 0
/*
// list of possible cufft errors
        if (cufft_res == CUFFT_INVALID_PLAN) {printf("CUFFT_INVALID_PLAN\n");}
        else if (cufft_res == CUFFT_ALLOC_FAILED) {printf("CUFFT_ALLOC_FAILED\n");}
        else if (cufft_res == CUFFT_INVALID_TYPE) {printf("CUFFT_INVALID_TYPE\n");}
        else if (cufft_res == CUFFT_INVALID_VALUE) {printf("CUFFT_INVALID_VALUE\n");}
        else if (cufft_res == CUFFT_INTERNAL_ERROR) {printf("CUFFT_INTERNAL_ERROR\n");}
        else if (cufft_res == CUFFT_EXEC_FAILED) {printf("CUFFT_EXEC_FAILED\n");}
        else if (cufft_res == CUFFT_SETUP_FAILED) {printf("CUFFT_SETUP_FAILED\n");}
        else if (cufft_res == CUFFT_INVALID_SIZE) {printf("CUFFT_INVALID_SIZE\n");}
        else if (cufft_res == CUFFT_UNALIGNED_DATA) {printf("CUFFT_UNALIGNED_DATA\n");}
        else if (cufft_res == CUFFT_INCOMPLETE_PARAMETER_LIST) {printf("INCOMPLETE_PARAMETER_LIST\n");}
        else if (cufft_res == CUFFT_INVALID_DEVICE) {printf("CUFFT_INVALID_DEVICE\n");}
        else if (cufft_res == CUFFT_NO_WORKSPACE) {printf("CUFFT_NO_WORKSPACE\n");}
        else if (cufft_res == CUFFT_NOT_IMPLEMENTED) {printf("CUFFT_NOT_IMPLEMENTED\n");}
        else if (cufft_res == CUFFT_PARSE_ERROR) {printf("PARSE_ERROR\n");}
        else if (cufft_res == CUFFT_LICENSE_ERROR) {printf("CUFFT_LICENSE_ERROR\n");}
        
*/



/* ************************************************************************************ *
 *                                                                                      *
 *                                    GLOBAL MEM                                        *
 *                                                                                      *
 * ************************************************************************************ */

extern gpe_mem_t gpe_mem;




/************************************************************************************/ 
/****************************** GPE ENGINE NTERFACE *********************************/
/************************************************************************************/
/**
 * Function provides sizes of mesh 
 * provided in compilation process.
 * @param *_nx size of mesh in x direction [OUTPUT]
 * @param *_ny size of mesh in y direction [OUTPUT]
 * @param *_nz size of mesh in z direction [OUTPUT]
 * */
void gpe_get_lattice(int *_nx, int *_ny, int *_nz);

/**
 * Function creates GPE engine.
 * @param alpha \f$\alpha\f$ parameter of GPE equation [INPUT]
 * @param beta \f$\beta\f$ of GPE equation [INPUT]
 * @param dt integration step [INPUT]
 * @param npart number of particles [INPUT]
 * @param nthreads number of GPU threads. It has to be power of 2. Recommended 512 or 1024 [INPUT]
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_create_engine(double alpha, double beta, double dt, double npart , int nthreads = 1024);


/**
 * Function changes values of GPE parameters
 * @param alpha \f$\alpha\f$ parameter of GPE equation [INPUT]
 * @param beta \f$\beta\f$ of GPE equation [INPUT]
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_change_alpha_beta(double alpha, double beta);


/**
 * Function sets evolution to real time one (enables investigating dynamics of condensate).
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_set_rte_evolution();


/**
 * Function sets evolution to imaginary time one (enables finding ground state of wavefunction).
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_set_ite_evolution();


/**
 * Function sets new value of time for present wave function, i.e.  \f$\Psi(t)\rightarrow\Psi(t_0)\f$
 * @param t0 new value of time [INPUT]
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_set_time(double t0);


/**
 * Function sets user parameters. Maximal number of user parameters is 32.
 * User parameters are accessible in functions gpe_modify_psi(), gpe_external_potential(), gpe_EDF() and gpe_dEDFdn()
 * through array d_user_param.
 * @param size number of parameters, not bigger than 32 [INPUT]
 * @param params pointer to array with parameters [INPUT]
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_set_user_params(int size, double *params);


/**
 * Functions sets value of quantum friction coefficient.
 * Note - quantum friction produces extra cost in computation process - overhead is about 50%! 
 * @param qfcoeff - quantum friction coefficient \f$\gamma\f$. If qfcoeff==0.0 then quantum friction is deactivated. [INPUT]
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_set_quantum_friction_coeff(double qfcoeff);


/**
 * Function copies vortex parameters to constant memory
 * @param vortex_x0 
 * @param vortex_y0 
 * @param Q topological charge of vortex
 */
int gpe_set_vortex(const double vortex_x0, const double vortex_y0, const int8_t Q = 1);

/**
 * Function evaluates phase of wavefunction for specified time i.e. \f$\Psi(t)\f$ and save in additional array on the device.
 * If given pointer to host array also copies phase to host.
 * @param h_phase pointer to an array to store phase on host or NULL (by default) [INPUT]
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_get_phase(double* h_phase = NULL);

/**
 * Function fills additional array on device for storing phase of wavefunctio with given host array.
 * @param h_phase pointer to an array to store phase on host or NULL (by default) [INPUT]
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_set_phase(double* h_phase);

/**
 * Function sets wave function for specified time i.e. \f$\Psi(t)\f$
 * @param t value of time [INPUT]
 * @param psi corresponding wave function [INPUT]
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_set_psi(double t, cuCplx * psi);


/**
 * Function returns wave function and corresponding time i.e. \f$\Psi(t)\f$
 * @param t value of time [OUTPUT]
 * @param psi corresponding wave function [OUTPUT]
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_get_psi(double *t, cuCplx * psi);


/**
 * Function returns density computed from wave function \f$n(\vec{r}, t)=\kappa |\Psi(\vec{r}, t)|^{2}\f$ and corresponding time.
 * \f$\kappa\f$ is equal 1 for particles and 2 for dimers.
 * @param t value of time [OUTPUT]
 * @param density corresponding density [OUTPUT]
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_get_density(double *t, double * density);


/**
 * Function returns currents computed from wave function \f$\vec{j}(\vec{r}, t)=\frac{1}{\kappa}\textrm{Im}[\Psi^*(\vec{r}, t)\vec{\nabla}\Psi(\vec{r}, t)]\f$ 
 * and corresponding time.
 * \f$\kappa\f$ is equal 1 for particles and 2 for dimers.
 * @param t value of time [OUTPUT]
 * @param jx x coordinate of currents i.e. j_x(r,t) [OUTPUT]
 * @param jy y coordinate of currents i.e. j_y(r,t) [OUTPUT]
 * @param jz z coordinate of currents i.e. j_z(r,t) [OUTPUT]
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_get_currents(double *t, double * jx, double * jy, double * jz);


/**
 * Function normalizes state
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_normalize_psi();


/**
 * Function evolves state nt steps in time i.e. \f$\Psi(t)\rightarrow\Psi(t+n_t dt)\f$
 * @param nt number of steps to evolve [INPUT]
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_evolve(int nt);


/**
 * Function evolves state nt steps in time i.e. \f$\Psi(t)\rightarrow\Psi(t+n_t dt)\f$ with option evolution by quantum friction potential.
 * @param nt number of steps to evolve [INPUT]
 * @param chemical_potential chemical potential of a system in ITE [INPUT]
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_evolve_qf(int nt, double* chemical_potential = NULL);


/**
 * Function evolves state nt steps in time i.e. \f$\Psi(t)\rightarrow\Psi(t+n_t dt)\f$ with imprinting vortex (in ite).
 * NOTE: Assuming that function gpe_set_vortex has already been used.
 * @param nt number of steps to evolve [INPUT]
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_evolve_vortex(int nt, double* chemical_potential = NULL);


/**
 * Function evolves state nt steps in time i.e. \f$\Psi(t)\rightarrow\Psi(t+n_t dt)\f$ with given phase (in ite).
 * NOTE: Assuming that function gpe_set_phase or gpe_get_phase has already been used.
 * @param nt number of steps to evolve [INPUT]
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_evolve_enforced_phase(int nt, double* chemical_potential = NULL);

#ifdef DIPOLAR

/**
 * Function evolves state nt steps in time i.e. \f$\Psi(t)\rightarrow\Psi(t+n_t dt)\f$ with dipolar interactions specified by a_dip constant.
 * (a_dip
 * NOTE: Assuming that a_dip is declared in d_user_params.
 * @param nt number of steps to evolve [INPUT]
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_evolve_dipolar(int nt);

/**
 * Function evolves state nt steps in time i.e. \f$\Psi(t)\rightarrow\Psi(t+n_t dt)\f$ with dipolar interactions specified by a_dip constant.
 * (a_dip
 * NOTE: Assuming that a_dip is declared in d_user_params.
 * @param nt number of steps to evolve [INPUT]
 * @param chemical_potential chemical potential of a system in ITE
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_evolve_dipolar(int nt, double* chemical_potential);

/**
 * Function returns energy of dipolar interactions computed from wave function and Vdip operator.
 * NOTE: Assuming that a_dip is declared in d_user_params.
 * 
 * @param t value of time [OUTPUT]
 * @param edip expectation value of dipolar interactions energy [OUTPUT]
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_energy_dipolar(double *t, double *edip);
#endif

/**
 * Function returns energy computed from wave function and corresponding time.
 * Total energy is equal etot=ekin+eint+eext.
 * @param t value of time [OUTPUT]
 * @param ekin expectation value of kinetic energy operator [OUTPUT]
 * @param eint expectation value of internal energy [OUTPUT]
 * @param eext expectation value of external energy [OUTPUT]
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_energy(double *t, double *ekin, double *eint, double *eext);


/**
 * Function destroys engine.
 * @return It returns 0 if success otherwise error code is returned.
 * */
int gpe_destroy_engine();

#endif
