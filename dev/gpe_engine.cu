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
#include <math.h>
#include <complex.h>      // not std::complex!


#include "reductions.cuh"
#include "gpe_engine.cuh"


    
/***************************************************************************/ 
/******************************** GLOBALS **********************************/
/***************************************************************************/


gpe_mem_t gpe_mem;
gpe_flags_t gpe_flags;


// ============================ CONSTANT MEMORY ALLOCATION ==================================================

/*
 * TODO: check available size of constant memory
 *       think how to "dynamically allocate constant memory" - in runtime
 */
// device constants
__constant__ double d_alpha;
__constant__ double d_beta;
__constant__ double d_qfcoeff; // quantum friction coeff
__constant__ cuCplx d_step_coeff; // 0.5*dt/(i*alpha-beta)

// vortex properties
__constant__ double d_vortex_x0;
__constant__ double d_vortex_y0;
__constant__ int8_t d_vortex_Q;

// reciprocal lattice constants
__constant__ double d_kkx[NX];
__constant__ double d_kky[NY];
__constant__ double d_kkz[NZ];
__constant__ cuCplx d_exp_kkx2[NX]; // exp( (dt/(i*alpha-beta)) * 1/(2gamma) * kx^2 )
__constant__ cuCplx d_exp_kky2[NY]; // exp( (dt/(i*alpha-beta)) * 1/(2gamma) * ky^2 )
__constant__ cuCplx d_exp_kkz2_over_nxyz[NZ]; // exp( (dt/(i*alpha-beta)) * 1/(2gamma) * kz^2 ) / nxyz


#ifndef MAX_USER_PARAMS
#define MAX_USER_PARAMS 32
#endif
__constant__ double d_user_param[MAX_USER_PARAMS];
__constant__ uint d_nx; // lattice size in x direction
__constant__ uint d_ny; // lattice size in y direction
__constant__ uint d_nz; // lattice size in z direction
__constant__ double d_dt;
__constant__ double d_t0;
__constant__ double d_npart;

/***************************************************************************/ 
/****************************** FUNCTIONS **********************************/
/***************************************************************************/

// =========================== Lattice ========================================================================

void gpe_get_lattice(int *_nx, int *_ny, int *_nz)
{
    *_nx = nx;
    *_ny = ny;
    *_nz = nz;
}

/*
 * Allocates memory and creates arrays containing reciprocal lattice points' coordinates.
 */
inline gpe_result_t gpe_reciprocal_lattice_init( double alpha, double beta)
{
    /* ***************************************************************************************
     * TODO:
     *      - ask if c3/(GAMMA*GAMMA) could not be change to simplier form (NOT IMPORTANT ...)
     *      - think of adding OpenMP sections (probably not parallel for!)
     */
    
    /* NOTE : nx , ny , nz = 2j forall j integers (e.g. even numbers for the lattice dimensions) */
    // Initialize lattice in momentum space (first Brullion zone)
    /* initialize the k-space lattice */
    const double dt = gpe_mem.dt;
    uint ui;
    int i,j;
    double r;
    
    // Generate arrays on host
    gpemalloc(gpe_mem.kkx,nx,double);
    gpemalloc(gpe_mem.kky,ny,double);
    gpemalloc(gpe_mem.kkz,nz,double);  
    
    for ( i = 0 ; i <= nx / 2 - 1 ; i++ ) {
        gpe_mem.kkx[ i ] = 2. * ( double ) M_PI / nx * ( double ) i ;  }
    j = - i ;
    for ( i = nx / 2 ; i < nx ; i++ ) 
    {
        gpe_mem.kkx[ i ] = 2. * ( double ) M_PI / nx * ( double ) j ; 
        j++ ;
    }
    cuErrCheck( cudaMemcpyToSymbol(d_kkx, gpe_mem.kkx, nx*sizeof(double)) ) ;

    for ( i = 0 ; i <= ny / 2 - 1 ; i++ ) {
        gpe_mem.kky[ i ] = 2. * ( double ) M_PI / ny * ( double ) i ;  }
    j = - i ;
    for ( i = ny / 2 ; i < ny ; i++ ) 
    {
        gpe_mem.kky[ i ] = 2. * ( double ) M_PI / ny * ( double ) j ; 
        j++ ;
    }
    cuErrCheck( cudaMemcpyToSymbol(d_kky, gpe_mem.kky, ny*sizeof(double)) ) ;

    for ( i = 0 ; i <= nz / 2 - 1 ; i++ ) {
        gpe_mem.kkz[ i ] = 2. * ( double ) M_PI / nz * ( double ) i ;  }
    j = - i ;
    for ( i = nz / 2 ; i < nz ; i++ ) 
    {
        gpe_mem.kkz[ i ] = 2. * ( double ) M_PI / nz * ( double ) j ; 
        j++ ;
    }    
    cuErrCheck( cudaMemcpyToSymbol(d_kkz, gpe_mem.kkz, nz*sizeof(double)) ) ;
    
    // 0.5*dt/(i*alpha-beta)*GAMMA
    cplx c1=GAMMA*0.5*dt + I*0.0;
    cplx c2=-1.0*beta + I*alpha;
    cplx c3=c1/c2;
    cuErrCheck( cudaMemcpyToSymbol(d_step_coeff, &c3, sizeof(cuCplx)) ) ;
    
    // kinetic operator mulipliers
    cuCplx *carr;
    
    // nx direction
    gpemalloc(carr,nx,cuCplx);
    for(ui=0; ui<nx; ui++)
    {
        c1=cexp(c3*gpe_mem.kkx[ui]*gpe_mem.kkx[ui]/(GAMMA*GAMMA));
        carr[ui].x=creal(c1); carr[ui].y=cimag(c1);
        //carr[ui] = (cuCplx) c1; // cuCplx and cplx should be binary-compatible
    }
    cuErrCheck( cudaMemcpyToSymbol(d_exp_kkx2, carr, nx*sizeof(cuCplx)) ) ;
    free(carr);
    
    // ny direction
    gpemalloc(carr,ny,cuCplx);
    for(ui=0; ui<ny; ui++)
    {
        c1=cexp(c3*gpe_mem.kky[ui]*gpe_mem.kky[ui]/(GAMMA*GAMMA));
        carr[ui].x=creal(c1); carr[ui].y=cimag(c1);
        //carr[ui] = (cuCplx) c1; // cuCplx and cplx should be binary-compatible
    }
    cuErrCheck( cudaMemcpyToSymbol(d_exp_kky2, carr, ny*sizeof(cuCplx)) ) ;
    free(carr);
    
    // nz direction
    gpemalloc(carr,nz,cuCplx);
    for(ui=0; ui<nz; ui++)
    {
        c1=cexp(c3*gpe_mem.kkz[ui]*gpe_mem.kkz[ui]/(GAMMA*GAMMA)) / (double)(nxyz); // NOTE: here we divide to 
        carr[ui].x=creal(c1); carr[ui].y=cimag(c1);
        //carr[ui] = (cuCplx) c1; // cuCplx and cplx should be binary-compatible
    }
    cuErrCheck( cudaMemcpyToSymbol(d_exp_kkz2_over_nxyz, carr, nz*sizeof(cuCplx)) ) ;
    free(carr);
    
    return GPE_SUCCESS;
}

/*
 * Needed when changing evolution type without rectreating whole engine.
 */
inline gpe_result_t gpe_reciprocal_lattice_change( double alpha, double beta)
{
    /* ***************************************************************************************
     * TODO:
     *      - ask if c3/(GAMMA*GAMMA) could not be change to simplier form (NOT IMPORTANT ...)
     *      - think of adding OpenMP sections (probably not parallel for!)
     */
    
    double dt = gpe_mem.dt;
    uint ui;
    int i,j;
    double r;
    
    // 0.5*dt/(i*alpha-beta)*GAMMA
    cplx c1=GAMMA*0.5*dt + I*0.0;
    cplx c2=-1.0*beta + I*alpha;
    cplx c3=c1/c2;
    cuErrCheck( cudaMemcpyToSymbol(d_step_coeff, &c3, sizeof(cuCplx)) ) ;
    
    // kinetic operator mulipliers
    cuCplx *carr;
    
    // nx direction
    gpemalloc(carr,nx,cuCplx);
    for(ui=0; ui<nx; ui++)
    {
        c1=cexp(c3*gpe_mem.kkx[ui]*gpe_mem.kkx[ui]/(GAMMA*GAMMA));
        carr[ui].x=creal(c1); carr[ui].y=cimag(c1);
        //carr[ui] = (cuCplx) c1; // cuCplx and cplx should be binary-compatible
    }
    cuErrCheck( cudaMemcpyToSymbol(d_exp_kkx2, carr, nx*sizeof(cuCplx)) ) ;
    free(carr);
    
    // ny direction
    gpemalloc(carr,ny,cuCplx);
    for(ui=0; ui<ny; ui++)
    {
        c1=cexp(c3*gpe_mem.kky[ui]*gpe_mem.kky[ui]/(GAMMA*GAMMA));
        carr[ui].x=creal(c1); carr[ui].y=cimag(c1);
        //carr[ui] = (cuCplx) c1; // cuCplx and cplx should be binary-compatible
    }
    cuErrCheck( cudaMemcpyToSymbol(d_exp_kky2, carr, ny*sizeof(cuCplx)) ) ;
    free(carr);
    
    // nz direction
    gpemalloc(carr,nz,cuCplx);
    for(ui=0; ui<nz; ui++)
    {
        c1=cexp(c3*gpe_mem.kkz[ui]*gpe_mem.kkz[ui]/(GAMMA*GAMMA)) / (double)(nxyz); // NOTE: here we divide to normalize CUFFT
        carr[ui].x=creal(c1); carr[ui].y=cimag(c1);
        //carr[ui] = (cuCplx) c1; // cuCplx and cplx should be binary-compatible
    }
    cuErrCheck( cudaMemcpyToSymbol(d_exp_kkz2_over_nxyz, carr, nz*sizeof(cuCplx)) ) ;
    free(carr);
    
    return GPE_SUCCESS;
}


// =========================== User interface ========================================================================

int gpe_create_engine(double alpha, double beta, double dt, double npart, int nthreads)
{
    uint ui;
    int i,j;
    double r;
    gpe_result_t res;
    
    // check if mode is right
    #ifndef GAMMA
        return -99; // not supported mode
    #endif
    gpe_check_particle_type();
    gpe_print_interactions_type();
    
    // Set flags
    gpe_flags.vortex_set = 0;
    gpe_flags.phase_set = 0;
    
    // Set number of blocks, if number of threads is given
    gpe_mem.threads=nthreads;
    gpe_mem.blocks=(int)ceil((float)nxyz/nthreads);

//     printf("# GPU SETTING: THREADS=%d, BLOCKS=%d, THREADS*BLOCKS=%d, nxyz=%d\n",gpe_mem.threads,gpe_mem.blocks,gpe_mem.threads*gpe_mem.blocks,nxyz);
    printf("\n");
    
    // Fill const memory
    ui=nx;
    cuErrCheck( cudaMemcpyToSymbol(d_nx, &ui, sizeof(uint)) ) ;
    ui=ny;
    cuErrCheck( cudaMemcpyToSymbol(d_ny, &ui, sizeof(uint)) ) ;
    ui=nz;
    cuErrCheck( cudaMemcpyToSymbol(d_nz, &ui, sizeof(uint)) ) ;   
    cuErrCheck( cudaMemcpyToSymbol(d_alpha, &alpha, sizeof(double)) ) ;
    gpe_mem.alpha=alpha;
    cuErrCheck( cudaMemcpyToSymbol(d_beta, &beta, sizeof(double)) ) ;
    gpe_mem.beta=beta;
    cuErrCheck( cudaMemcpyToSymbol(d_dt, &dt, sizeof(double)) ) ;
    gpe_mem.dt=dt;
    r=0.0;
    cuErrCheck( cudaMemcpyToSymbol(d_t0, &r, sizeof(double)) ) ;
    cuErrCheck( cudaMemcpyToSymbol(d_qfcoeff, &r, sizeof(double)) ) ;
    gpe_mem.t0=0.0;
    gpe_mem.it=0;
    gpe_mem.qfcoeff=0.0;
    cuErrCheck( cudaMemcpyToSymbol(d_npart, &npart, sizeof(double)) ) ;
    gpe_mem.npart=npart;
    
    // create reciprocal lattice (in bonduary of first Brullion zone)
    res = gpe_reciprocal_lattice_init(alpha, beta);
    
    
    
    // TODO: Create separate function for this and probably create array of plans...
    // create cufft plans
    cufftResult cufft_result;
    cufft_result=cufftCreate(&gpe_mem.plan); if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
    cufft_result=cufftSetAutoAllocation(gpe_mem.plan, 0); if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
    size_t workSize;
    cufft_result=cufftMakePlan3d(gpe_mem.plan, nx, ny, nz, CUFFT_Z2Z, &workSize);
    if(workSize<sizeof(cuCplx)*nxyz) workSize=sizeof(cuCplx)*nxyz;
    cuErrCheck( cudaMalloc( &gpe_mem.d_wrk , workSize ) );
    cufft_result=cufftSetWorkArea(gpe_mem.plan, gpe_mem.d_wrk); if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
    
    // allocate memory for workspace on device
    cuErrCheck( cudaMalloc( &gpe_mem.d_wrk2, sizeof(cuCplx)*nxyz ) );
    cuErrCheck( cudaMalloc( &gpe_mem.d_psi,  sizeof(cuCplx)*nxyz ) );
    cuErrCheck( cudaMalloc( &gpe_mem.d_psi2, sizeof(cuCplx)*nxyz ) );
    gpe_mem.d_wrk2R = (double *) gpe_mem.d_wrk2; 
    
    gpe_mem.d_wrk3R = NULL;
    gpe_mem.d_wrk3C = NULL;
    gpe_mem.d_phase = NULL;
    
#ifdef DIPOLAR
    // TODO: Check if it is not necessary!
    //cuErrCheck( cudaMalloc( &gpe_mem.d_dipolar_wrk, sizeof(cuCplx)*nxyz) );
#endif
    
    return GPE_SUCCESS; // success
}

int gpe_destroy_engine()
{
    
    cufftResult cufft_result;
    free(gpe_mem.kkx);
    free(gpe_mem.kky);
    free(gpe_mem.kkz);
    cufft_result=cufftDestroy(gpe_mem.plan); if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
    cuErrCheck( cudaFree(gpe_mem.d_wrk) );
    cuErrCheck( cudaFree(gpe_mem.d_wrk2) );
    cuErrCheck( cudaFree(gpe_mem.d_psi) );
    cuErrCheck( cudaFree(gpe_mem.d_psi2) );
    if(gpe_mem.d_wrk3R != NULL) cuErrCheck( cudaFree(gpe_mem.d_wrk3R) );
    if(gpe_mem.d_wrk3C != NULL) cuErrCheck( cudaFree(gpe_mem.d_wrk3C) );
    if(gpe_mem.d_phase != NULL) cuErrCheck( cudaFree(gpe_mem.d_phase) );
    
    return GPE_SUCCESS; // success
}

int gpe_change_alpha_beta(double alpha, double beta)
{
    
    uint ui;
    
    cuErrCheck( cudaMemcpyToSymbol(d_alpha, &alpha, sizeof(double)) ) ;
    gpe_mem.alpha=alpha;
    cuErrCheck( cudaMemcpyToSymbol(d_beta, &beta, sizeof(double)) ) ;
    gpe_mem.beta=beta;
    
    // update reciprocal lattice
    gpe_reciprocal_lattice_change(alpha, beta);
    
    return 0;
}

int gpe_set_rte_evolution()
{
    gpe_change_alpha_beta(1.0,0.0);
    return GPE_SUCCESS;
}

int gpe_set_ite_evolution()
{
    gpe_change_alpha_beta(0.0,1.0);
    return GPE_SUCCESS;
}

int gpe_set_time(double t0)
{
    
    
    cuErrCheck( cudaMemcpyToSymbol(d_t0, &t0, sizeof(double)) ) ;
    gpe_mem.t0=t0;
    gpe_mem.it=0;    
    
    return 0;
}

int gpe_set_user_params(int size, double *params)
{
    if(size>MAX_USER_PARAMS) return -9;
    
    cuErrCheck( cudaMemcpyToSymbol(d_user_param, params, MAX_USER_PARAMS*sizeof(double)) );
    
    return 0;
}

int gpe_set_quantum_friction_coeff(double qfcoeff)
{
    
    if(qfcoeff!=0.0)
    {
        qfcoeff=qfcoeff/( GAMMA*(double)(nxyz) );
        cuErrCheck( cudaMemcpyToSymbol(d_qfcoeff, &qfcoeff, sizeof(double)) ) ;
        gpe_mem.qfcoeff=qfcoeff;
        
        if(gpe_mem.d_wrk3R==NULL) cuErrCheck( cudaMalloc( &gpe_mem.d_wrk3R , sizeof(double)*nxyz ) );
        if(gpe_mem.d_wrk3C==NULL) cuErrCheck( cudaMalloc( &gpe_mem.d_wrk3C , sizeof(cuCplx)*nxyz ) );
    }
    else
    {
        cuErrCheck( cudaMemcpyToSymbol(d_qfcoeff, &qfcoeff, sizeof(double)) ) ;
        gpe_mem.qfcoeff=qfcoeff;
        
        if(gpe_mem.d_wrk3R != NULL) cuErrCheck( cudaFree(gpe_mem.d_wrk3R) );
        if(gpe_mem.d_wrk3C != NULL) cuErrCheck( cudaFree(gpe_mem.d_wrk3C) );   
        
        gpe_mem.d_wrk3R = NULL;
        gpe_mem.d_wrk3C = NULL;
    }
    
    return 0;
}


// ======================= Quantum vortices interface =======================================================

int gpe_set_vortex(const double vortex_x0, const double vortex_y0, const int8_t Q) 
{
    cudaError err;
    
    cuErrCheck( cudaMemcpyToSymbol(d_vortex_x0, &vortex_x0, sizeof(double)) ) ;
    cuErrCheck( cudaMemcpyToSymbol(d_vortex_y0, &vortex_y0, sizeof(double)) ) ;
    cuErrCheck( cudaMemcpyToSymbol(d_vortex_Q, &Q, sizeof(int8_t)) ) ;
    
    gpe_flags.vortex_set = 1;
    
    return 0; // success
}

/*
 * This function imprints vortex parallel to z axis crossing x,y plane in (x0,y0) point
 * double d_vortex_x0, d_vortex_y0 - position of vortex in xy plane
 * uint8_t d_Q_vortex - topological charge of vortex
 * NOTE: It is considered that x0 and y0 should be chosen out of lattice points in case 
 *       phase is corectly (mathematically) defined in every lattice point (check atan2).
 */
__global__ void __gpe_imprint_vortexline_zdir_(cuCplx *psi)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    uint ix, iy, iz, i;
    
    // registers
    cuCplx lpsi = psi[ixyz];
    double abs_psi, phase;
    double _x,_y;
    
    if(ixyz<nxyz)
    {
        ixyz2ixiyiz(ixyz,ix,iy,iz,i);
        
        _x = constgpu(ix) - 1.0*(NX/2) - d_vortex_x0;
        _y = constgpu(iy) - 1.0*(NY/2) - d_vortex_y0;
        
        abs_psi = hypot(lpsi.x, lpsi.y); //abs_psi = sqrt(lpsi.x*lpsi.x + lpsi.y*lpsi.y); (intrinsic should be faster)
        phase = atan2(_x,_y); // atan2(0,0) == -pi/2
        phase *= (double) (d_vortex_Q); //if (d_vortex_Q != 1) phase *= (double) (d_vortex_Q);
        lpsi.x = abs_psi*cos(phase);
        lpsi.y = abs_psi*sin(phase);
        
        psi[ixyz] = lpsi;
	    
    }
}

/*
 * This function imprints vortex parallel to z axis crossing x,y plane in (x0,y0) point
 * double d_vortex_x0, d_vortex_y0 - position of vortex in xy plane
 * uint8_t d_Q_vortex - topological charge of vortex
 * NOTE: It is considered that x0 and y0 should be chosen out of lattice points in case 
 *       phase is corectly (mathematically) defined in every lattice point (check atan2).
 */
__global__ void __gpe_imprint2_vortexline_zdir_(cuCplx *psi)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    uint ix, iy, iz, i;
    
    // registers
    cuCplx lpsi = psi[ixyz];
    double abs_psi, phase;
    double _x,_y;
    
    if(ixyz<nxyz)
    {
        ixyz2ixiyiz(ixyz,ix,iy,iz,i);
        
        _x = constgpu(ix) - 1.0*(NX/2) - d_vortex_x0;
        _y = constgpu(iy) - 1.0*(NY/2) - d_vortex_y0;
        
        abs_psi = hypot(lpsi.x, lpsi.y); //abs_psi = sqrt(lpsi.x*lpsi.x + lpsi.y*lpsi.y); (intrinsic should be faster)
        if (abs_psi > 1e-15)
        {
            phase = atan2(_x,_y); // atan2(0,0) == -pi/2
            phase *= (double) (d_vortex_Q); //if (d_vortex_Q != 1) phase *= (double) (d_vortex_Q);
            lpsi.x = abs_psi*cos(phase);
            lpsi.y = abs_psi*sin(phase);
            
            psi[ixyz] = lpsi;
	    }
    }
}

// ======================= More general enforcing phase =====================================================

__global__ void __gpe_compute_phase__(cuCplx* psi, double* d_phase)
{
	size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
	if(ixyz<nxyz)
    {
        d_phase[ixyz] = cplxArg(psi[ixyz]);
    }
}

int gpe_set_phase(double* h_phase)
{
	
    if (gpe_mem.d_phase == NULL) cuErrCheck( cudaMalloc( &gpe_mem.d_phase, sizeof(double)*nxyz ) );
    
    cuErrCheck( cudaMemcpy( gpe_mem.d_phase, h_phase, sizeof(double)*nxyz, cudaMemcpyHostToDevice) ); 
    gpe_flags.phase_set=1;
    
    return GPE_SUCCESS;
}

int gpe_get_phase(double* h_phase)
{
	
    if (gpe_mem.d_phase == NULL) cuErrCheck( cudaMalloc( &gpe_mem.d_phase, sizeof(double)*nxyz ) );
    
    __gpe_compute_phase__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi,gpe_mem.d_phase);
    if (h_phase) cuErrCheck( cudaMemcpy( h_phase, gpe_mem.d_phase, sizeof(double)*nxyz, cudaMemcpyDeviceToHost) );  // if h_phase != NULL copy phase to host
    gpe_flags.phase_set=1;
    
    return GPE_SUCCESS;
}

__global__ void __gpe_enforce_phase__(cuCplx* psi, double* d_phase)
{
	size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
	if(ixyz<nxyz)
    {
        psi[ixyz] = cplxScale( cplxExpi(d_phase[ixyz]), cplxAbs(psi[ixyz]) );
    }
}

// ======================= Density/Normalization ============================================================

// TODO: Test speed with cublas

/**
 * Function computes density from wave function psi
 * */
inline __device__  double gpe_density(cuCplx psi)
{
    return GAMMA * (psi.x*psi.x + psi.y*psi.y); // |psi|^2 * GAMMA, where GAMMA=1 for particles, GAMMA=2 for dimers
}

__global__ void __gpe_compute_density__(cuCplx *psi_in, double *rho_out)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    if(ixyz<nxyz)
    {
        rho_out[ixyz] = gpe_density(psi_in[ixyz]);
    }
}

/**
 * Computes density and saves in array of complex numbers (as real part).
 * Suitable for dipolar interactions.
 * */
__global__ void __gpe_compute_density2C__(cuCplx *psi_in, cuCplx *rho_out)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    if(ixyz<nxyz)
    {
        rho_out[ixyz].x = gpe_density(psi_in[ixyz]);
        rho_out[ixyz].y = 0.;
    }
}

__global__ void __gpe_normalize__(cuCplx *psi_inout, double *sumrho)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    if(ixyz<nxyz)
    {
//         if(ixyz==0) printf("sumrho[0]=%f\n", sumrho[0]);
        psi_inout[ixyz] = cplxScale(psi_inout[ixyz], sqrt(d_npart/sumrho[0]));
    }
}

// Normalizes wavefunction
int gpe_normalize(cuCplx *psi, double *wrk)
{
    __gpe_compute_density__<<<gpe_mem.blocks, gpe_mem.threads>>>(psi, wrk);
    cuErrCheck( local_reduction(wrk, nxyz, wrk, gpe_mem.threads, 0) );
    __gpe_normalize__<<<gpe_mem.blocks, gpe_mem.threads>>>(psi, wrk);
    
    return GPE_SUCCESS;
}

int gpe_normalize_psi()
{
    return gpe_normalize(gpe_mem.d_psi, gpe_mem.d_wrk2R);
}

static inline int gpe_normalize_psi(double *chemical_potential)
{    
    __gpe_compute_density__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi, gpe_mem.d_wrk2R);
    cuErrCheck( local_reduction(gpe_mem.d_wrk2R, nxyz, gpe_mem.d_wrk2R, gpe_mem.threads, 0) );
    __gpe_normalize__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi, gpe_mem.d_wrk2R);
    
    if (chemical_potential)
    {
        double norm;
        cuErrCheck( cudaMemcpy( &norm, gpe_mem.d_wrk2R, sizeof(double), cudaMemcpyDeviceToHost) ); 
        *chemical_potential = -.5*log(norm/gpe_mem.npart)/gpe_mem.dt;
    }
    
    return GPE_SUCCESS;
}

int gpe_get_density(double *t, double * density)
{
    __gpe_compute_density__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi, gpe_mem.d_wrk2R);
    
    cuErrCheck( cudaMemcpy( density , gpe_mem.d_wrk2R , sizeof(double)*nxyz, cudaMemcpyDeviceToHost ) );
    
    *t = gpe_mem.t0 + gpe_mem.dt*gpe_mem.it;
    return 0;
}


// =================== Accesing wavefunction =============================================================================

int gpe_set_psi(double t, cuCplx * psi)
{
    
    cuErrCheck( cudaMemcpyToSymbol(d_t0, &t, sizeof(double)) ) ;
    gpe_mem.it=0;
    gpe_mem.t0=t;
    cuErrCheck( cudaMemcpy( gpe_mem.d_psi , psi , sizeof(cuCplx)*nxyz, cudaMemcpyHostToDevice ) );
    
    return 0;
}

int gpe_get_psi(double *t, cuCplx * psi)
{
    
    cuErrCheck( cudaMemcpy( psi , gpe_mem.d_psi , sizeof(cuCplx)*nxyz, cudaMemcpyDeviceToHost ) );
    *t = gpe_mem.t0 + gpe_mem.dt*gpe_mem.it;
    
    return 0;
}

// ======================= Evolution algorithm =============================================================================

/**
 * construct  exp(-i*dt*V/2) and apply exp(-i*dt*V/2) * psi 
 * */
__global__ void __gpe_exp_Vstep1_(uint it, cuCplx *psi_in, cuCplx *psi_out, double * wrkR)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    uint ix, iy, iz, i;
    
    // registers
    cuCplx lpsi, exp_lv;
    double lrho, lv;
    
    if(ixyz<nxyz)
    {
        ixyz2ixiyiz(ixyz,ix,iy,iz,i); 
        
        lpsi = psi_in[ixyz]; // psi to register
        lpsi=gpe_modify_psi(ix, iy, iz, it, lpsi); // modify psi
        lrho = gpe_density(lpsi); // compute density
        lv=gpe_external_potential(ix, iy, iz, it) + gpe_dEDFdn(lrho,it); // external potential + mean field
        
        wrkR[ixyz]=lv; // it will be use later
        exp_lv = cplxExp( cplxScale(d_step_coeff,lv) );
        
        psi_out[ixyz] = cplxMul(lpsi, exp_lv); // apply and save
    }
}

__global__ void __gpe_exp_Vstep1_qf_(uint it, cuCplx *psi_in, cuCplx *psi_out, double * wrkR, double *qfpotential)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    uint ix, iy, iz, i;
    
    // registers
    cuCplx lpsi, exp_lv;
    double lrho, lv;
    
    if(ixyz<nxyz)
    {
        ixyz2ixiyiz(ixyz,ix,iy,iz,i); 
        
        lpsi = psi_in[ixyz]; // psi to register
        lpsi=gpe_modify_psi(ix, iy, iz, it, lpsi); // modify psi
        lrho = gpe_density(lpsi); // compute density
        lv=gpe_external_potential(ix, iy, iz, it) + gpe_dEDFdn(lrho,it) + qfpotential[ixyz]; 
           // external potential + mean field + quantum friction potential
        
        wrkR[ixyz]=lv; // it will be use later
        exp_lv = cplxExp( cplxScale(d_step_coeff,lv) );
        
        psi_out[ixyz] = cplxMul(lpsi, exp_lv); // apply and save
    }    
}

__global__ void __gpe_exp_Vstep2_(uint it, cuCplx *psi_in, cuCplx *psi_out, double * wrkR, cuCplx * wrkC)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    uint ix, iy, iz, i;
    
    // registers
    cuCplx lpsi, exp_lv;
    double lrho, lv;
    
    if(ixyz<nxyz)
    {
        ixyz2ixiyiz(ixyz,ix,iy,iz,i); 
        
        lpsi = psi_in[ixyz]; // psi to register
        lv = wrkR[ixyz]; // potentials to register
        exp_lv = cplxExp( cplxScale(d_step_coeff,lv) );
        lpsi=cplxMul(lpsi, exp_lv); // finalize step from predictor
        
        lrho = gpe_density(lpsi); // compute density
        lv=0.5*(lv + gpe_external_potential(ix, iy, iz, it+1) + gpe_dEDFdn(lrho,it+1)); // external potential + mean field - take average
        exp_lv = cplxExp( cplxScale(d_step_coeff,lv) );
        wrkC[ixyz]=exp_lv; // it will be used later
        
        lpsi = psi_out[ixyz]; // psi to register
        lpsi=gpe_modify_psi(ix, iy, iz, it, lpsi); // modify psi
        psi_out[ixyz] = cplxMul(lpsi, exp_lv); // apply and save      
    }    
}

__global__ void __gpe_exp_Vstep2_part1_(cuCplx *psi_in, cuCplx *psi_out, double * wrkR)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    
    // registers
    cuCplx exp_lv;

    if(ixyz<nxyz)
    {
        exp_lv = cplxExp( cplxScale(d_step_coeff, wrkR[ixyz]) );
        psi_out[ixyz]=cplxMul(psi_in[ixyz], exp_lv); // finalize step from predictor     
    }    
}

__global__ void __gpe_exp_Vstep2_part2_(uint it, cuCplx *psi_in, cuCplx *psi_out, double * wrkR, cuCplx * wrkC)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    uint ix, iy, iz, i;
    
    // registers
    cuCplx lpsi, exp_lv;
    double lrho, lv;
    
    if(ixyz<nxyz)
    {
        ixyz2ixiyiz(ixyz,ix,iy,iz,i); 
        
        lpsi = psi_in[ixyz]; // psi to register
        lv = wrkR[ixyz]; // potentials to register
        lrho = gpe_density(lpsi); // compute density
        lv=0.5*(lv + gpe_external_potential(ix, iy, iz, it+1) + gpe_dEDFdn(lrho,it+1)); // external potential + mean field - take average
        exp_lv = cplxExp( cplxScale(d_step_coeff,lv) );
        wrkC[ixyz]=exp_lv; // it will be used later
        
        lpsi = psi_out[ixyz]; // psi to register
        lpsi=gpe_modify_psi(ix, iy, iz, it, lpsi); // modify psi
        psi_out[ixyz] = cplxMul(lpsi, exp_lv); // apply and save      
    }    
}

__global__ void __gpe_exp_Vstep2_part2_qf_(uint it, cuCplx *psi_in, cuCplx *psi_out, double * wrkR, cuCplx * wrkC, double *qfpotential)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    uint ix, iy, iz, i;
    
    // registers
    cuCplx lpsi, exp_lv;
    double lrho, lv;
    
    if(ixyz<nxyz)
    {
        ixyz2ixiyiz(ixyz,ix,iy,iz,i); 
        
        lpsi = psi_in[ixyz]; // psi to register
        lv = wrkR[ixyz]; // potentials to register
        lrho = gpe_density(lpsi); // compute density
        lv=0.5*(lv + gpe_external_potential(ix, iy, iz, it+1) + gpe_dEDFdn(lrho,it+1) + qfpotential[ixyz]) ; 
          // external potential + mean field + quantum friction potential - take average 
        exp_lv = cplxExp( cplxScale(d_step_coeff,lv) );
        wrkC[ixyz]=exp_lv; // it will be used later
        
        lpsi = psi_out[ixyz]; // psi to register
        lpsi=gpe_modify_psi(ix, iy, iz, it, lpsi); // modify psi
        psi_out[ixyz] = cplxMul(lpsi, exp_lv); // apply and save      
    }    
}

__global__ void __gpe_exp_Vstep3_(cuCplx *psi_inout, cuCplx * wrkC)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    
    if(ixyz<nxyz)
    {
        psi_inout[ixyz] = cplxMul(psi_inout[ixyz], wrkC[ixyz]); // apply and save      
    }    
}

__global__ void __gpe_multiply_by_expT__(cuCplx *psi_in, cuCplx *psi_out)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    uint ix, iy, iz, i;
    cuCplx _wavef;
    
    if(ixyz<nxyz)
    {
        ixyz2ixiyiz(ixyz,ix,iy,iz,i); 
        _wavef=psi_in[ixyz]; // bring to register
        _wavef=cplxMul(_wavef,d_exp_kkx2[ix]);
        _wavef=cplxMul(_wavef,d_exp_kky2[iy]);
        _wavef=cplxMul(_wavef,d_exp_kkz2_over_nxyz[iz]); // note - normalization factor is included here
        psi_out[ixyz]=_wavef; // send to global memory
    }    
}

__global__ void __gpe_multiply_by_k2_qf__(cuCplx *psi_in, cuCplx *psi_out)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    uint ix, iy, iz, i;
    
    if(ixyz<nxyz)
    {
        ixyz2ixiyiz(ixyz,ix,iy,iz,i); 

        psi_out[ixyz]=cplxScale(psi_in[ixyz], d_qfcoeff*( d_kkx[ix]*d_kkx[ix] + d_kky[iy]*d_kky[iy] + d_kkz[iz]*d_kkz[iz] ) ); 
            
    }    
}

__global__ void __gpe_overlap_imag_qf__(cuCplx *psi1, cuCplx *psi2, double *overlap)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    double lrho;
    cuCplx lpsi;
    if(ixyz<nxyz)
    {
        lpsi = psi1[ixyz]; // psi to register
        lrho = gpe_density(lpsi); // compute density
        overlap[ixyz]= cplxMulI( cplxConj(lpsi),  psi2[ixyz] )/(lrho+GPE_QF_EPSILON);
    }
}

int gpe_compute_qf_potential(cuCplx *psi, cuCplx *wrk, double *qfpotential)
{
    cufftResult cufft_result;
    
        cufft_result=cufftExecZ2Z(gpe_mem.plan, psi, wrk, CUFFT_FORWARD);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        __gpe_multiply_by_k2_qf__<<<gpe_mem.blocks, gpe_mem.threads>>>(wrk, wrk);
        cufft_result=cufftExecZ2Z(gpe_mem.plan, wrk, wrk, CUFFT_INVERSE);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        __gpe_overlap_imag_qf__<<<gpe_mem.blocks, gpe_mem.threads>>>(psi, wrk, qfpotential);

        
    return 0;
}


// ===================== Evolution interface ===========================================

/**
 * Function evolves wave funcion nt steps 
 * */
int gpe_evolve(int nt)
{
    printf("Function not implemented!\n");
    return GPE_SUCCESS;
}

/**
 * Function evolves wave funcion nt steps 
 * */
int gpe_evolve_qf(int nt, double* chemical_potential)
{
    cufftResult cufft_result;
    int i;
        
    for(i=0; i<nt; i++)
    {
                
        if(gpe_mem.qfcoeff!=0.0) // quantum friction is active
        {
            cuErrCheck( gpe_compute_qf_potential(gpe_mem.d_psi, gpe_mem.d_wrk3C, gpe_mem.d_wrk3R) );
            __gpe_exp_Vstep1_qf_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi, gpe_mem.d_psi2, gpe_mem.d_wrk2R, gpe_mem.d_wrk3R);
        }
        else
        {
            // potential part exp(V/2)
            __gpe_exp_Vstep1_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi, gpe_mem.d_psi2, gpe_mem.d_wrk2R);
        }
        
        // kinetic part exp(T)
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi2, gpe_mem.d_psi2, CUFFT_FORWARD);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        __gpe_multiply_by_expT__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi2, gpe_mem.d_psi2);
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi2, gpe_mem.d_psi2, CUFFT_INVERSE);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        
        // potential part exp(V/2)
        if(gpe_mem.beta==0.0 && gpe_mem.qfcoeff==0.0)
        {
            // without normalization
            __gpe_exp_Vstep2_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi2, gpe_mem.d_psi, gpe_mem.d_wrk2R, gpe_mem.d_psi2);
        }
        else
        {
            __gpe_exp_Vstep2_part1_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi2, gpe_mem.d_psi2, gpe_mem.d_wrk2R);
            
            if(gpe_mem.beta!=0.0)
            {
                // with normalization between
                cuErrCheck( gpe_normalize(gpe_mem.d_psi2, gpe_mem.d_wrk2R+nxyz));
            }
            
            if(gpe_mem.qfcoeff!=0.0) // quantum friction is active
            {
                cuErrCheck( gpe_compute_qf_potential(gpe_mem.d_psi, gpe_mem.d_wrk3C, gpe_mem.d_wrk3R));
                __gpe_exp_Vstep2_part2_qf_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi2, gpe_mem.d_psi, gpe_mem.d_wrk2R, gpe_mem.d_psi2, gpe_mem.d_wrk3R);
            }
            else
            {
                __gpe_exp_Vstep2_part2_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi2, gpe_mem.d_psi, gpe_mem.d_wrk2R, gpe_mem.d_psi2);
            }
        }
        
        // kinetic part exp(T)
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi, gpe_mem.d_psi, CUFFT_FORWARD);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        __gpe_multiply_by_expT__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi, gpe_mem.d_psi);
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi, gpe_mem.d_psi, CUFFT_INVERSE);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        
        // potential part exp(V/2)
        __gpe_exp_Vstep3_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi, gpe_mem.d_psi2);
        
        if(gpe_mem.beta!=0.0)
        {
            // normalize
            cuErrCheck( gpe_normalize(gpe_mem.d_psi, gpe_mem.d_wrk2R+nxyz));
            if (chemical_potential)
            {
                double norm;
                cuErrCheck( cudaMemcpy( &norm, gpe_mem.d_wrk2R+nxyz, sizeof(double), cudaMemcpyDeviceToHost) ); 
                *chemical_potential = -.5*log(norm/gpe_mem.npart)/gpe_mem.dt;
            } 
        }
        
        gpe_mem.it = gpe_mem.it + 1;
    }
    
    return 0;
}


/**
 * Simplest enforcing vortex phase with with method in both predictor and normal steps.
 */
int gpe_evolve_vortex(int nt, double* chemical_potential)
{
    cufftResult cufft_result;
    int i;
        
    for(i=0; i<nt; i++)
    {
        // changing the phase
        __gpe_imprint_vortexline_zdir_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi);
        cuErrCheck( cudaGetLastError() );
        
        
        if(gpe_mem.qfcoeff!=0.0) // quantum friction is active
        {
            cuErrCheck( gpe_compute_qf_potential(gpe_mem.d_psi, gpe_mem.d_wrk3C, gpe_mem.d_wrk3R) );
            __gpe_exp_Vstep1_qf_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi, gpe_mem.d_psi2, gpe_mem.d_wrk2R, gpe_mem.d_wrk3R);
        }
        else
        {
            // potential part exp(V/2)
            __gpe_exp_Vstep1_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi, gpe_mem.d_psi2, gpe_mem.d_wrk2R);
        }
        
        // kinetic part exp(T)
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi2, gpe_mem.d_psi2, CUFFT_FORWARD);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        __gpe_multiply_by_expT__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi2, gpe_mem.d_psi2);
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi2, gpe_mem.d_psi2, CUFFT_INVERSE);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        
        // potential part exp(V/2)
        if(gpe_mem.beta==0.0 && gpe_mem.qfcoeff==0.0)
        {
            // without normalization
            __gpe_exp_Vstep2_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi2, gpe_mem.d_psi, gpe_mem.d_wrk2R, gpe_mem.d_psi2);
        }
        else
        {
            __gpe_exp_Vstep2_part1_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi2, gpe_mem.d_psi2, gpe_mem.d_wrk2R);
            
            if(gpe_mem.beta!=0.0)
            {
                // with normalization between
                cuErrCheck( gpe_normalize(gpe_mem.d_psi2, gpe_mem.d_wrk2R+nxyz));
                
                // changing the phase
                __gpe_imprint_vortexline_zdir_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi);
                cuErrCheck( cudaGetLastError() );
            }
            
            if(gpe_mem.qfcoeff!=0.0) // quantum friction is active
            {
                cuErrCheck( gpe_compute_qf_potential(gpe_mem.d_psi, gpe_mem.d_wrk3C, gpe_mem.d_wrk3R));
                __gpe_exp_Vstep2_part2_qf_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi2, gpe_mem.d_psi, gpe_mem.d_wrk2R, gpe_mem.d_psi2, gpe_mem.d_wrk3R);
            }
            else
            {
                __gpe_exp_Vstep2_part2_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi2, gpe_mem.d_psi, gpe_mem.d_wrk2R, gpe_mem.d_psi2);
            }
        }
        
        // kinetic part exp(T)
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi, gpe_mem.d_psi, CUFFT_FORWARD);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        __gpe_multiply_by_expT__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi, gpe_mem.d_psi);
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi, gpe_mem.d_psi, CUFFT_INVERSE);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        
        // potential part exp(V/2)
        __gpe_exp_Vstep3_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi, gpe_mem.d_psi2);
        
        if(gpe_mem.beta!=0.0)
        {
            // normalize
            cuErrCheck( gpe_normalize(gpe_mem.d_psi, gpe_mem.d_wrk2R+nxyz));
            if (chemical_potential)
            {
                double norm;
                cuErrCheck( cudaMemcpy( &norm, gpe_mem.d_wrk2R+nxyz, sizeof(double), cudaMemcpyDeviceToHost) ); 
                *chemical_potential = -.5*log(norm/gpe_mem.npart)/gpe_mem.dt;
            } 
        }
        
        gpe_mem.it = gpe_mem.it + 1;
    }
    
    return GPE_SUCCESS;
}

/**
 * Enforces vortex phase with second method.
 */
int gpe_evolve_vortex2(int nt, double* chemical_potential)
{
    cufftResult cufft_result;
    int i;
        
    for(i=0; i<nt; i++)
    {
        // changing the phase
        __gpe_imprint2_vortexline_zdir_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi);
        cuErrCheck( cudaGetLastError() );
        
        
        if(gpe_mem.qfcoeff!=0.0) // quantum friction is active
        {
            cuErrCheck( gpe_compute_qf_potential(gpe_mem.d_psi, gpe_mem.d_wrk3C, gpe_mem.d_wrk3R) );
            __gpe_exp_Vstep1_qf_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi, gpe_mem.d_psi2, gpe_mem.d_wrk2R, gpe_mem.d_wrk3R);
        }
        else
        {
            // potential part exp(V/2)
            __gpe_exp_Vstep1_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi, gpe_mem.d_psi2, gpe_mem.d_wrk2R);
        }
        
        // kinetic part exp(T)
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi2, gpe_mem.d_psi2, CUFFT_FORWARD);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        __gpe_multiply_by_expT__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi2, gpe_mem.d_psi2);
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi2, gpe_mem.d_psi2, CUFFT_INVERSE);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        
        // potential part exp(V/2)
        if(gpe_mem.beta==0.0 && gpe_mem.qfcoeff==0.0)
        {
            // without normalization
            __gpe_exp_Vstep2_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi2, gpe_mem.d_psi, gpe_mem.d_wrk2R, gpe_mem.d_psi2);
        }
        else
        {
            __gpe_exp_Vstep2_part1_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi2, gpe_mem.d_psi2, gpe_mem.d_wrk2R);
            
            if(gpe_mem.beta!=0.0)
            {
                // with normalization between
                cuErrCheck( gpe_normalize(gpe_mem.d_psi2, gpe_mem.d_wrk2R+nxyz));
                
                // changing the phase
                __gpe_imprint2_vortexline_zdir_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi);
                cuErrCheck( cudaGetLastError() );
            }
            
            if(gpe_mem.qfcoeff!=0.0) // quantum friction is active
            {
                cuErrCheck( gpe_compute_qf_potential(gpe_mem.d_psi, gpe_mem.d_wrk3C, gpe_mem.d_wrk3R));
                __gpe_exp_Vstep2_part2_qf_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi2, gpe_mem.d_psi, gpe_mem.d_wrk2R, gpe_mem.d_psi2, gpe_mem.d_wrk3R);
            }
            else
            {
                __gpe_exp_Vstep2_part2_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi2, gpe_mem.d_psi, gpe_mem.d_wrk2R, gpe_mem.d_psi2);
            }
        }
        
        // kinetic part exp(T)
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi, gpe_mem.d_psi, CUFFT_FORWARD);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        __gpe_multiply_by_expT__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi, gpe_mem.d_psi);
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi, gpe_mem.d_psi, CUFFT_INVERSE);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        
        // potential part exp(V/2)
        __gpe_exp_Vstep3_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi, gpe_mem.d_psi2);
        
        if(gpe_mem.beta!=0.0)
        {
            // normalize
            cuErrCheck( gpe_normalize(gpe_mem.d_psi, gpe_mem.d_wrk2R+nxyz));
            if (chemical_potential)
            {
                double norm;
                cuErrCheck( cudaMemcpy( &norm, gpe_mem.d_wrk2R+nxyz, sizeof(double), cudaMemcpyDeviceToHost) ); 
                *chemical_potential = -.5*log(norm/gpe_mem.npart)/gpe_mem.dt;
            } 
        }
        
        gpe_mem.it = gpe_mem.it + 1;
    }
    
    return GPE_SUCCESS;
}

/**
 * Enforces vortex phase only in predictor step (first method).
 */
int gpe_evolve_vortex3(int nt, double* chemical_potential)
{
    cufftResult cufft_result;
    int i;
        
    for(i=0; i<nt; i++)
    {
        // changing the phase
        __gpe_imprint_vortexline_zdir_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi);
        cuErrCheck( cudaGetLastError() );
        
        
        if(gpe_mem.qfcoeff!=0.0) // quantum friction is active
        {
            cuErrCheck( gpe_compute_qf_potential(gpe_mem.d_psi, gpe_mem.d_wrk3C, gpe_mem.d_wrk3R) );
            __gpe_exp_Vstep1_qf_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi, gpe_mem.d_psi2, gpe_mem.d_wrk2R, gpe_mem.d_wrk3R);
        }
        else
        {
            // potential part exp(V/2)
            __gpe_exp_Vstep1_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi, gpe_mem.d_psi2, gpe_mem.d_wrk2R);
        }
        
        // kinetic part exp(T)
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi2, gpe_mem.d_psi2, CUFFT_FORWARD);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        __gpe_multiply_by_expT__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi2, gpe_mem.d_psi2);
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi2, gpe_mem.d_psi2, CUFFT_INVERSE);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        
        // potential part exp(V/2)
        if(gpe_mem.beta==0.0 && gpe_mem.qfcoeff==0.0)
        {
            // without normalization
            __gpe_exp_Vstep2_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi2, gpe_mem.d_psi, gpe_mem.d_wrk2R, gpe_mem.d_psi2);
        }
        else
        {
            __gpe_exp_Vstep2_part1_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi2, gpe_mem.d_psi2, gpe_mem.d_wrk2R);
            
            if(gpe_mem.beta!=0.0)
            {
                // with normalization between
                cuErrCheck( gpe_normalize(gpe_mem.d_psi2, gpe_mem.d_wrk2R+nxyz));
                
                // changing the phase
                //__gpe_imprint_vortexline_zdir_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi);
                //cuErrCheck( cudaGetLastError() );
            }
            
            if(gpe_mem.qfcoeff!=0.0) // quantum friction is active
            {
                cuErrCheck( gpe_compute_qf_potential(gpe_mem.d_psi, gpe_mem.d_wrk3C, gpe_mem.d_wrk3R));
                __gpe_exp_Vstep2_part2_qf_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi2, gpe_mem.d_psi, gpe_mem.d_wrk2R, gpe_mem.d_psi2, gpe_mem.d_wrk3R);
            }
            else
            {
                __gpe_exp_Vstep2_part2_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi2, gpe_mem.d_psi, gpe_mem.d_wrk2R, gpe_mem.d_psi2);
            }
        }
        
        // kinetic part exp(T)
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi, gpe_mem.d_psi, CUFFT_FORWARD);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        __gpe_multiply_by_expT__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi, gpe_mem.d_psi);
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi, gpe_mem.d_psi, CUFFT_INVERSE);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        
        // potential part exp(V/2)
        __gpe_exp_Vstep3_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi, gpe_mem.d_psi2);
        
        if(gpe_mem.beta!=0.0)
        {
            // normalize
            cuErrCheck( gpe_normalize(gpe_mem.d_psi, gpe_mem.d_wrk2R+nxyz));
            if (chemical_potential)
            {
                double norm;
                cuErrCheck( cudaMemcpy( &norm, gpe_mem.d_wrk2R+nxyz, sizeof(double), cudaMemcpyDeviceToHost) ); 
                *chemical_potential = -.5*log(norm/gpe_mem.npart)/gpe_mem.dt;
            } 
        }
        
        gpe_mem.it = gpe_mem.it + 1;
    }
    
    return GPE_SUCCESS;
}


// TODO: Check if this works better!
int gpe_evolve_enforced_phase(int nt, double* chemical_potential)
{
    cufftResult cufft_result;
    int i;
        
    for(i=0; i<nt; i++)
    {
        // changing the phase
        __gpe_enforce_phase__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi,gpe_mem.d_phase);
        cuErrCheck( cudaGetLastError() );
        
        
        if(gpe_mem.qfcoeff!=0.0) // quantum friction is active
        {
            cuErrCheck( gpe_compute_qf_potential(gpe_mem.d_psi, gpe_mem.d_wrk3C, gpe_mem.d_wrk3R) );
            __gpe_exp_Vstep1_qf_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi, gpe_mem.d_psi2, gpe_mem.d_wrk2R, gpe_mem.d_wrk3R);
        }
        else
        {
            // potential part exp(V/2)
            __gpe_exp_Vstep1_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi, gpe_mem.d_psi2, gpe_mem.d_wrk2R);
        }
        
        // kinetic part exp(T)
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi2, gpe_mem.d_psi2, CUFFT_FORWARD);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        __gpe_multiply_by_expT__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi2, gpe_mem.d_psi2);
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi2, gpe_mem.d_psi2, CUFFT_INVERSE);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        
        // potential part exp(V/2)
        if(gpe_mem.beta==0.0 && gpe_mem.qfcoeff==0.0)
        {
            // without normalization
            __gpe_exp_Vstep2_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi2, gpe_mem.d_psi, gpe_mem.d_wrk2R, gpe_mem.d_psi2);
        }
        else
        {
            __gpe_exp_Vstep2_part1_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi2, gpe_mem.d_psi2, gpe_mem.d_wrk2R);
            
            if(gpe_mem.beta!=0.0)
            {
                // with normalization between
                cuErrCheck( gpe_normalize(gpe_mem.d_psi2, gpe_mem.d_wrk2R+nxyz));
                
                // changing the phase
                __gpe_enforce_phase__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi,gpe_mem.d_phase);
                cuErrCheck( cudaGetLastError() );
            }
            
            if(gpe_mem.qfcoeff!=0.0) // quantum friction is active
            {
                cuErrCheck( gpe_compute_qf_potential(gpe_mem.d_psi, gpe_mem.d_wrk3C, gpe_mem.d_wrk3R));
                __gpe_exp_Vstep2_part2_qf_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi2, gpe_mem.d_psi, gpe_mem.d_wrk2R, gpe_mem.d_psi2, gpe_mem.d_wrk3R);
            }
            else
            {
                __gpe_exp_Vstep2_part2_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_psi2, gpe_mem.d_psi, gpe_mem.d_wrk2R, gpe_mem.d_psi2);
            }
        }
        
        // kinetic part exp(T)
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi, gpe_mem.d_psi, CUFFT_FORWARD);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        __gpe_multiply_by_expT__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi, gpe_mem.d_psi);
        cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi, gpe_mem.d_psi, CUFFT_INVERSE);
        if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
        
        // potential part exp(V/2)
        __gpe_exp_Vstep3_<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi, gpe_mem.d_psi2);
        
        if(gpe_mem.beta!=0.0)
        {
            // normalize
            cuErrCheck( gpe_normalize(gpe_mem.d_psi, gpe_mem.d_wrk2R+nxyz));
            if (chemical_potential)
            {
                double norm;
                cuErrCheck( cudaMemcpy( &norm, gpe_mem.d_wrk2R+nxyz, sizeof(double), cudaMemcpyDeviceToHost) ); 
                *chemical_potential = -.5*log(norm/gpe_mem.npart)/gpe_mem.dt;
            } 
        }
        
        gpe_mem.it = gpe_mem.it + 1;
    }
    
    return GPE_SUCCESS;
}




// ========================= Energy counting =================================================

__global__ void __gpe_compute_vext__(uint it, double *rho, double *wrk)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    uint ix, iy, iz, i;
    if(ixyz<nxyz)
    {
        ixyz2ixiyiz(ixyz,ix,iy,iz,i); 
        wrk[ixyz]=rho[ixyz]*gpe_external_potential(ix, iy, iz, it);
    }
}

__global__ void __gpe_compute_vint__(uint it, double *rho, double *wrk)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    if(ixyz<nxyz)
    {
        wrk[ixyz]=gpe_EDF(rho[ixyz], it);
    }
}

__global__ void __gpe_compute_vext_vint__(uint it, double *rho, double *wrk1, double *wrk2)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    uint ix, iy, iz, i;
    double lrho;
    if(ixyz<nxyz)
    {
        ixyz2ixiyiz(ixyz,ix,iy,iz,i); 
        lrho=rho[ixyz];
        wrk1[ixyz]=lrho*gpe_external_potential(ix, iy, iz, it);
        wrk2[ixyz]=gpe_EDF(lrho, it);
    }
}

__global__ void __gpe_multiply_by_k2__(cuCplx *psi_in, cuCplx *psi_out)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    uint ix, iy, iz, i;
    
    if(ixyz<nxyz)
    {
        ixyz2ixiyiz(ixyz,ix,iy,iz,i); 

        psi_out[ixyz]=cplxScale(psi_in[ixyz], ( d_kkx[ix]*d_kkx[ix] + d_kky[iy]*d_kky[iy] + d_kkz[iz]*d_kkz[iz] )/(constgpu(2.0*GAMMA*nxyz)) ); 
            
    }    
}

__global__ void __gpe_overlap_real__(cuCplx *psi1, cuCplx *psi2, double *overlap)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    if(ixyz<nxyz)
    {
        overlap[ixyz]= cplxMulR( cplxConj(psi1[ixyz]),  psi2[ixyz] );
    }
}

int gpe_energy(double *t, double *ekin, double *eint, double *eext)
{
    int ierr;
    cufftResult cufft_result;
   
    *t = gpe_mem.t0 + gpe_mem.dt*gpe_mem.it;
    
    if(gpe_mem.beta==0.0) // normalize - otherwise is normalized every time step
    {
        ierr=gpe_normalize_psi();
        if(ierr!=0) return ierr;
    }
    
    // Compute density
    __gpe_compute_density__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi, gpe_mem.d_wrk2R);
    
    // Compute <V_ext> and <V_int>
    double * wrk1 = (double *)gpe_mem.d_wrk;
    double * wrk2 = wrk1 + nxyz;
    __gpe_compute_vext_vint__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.it, gpe_mem.d_wrk2R, wrk1, wrk2);
    cuErrCheck( local_reduction(wrk1, nxyz, wrk1, gpe_mem.threads, 0) );
    cuErrCheck( local_reduction(wrk2, nxyz, wrk2, gpe_mem.threads, 0) );
    cuErrCheck( cudaMemcpy( eext , wrk1 , sizeof(double), cudaMemcpyDeviceToHost ) );
    cuErrCheck( cudaMemcpy( eint , wrk2 , sizeof(double), cudaMemcpyDeviceToHost ) );
        
    // Compute <T> - kinetic energy
    cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi, gpe_mem.d_psi2, CUFFT_FORWARD);
    if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
    __gpe_multiply_by_k2__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi2, gpe_mem.d_psi2);
    cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi2, gpe_mem.d_psi2, CUFFT_INVERSE);
    if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;    
    __gpe_overlap_real__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi, gpe_mem.d_psi2, gpe_mem.d_wrk2R);
    cuErrCheck( local_reduction(gpe_mem.d_wrk2R, nxyz, gpe_mem.d_wrk2R, gpe_mem.threads, 0) );
    cuErrCheck( cudaMemcpy( ekin , gpe_mem.d_wrk2R , sizeof(double), cudaMemcpyDeviceToHost ) );
    
    return 0;
}

// ========================== Currents of probability ================================

__global__ void __gpe_multiply_by_kx__(cuCplx *psi_in, cuCplx *psi_out)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    uint ix, iy, iz, i;
    
    if(ixyz<nxyz)
    {
        ixyz2ixiyiz(ixyz,ix,iy,iz,i); 

        psi_out[ixyz]=cplxScale(psi_in[ixyz], d_kkx[ix]/(constgpu(GAMMA*nxyz)) ); 
    }    
}

__global__ void __gpe_multiply_by_ky__(cuCplx *psi_in, cuCplx *psi_out)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    uint ix, iy, iz, i;
    
    if(ixyz<nxyz)
    {
        ixyz2ixiyiz(ixyz,ix,iy,iz,i); 

        psi_out[ixyz]=cplxScale(psi_in[ixyz], d_kky[iy]/(constgpu(GAMMA*nxyz)) ); 
    }    
}

__global__ void __gpe_multiply_by_kz__(cuCplx *psi_in, cuCplx *psi_out)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    uint ix, iy, iz, i;
    
    if(ixyz<nxyz)
    {
        ixyz2ixiyiz(ixyz,ix,iy,iz,i); 

        psi_out[ixyz]=cplxScale(psi_in[ixyz], d_kkz[iz]/(constgpu(GAMMA*nxyz)) ); 
    }    
}

int gpe_get_currents(double* t, double* jx, double* jy, double* jz)
{
    int ierr;
    
    cufftResult cufft_result;
   
    *t = gpe_mem.t0 + gpe_mem.dt*gpe_mem.it;
    
    int alloc=0;
    if(gpe_mem.d_wrk3R==NULL) // I need extra memory
    {
        alloc=1;
        cuErrCheck( cudaMalloc( &gpe_mem.d_wrk3R , sizeof(double)*nxyz ) );
    }
    
    // move to momentum space
    cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_psi, gpe_mem.d_psi2, CUFFT_FORWARD);  
    if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;
    
    // Compute d / dx and jx
    __gpe_multiply_by_kx__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi2, gpe_mem.d_wrk2);
    cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_wrk2, gpe_mem.d_wrk2, CUFFT_INVERSE);
    if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;    
    __gpe_overlap_real__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi, gpe_mem.d_wrk2, gpe_mem.d_wrk3R);    
    cuErrCheck( cudaMemcpy( jx , gpe_mem.d_wrk3R , sizeof(double)*nxyz, cudaMemcpyDeviceToHost ) );    
    
    // Compute d / dy and jy
    __gpe_multiply_by_ky__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi2, gpe_mem.d_wrk2);
    cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_wrk2, gpe_mem.d_wrk2, CUFFT_INVERSE);
    if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;    
    __gpe_overlap_real__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi, gpe_mem.d_wrk2, gpe_mem.d_wrk3R);    
    cuErrCheck( cudaMemcpy( jy , gpe_mem.d_wrk3R , sizeof(double)*nxyz, cudaMemcpyDeviceToHost ) );
    
    // Compute d / dz and jz
    __gpe_multiply_by_kz__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi2, gpe_mem.d_wrk2);
    cufft_result=cufftExecZ2Z(gpe_mem.plan, gpe_mem.d_wrk2, gpe_mem.d_wrk2, CUFFT_INVERSE);
    if(cufft_result!= CUFFT_SUCCESS) return (int)cufft_result;    
    __gpe_overlap_real__<<<gpe_mem.blocks, gpe_mem.threads>>>(gpe_mem.d_psi, gpe_mem.d_wrk2, gpe_mem.d_wrk3R);    
    cuErrCheck( cudaMemcpy( jz , gpe_mem.d_wrk3R , sizeof(double)*nxyz, cudaMemcpyDeviceToHost ) );

    // Free memory
    if(alloc) 
    {
        cuErrCheck( cudaFree(gpe_mem.d_wrk3R) );  
        gpe_mem.d_wrk3R = NULL;      
    }
    
    return GPE_SUCCESS;
}



// ======================================= TESTING ================================================================================

__global__ void print_gpu_array_nans( cuCplx* psi, int size)
{
    size_t ixyz= threadIdx.x + blockIdx.x * blockDim.x;
    uint ix, iy, iz, i;
    
    if( ixyz<nxyz && ixyz < size)
    {
        ixyz2ixiyiz(ixyz,ix,iy,iz,i);
        ix -= nx/2;
        iy -= ny/2;
        iz -= nz/2;
        
        if (isnan(psi[i].x) || isnan(psi[i].y)) printf("x: %d\ty: %d\tz: %d\t\tpsi %e + %ej\n",ix,iy,iz,psi[i].x,psi[i].y);
    }
}

// ==================================================================================================================================
