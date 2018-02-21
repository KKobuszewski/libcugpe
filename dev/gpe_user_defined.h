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
 * @brief cuGPE library - user defined functions
 * */
#include <cuComplex.h>
#include <float.h>

#ifndef __GPE_USER_DEFINED__
#define __GPE_USER_DEFINED__

#define PARTICLES 1
#define DIMERS 2

#define M_PI 3.14159265358979323846




/***************************************************************************/ 
/**************************** USER DEFINED *********************************/
/***************************************************************************/

/**
 * GPE_FOR can be either PARTICLES or DIMERS. If PARTICLES then \f$\kappa=1\f$, if DIMERS then \f$\kappa=2\f$
 * 
 * Create your own type of density functionals in parenthesis: 
 * #if (INTERACTIONS == <number>) 
 * ... 
 * #endif
 * 
 * */



// definitions of device constants that could be 
extern __device__  double d_user_param[];
extern __device__  uint d_nx; // lattice size in x direction
extern __device__  uint d_ny; // lattice size in y direction
extern __device__  uint d_nz; // lattice size in z direction
extern __device__  double d_dt;
extern __device__  double d_t0;
extern __device__  double d_npart;



/**
 * Function changes wave function.
 * This function is called before each integration step.
 * NOTE: This function assumes that norm is not changed after modification.
 * @param ix - x coordinate, ix=0,1,...,d_nx-1, where d_nx is global variable
 * @param iy - y coordinate, iy=0,1,...,d_ny-1, where d_ny is global variable
 * @param iz - z coordinate, iy=0,1,...,d_nz-1, where d_ny is global variable
 * @param it - time value, ie. time = d_t0 + it*d_dt, d_t0 and d_dt are global variables
 * @param psi - psi(ix, iy, iz, it) - psi is normalized, i.e. int n(r) d^3r = npart, where n(r) computed according gpe_density(psi)
 * @return value of wave function after modification
 * */
inline __device__  cuDoubleComplex gpe_modify_psi(uint ix, uint iy, uint iz, uint it, cuDoubleComplex psi)
{
    return psi; // no change
}



// a little magic with preprocessor

#if (INTERACTIONS==0) // unitary fermi gas for dimers


/* ***************************************************************************************************** *
 *                                                                                                       *
 *                            UNITARY REGIME DENSITY FUNCTIONAL                                          *
 *                                                                                                       *
 * ***************************************************************************************************** */

// here you can #define MAX_USER_PARAMS
#define MAX_USER_PARAMS 4
typedef enum {OMEGA_X, OMEGA_Y, OMEGA_Z, A_SCAT} user_params_t;

#ifndef GPE_FOR
#define GPE_FOR DIMERS // by default for bosonic dimers consisted of pair fermion-fermion
#endif

#ifndef UNITARY
#define UNITARY
#endif
#pragma message ()"Compiling for unitary Fermi gas")

/**
 * Function returns value of energy density functional EDF
 * @param rho - density, computed according gpe_density(psi)
 * @param it - time value, ie. time = d_t0 + it*d_dt, d_t0 and d_dt are global variables
 * @return value of energy density functional
 * */
inline __device__  double gpe_EDF(double rho, uint it)
{
    // Density energy functional for unitary Fermi gas
    // see: Phys. Rev. A 90, 043638 (2014)
    return 0.37*0.6*rho*pow(3.0*M_PI*M_PI*rho, 2.0/3.0)/2.; // unitary limit
}

/**
 * Function returns value of local part of mean field, i.e U= d_EDF / dn - variational derivative of EDF with respect to density
 * @param rho - density, computed according gpe_density(psi)
 * @param it - time value, ie. time = d_t0 + it*d_dt, d_t0 and d_dt are global variables
 * @return value of mean field
 * */
inline __device__  double gpe_dEDFdn(double rho, uint it)
{
    // see: Phys. Rev. A 90, 043638 (2014)
    return 0.37*pow(3.0*M_PI*M_PI*rho, 2.0/3.0)/2.0; // unitary limit
}

/**
 * Function returns value of Fourier transform of particle-particle interaction potential.
 * Used for evaluation F^{-1} [ Vpp_k * F[ density ] ].
 *
 * @param kx  - k_x coordinate in momentum space
 * @param ky  - k_y coordinate in momentum space
 * @param kz  - k_z coordinate in momentum space
 *
 * @return value of F[V_pp]
 */
inline __device__ double gpe_Vpp_k(double kx, double ky, double kz, uint it)
{
  return 0; // no long-range term.
}

/**
 * Function returns value of gradient corrections.
 * @param
 *
 * @return
 */
inline __device__ double gpe_gradient_corrections(double rho, double grad_density_x, double grad_density_y, double grad_density_z,
                                           double laplacian_density, double it)
{
  return 0; // no gradient corrections
}

static inline void gpe_print_interactions_type()
{
    printf("# GPE FOR FERMIONIC DIMERS IN UNITARY LIMIT\n");
}

#elif (INTERACTIONS==1) // BEC regime for fermionic dimers

/* ***************************************************************************************************** *
 *                                                                                                       *
 *                                ETF BEC REGIME DENSITY FUNCTIONAL                                      *
 *                                                                                                       *
 * ***************************************************************************************************** */

// here you can #define MAX_USER_PARAMS
#define MAX_USER_PARAMS 4
typedef enum {OMEGA_X, OMEGA_Y, OMEGA_Z, A_SCAT} user_params_t;

#ifndef GPE_FOR
#define GPE_FOR DIMERS // by default for bosonic dimers consisted of pair fermion-fermion
#endif

#pragma message ("Compiling for Fermi gas on BEC side of crossover")

#define XI 0.37           // Bertsch's parameter
#define CONTACT 0.901     // Tan's contact

/**
 * Function returns value of energy density functional EDF
 * @param rho - density, computed according gpe_density(psi)
 * @param it - time value, ie. time = d_t0 + it*d_dt, d_t0 and d_dt are global variables
 * @return value of energy density functional
 * */
inline __device__  double gpe_EDF(double rho, uint it)
{    
    // Density energy functional for fermionic cold atoms (out of unitary regime and only positive scattering lengths)
    // see: Phys. Rev. Lett. 112, 025301 (2014)
    const double a = d_user_param[A_SCAT]; // scattering length
    const double kF=pow(3.0*M_PI*M_PI*rho , 1.0/3.0);
    const double eF=0.5*kF*kF;
    //const double x  = 1.0/(a*kF);
    const double x  = (kF > DBL_MIN) ? 1.0/(a*kF) : DBL_MAX;
    return 0.6*eF*rho*XI*(XI+x) / ( XI + x*(1.0+CONTACT) + 3.0*M_PI*XI*x*x ) - rho/(2.0*a*a);
}

/**
 * Function returns value of local part of mean field, i.e U= d_EDF / dn - variational derivative of EDF with respect to density
 * @param rho - density, computed according gpe_density(psi)
 * @param it - time value, ie. time = d_t0 + it*d_dt, d_t0 and d_dt are global variables
 * @return value of mean field
 * */
inline __device__  double gpe_dEDFdn(double rho, uint it)
{
    // Density energy functional for fermionic cold atoms
    // see: Phys. Rev. Lett. 112, 025301 (2014)
    //const double a = d_user_param[A_SCAT]; // scattering length passed via user params array
    //const double kF= pow(3.0*M_PI*M_PI*rho , 1.0/3.0);
    //const double eF= 0.5*kF*kF;
    //const double x = 1.0/(a*kF);
    //const double D = ( XI + x*(1.0+CONTACT) + 3.0*M_PI*XI*x*x);
    //return XI*eF*(XI+0.8*x)/D + 0.2*XI*eF*(XI+x)*x*( (1.0+CONTACT)+6.0*M_PI*XI*x )/(D*D) - 1.0/(2.0*a*a);
    
    // Density energy functional for fermionic cold atoms
    // see: Phys. Rev. Lett. 112, 025301 (2014)
    
    const double a = d_user_param[A_SCAT]; // scattering length passed via user params array
    
    if (rho > 1e-15)
    {    
        const double kF = pow(3.0*M_PI*M_PI*rho , 1.0/3.0);
        const double eF = 0.5*kF*kF;
        //const double x  = 1.0/(a*kF);  
        const double x  = (kF > DBL_MIN) ? 1.0/(a*kF) : DBL_MAX;
        const double D = ( XI + x*(1.0+CONTACT) + 3.0*M_PI*XI*x*x);
        return XI*eF*(XI+0.8*x)/D + 0.2*XI*eF*(XI+x)*x*( (1.0+CONTACT)+6.0*M_PI*XI*x )/(D*D) - 1.0/(2.0*a*a);
    }
    else
    {
        return - 1.0/(2.0*a*a);
    }
}

/**
 * Function returns value of Fourier transform of particle-particle interaction potential.
 * Used for evaluation F^{-1} [ Vpp_k * F[ density ] ].
 *
 * @param kx  - k_x coordinate in momentum space
 * @param ky  - k_y coordinate in momentum space
 * @param kz  - k_z coordinate in momentum space
 *
 * @return value of F[V_pp]
 */
inline __device__ double gpe_Vpp_k(double kx, double ky, double kz)
{
  return 0; // no long-range term.
}

/**
 * Function returns value of gradient corrections.
 */
inline __device__ double gpe_gradient_corrections(double rho, double grad_density_x, double grad_density_y, double grad_density_z,
                                           double laplacian_density, double it)
{
  return 0; // no gradient corrections
}


static inline void gpe_print_interactions_type()
{
    printf("# GPE FOR FERMIONIC DIMERS IN BEC LIMIT\n");
}


#elif (INTERACTIONS == -1) // simple contact bosonic BEC

/* ***************************************************************************************************** *
 *                                                                                                       *
 *                                BOSONIC BEC DENSITY FUNCTIONAL                                         *
 *                                                                                                       *
 * ***************************************************************************************************** */

// here you can #define MAX_USER_PARAMS
#define MAX_USER_PARAMS 4
typedef enum {OMEGA_X, OMEGA_Y, OMEGA_Z, A_SCAT} user_params_t;

#ifndef GPE_FOR
#define GPE_FOR PARTICLES // simple bosonic GP
#endif

#ifndef CONTACT
#define CONTACT
#endif

#pragma message ("Compiling for contact Bose gas.")


/**
 * Function returns value of energy density functional EDF
 * @param rho - density, computed according gpe_density(psi)
 * @param it - time value, ie. time = d_t0 + it*d_dt, d_t0 and d_dt are global variables
 * @return value of energy density functional
 * */
inline __device__  double gpe_EDF(double rho, uint it)
{
    const double a = d_user_param[A_SCAT]; // scattering length passed via user params array
    return 2*M_PI*a*rho*rho; // 1/2 g N |psi|^2
}

/**
 * Function returns value of local part of mean field, i.e U= d_EDF / dn - variational derivative of EDF with respect to density
 * @param rho - density, computed according gpe_density(psi)
 * @param it - time value, ie. time = d_t0 + it*d_dt, d_t0 and d_dt are global variables
 * @return value of mean field
 * */
inline __device__  double gpe_dEDFdn(double rho, uint it)
{
    const double a = d_user_param[A_SCAT]; // scattering length passed via user params array
    // TODO: Change constant !!!
    return 4*M_PI*a*rho; // contact interaction term is considered as a mean field value
}

/**
 * Function returns value of Fourier transform of particle-particle interaction potential.
 * Used for evaluation F^{-1} [ Vpp_k * F[ density ] ].
 *
 * @param kx  - k_x coordinate in momentum space
 * @param ky  - k_y coordinate in momentum space
 * @param kz  - k_z coordinate in momentum space
 *
 * @return value of F[V_pp]
 */
inline __device__ double gpe_Vpp_k(double kx, double ky, double kz)
{
  return 0; // no long-range term.
}

/**
 * Function returns value of gradient corrections.
 */
inline __device__ double gpe_gradient_corrections(double rho, double grad_density_x, double grad_density_y, double grad_density_z,
                                           double laplacian_density, double it)
{
  return 0; // no gradient corrections
}


static inline void gpe_print_interactions_type()
{
    printf("# GPE FOR BOSONS IN BEC\n");
}



#elif (INTERACTIONS == -2)

/* ***************************************************************************************************** *
 *                                                                                                       *
 *                       BOSONIC BEC WITH DIPOLAR INTERACTIONS DENSITY FUNCTIONAL                        *
 *                                                                                                       *
 * ***************************************************************************************************** */

// here you can #define MAX_USER_PARAMS
#define MAX_USER_PARAMS 5
typedef enum {OMEGA_X, OMEGA_Y, OMEGA_Z, A_CON, A_DIP} user_params_t;

#ifndef GPE_FOR
#define GPE_FOR PARTICLES // simple bosonic GP
#endif

#ifndef DIPOLAR
#define DIPOLAR
#endif

#pragma message ("Compiling for Bose gas with dipolar interactions.")

#include <float.h> // for macro DBL_EPSILON

/**
 * Function returns value of energy density functional EDF for contat interactions only. 
 * This function is used to compute energy, so that we can get contact part of interactions energy.
 * @param rho - density, computed according gpe_density(psi)
 * @param it - time value, ie. time = d_t0 + it*d_dt, d_t0 and d_dt are global variables
 * @return value of energy density functional
 * */
inline __device__  double gpe_EDF(double rho, uint it)
{
    const double a = d_user_param[A_CON]; // scattering length passed via user params array
    // TODO: Change constant !!!
    return 2*M_PI*a*rho*rho; // 1/2 g_dd |psi|^4
}

/**
 * Function returns value of local part of mean field, i.e U= d_EDF / dn - variational derivative of EDF with respect to density
 * @param rho - density, computed according gpe_density(psi)
 * @param it - time value, ie. time = d_t0 + it*d_dt, d_t0 and d_dt are global variables
 * @return value of mean field
 * */
inline __device__  double gpe_dEDFdn(double rho, uint it)
{
    //const double a = d_user_param[A_SCAT]; // scattering length passed via user params array
    return 0.; // unitary limit
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// TODO: optimize contact interactions
#define DIRAC_DELTA_TRANSFORM ((double) 1.)
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


/**
 * Function returns value of Fourier transform of particle-particle interaction potential.
 * Used for evaluation F^{-1} [ Vpp_k * F[ density ] ].
 *
 * @param kx  - k_x coordinate in momentum space
 * @param ky  - k_y coordinate in momentum space
 * @param kz  - k_z coordinate in momentum space
 *
 * @return value of F[V_pp]
 */
inline __device__ double gpe_Vpp_k(double kx, double ky, double kz)
{
    // registers
    const double a_con = d_user_param[A_CON];
    const double a_dip = d_user_param[A_DIP];
    const double k_sq = kx*kx + ky*ky + kz*kz;

    if ( k_sq < DBL_EPSILON )
    {
        return 4.*M_PI*a_con*DIRAC_DELTA_TRANSFORM;
    }
    else
    {
        return 4.*M_PI*(  a_con*DIRAC_DELTA_TRANSFORM + a_dip*( 3.*kz*kz/k_sq - 1. )  );
        //  g_con/(2 \pi)^3/2 - g_dd * ( 1 - 3*cos^2(\mu,k) )  NOTE: dipoles polarized in z direction!
    }
}

/**
 * Function returns value of gradient corrections.
 */
inline __device__ double gpe_gradient_corrections(double rho, double grad_density_x, double grad_density_y, double grad_density_z,
                                           double laplacian_density, double it)
{
  return 0; // no gradient corrections
}

static inline void gpe_print_interactions_type()
{
    printf("# GPE FOR BOSONS IN BEC WITH DIPOLAR INTERACTIONS\n");
}


#endif // end choosing type





// ===========================================================================================================================

/* ************************************************************************************************************************* *
 *                                                                                                                           *
 *                                                  EXTERNAL POTENTIALS                                                      *
 *                                                                                                                           *
 * ************************************************************************************************************************* */

// TODO:
// 1. Make another arrays d_vext_params and d_interactions_params instead of d_user_params

#ifndef VEXT
#define VEXT 0
#endif

#if (VEXT == 0) // HARMONIC POTENTIAL

#pragma message ("Compiling for harmonic potential.")

/**
 * Function computes value of external potential V_ext(x,y,z,t)
 * @param ix - x coordinate, ix=0,1,...,d_nx-1, where d_nx is global variable 
 * @param iy - y coordinate, iy=0,1,...,d_ny-1, where d_ny is global variable 
 * @param iz - z coordinate, iy=0,1,...,d_nz-1, where d_ny is global variable 
 * @param it - time value, ie. time = d_t0 + it*d_dt, d_t0 and d_dt are global variables
 * @return value of external potential
 * */
inline __device__  double gpe_external_potential(uint ix, uint iy, uint iz, uint it)
{

    // harmonic trap:
    // V(x,y,z) = 0.5*(omega_x*x)^2 + 0.5*(omega_y*y)^2 + 0.5*(omega_z*z)^2

    // frequencies are passed through d_user_param array
    const double omega_x = d_user_param[OMEGA_X];
    const double omega_y = d_user_param[OMEGA_Y];
    const double omega_z = d_user_param[OMEGA_Z];

    // coordinate with respect to center of the box
    const double _ix = (double)(ix) - 1.0*(NX/2);
    const double _iy = (double)(iy) - 1.0*(NY/2);
    const double _iz = (double)(iz) - 1.0*(NZ/2);


    // TODO: use exp or anharmonicity + check HOW GRAVITY CHANGES THE MOTION
    // TODO: maybe passing lambda will be more effective?
    // TODO: maybe this should be done in another array? <- quicker?
    double V_trap =   0.5*_ix*_ix*omega_x*omega_x
                    + 0.5*_iy*_iy*omega_y*omega_y
                    + 0.5*_iz*_iz*omega_z*omega_z;

    return V_trap;
}

#ifdef __CUDACC__
const char* potential_type = "HARMONIC POTENTIAL\n";
#endif


#elif (VEXT == 1) // WOODS-SAXON TUBE
#pragma message ("Compiling for periodic tube.")

typedef enum {V0,A,R} potential_params_t;

/**
 * Function computes value of external potential V_ext(x,y,z,t)
 * @param ix - x coordinate, ix=0,1,...,d_nx-1, where d_nx is global variable 
 * @param iy - y coordinate, iy=0,1,...,d_ny-1, where d_ny is global variable 
 * @param iz - z coordinate, iy=0,1,...,d_nz-1, where d_ny is global variable 
 * @param it - time value, ie. time = d_t0 + it*d_dt, d_t0 and d_dt are global variables
 * @return value of external potential 
 * */
inline __device__  double gpe_external_potential(const uint ix,const uint iy,const uint iz,const uint it)
{
    // tube with Woods-Saxon potential:
    // V(x,y,z) = -Vo/(1+exp( (sqrt(x^2+y^2)-R)/a )
    
    // frequencies are passed through d_user_param array
    const double V_0 = d_user_param[V0];
    const double a   = d_user_param[A];
    const double r   = d_user_param[R];
    
    // coordinate with respect to center of the box
    const double _ix = (double)(ix) - 1.0*(NX/2);
    const double _iy = (double)(iy) - 1.0*(NY/2);
    const double _iz = (double)(iz) - 1.0*(NZ/2);
    
    // TODO: use exp or anharmonicity + check HOW GRAVITY CHANGES THE MOTION
    // TODO: maybe passing lambda will be more effective?
    // TODO: maybe this should be done in another array? <- quicker?
    const double V_trap = V_0*(1.-1./(  1 + exp( (sqrt(_ix*_ix + _iy*_iy) - r)/a )  ));
    
    return V_trap;
}

#ifdef __CUDACC__
const char* potential_type = "WOODS-SAXON TUBE\n";
#endif

//elif == 2 ...
#endif // VEXT (POTENTIALS)

#endif
