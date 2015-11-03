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

#include <cuda.h>
#include <math.h>
#include <cuComplex.h>

#ifndef __CU_CPLX_MATH_H__
#define __CU_CPLX_MATH_H__

#define constgpu(c) (double)(c)

typedef cuDoubleComplex cuCplx; // structure with two doubles x and y for real and imaginary parts, binary compatible with <ccomplex>
typedef double complex cplx;    // <ccomplex> complex number type, not binary compatible with c++ <complex> class


////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline cuCplx cplxAdd(cuCplx a, cuCplx b)
{
    cuCplx c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

static __device__ __host__ inline cuCplx cplxSub(cuCplx a, cuCplx b)
{
    cuCplx c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    return c;
}

// Complex scale
static __device__ __host__ inline cuCplx cplxScale(cuCplx a, double s)
{
    cuCplx c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex scalei = multiplication  by purely imaginay number, i.e: c = a*(i*s)
static __device__ __host__ inline cuCplx cplxScalei(cuCplx a, double s)
{
    cuCplx c;
    c.x = s * a.y * constgpu(-1.0);
    c.y = s * a.x;
    return c;
}

static __device__ __host__ inline cuCplx cplxConj(cuCplx a)
{
    a.y=-1.0*a.y;
    return a;
}

// norm2=|a|^2
static __device__ __host__ inline double cplxNorm2(cuCplx a)
{
    return (double)(a.x*a.x + a.y*a.y);
}

// abs=|a|
static __device__ __host__ inline double cplxAbs(cuCplx a)
{
    return (double)(sqrt(cplxNorm2(a)));
}

static __device__ __host__ inline double cplxArg(cuCplx a)
{
    return (double)(atan2(a.y,a.x));
}


// Complex multiplication
static __device__ __host__ inline cuCplx cplxMul(cuCplx a, cuCplx b)
{
    cuCplx c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex multiplication - and return only real part
static __device__ __host__ inline double cplxMulR(cuCplx a, cuCplx b)
{
    double c;
    c = a.x * b.x - a.y * b.y;
    return c;
}

// Complex multiplication - and return only imag part
static __device__ __host__ inline double cplxMulI(cuCplx a, cuCplx b)
{
    double c;
    c = a.x * b.y + a.y * b.x;
    return c;
}

static __device__ __host__ inline cuCplx cplxDiv(cuCplx a, cuCplx b)
{
    cuCplx c;
    double t=b.x*b.x + b.y*b.y;
    c.x = (a.x * b.x + a.y * b.y)/t;
    c.y = (a.y * b.x - a.x * b.y)/t;
    return c;
}

static __device__ __host__ inline cuCplx cplxSqrt(cuCplx z)
{
    cuCplx r;
    double norm=sqrt(z.x*z.x + z.y*z.y);

    r.x=sqrt((norm+z.x)/2.);
    if(z.y>0.0) r.y=sqrt((norm-z.x)/2.);
    else r.y=constgpu(-1.)*sqrt((norm-z.x)/2.);
    return r;
}

static __device__ __host__ inline cuCplx cplxLog(cuCplx z)
{
    cuCplx r;
    double norm=z.x*z.x + z.y*z.y;
    double arg=atan2(z.y,z.x);
    r.x=constgpu(0.5)*log(norm);
    r.y=arg;
    return r;
}

// z=exp(i*x)=cos(x)+i*sin(x), x - real number
static __device__ __host__ inline cuCplx cplxExpi(double x)
{
    cuCplx r;
    r.x=cos(x);
    r.y=sin(x);
    return r;
}

// z=exp(x), x - complex number
static __device__ __host__ inline cuCplx cplxExp(cuCplx x)
{
    double t;
    cuCplx r;
    t=exp(x.x);
    r.x=t*cos(x.y);
    r.y=t*sin(x.y);
    return r;
}

// returns 1/c
static __device__ __host__ inline cuCplx cplxInv(cuCplx c)
{
    cuCplx r;
    double n=c.x*c.x + c.y*c.y;
    r.x=c.x/n;
    r.y=constgpu(-1.0)*c.y/n;
    return r;
}


#endif