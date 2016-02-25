from __future__ import print_function
from read_dftc_info import read_dftc_info

# math
import numpy as np
import math

# file processing
import os.path
import sys
import string
import csv
import re

"""
SOME RULES:

1. Always use path to .dftc file (not for other files in this directory)
2. Data are given in oscilatory units (except psi)
"""


###########################################################################################################
#
#     LOADING DATA
#
############################

def get_data(datafile,measure_it=0):
    
    print('loading file {} ...'.format(datafile))
    Nx,Ny,Nz,dt,nom,a_scat,aspect,omega_x,r0,npart,time_tot = read_dftc_info(datafile)
    a_ho = 1./math.sqrt(omega_x)
    
    
    data = np.memmap(datafile,dtype=np.complex128)
    if (len(data) == nom*Nx*Ny*Nz):
        print('# lenght of array consistent.\t\tnom*Nx*Ny*Nz =',nom*Nx*Ny*Nz,'\tarray shape:',data.shape)
    else:
        print('# ERROR: lenght of array incosistent.\tnom*Nx*Ny*Nz =',nom*Nx*Ny*Nz,'\tarray shape:',data.shape)
    data = np.reshape(data,[nom,Nx,Ny,Nz])
    density = np.abs(data[measure_it,:,:,:])**2
    phase   = np.angle(data[measure_it,:,:,:])
    print('data shape:',density.shape)
    
    return density, phase

def get_data_all(datafile):
    
    print('loading file {} ...'.format(datafile))
    Nx,Ny,Nz,dt,nom,a_scat,aspect,omega_x,r0,npart,time_tot = read_dftc_info(datafile)
    a_ho = 1./math.sqrt(omega_x)
    
    
    data = np.memmap(datafile,dtype=np.complex128)
    if (len(data) == nom*Nx*Ny*Nz):
        print('# lenght of array consistent.\t\tnom*Nx*Ny*Nz =',nom*Nx*Ny*Nz,'\tarray shape:',data.shape)
    else:
        print('# ERROR: lenght of array incosistent.\tnom*Nx*Ny*Nz =',nom*Nx*Ny*Nz,'\tarray shape:',data.shape)
    data = np.reshape(data,[nom,Nx,Ny,Nz])
    density = np.abs(data[:,:,:,:])**2
    phase   = np.angle(data[:,:,:,:])
    print('data shape:',density.shape)
    
    return density, phase


# TODO: Some more processing for trajectory needed -> return array indexed by time
def get_trajectory(datafile,show=False):
    trajectory_file = datafile+'.trajectory'
    print('loading file {} ...'.format(trajectory_file))
    data = np.loadtxt(trajectory_file,skiprows=1)
    
    if show is True:
        print('t\t  x\t y\t z')
        for t,x,y,z in zip(data[:,0],data[:,1],data[:,2],data[:,3]):
            print(t,'\t','{:4.2f}'.format(x),'\t','{:4.2f}'.format(y),'\t','{:4.2f}'.format(z))
    
    return data[:,0],data[:,1],data[:,2],data[:,3]

def trajectory_in_xy(T,X,Y,Z,slice_z):
	t = T[np.where(Z == slice_z)]
	x = X[np.where(Z == slice_z)]
	y = Y[np.where(Z == slice_z)]
	
	return t,x,y

def trajectory_at_t(T,X,Y,Z,slice_t):
	x = X[np.where(T == slice_t)]
	y = Y[np.where(T == slice_t)]
	z = T[np.where(T == slice_t)]
	
	return x,y,z

def get_energy(datafile,mode='ite'):
    datafile = datafile+'.energy'
    print('loading file {} ...'.format(datafile))
    data = np.loadtxt(datafile,skiprows=1)
    
    if (mode == 'ite'):
        return data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5]
    if (mode == 'rte'):
        pass #return data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5]

###########################################################################################################
#
#     SLICES
#
############################

def get_slice_x(data,y_slice=32,z_slice=32):
    return data[:,y_slice,z_slice]

def get_slice_y(data,x_slice=32,z_slice=32):
    return data[x_slice,:,z_slice]

def get_slice_z(data,x_slice=32,y_slice=32):
    return data[x_slice,y_slice,:]

def get_slice_xy(data,z_slice=32):
    return data[:,:,z_slice]

def get_slice_xz(data,y_slice=32):
    return data[:,y_slice,:]

def get_slice_yz(data,x_slice=32):
    return data[x_slice,:,:]

def create_grid(Nx=8,Ny=8,Nz=8,scale=1.):
    x = np.linspace(-Nx/2.,Nx/2.-1.,Nx,endpoint=True) * scale
    y = np.linspace(-Ny/2.,Ny/2.-1.,Ny,endpoint=True) * scale
    z = np.linspace(-Nz/2.,Nz/2.-1.,Nz,endpoint=True) * scale
    
    return x,y,z

def create_meshgrid(Nx=8,Ny=8,Nz=8,scale=1.):
    x,y,z = create_grid(Nx=Nx,Ny=Ny,Nz=Nz,scale=scale)
    
    if (Nz == 0):
        X,Y = np.meshgrid(x,y, indexing='ij')
        return X,Y
    else:
        X,Y,Z = np.meshgrid(x,y,z, indexing='ij')
        return X,Y,Z
