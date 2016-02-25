from __future__ import print_function
import re

def read_dftc_info(infofile,show=False):
    numeric_const_pattern = r"""
    [-+]? # optional sign
    (?:
        (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
        |
        (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
    )
    # followed by optional exponent part if desired
    (?: [Ee] [+-]? \d+ ) ?
    """
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    
    infofile=infofile+'.info'
    
    time_tot = 0.
    
    with open(infofile,'r') as f:
        info = f.readlines()
        info = info[2:]                                                 # passes by two lines describing type of simulation. TODO: Make use of it
        Nx = int(re.findall(r'\d+', info[0])[0])
        Ny = int(re.findall(r'\d+', info[1])[0])
        Nz = int(re.findall(r'\d+', info[2])[0])
        dt = float(rx.findall(info[3])[0])
        nom = int(re.findall(r'\d+', info[4])[0])
        a_scat = float(rx.findall(info[5])[0])
        aspect = float(rx.findall(info[6])[0])
        omega_x = float(rx.findall(info[7])[0])
        r0 = float(rx.findall(info[8])[0])
        npart = float(rx.findall(info[9])[0])
        if (len(info) > 9):
            time_tot = float(rx.findall(info[9])[0])
    
    if (show is True):
        print('lattice:\t',Nx,'x',Ny,'x',Nz)
        print('dt:\t\t',dt)
        print('nom:\t\t',nom)
        print('a_scat:\t\t',a_scat)
        print('aspect ratio:\t',aspect)
        print('omega x:\t',omega_x)
        print('r0:\t\t',r0)
        print('npart:\t\t',npart)
    
    return Nx,Ny,Nz,dt,nom,a_scat,aspect,omega_x,r0,npart,time_tot


if __name__ == '__main__':
    # execute only if run as a script
    import glob
    files = glob.glob('*.dftc.info')
    print()
    for f in files:
        print('reading file:',f)
        Nx,Ny,Nz,dt,nom,a_scat,aspect,omega_x,r0,npart = read_dftc_info(f)
        print('lattice:\t',Nx,'x',Ny,'x',Nz,'\t',type(Nx))
        print('dt:\t\t',dt,'\t\t',type(dt))
        print('nom:\t\t',nom,'\t\t',type(nom))
        print('a_scat:\t\t',a_scat,'',type(a_scat))
        print('aspect ratio:\t',aspect,'\t\t',type(aspect))
        print('omega x:\t',omega_x,'\t\t',type(omega_x))
        print('r0:\t\t',r0,'\t\t',type(r0))
        print('npart:\t\t',npart,'\t',type(npart))
        a_ho = 1./math.sqrt(omega_x)
        print('a_ho:\t\t',a_ho)
        print()
