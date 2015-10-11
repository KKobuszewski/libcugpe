# math
import numpy as np
import math

# file processing
import os.path
import glob
import string
import csv


def compare_binary(pattern,testname):
    
    files = glob.glob(pattern)
    
    data1 = np.memmap(files[0],dtype=np.complex128)
    data2 = np.memmap(files[1],dtype=np.complex128)
    
    diff = data1 - data2
    diff_real = np.abs(np.real(diff))
    diff_imag = np.abs(np.imag(diff))
    diff_abs = np.abs(data1) - np.abs(data2)
    diff_phase = np.angle(data1) - np.angle(data2)
    diff_phase[diff_phase > math.pi] -= (2*math.pi)
    diff_phase = np.abs(diff_phase)
    
    
    if np.max(diff_real) < 1e-15 and np.max(diff_imag) < 1e-15:
        print('TEST {}:\tOK'.format(testname))
    else:
        print('TEST {}:\tFAILED'.format(testname))
        print('differences between {} and {}'.format(files[0],files[1]))
        print('max difference real part:\t',np.max(diff_real ))
        print('max difference imag part:\t',np.max(diff_imag ))
        print('max difference modulus:\t\t',np.max(diff_abs  ))
        print('max difference phase:\t\t',  np.max(diff_phase))
    print()

def compare_columns(column1,column2,testname):
    if np.allclose(column1,column2,atol=1e-15):
        print('TEST {}:\tOK'.format(testname))
    else:
        print('TEST {}:\tFAILED'.format(testname))
        
        diff = np.abs(column1 - column2)
        print('maximal difference:',np.max(diff))

def compare_txt(pattern,testname):
    print('TESTING {}'.format(testname))
    
    files = glob.glob(pattern)
    
    data1 = np.loadtxt(files[0],skiprows=1)
    data2 = np.loadtxt(files[1],skiprows=1)
    
    if len(data1) == len(data2):
        print('TEST COMPARISON EVOLUTION LENGTH:\tOK')
        compare_columns(data1[:,1],data2[:,1],"TOTAL ENERGY")
        compare_columns(data1[:,2],data2[:,2],"KINETIC ENERGY")
        compare_columns(data1[:,3],data2[:,3],"POTENTIAL ENERGY")
        compare_columns(data1[:,4],data2[:,4],"INTERACT. ENERGY")
    else:
        print('TEST COMPARISON EVOLUTION LENGTH:\tFAILED')
        print('evolution steps in file {}:\t'.format(files[0]),len(data1))
        print('evolution steps in file {}:\t'.format(files[1]),len(data2))
    print()


compare_binary('*.bin','ITE PSI')
compare_binary('*.dat','RTE PSI')
compare_txt('energy_ite*.txt','ITE STATISTICS')
compare_txt('energy_rte*.txt','RTE STATISTICS')