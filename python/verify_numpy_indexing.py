#!/usr/bin/python3
import numpy as np

###################################################################################
#                                                                                 #
#  Example form: http://stackoverflow.com/questions/1827489/numpy-meshgrid-in-3d  #
#                                                                                 #
###################################################################################

x_ = np.linspace(0., 1., 10)
y_ = np.linspace(1., 2., 20)
z_ = np.linspace(3., 4., 30)

# first option is 'ij' for array-like indexing
# second option is 'xy' for cartesian coordinates
x, y, z = np.meshgrid(x_, y_, z_, indexing='ij') 

print(x[:,0,0])
print(x_)

assert np.all(x[:,0,0] == x_)
assert np.all(y[0,:,0] == y_)
assert np.all(z[0,0,:] == z_)
