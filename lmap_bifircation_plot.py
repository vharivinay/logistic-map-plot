'''
This program computes the logistic map equation.
The logistic map equation is a second degree polynomial equation often used as an example in the discussions of chaos

More information:
wiki: https://en.wikipedia.org/wiki/Logistic_map#Finding_cycles_of_any_length_when_r_=_4

Author: Harivinay V
github: https://github.com/M87K452b
'''

# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

@jit(nopython=True) # Compute this line out in the absence of a supported GPU
def lmap_compute(xn=4,r=0.0015):
    '''
    This functions computes the Logistic Map equationexit
    '''
    rvals = []
    xvals = []

    for r in np.arange(0,xn,r):
        #print('r = {}\r'.format(r), end="") # Disabled because jit doesnt like it! 
        xold = 0.5
        # To get equlibrium value
        for i in range(2000):
            xnew = (xold-(xold**2))*r
            xold = xnew
            
        # Save equilibrium values
        xsteady = xnew
        for i in range(1001):
            xnew = (xold-(xold**2))*r
            xold = xnew
            rvals.append(r)
            xvals.append(xnew)
            if abs(xnew - xsteady) < 0.001:
                break
    return rvals,xvals

# Run the main function

## Define Inputs
xn = 4
r = 0.0025

tic = time.perf_counter()
rvals,xvals = lmap_compute(xn,r)
toc = time.perf_counter()

print('computation time: ',abs(toc - tic))

# Visualization

f = plt.figure(figsize = (16,12))
plt.subplot(111)
ax1 = plt.scatter(rvals,xvals, s = 0.05)
plt.xlim(3.447,4.0)
plt.ylim(0,1)
plt.axis('off')
plt.show()

f.savefig("bifircation-plot_r{}.png".format(r), bbox_inches='tight', dpi=400)