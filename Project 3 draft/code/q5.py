# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:11:48 2018

@author: alrlc
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.integrate import solve_ivp
from functools import partial
import multiprocessing
from time import time

def unlink_wrap(dat, lims=[-np.pi, np.pi], thresh = 1.5):
    """
    Iterate over contiguous regions of `dat` (i.e. where it does not
    jump from near one limit to the other).

    This function returns an iterator object that yields slice
    objects, which index the contiguous portions of `dat`.

    This function implicitly assumes that all points in `dat` fall
    within `lims`.

    """
    jump = np.nonzero(np.abs(np.diff(dat)) > thresh)[0]
    lasti = 0
    for ind in jump:
        yield slice(lasti, ind + 1)
        lasti = ind + 1
    yield slice(lasti, len(dat))

def test_function(A,w,v,t,y):
    """
    Test function for RK4
    t:   time we are checking right now
    y:   input vector
    Case-specific:
    A:   Amplitude of driving force
    w:   Angular Frequency of friving force
    v:   Damping Force
    """
    y_0 = y[0]  #Value of y
    y_1 = y[1]  #Valueof y'
    
    yp  = y_1
    ypp = (A*np.sin(w*t) - v*(y_1) - np.sin(y_0))
    return np.array([yp,ypp])

def helper(A):
    start = time()
    v = 0.5
    w = 2/3
    f = partial(test_function, A,w,v)
    
    x = [0.2,0]
    nrange = np.arange(0,3000,5)
    newt = (nrange)*2*np.pi/w
    ntmax = max(newt) + 2*np.pi/w
    
    xvals = solve_ivp(f,[0,ntmax], x,rtol=1e-13, atol =1e-14,t_eval=newt)    
    t = xvals.t
    y = xvals.y
    
    #Case-specific: Just to keep θ between -pi and pi
    y[0] = ( y[0] + np.pi) % (2 * np.pi ) - np.pi
    
    plt.figure()
    plt.scatter(nrange, y[0], c='b', s=5)
    plt.title(r'$A = $'+str(A))
    plt.xlabel("n")
    plt.ylabel(r'$\theta$ (rad)')
    fname = fname = os.path.join('..','figs','q5_A_'+str(A)+'_Poincare.pdf')
    plt.savefig(fname)
    
    print(A,": took ",time()-start," seconds to finish")
    
    return t,y

def start_p():
    print('Starting', multiprocessing.current_process().name)
    
if __name__ == '__main__':

    A = [0.5,1.2,1.35,1.44,1.465]
    pool = multiprocessing.Pool(processes=3,initializer=start_p)
    pool_output = pool.map(helper, A)
    pool.close()
    pool.join()