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
    if A==0: ypp = (-v*(y_1) - np.sin(y_0))
    else: ypp = (A*np.sin(w*t) - v*(y_1) - np.sin(y_0))
    return np.array([yp,ypp])

def helper(v):
    start = time()
    y0 = 0.2
    y0p = 0
    ti = 0
    tf = 100
    A=0
    w = 0
    
    f = partial(test_function, A,w,v)
    bounce = solve_ivp(f, (ti,tf), y0=[y0,y0p],
                       dense_output=True, vectorized=True)
    
    t = bounce.t
    y = bounce.y
    
    plt.figure()
    plt.plot(t,y[0])
    plt.title(r'$\nu = $'+str(v))
    plt.xlabel("t (sec)")
    plt.ylabel(r'$\theta$ (rad)')
    fname = fname = os.path.join('..','figs','q1_v_'+str(v)+'_no_wrap.pdf')
    plt.savefig(fname)
    
    #Case-specific: Just to keep θ between -pi and pi
    y[:,0] = ( y[:,0] + np.pi) % (2 * np.pi ) - np.pi
    
    plt.figure()
    plt.plot(t,y[0])
    plt.title(r'$\nu = $'+str(v))
    plt.xlabel("t (sec)")
    plt.ylabel(r'$\theta$ (rad)')
    fname = fname = os.path.join('..','figs','q1_v_'+str(v)+'.pdf')
    plt.savefig(fname)
    
    plt.figure()
    plt.plot(y[1],y[0])
    plt.title(r'$\nu = $'+str(v)+" Phase portrait")
    plt.ylabel(r'$\theta$ (rad)')
    plt.xlabel(r'$v$ (rad/sec)')
    fname = fname = os.path.join('..','figs','q1_v_'+str(v)+'_phase.pdf')
    plt.savefig(fname)
    print(v,": took ",time()-start," seconds to finish")
    
    return t,y

def start_p():
    print('Starting', multiprocessing.current_process().name)
    
if __name__ == '__main__':
    
    # 0.5 <= A <= 1.2
    V = [1,5,10]
    pool = multiprocessing.Pool(processes=3,initializer=start_p)
    pool_output = pool.map(helper, V)
    pool.close()
    pool.join()