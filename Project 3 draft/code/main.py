# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 12:52:00 2018

@author: alrlc
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.integrate import solve_ivp

def unlink_wrap(dat, lims=[-np.pi, np.pi], thresh = 0.95):
    """
    Iterate over contiguous regions of `dat` (i.e. where it does not
    jump from near one limit to the other).

    This function returns an iterator object that yields slice
    objects, which index the contiguous portions of `dat`.

    This function implicitly assumes that all points in `dat` fall
    within `lims`.

    """
    jump = np.nonzero(np.abs(np.diff(dat)) > ((lims[1] - lims[0]) * thresh))[0]
    lasti = 0
    for ind in jump:
        yield slice(lasti, ind + 1)
        lasti = ind + 1
    yield slice(lasti, len(dat))


def test_function(t,y,A=0,w=1,v=1):
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

def RK4(f, ti, tf, y0,A=1,w=1,v=1, h=1e-2):
    """
    f: function evaluating the time derivatives; y'(t) = f(t,y)
    Example f=@(t,y) y, uses function handle to solve y'=y;
    Note that f must accept TWO arguments: t and y(t). For systems of N first
    order ODEs, f and y should both output row vectors with N components.
    y0: row vector of initial conditions
    [ti tf] is the time interval over which we solve the ODE
    
    Case-specific:
    A:   Amplitude of driving force
    w:   Angular Frequency of friving force
    v:   Damping Force
    h:   Step Size
    """
    
    numSteps = int((tf-ti)/h) # Rounded to smaller division to not overshoot
    #Initialization
    y = np.zeros((numSteps+1,len(y0))) #Pre-allocating for y
    y[0] = y0
    for i in range(numSteps):
        k1 = f(ti + i*h, y[i],A,w,v)
        k2 = f(ti + i*h + h/2, y[i] + (h/2)*k1,A,w,v)
        k3 = f(ti + i*h + h/2, y[i] + (h/2)*k2,A,w,v)
        k4 = f(ti + (i+1)*h,   y[i] + h*k3,A,w,v    )
        y[i+1] = y[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        
    t = ti + h*np.array(range(numSteps+1))
    
    return t, y        
        
        
"""
Question 5:
"""
def q5():
    """
    Initial conditions:
        θ  = 0.2 rad
        θ' = 0 since v=0
        A  = 0
    """
    y0 = 0.2
    y0p = 0
    ti = 0
    tf = 3000    # More than 300 periods for 2/3 rad/sec 
    v = 0.5
    w = 2/3
    for A in [0.5,1.2,1.35,1.44,1.465]:
        t,y = RK4(test_function, ti,tf, [y0,y0p],A=A,w=w,v=v,h=1e-3)
        t = np.array(t)
        y = np.array(y)
        crit = np.array([math.isclose(x%(3*np.pi),0,abs_tol=1e-3) or math.isclose(x%(3*np.pi),3*np.pi,abs_tol=1e-3) for x in t])
        y = y[crit]
        t = t[crit]
        n = [round(x/(3*np.pi)) for x in t]
        n,y = n[-150:], y[-150:]
        plt.figure()
        for slc in unlink_wrap(y[:,0], [-np.pi, np.pi]):
            plt.plot(n[slc], y[slc,0], 'b-', linewidth=1)
        plt.title(r'$A = $'+str(A))
        plt.xlabel("n")
        plt.ylabel(r'$\theta$ (rad)')
        fname = fname = os.path.join('..','figs','q5_A_'+str(A)+'_Poincare.pdf')
        plt.savefig(fname)
        
#q1()
#q2()
q3()
#q4()
#q5()