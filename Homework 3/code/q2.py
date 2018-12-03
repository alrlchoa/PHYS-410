# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 10:41:48 2018

@author: alrlc
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os
from numpy.linalg import cond # Import condition number function

def test_function(x):
    return np.sin(x)

def actual_integral(a,b):
    return -(np.cos(b) - np.cos(a))
T = dict() #Storage for solved Rombergs

def trapezoid(f,a,b,m):
    """
    Composite Tradezoid Integration on function f
    f: function to integrate
    [a,b]: interval to integrate on
    m: 2**m = number of sub-intervals
    """
    if b < a:
        return -trapezoid( f, b, a, m )
    
    w = (b-a)/(2**m)    #width of sub-interval
    x = np.append(np.arange(a,b,w),b)
    y = np.apply_along_axis(f,0,x)
    y[0] /= 2
    y[-1] /= 2
    
    return w*np.sum(y)
    
def Romberg(f, a, b, m, k):
    if m > 27:
        raise MemoryError
    """
    Romberg Integration on function f
    f: function to integrate
    [a,b]: interval to integrate on
    m: 2**m = number of sub-intervals
    k: order of Romberg
    """
    if (m,k) in T.keys():
        return T[(m,k)]
    else:
        if k==0:
            T[(m,k)] = trapezoid(f,a,b,m)
        else:
            T[(m,k)] = ((4**k)*Romberg(f,a,b,m,k-1) - Romberg(f,a,b,m-1,k-1))/(4**k - 1)
        return T[(m,k)]
    


M = np.arange(1,25,1)  #2**m = number of sub-intervals for the original trapezoid integration
k = 2   #2*(k+1) = order of error we want to get rid of
a = 0
b = 1

for m in M:
    print(m,": ",Romberg(test_function,a,b,m+k,k), abs(actual_integral(a,b)-Romberg(test_function,a,b,m+k,k)))