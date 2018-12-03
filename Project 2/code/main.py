# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:56:28 2018

@author: alrlc
"""

import numpy as np
import matplotlib.pyplot as plt
from ising2D import ising2D
import os
from time import time
import multiprocessing
from functools import partial

def helper(N,t):
    print(t)
    start = time()
    EOL, MOL = ising2D(T=t, N=N, J=1, c=10000)
    EL = EOL[-5000*N**2:]
    ML = MOL[-5000*N**2:]
    
    E = np.mean(EL)
    M = np.abs(np.mean(ML))
    X = (np.mean((ML)**2) - M**2)/(t)
    C = (np.mean((EL)**2) - E**2)/(t)
    
    print(t, ": Took ", time() - start, "seconds")
    
    return [E,M,X,C]

def f(t):
    print(t)
    return [t,t+1]

def start_p():
    print('Starting', multiprocessing.current_process().name)
 
if __name__ == '__main__':
    T = np.arange(2,2.6,0.02)
    for N in [10,20,50]:
        whole = time()
        # start 4 worker processes
        pool = multiprocessing.Pool(processes=4,initializer=start_p)
        func = partial(helper, N)
        print("Hi")
        pool_output = pool.map(func, T)
        pool.close()
        pool.join()
        print(pool_output)
        resultE = np.array([item[0] for item in pool_output])    # Total Energy
        resultM = np.array([item[1] for item in pool_output])    # Total Magnetization
        resultX = np.array([item[2] for item in pool_output])    # Magnetic Susceptibility
        resultC = np.array([item[3] for item in pool_output])    # Heat Capacity
            
        plt.figure()
        plt.plot(T, resultC)
        plt.title("Heat Capacity for N="+str(N))
        plt.xlabel("Temperature")
        plt.ylabel("Heat Capacity")
        fname = os.path.join("..", "figs", "q3_N"+str(N)+"_C.pdf")
        plt.savefig(fname)
        
        plt.figure()
        plt.plot(T, resultE)
        plt.title("Energy for N="+str(N))
        plt.xlabel("Temperature")
        plt.ylabel("Energy")
        fname = os.path.join("..", "figs", "q3_N"+str(N)+"_EL.pdf")
        plt.savefig(fname)
        
        plt.figure()
        plt.plot(T, np.abs(resultM))
        plt.title("Magnetization for N="+str(N))
        plt.xlabel("Temperature")
        plt.ylabel("Magnetization")
        fname = os.path.join("..", "figs", "q3_N"+str(N)+"_ML.pdf")
        plt.savefig(fname)
        
        plt.figure()
        plt.plot(T, resultX)
        plt.title("Magnetic Susceptibility for N="+str(N))
        plt.xlabel("Temperature")
        plt.ylabel("Magnetic Susceptibility")
        fname = os.path.join("..", "figs", "q3_N"+str(N)+"_X.pdf")
        plt.savefig(fname)
        
        print("N=",N," took ",time()-whole," seconds")