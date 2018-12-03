# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 22:13:37 2018

@author: alrlc
"""

import numpy as np
import matplotlib.pyplot as plt
from ising2D import ising2D
import os
from time import time


def q2():
    print("Plotting Total Energy for lattice N=20")
    E20, M20 = ising2D(T=1, N=20, J = 1, c=1000)
    """ Cleaning up to only take spin flips"""
    plt.figure()
    plt.plot(E20)
    plt.title("Total Energy v Spin Flips for N=20")
    plt.xlabel("Spin Iterations")
    plt.ylabel("Total Energy")
    fname = os.path.join("..", "figs", "q2_N20.pdf")
    plt.savefig(fname)
    
    print("Plotting Total Energy for lattice N=40")
    E40, _ = ising2D(T=3, N=40, J = 1, c=1000)
    """ Cleaning up to only take spin flips"""
    plt.figure()
    plt.plot(E40)
    plt.title("Total Energy v Spin Flips for N=40")
    plt.xlabel("Spin Iterations")
    plt.ylabel("Total Energy")
    fname = os.path.join("..", "figs", "q2_N40.pdf")
    plt.savefig(fname)
    
    print("Plotting Total Energy for lattice N=80")
    E80, _ = ising2D(T=3, N=80, J = 1, c=1000)
    """ Cleaning up to only take spin flips"""
    plt.figure()
    plt.plot(E80)
    plt.title("Total Energy v Spin Flips for N=80")
    plt.xlabel("Spin Iterations")
    plt.ylabel("Total Energy")
    fname = os.path.join("..", "figs", "q2_N80.pdf")
    plt.savefig(fname)
    
    print("Finished Q2")
    return

def q3():
    T = np.arange(2,2.6,0.02)
    for N in [10,20,50]:
        whole = time()
        print("Plotting for N=",N)
        resultE = []    # Total Energy
        resultM = []    # Total Magnetization
        resultX = []    # Magnetic Susceptibility
        resultC = []    # Heat Capacity
        for t in T:
            start = time()
            EOL, MOL = ising2D(T=t, N=N, J=1, c=10000)
            EL = EOL[-5000*N**2:]
            ML = MOL[-5000*N**2:]
            
            E = np.mean(EL)
            M = np.abs(np.mean(ML))
            X = (np.mean((ML)**2) - M**2)/(t)
            C = (np.mean((EL)**2) - E**2)/(t)
            resultE.append(E)
            resultM.append(M)
            resultX.append(X)
            resultC.append(C)
            print(t, ": Took ", time() - start, "seconds")
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
    print("Finished Q3")
    return

def q4():
    T = np.arange(1.6,3,0.01)
    for N in [50]:
        whole = time()
        print("Plotting for N=",N)
        resultE = []    # Total Energy
        resultM = []    # Total Magnetization
        resultX = []    # Magnetic Susceptibility
        resultC = []    # Heat Capacity
        for t in T:
            start = time()
            EOL, MOL = ising2D(T=t, N=N, J=1, c=15000)
            EL = EOL[-10000*N**2:]
            ML = MOL[-10000*N**2:]
            
            E = np.mean(EL)
            M = np.abs(np.mean(ML))
            X = (np.mean((ML)**2) - M**2)/(t)
            C = (np.mean((EL)**2) - E**2)/(t)
            resultE.append(E)
            resultM.append(M)
            resultX.append(X)
            resultC.append(C)
            print(t, ": Took ", time() - start, "seconds")
        plt.figure()
        plt.plot(T, resultC)
        plt.title("Heat Capacity for N="+str(N))
        plt.xlabel("Temperature")
        plt.ylabel("Heat Capacity")
        fname = os.path.join("..", "figs", "q4_N"+str(N)+"_C.pdf")
        plt.savefig(fname)
        
        plt.figure()
        plt.plot(T, resultE)
        plt.title("Energy for N="+str(N))
        plt.xlabel("Temperature")
        plt.ylabel("Energy")
        fname = os.path.join("..", "figs", "q4_N"+str(N)+"_EL.pdf")
        plt.savefig(fname)
        
        plt.figure()
        plt.plot(T, np.abs(resultM))
        plt.title("Magnetization for N="+str(N))
        plt.xlabel("Temperature")
        plt.ylabel("Magnetization")
        fname = os.path.join("..", "figs", "q4_N"+str(N)+"_ML.pdf")
        plt.savefig(fname)
        
        plt.figure()
        plt.plot(T, resultX)
        plt.title("Magnetic Susceptibility for N="+str(N))
        plt.xlabel("Temperature")
        plt.ylabel("Magnetic Susceptibility")
        fname = os.path.join("..", "figs", "q4_N"+str(N)+"_X.pdf")
        plt.savefig(fname)
        
        print("N=",N," took ",time()-whole," seconds")
    print("Finished Q4")
    return

plt.close("all")
q2()
q3()
q4()