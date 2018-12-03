# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 18:43:04 2018

@author: alrlc
"""
import numpy as np

def ising2D(T,N,J,c=10000):
    """
    Function to calculate average energy and magnetization of 1D Ising model
    T: Temperature
    N: dimension of one side of lattice. We are making an NxN lattice
    J: Ising couplling
    c: constant factor to scale number of steps
    """
    """ Initialize configuration """
    grid = (np.random.randint(low=0, high=2, size=(N,N))*2) -1 #Takes -1 or 1 with equal probability; 
    t = c*N**2 #Number of steps
    Elist = np.zeros(t)
    Mlist = np.zeros(t)
    Energy = -J*np.sum(grid*np.roll(grid,(0,1))) - J* np.sum(grid*np.roll(grid,(1,0))) #np.roll is the Python equivalent for circshift
    Magnet = np.sum(grid)
    """ Generate horizontal and vertical positions separately. Still uniform."""
    trialsh = np.random.randint(low=0, high = N, size=t)
    trialsv = np.random.randint(low=0, high = N, size=t)
    
    """ Metropolis algorithm """
    for i in range(t):
        h = trialsh[i]
        v = trialsv[i]
        """ Compensate for periodic boundaries """
        left = grid[v][(h-1)%N]
        right = grid[v][(h+1)%N]
        up = grid[(v-1)%N][h]
        down = grid[(v+1)%N][h]
        """ Change in Energy and Declaration of relevant probability """
        dE = 2*J*grid[v][h]*(left + right) +  2*J*grid[v][h]*(up + down)
        p = np.exp(-dE/T) # If dE<=0, p >=1 and is automatically accepted
        """ Acceptance test (automatically passes for dE<=0) """
        if np.random.uniform() <= p:
            grid[v][h] *= -1   #Flip the sign
            Energy += dE
            Magnet += 2*grid[v][h]
        """ Update energy and magnetization lists """
        Mlist[i] = Magnet
        Elist[i] = Energy
    return Elist, Mlist #Stopped here to make other parts easier to factorize