# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 13:26:30 2018

@author: alrlc
"""
import numpy as np
import matplotlib.pyplot as plt
import os

#Constants
h2 = 0.076199682 #m_e eV nm^2 <-- h-bar squared
# Inputs
V0 = 10 # eV
W = 0.6 # nm
S = 0.2 # nm
M = 1 # m_e
delta = 0.0000000001
"""
Code for a simple bisection
[a,b] signifies the starting interval.
givenFunction is the function of the sepcific problem we want
delta is a threshold value to function(x) for saying approximation is good enough
"""
def bisection(a,b,n, givenFunction, delta = 0.01):
    if a>b:
        # Checking if a <= b
        return bisection(b,a, givenFunction, delta)
    mid = (a+b)/2
    while abs(givenFunction(mid,n)) >= delta: # Test delta against value of root
        mid = (a+b)/2
        if givenFunction(a,n)*givenFunction(mid,n) < 0:
            b = mid
        else:
            a = mid
    return mid # Return midpoint of interval if bisection is  satisfied

def alpha(E):
    return np.sqrt(2*M*E/h2)

def beta(E):
    return np.sqrt(2*M*(V0 - E)/h2)

def P_allowed(E):
    a = alpha(E)
    dx = a*W
    return np.matrix([[ np.cos(dx), np.sin(dx)/a],
                      [ -a*np.sin(dx), np.cos(dx)]])
    
def P_forbidden(E):
    b = beta(E)
    dx = b*S
    return np.matrix([[ np.cosh(dx), np.sinh(dx)/b],
                      [ b*np.sinh(dx), np.cosh(dx)]])

def starting_value(E):
    return np.matrix([[1],[beta(E)]])
    
def zero_function(E,N):
    X = (P_allowed(E)*((P_forbidden(E)*P_allowed(E))**(N-1))*starting_value(E))
    return X[1,0] + beta(E)*X[0,0]

x = np.arange(0.001,10,0.001) # Doing a grid search

N = [1,2,15,30,35]#input for number of wells

for n in N:
    plt.figure()
    zeroes = []
    print("Here")
    y = [zero_function(q,n) for q in x]
    print("There")
    for i in range(0,len(x)-1):
        a = x[i]
        b = x[i+1]
        
        if y[i]*y[i+1] <=0:
            zeroes.append(bisection(a,b,n,zero_function,delta))
            print('Found zero %d'% len(zeroes))
    
    print("N = ",n," Well:")
    print("Number of Zeroes", len(zeroes))
    plt.plot(x, y)
    plt.scatter(zeroes, [zero_function(x,n) for x in zeroes], c='r')
    plt.xlabel("Energy (eV)")
    plt.ylabel("f(E)")
    plt.title('f(E) for %d Well System'% (n))
    plt.show()