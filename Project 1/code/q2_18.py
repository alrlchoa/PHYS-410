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
def bisection(a,b, givenFunction, delta = 0.01):
    if a>b:
        # Checking if a <= b
        return bisection(b,a, givenFunction, delta)
    mid = (a+b)/2
    if abs(givenFunction(mid)) < delta: # Test delta against value of root
        return mid # Return midpoint of interval if bisection is  satisfied
    else:
        if givenFunction(a)*givenFunction(mid) < 0:
            return bisection(a, mid, givenFunction, delta)
        else:
            return bisection(mid, b, givenFunction, delta)

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
    
def zero_function(E):
    X = (P_allowed(E)*((P_forbidden(E)*P_allowed(E))**1)*starting_value(E))
    return X[1,0] + beta(E)*X[0,0]

x = np.arange(0.001,10,0.001) # Doing a grid search

# Finding all zeroes for original specifications
zeroes = []

for i in range(0,len(x)-1):
    a = x[i]
    b = x[i+1]
    
    if zero_function(a)*zero_function(b) <=0:
        zeroes.append(bisection(a,b,zero_function,delta))

print("Zeroes for original specifications")
print("Zeroes", zeroes)
print("Value of Approximate Zeroes", [zero_function(x) for x in zeroes])

# Plotting lowest two energies vs separation distance
s_candidates = np.arange(0.2,3.5, 0.1)
low1 = []
low2 = []


for s in s_candidates:
    S = s
    zeroes = []
    y = [zero_function(a) for a in x]
    for i in range(0,len(x)-1):
        a = x[i]
        b = x[i+1]
        if len(zeroes) == 2:
            break
        if y[i]*y[i+1] <=0:
            zeroes.append(bisection(a,b,zero_function,delta))
    low1.append(zeroes[0])
    low2.append(zeroes[1])

plt.figure()
plt.scatter(s_candidates, low1, label="Lowest Energy", s=30)
plt.scatter(s_candidates, low2, label="2nd Lowest Energy", s=10)
plt.xlabel("Separation distance (nm)")
plt.ylabel("Energy (eV)")
plt.title("Separation distance vs. Energy of Lowest Two States")
plt.legend()
fname = os.path.join('..', 'figs', 'q2_18.png')
plt.savefig(fname)