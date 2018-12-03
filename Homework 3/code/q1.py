# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 19:50:53 2018

@author: alrlc
"""

import numpy as np
import matplotlib.pyplot as plt

def test_function(x):
    return(np.sin(x**2))

def actual_derivative(x):
    return(2*x*np.cos(x**2))

def estimator_function(x,h):
    n3 = (test_function(x + 3*h) - test_function(x - 3*h))/60
    n2 = 3*(test_function(x + 2*h) - test_function(x - 2*h))/20
    n1 = 3*(test_function(x + h) - test_function(x - h))/4
    return (n3 - n2 + n1)/h

h = 3 * 10**-3 #Optimal h value found by optimization
x = np.linspace(0,1,10000)
y_actual = np.apply_along_axis(actual_derivative,0,x)
y_estimator = np.apply_along_axis(estimator_function,0,x,h)

#Plotting Estimated and Actual Derivative
plt.figure()
plt.plot(x, y_actual, 'r-', label='Actual Derivative')
plt.plot(x, y_estimator, 'g--', label='Estimated Derivative')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Comparison of Estimator to Actual Derivate')
plt.legend()
plt.savefig('..\\figs\\q1_comparison.pdf')

Y = []
c = np.arange(10**-4,10**-2, 10**-3)
for h in c:
    Y.append(np.apply_along_axis(estimator_function,0,x,h))
#Plotting Error of Estimated and Actual Derivative
plt.figure()
for i in range(0,len(Y)):
    abs_error = np.abs(y_actual - Y[i])
    
    plt.plot(x, np.cumsum(abs_error), label='h = '+str(c[i]))
plt.xlabel('X')
plt.ylabel('Cumulative Error from x=0 (log scale)')
plt.yscale('log')
plt.title('Cumulative Error of Estimator to Actual Derivate')
plt.legend()
plt.savefig('..\\figs\\q1_error.pdf')