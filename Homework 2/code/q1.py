# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 19:50:53 2018

@author: alrlc
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from prettytable import PrettyTable
from decimal import Decimal, getcontext # A more precise way on modelling floats

getcontext().prec = 15 #Shows 15 significant digits after first non-zero one

exact = np.array([Decimal(0.5)])                # Exact sequence
r_per = np.array([Decimal(0.994)])              # Pertubation on r sequence
p_per = np.array([Decimal(1), Decimal(0.497)])  # Pertubation on p sequence
q_per = np.array([Decimal(1), Decimal(0.497)])  # Pertubation on q sequence

def r_gen(x):
    # Recursive generator for r sequence
    return x/2

def p_gen(x1, x2):
    # Recursive generator for p sequence
    return 3*x1/2 - x2/2

def q_gen(x1, x2):
    # Recursive generator for q sequence
    return 5*x1/2 - x2

while len(exact) < 20:
    exact = np.append(exact, r_gen(exact[-1]))
    r_per = np.append(r_per, r_gen(r_per[-1]))
    

while len(p_per) < 20:
    p_per = np.append(p_per, p_gen(p_per[-1], p_per[-2]))
    q_per = np.append(q_per, q_gen(q_per[-1], q_per[-2]))

#Printing in tabular form
t = PrettyTable(['Exact', 'R', 'P', 'Q'])

for i in range(0,len(exact)):
    t.add_row([ exact[i] , r_per[i] , p_per[i] , q_per[i] ])
print(t)

#Plotting errors
plt.figure()
plt.plot(np.arange(0,20), exact - exact, c='k', label='Exact')
plt.plot(np.arange(0,20), exact - r_per, c='r', label='R')
plt.plot(np.arange(0,20), exact - p_per, c='b', label='P')
plt.plot(np.arange(0,20), exact - q_per, c='g', label='Q')
plt.xlabel('Iteration (n)')
plt.ylabel('Error against Exact')
plt.xticks(np.arange(0,20))
plt.title('Comparison of Different Recursive methods')
plt.legend()
plt.savefig('..\\figs\\q1_error.png')