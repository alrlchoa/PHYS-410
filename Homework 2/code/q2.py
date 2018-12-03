# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 10:41:48 2018

@author: alrlc
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from numpy.linalg import cond # Import condition number function

A = np.matrix([[1/2, 1/3],[1/3, 1/4]]) # Matrix for First System
B = np.matrix([[1/2, 1/3],[1/3, -1/2]]) # Matrix for Second System

print("Condition number for first system: ", cond(A,p=2))
print("Condition number for second system: ", cond(B,p=2))

#Geometrical representation for original and perturbed systems
x = np.linspace(-10,300)

#First System
y11 = [3 - 3*i/2 for i in x]
y1per = [100/33 - 50*i/33 for i in x]
y12 = [-4*i/3 - 32 for i in x]

#Second System
y21 = y11       #By construction of question, they are the same
y2per = y1per   #By construction of question, they are the same
y22 = [2*i/3 + 16 for i in x]

#Plotting first system
plt.figure()
plt.plot(x,y11, c='b', label="Original First Equation")
plt.plot(x,y1per, c='r', label="Perturbed First Equation")
plt.plot(x,y12, c='g', label="Second Equation")
plt.scatter([210],[-312], c='c', label="Original Intersection")
plt.scatter([578/3],[-2600/9], c='y', label="Perturbed Intersection")
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(180,220)
plt.ylim(-320, -280)
plt.title('First System of Equations')
plt.legend()
plt.savefig('..\\figs\\q2_first.png')

#Plotting second system
plt.figure()
plt.plot(x,y21, c='b', label="Original First Equation")
plt.plot(x,y2per, c='r', label="Perturbed First Equation")
plt.plot(x,y22, c='g', label="Second Equation")
plt.scatter([-6],[12], c='c', label="Original Intersection")
plt.scatter([-107/18],[325/27], c='y', label="Perturbed Intersection")
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-7,-5)
plt.ylim(11, 13)
plt.title('Second System of Equations')
plt.legend()
plt.savefig('..\\figs\\q2_second.png')