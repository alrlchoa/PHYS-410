# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 12:52:00 2018

@author: alrlc
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def test_function(t,y,E=1):
    y_0 = y[0]  #Value of y
    y_1 = y[1]  #Valueof y'
    
    #Parameters
    h2 = 0.076199682
    a  = 500
    b  = 3500
    #me is cancelled by h2 units
    
    yp  = y_1
    ypp = (-2/h2)*(E + a*t**2 - b*t**4 - ((a**2)/(4*b)))*y_0
    return np.array([yp,ypp])

def RK4(f, ti, tf, y0,E=1):
    """
    f: function evaluating the time derivatives; y'(t) = f(t,y)
    Example f=@(t,y) y, uses function handle to solve y'=y;
    Note that f must accept TWO arguments: t and y(t). For systems of N first
    order ODEs, f and y should both output row vectors with N components.
    y0: row vector of initial conditions
    [ti tf] is the time interval over which we solve the ODE
    """
    
    #Step size
    h = 1e-2
    numSteps = int((tf-ti)/h) # Rounded to smaller division to not overshoot
    #Initialization
    y = np.zeros((numSteps+1,len(y0))) #Pre-allocating for y
    y[0] = y0
    
    for i in range(numSteps):
        k1 = f(ti + i*h, y[i],E)
        k2 = f(ti + i*h + h/2, y[i] + (h/2)*k1,E)
        k3 = f(ti + i*h + h/2, y[i] + (h/2)*k2,E)
        k4 = f(ti + (i+1)*h,   y[i] + h*k3,E    )
        y[i+1] = y[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        
    t = ti + h*np.array(range(numSteps+1))
    
    return t, y

"""
Question 1:
"""
epsilon = 1e-5
t,y = RK4(test_function, -0.6,0.6, [0,epsilon],E=1)

plt.figure()
plt.plot(t,y[:,0])
plt.title("E = 1 eV")
plt.xlabel("x (nm)")
plt.ylabel(r'$\Psi(x)$')
fname = fname = os.path.join('..','figs','q1_linscale.pdf')
plt.savefig(fname)

plt.figure()
plt.plot(t,y[:,0])
plt.title("E = 1 eV")
plt.xlabel("x (nm)")
plt.ylabel(r'$\log \Psi(x)$')
plt.yscale("log")
fname = fname = os.path.join('..','figs','q1_logscale.pdf')
plt.savefig(fname)

"""
Question 3:
"""
epsilon = 1e-5

#Different step sizes to speed up search
delta0 = 1e-2
delta1 = 1e-4
delta2 = 1e-6

crit = 1e-6
eigens = []
ts = []
ys = []
E0 = 0
count = 0
while len(eigens) <6:
    t,y = RK4(test_function, -0.6,0, [0,epsilon], E=E0)
    ty = np.abs(y[-1])
    if count%1000 == 0: print(E0, ty)
    if ty[0]<crit or ty[1]<crit:
        print("Energy Eigenvalue:",E0, ty)
        eigens.append(E0)
        ts.append(np.append(t,-t[::-1]))
        if ty[1]<crit:
            ys.append(np.append(y[:,0],(y[:,0])[::-1])) #Even Case
        else:
            ys.append(np.append(y[:,0],(-y[:,0])[::-1])) #Odd Case
        E0 += 0.003 #Just to nudge the eigenstate further to get non-degenerate
    if ty[0]<0.0001 or ty[1]<0.0001:
        E0 += delta2
    elif ty[0]<0.005 or ty[1]<0.005:
        E0 += delta1
    else:
        E0 += delta0
    count +=1

print("The 6 lowest eigenvalues are: ", eigens)

"""
Question 4
"""
magnitudes = np.power(ys,2)
A = np.array([np.trapz(ys[i]**2,ts[i]) for i in range(len(ys))])
A = np.sqrt(A)
normalized = np.array([eigens[i] + (ys[i]/A[i]) for i in range(len(ys))])

plt.figure()
for i in range(len(ys)):
    plt.plot(ts[i], normalized[i], label="Eigenstate "+str(i+1))
x = ts[0]
a  = 500
b  = 3500
plt.plot(x,-a*x**2 + b*x**4 + ((a**2)/(4*b)), label="Potential")
plt.ylim(np.min(normalized)-5,np.max(normalized)+5)
plt.ylabel("Energy (eV)")
plt.xlabel("x (nm)")
plt.title("Eigenstates on Energy")
plt.legend()
fname = fname = os.path.join('..','figs','q4_eigenstates.pdf')
plt.savefig(fname)