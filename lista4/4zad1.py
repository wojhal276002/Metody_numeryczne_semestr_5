import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats
import cmath
import sys
import math
import scipy

def f(x): 
    return np.tan(np.pi-x) - x 

def f_prime(x):
    return -1/np.cos(np.pi - x)**2 - 1

def bisekcja(f,x1,x2,epsilon):
    liczba = 0
    fa,fb = f(x1),f(x2)
    liczba += 4
    if fa == 0.0: 
        return x1
    if fb == 0.0: 
        return x2
    iter = 0
    while abs(x2-x1) > epsilon:
        mk = 1/2*(x1 + x2)
        liczba +=3
        f3 = f(mk)
        liczba += 2
        if f3 == 0.0: 
            return mk
        else: 
            if fa*f3 < 0:
                x2 = mk
                fb = f3
            elif f3*fb < 0:
                x1 = mk 
                fa = f3
            liczba += 1
        iter += 1
    liczba += 2
    return (x1 + x2)/2.0, iter, liczba

x_biseks,iter_biseks, liczba_biseks = bisekcja(f, -1.6, -2.2, 1.0e-8)
print(x_biseks,f(x_biseks),iter_biseks,liczba_biseks)

def brent(f, x1, x2, epsilon):
    liczba = 0
    fa,fb = f(x1),f(x2)
    liczba += 4
    if abs(fa) < abs(fb):
        x1, x2 = x2, x1
        fa, fb = fb, fa
    iter = 0
    c = (x1+x2)/2
    fc = f(c)
    liczba += 4
    while abs(x2 - x1) > epsilon:
        if fc*fa < 0:
            przedzial = (x1,c)
            dol = True
        else:
            przedzial = (c,x2)
            dol = False
        liczba += 1
        x = -(x1 * fb * fc / ((fa - fb) * (fa - fc)) + x2 * fa * fc / ((fb - fa) * (fb - fc)) + c * fa * fb / ((fc - fa) * (fc - fb)))
        liczba += 21
        if not dol:
            if przedzial[0] < x < przedzial[1]:
                x1,c,x2 = c,x,x2
                fa,fc,fb = fc,f(x),fb
            else:
                x = (przedzial[0] + przedzial[1]) / 2
                x1,c,x2 = c,x,x2
                fa,fc,fb = fc,f(x),fb
                liczba += 2
        else:
            if przedzial[0] < x < przedzial[1]:
                x1,c,x2 = x1,x,c
                fa,fc,fb = fa,f(x),fc
            else:
                x = (przedzial[0] + przedzial[1]) / 2
                x1,c,x2 = x1,x,c
                fa,fc,fb = fa,f(x),fc
                liczba += 2
        iter += 1
    return x,iter,liczba

x_brent,iter_brent,liczba_brent = brent(f,-1.6,-2.2,1.0e-8)
print(x_brent,f(x_brent),iter_brent,liczba_brent)

def sieczne(f,x1,x2,epsilon):
    liczba = 0
    fa,fb = f(x1),f(x2)
    liczba+= 4
    iter = 0
    while abs(x1-x2) > epsilon:
        x = x1 - fa*(x1 - x2)/(fa - fb)
        fx = f(x)
        liczba += 6
        x1,x2 = x,x1
        fa,fb = fx,fa
        iter+= 1
    return x,iter,liczba

x_sieczne,iter_sieczne,liczba_sieczne = sieczne(f,-1.6,-2.2,1.0e-8)
print(x_sieczne,f(x_sieczne),iter_sieczne,liczba_sieczne)

def newton(f,fpier,x1,epsilon):
    liczba = 0
    x0 = x1
    fa = f(x0)
    liczba += 2
    iter = 0
    while abs(fa) > epsilon:
        Dfxn = fpier(x0)
        liczba += 4
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        x0 = x0 - fa/Dfxn
        fa = f(x0)
        liczba += 4
        iter += 1
    return x0,iter,liczba

x_newton,iter_newton,liczba_newton = newton(f,f_prime,-2.2,1.0e-8)
print(x_newton,f(x_newton),iter_newton,liczba_newton)

print(scipy.optimize.fsolve(f,-2))
