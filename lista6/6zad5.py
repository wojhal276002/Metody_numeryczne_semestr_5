import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve
from numpy.linalg import solve
from functools import reduce
from operator import mul


def wielomian(n):
    if n == 0:
        return 1
    if n == 1:
        return np.poly1d([1,0])
    wiel = (((2*n-1)*np.poly1d([1,0])*(wielomian(n-1))-(n-1)*wielomian(n-2))/n)
    return wiel
def gauss_legendarny(n,start,stop,funkcja):
    wiel = wielomian(n)
    roots = np.roots(list(wiel.coefficients))
    new_roots = []
    for i in roots:
        new_roots.append((stop-start)/2*i+(stop+start)/2)
    wagi = []
    for j in roots:
        wagi.append(2/((1-j**2)*(np.poly1d.deriv(wiel)(j))**2))
    sumka = 0
    for k in range(len(new_roots)):
        sumka += wagi[k]*funkcja(new_roots[k])
    return (stop-start)/2*sumka

print(f"2 wezly:{gauss_legendarny(2,1,np.pi,lambda x: np.log(x)/(x**2 - 2*x + 2))}\n4 wezly:{gauss_legendarny(4,1,np.pi,lambda x: np.log(x)/(x**2 - 2*x + 2))}")

