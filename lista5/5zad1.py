import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.optimize import fsolve
from numpy.linalg import solve
from functools import reduce
from operator import mul
from scipy.interpolate import interp1d

def mnozonko(pol1,pol2):
    polynomial = np.zeros((max(len(pol1),len(pol2)))+1)
    pol1,pol2= pol1[::-1],pol2[::-1]
    for i in range(len(pol1)):
        for j in range(len(pol2)):
            polynomial[i+j]+=pol1[i]*pol2[j]
    return list(polynomial[::-1])

def mnozonki(lista):
    pol1,pol2 = lista[0],lista[1]
    idx = 2
    c = 0
    for i in range(len(lista)-1):
        c = mnozonko(pol1,pol2)
        if idx<len(lista):
            pol1,pol2 = c,lista[idx]
            idx+=1
    return c

def zad1(x,y):
    pol = []
    for k in range(len(x)):
        pol.append([1,-1*x[k]])
    nowy = np.array(np.zeros(len(pol)))
    for k in range(len(pol)):
        d = pol.copy()
        d.pop(k)
        mianownik = reduce(mul,[sum([x[k],i[1]]) for i in d])
        liczba = (y[k]/mianownik)
        rezult = mnozonki(d)
        nowy_rezult = [i * liczba for i in rezult]
        nowy += nowy_rezult
    return np.poly1d(nowy)

f = zad1([0,3000,6000],[1.225,0.905,0.652])
print(f(0),f(3000),f(6000))
plt.scatter([0,3000,6000],[1.225,0.905,0.652])
xs = np.linspace(0,6000,6000)
wbudowany = interp1d([0,3000,6000],[1.225,0.905,0.652],"quadratic")
plt.plot(xs, f(xs))
plt.plot(xs,wbudowany(xs))
plt.show()
# def lagrange(x,xData,yData):
#     n = len(xData)
#     y = 0
#     for i in range(n):
#         w = 1.0
#         for j in range(n):
#             if i != j:
#                 w = w*(x-xData[j])/(xData[i]-xData[j])
#         y = y + w*yData[i]
#     return y
# print(lagrange(np.poly1d([1,0]),[0,3000,6000], [1.255, 0.905, 0.652]))
