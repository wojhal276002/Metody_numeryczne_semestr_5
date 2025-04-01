import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.optimize import fsolve
from numpy.linalg import solve
from functools import reduce
from operator import mul

def lagrange(x,xData,yData):
    n = len(xData)
    y = 0
    for i in range(n):
        w = 1.0
        for j in range(n):
            if i != j:
                w = w*(x-xData[j])/(xData[i]-xData[j])
        y = y + w*yData[i]
    return y

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

c = lagrange(np.poly1d([1,0]),[0.2, 2, 20, 200, 2000, 20000], [103, 13.9, 2.72, 0.8, 0.401, 0.433])

for r in [5,50,5000]:
    print(r,c(r))

d = zad1([0.2, 2, 20, 200, 2000, 20000], [103, 13.9, 2.72, 0.8, 0.401, 0.433])
for r in [5,50,5000]:
    print(r,d(r))