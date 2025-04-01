import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.optimize import fsolve
from numpy.linalg import solve
from functools import reduce
from operator import mul
import sympy

def Df(potega,h,wartosc1,wartosc2,wartosc3=None):
    if potega == 1:
        return (wartosc1-wartosc2)/h
    if potega == 2:
        return ((2**(potega-1))*Df(potega-1,h,wartosc1,wartosc2)-Df(potega-1,2*h,wartosc3,wartosc2))/(2**(potega-1)-1)
    else:
        raise ValueError("Za mało punktów ciole")

def Db(potega,h,wartosc1,wartosc2,wartosc3=None):
    if potega == 1:
        return (wartosc2-wartosc1)/h
    if potega == 2:
        return ((2**(potega-1))*Db(potega-1,h,wartosc1,wartosc2)-Db(potega-1,2*h,wartosc3,wartosc2))/(2**(potega-1)-1)
    else:
        raise ValueError("Za mało punktów ciole")

def Dc2(wartosc1,wartosc2,h):
    return (wartosc1-wartosc2)/(2*h)

def Dc4(wartosc1,wartosc2,wartosc3,wartosc4,h):
    return (4*Dc2(wartosc1,wartosc2,h)-Dc2(wartosc3,wartosc4,2*h))/3


print("Df1",Df(1,0.1,0.192916,0.138910,0.244981),"\n","Df2",Df(2,0.1,0.192916,0.138910,0.244981),"\n","Dc2",Dc2(0.192916 ,0.078348 ,0.1),"\n",
      "Dc22",Dc2(0.244981,0.000000 ,0.2),"\n","Db1",Db(1,0.1,0.078348,0.138910),"\n","Db2",Db(2,0.1,0.078348,0.138910,0.000000),"\n","Dc4",Dc4(0.192916 ,0.078348,0.244981,0.000000,0.1))

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

f = zad1([0,0.1,0.2,0.3,0.4], [0.000000,0.078348,0.138910,0.192916,0.244981])
f1 = np.poly1d.deriv(f)
print(f"f_prim:{f1(0.2)}")

