import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.optimize import fsolve
from numpy.linalg import solve
from functools import reduce
from operator import mul

def gauSS(macierz_a,macierz_b):
    pierwszy_a = macierz_a.shape[0]
    if macierz_a.shape[1] != pierwszy_a:
        print('niespójny kształt macierzy kwadratowej')
        return
    if macierz_b.shape[1] > 1 or macierz_b.shape[0] != macierz_a.shape[0]:
            print('zła forma macierzy z wynikami')
            return
    n = len(macierz_b)
    m = n-1
    i = 0
    j = i-1 
    x = np.zeros(n)
    macierz_rozszerzona = np.concatenate((macierz_a,macierz_b),axis=1,dtype=float)
    while i < n:
        if macierz_rozszerzona[0][0] == 0.0:
            for c in range(i+1,n):
                nierzad = macierz_rozszerzona[c].copy()
                szukana_wartosc = macierz_rozszerzona[c][0]
                if szukana_wartosc != 0.0:
                    macierz_rozszerzona[c] = macierz_rozszerzona[0]
                    macierz_rozszerzona[0] = nierzad
                    break
                else:
                    if c == n-1:
                        print('Macierz jest osobliwa')
                        return 
        for j in range(i+1,n):
            czynnik = macierz_rozszerzona[j][i]/macierz_rozszerzona[i][i]
            macierz_rozszerzona[j] = macierz_rozszerzona[j] - (czynnik*macierz_rozszerzona[i])
        i+=1
    x[m] = macierz_rozszerzona[m][n]/macierz_rozszerzona[m][m]
    for k in range(n-2,-1,-1):
        x[k] = macierz_rozszerzona[k][n]
        for j in range(k+1,n):
            x[k] -= x[j]*macierz_rozszerzona[k][j]
        x[k] /= macierz_rozszerzona[k][k]
    return x

def poly_approximation(x,y,stopien):
    b = []
    sumaa = []
    for i in range(stopien+1):
        suma = []
        for k in range(stopien+1):
            suma.append(np.sum(x ** (i + k)))
        sumaa.append(suma)
        b.append([np.sum(y*x**i)])
    bety = gauSS(np.array(sumaa), np.array(b))
    return np.poly1d(bety[::-1])
x,y = np.array([1.0, 2.5, 3.5, 4.0, 1.1, 1.8, 2.2, 3.7]),np.array([6.008, 15.722, 27.13, 33.772, 5.257, 9.549, 11.098, 28.828])
liniowe= poly_approximation(x,y,1)
kwadratowe = poly_approximation(x,y,2)

xs = np.linspace(0.5, 4.5, 10000)
plt.scatter(x,y)
plt.plot(xs, liniowe(xs), label="stopien = 1")
plt.plot(xs, kwadratowe(xs), label="stopien = 2")
plt.show()

suma_bledu_liniowego = np.sum(abs(y-liniowe(x)))
suma_bledu_kwadratowego = np.sum(abs(y-kwadratowe(x)))
print("stopien=1:",suma_bledu_liniowego,"stopien=2:",suma_bledu_kwadratowego)