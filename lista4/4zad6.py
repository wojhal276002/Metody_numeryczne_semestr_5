import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats
import cmath

def laguerre_method(f_0, f_1, f_2, x0, n):
    xk = x0
    while abs(f_0(xk)) > 1*10**-8:
        G = f_1(xk) / f_0(xk)
        H = G ** 2 - f_2(xk) / f_0(xk)
        root = cmath.sqrt((n - 1) * (n * H - G ** 2))
        d = max([G + root, G - root], key=abs)
        a = n / d
        xk -= a
    return xk

f_0 = np.poly1d([1, 5+1j, -8+5j, 30-14j, -84])
f_1 = np.polyder(f_0, 1)
f_2 = np.polyder(f_0, 2)
xs = []
n = 4
for i in range(4):
    res = laguerre_method(f_0,f_1,f_2,0,n)
    xs.append(res)
    p = np.poly1d([1,-res])
    f_0 = np.polydiv(f_0, p)[0]
    f_1 = np.polyder(f_0, 1)
    f_2 = np.polyder(f_0, 2) 
    n -= 1
print(xs)