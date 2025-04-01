import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats
import math

X = np.linspace(4,8,1000)
Y = np.cosh(X)*np.cos(X)-1

def f(x):
    return np.cosh(x)*np.cos(x)-1

def fprime(x):
    return np.cos(x) * np.sinh(x) - np.sin(x) * np.cosh(x)

def fbis(x):
    return -2 * np.sin(x) * np.sinh(x)

plt.plot(X,Y)
plt.axhline(y=0,color='red',linestyle='dashed')
plt.axvline(x=4,color='red')
plt.axvline(x=5,color='red')
plt.show()

def newton(f,fpier,x1,epsilon):
    x0 = x1
    fa = f(x0)
    while abs(fa) > epsilon:
        Dfxn = fpier(x0)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        x0 = x0 - fa/Dfxn
        fa = f(x0)
    return x0

x = newton(f, fprime, 4, 1.0e-8)
print(x)