import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.optimize import fsolve
from numpy.linalg import solve
from functools import reduce
from operator import mul

x = np.array([1.2,2.8,4.3,5.4,6.8,7.9])
y = np.array([7.5,16.1,38.9,67.0,146.6,266.2])
zlogarytm_y = np.log(y)

def prosta_regresji(x,y):
    b_1 = np.sum(x*(y-np.mean(y)))/np.sum((x-np.mean(x))**2)
    b_0 = np.mean(y) - b_1 * np.mean(x)
    return b_1, b_0

b, c = prosta_regresji(x, zlogarytm_y)

def f(b,c,x):
    a = np.exp(c)
    return a * np.exp(b * x)

xs = np.linspace(0,9,1000)
func = f(b,c,xs)
plt.scatter(x, y,color='red')
plt.plot(xs, func)
plt.show()

func1 = f(b,c,x)
odchylenie = np.std(y-func1)
print(odchylenie)

