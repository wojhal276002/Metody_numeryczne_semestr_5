import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats
import cmath

xs = np.linspace(0,1.5,1000)
ys = np.linspace(0,1.5,1000)
def f1(x,y):
    return np.tan(x) - y - 1
def f2(x,y):
    return np.cos(x) - 3*np.sin(y)
def wzor1x(x):
    return 1/np.cos(x)**2
def wzor1y(x):
    return -1
def wzor2x(x):
    return -np.sin(x)
def wzor2y(x):
    return -3*np.cos(x)
plt.plot(xs,f1(xs,ys))
plt.plot(xs,f2(xs,ys))
plt.show()
x_start,y_start = 0,0
x0 = (x_start,y_start)
epsilon = 1
while epsilon > 1*10**-15:
    x,y = x0[0],x0[1]
    mac_f = np.array([f1(x,y),f2(x,y)])
    Df = np.array([[wzor1x(x),wzor1y(y)],[wzor2x(x),wzor2y(y)]])
    x0 = x0-1*np.dot(np.linalg.inv(Df),mac_f)
    epsilon= abs(x-x0[0])+abs(y-x0[1])
print(x0)
print(f1(x0[0],x0[1]))
print(f2(x0[0],x0[1]))