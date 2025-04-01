import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats
import math

def predkosc(t,u=2510,M_0=(2.8)*10**6,m=(13.3)*10**3,g=9.81):
    return u*np.log(M_0/(M_0-m*t))-g*t-335

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

x_sieczne,iter_sieczne,liczba_sieczne = sieczne(predkosc,60,80,1.0e-8)
print(x_sieczne)