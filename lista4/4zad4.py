import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import scipy.stats as stats
import math

def gibbs(T,R=8.31441,T_0=4.44418):
    return -1*R*T*np.log((T/T_0)**(5/2))+1*10**5

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


x_sieczne,iter_sieczne,liczba_sieczne = sieczne(gibbs,800,1000,1.0e-8)
print(x_sieczne)