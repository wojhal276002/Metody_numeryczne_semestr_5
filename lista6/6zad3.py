import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.optimize import fsolve
from numpy.linalg import solve
from functools import reduce
from operator import mul

def trapezy(wezly,funkcja,start,stop):
    h = (stop-start)/(wezly-1)
    wartosc = 0
    for i in range(wezly):
        moment = start+i*h
        if i in [0,wezly-1]:
            wartosc += funkcja(moment)
        else:
            wartosc += 2*funkcja(moment)
    return h/2*wartosc

print(f"6 wezlow:{trapezy(8,lambda x: (1/3)*(x**(-4/3))/(1+x**(-4/3)),10**-13,1)}")


