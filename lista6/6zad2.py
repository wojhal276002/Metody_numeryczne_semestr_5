import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.optimize import fsolve
from numpy.linalg import solve
from functools import reduce
from operator import mul


#cos(2cos^-1 (x)) = 2x^2-1
def simpson(wezly,funkcja,start,stop):
    h = (stop-start)/(wezly-1)
    wartosc = 0
    for i in range(wezly):
        moment = start+i*h
        if i in [0,wezly-1]:
            wartosc += funkcja(moment)
        elif i%2 == 1:
            wartosc += 4*funkcja(moment)
        else:
            wartosc += 2*funkcja(moment)
    return h/3*wartosc

print(f"3 wezly:{simpson(3,lambda x: np.cos(2*np.arccos(x)),-1,1)}\n5 wezlow: {simpson(5,lambda x: np.cos(2*np.arccos(x)),-1,1)}\n7 wezlow: {simpson(7,lambda x: np.cos(2*np.arccos(x)),-1,1)}")
