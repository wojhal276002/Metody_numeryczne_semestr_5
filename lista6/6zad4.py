import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.optimize import fsolve
from numpy.linalg import solve
from functools import reduce
from operator import mul

def caleczkunia(kat):
    return scipy.integrate.quad(lambda x: 1 / np.sqrt(1 - np.sin(np.deg2rad(kat)/2)**2 * np.sin(x)**2),0,np.pi/2)[0]

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

print(f"15°: {caleczkunia(15)}\n30°: {caleczkunia(30)}\n45°: {caleczkunia(45)}")
print(f"15°: {simpson(10**6,lambda x: 1 / np.sqrt(1 - np.sin(np.deg2rad(15)/2)**2 * np.sin(x)**2),0,np.pi/2)}\n30°: {simpson(10**6,lambda x: 1 / np.sqrt(1 - np.sin(np.deg2rad(30)/2)**2 * np.sin(x)**2),0,np.pi/2)}\n45°: {simpson(10**6,lambda x: 1 / np.sqrt(1 - np.sin(np.deg2rad(45)/2)**2 * np.sin(x)**2),0,np.pi/2)}")
print(caleczkunia(0),simpson(10**6,lambda x: 1 / np.sqrt(1 - np.sin(np.deg2rad(0)/2)**2 * np.sin(x)**2),0,np.pi/2),np.pi/2)