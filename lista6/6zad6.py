import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.optimize import fsolve
from numpy.linalg import solve
from functools import reduce
from operator import mul
import sympy

def Df1(funkcja,x,h):
    return (funkcja(x+h)-funkcja(x))/h

def Dc2(funkcja,x,h):
    return (funkcja(x+h)-funkcja(x-h))/(2*h)

def Dc4(funkcja,x,h):
    return ((4*Dc2(funkcja,x,h)-Dc2(funkcja,x,2*h))/3)
def rozniczkowanie(funkcja,x,aprox):
    wszystko = []
    for h in range(1,4):
        h1 = 10**-h
        Df1_ = Df1(funkcja,x,h1)
        Dc2_ = Dc2(funkcja,x,h1)
        Dc4_ = Dc4(funkcja,x,h1)
        cos = [f"h={h1}",abs(aprox-Df1_),abs(aprox-Dc2_),abs(aprox-Dc4_)]
        wszystko.append(cos)
    return wszystko

print(f"\nwielomian:{rozniczkowanie(lambda x: x**3 - 2*x,1,1)}\n \nsinus:{rozniczkowanie(lambda x: np.sin(x),np.pi/3,0.5)}\n \nexp:{rozniczkowanie(lambda x: np.exp(x),0,1)}")


