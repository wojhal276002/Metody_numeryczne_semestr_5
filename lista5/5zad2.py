import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy.optimize import fsolve
from numpy.linalg import solve
from scipy.interpolate import CubicSpline


funkcje_sklejane = CubicSpline([0.2, 2, 20, 200, 2000, 20000], [103, 13.9, 2.72, 0.8, 0.401, 0.433])
Re = [5, 50, 5000]
for r in Re:
    print("wartość r:",r,", wartość cD:",funkcje_sklejane(r))