import numpy as np
import matplotlib.pyplot as plt

def f1(x,y):
    return x**2 - 4*y

def euler_fixed_steps(f, y0, x, steps):
    h = (x - 0)/steps
    t = np.linspace(0, x, steps + 1)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(steps):
        y[i+1] = y[i] + h * f(t[i], y[i])
    return y[steps]

def rk2_fixed_steps(f, y0, x, steps):
    h = (x - 0) / steps  
    t = np.linspace(0, x, steps + 1)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(steps):
        F1 = h * f(t[i],y[i])
        F2 = h * f(t[i] + h / 2,y[i] + F1 / 2)
        y[i+1] = y[i] + F2
    return y[steps]

def rk4_fixed_steps(f, y0, x, steps):
    h = (x - 0) / steps
    t = np.linspace(0, x, steps + 1)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(steps):
        F1 = h * f(t[i],y[i])
        F2 = h * f(t[i] + h / 2,y[i] + F1 / 2)
        F3 = h * f( t[i] + h / 2,y[i] + F2 / 2)
        F4 = h * f(t[i] + h,y[i] + F3)
        y[i+1] = y[i] + (F1 + 2 * F2 + 2 * F3 + F4) / 6
    return y[steps]

y0 = 1.0

print(f"euler\n{euler_fixed_steps(f1, y0, 0.03,1)}\n{euler_fixed_steps(f1, y0, 0.03,2)}\n{euler_fixed_steps(f1, y0, 0.03,4)}\n")

print(f"rk2\n{rk2_fixed_steps(f1, y0, 0.03,1)}\n{rk2_fixed_steps(f1, y0, 0.03,2)}\n{rk2_fixed_steps(f1, y0, 0.03,4)}\n")

print(f"rk4\n{rk4_fixed_steps(f1, y0, 0.03,1)}\n{rk4_fixed_steps(f1, y0, 0.03,2)}\n{rk4_fixed_steps(f1, y0, 0.03,4)}\n")

print(f"analitycznie{31/32 * np.exp(-4*0.03) + 1/4 * 0.03**2 - 1/8 * 0.03 + 1/32}")



