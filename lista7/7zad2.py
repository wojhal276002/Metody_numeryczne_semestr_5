import numpy as np
import matplotlib.pyplot as plt

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
    return y


def euler_fixed_steps(f, y0, x, steps):
    h = (x - 0)/steps
    t = np.linspace(0, x, steps + 1)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(steps):
        y[i+1] = y[i] + h * f(t[i], y[i])
    return y

def f2(x,y):
    return np.sin(y)

euler_values = euler_fixed_steps(f2, 1,0.5, 5)
rk_values = rk4_fixed_steps(f2, 1,0.5, 5)

xs = np.arange(0,0.6,0.1)
plt.plot(xs, euler_values, label="Euler")
plt.plot(xs, rk_values, label="Runge")
plt.legend()
plt.show()