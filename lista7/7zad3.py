import numpy as np
import matplotlib.pyplot as plt

def runge_kutta_4th_order_for_2nd_order_ode(u1_0, u2_0, f_u1, f_u2, n, x):
    u1 = u1_0
    u2 = u2_0
    xs = np.linspace(0, x, n+1)
    h = xs[1] - xs[0]
    u1s = [u1_0]
    u2s = [u2_0]
    for i in range(n):
        k1_u1 = f_u1(xs[i], u1, u2)
        k1_u2 = f_u2(xs[i], u1, u2)
        k2_u1 = f_u1(xs[i] + h/2, u1 + h * k1_u1/2, u2 + h * k1_u2/2)
        k2_u2 = f_u2(xs[i] + h/2, u1 + h * k1_u1/2, u2 + h * k1_u2/2)
        k3_u1 = f_u1(xs[i] + h/2, u1 + h * k2_u1/2, u2 + h * k2_u2/2)
        k3_u2 = f_u2(xs[i] + h/2, u1 + h * k2_u1/2, u2 + h * k2_u2/2)
        k4_u1 = f_u1(xs[i] + h, u1 + h * k3_u1, u2 + h * k3_u2)
        k4_u2 = f_u2(xs[i] + h, u1 + h * k3_u1, u2 + h * k3_u2)
        u1 += h / 6 * (k1_u1 + 2*k2_u1 + 2*k3_u1 + k4_u1)
        u2 += h / 6 * (k1_u2 + 2*k2_u2 + 2*k3_u2 + k4_u2)
        u1s.append(u1)
        u2s.append(u2)
    return u1s, u2s
def func_u1(x, u1, u2):
    return u2
def func1_u2(x, u1, u2):
    return 0.5 * np.cos(2/3 * x) - np.sin(u1) - 1/2 * u2
def func2_u2(x, u1, u2):
    return 0.5 * np.cos(2/3 * x) - np.sin(u1) - 1/2 * u2
def func3_u2(x, u1, u2):
    return 1.35 * np.cos(2/3 * x) - np.sin(u1) - 1/2 * u2
u1_values1, u2_values1 = runge_kutta_4th_order_for_2nd_order_ode(0.01,0,func_u1,func1_u2,1000,30)
u1_values2, u2_values2 = runge_kutta_4th_order_for_2nd_order_ode(0.3,0,func_u1,func2_u2,1000,30)
u1_values3, u2_values3 = runge_kutta_4th_order_for_2nd_order_ode(0.3,0,func_u1,func3_u2,1000,30)
ts = np.linspace(0, 30, 1001)
plt.plot(ts, u1_values1)
plt.grid(True)
plt.title("Sytuacja 1: $\\theta(t)$")
plt.xlabel("t")
plt.ylabel("$\\theta(t)$")
plt.show()

plt.plot(u1_values1, u2_values1)
plt.grid(True)
plt.title("Sytuacja 1: Portret fazowy")
plt.ylabel("$\\frac{d\\theta}{dt}$")
plt.xlabel("$\\theta(t)$")
plt.show()

plt.plot(ts, u1_values2)
plt.grid(True)
plt.title("Sytuacja 2: $\\theta(t)$")
plt.xlabel("t")
plt.ylabel("$\\theta(t)$")
plt.show()

plt.plot(u1_values2, u2_values2)
plt.grid(True)
plt.title("Sytuacja 2: Portret fazowy")
plt.ylabel("$\\frac{d\\theta}{dt}$")
plt.xlabel("$\\theta(t)$")
plt.show()

plt.plot(ts, u1_values3)
plt.grid(True)
plt.title("Sytuacja 3: $\\theta(t)$")
plt.xlabel("t")
plt.ylabel("$\\theta(t)$")
plt.show()

plt.plot(u1_values3, u2_values3)
plt.grid(True)
plt.title("Sytuacja 3: Portret fazowy")
plt.ylabel("$\\frac{d\\theta}{dt}$")
plt.xlabel("$\\theta(t)$")
plt.show()