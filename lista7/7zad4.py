import numpy as np
import matplotlib.pyplot as plt

def ruch_bez_oporu(v_0, alpha, t_max, n_steps):
    ts = np.linspace(0, t_max, n_steps)
    ys = [0]
    xs = [0]
    vy = v_0 * np.sin(np.deg2rad(alpha))
    vx = v_0 * np.cos(np.deg2rad(alpha))
    for t in ts:
        if ys[-1] < 0:
            return xs, ys
        else:
            ys.append(vy * t - 1/2 * 9.81 * t**2)
            xs.append(vx * t)
    return xs, ys
v_0 = 20
alpha = 45
t_max = 40
steps = 1000
xs, ys = ruch_bez_oporu(v_0, alpha, t_max, steps)

plt.plot(xs, ys,label="bez oporu")

def rk4_fixed_steps(y_0,f, steps,x):
    h = (x - 0) / steps
    t = np.linspace(0, x, steps + 1)
    y = np.zeros((steps+1, len(y_0)))
    y[0] = y_0
    for i in range(steps):
        F1 = np.array(f(t[i],y[i]))
        F2 = np.array(f(t[i] + h/2,y[i] + h*F1/2))
        F3 = np.array(f(t[i] + h/2,y[i] + h*F2/2))
        F4 = np.array(f(t[i] + h,y[i] + h*F3))
        y[i+1] = y[i] + h / 6 *(F1 + 2 * F2 + 2 * F3 + F4) 
    return y

cw = 0.35
rho = 1.2
A = 0.1
m = 0.5
g = 9.81

def ruch_opor(t, values):
    x, y_pos, vx, vy = values
    V = np.sqrt(vx**2 + vy**2) 
    ax = -0.5 * cw * rho * A * V * vx / m  
    ay = -g - 0.5 * cw * rho * A * V * vy / m 
    return [vx, vy, ax, ay]
#param
tmax = 10
steps = 1000
y_0 = 0
x_0 = 0
alpha = 45
v0 = 20

vx0 = v0 * np.cos(np.deg2rad(alpha))
vy0 = v0 * np.sin(np.deg2rad(alpha))
values = rk4_fixed_steps([x_0, y_0, vx0, vy0], ruch_opor,steps,t_max)
print(values)
x, y = values[:, 0], values[:, 1]
valid_indices = y >= 0
x_vals = x[valid_indices]
y_vals = y[valid_indices]

plt.plot(x_vals, y_vals,label="opór")
plt.title("Bez oporów vs Opory")
plt.legend()
plt.show()