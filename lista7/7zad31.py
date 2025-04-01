import numpy as np
import matplotlib.pyplot as plt

def wahadlo(tau, Z, Q, A_hat, omega_hat):
    theta, dtheta = Z
    dtheta_dtau = dtheta
    d2theta_dtau = - (1 / Q) * dtheta - np.sin(theta) + A_hat * np.cos(omega_hat * tau)
    return [dtheta_dtau, d2theta_dtau]

def rk4_dla_wielu_param(f, Z0, t, params):
    Q, A_hat, omega_hat = params
    n = len(t)
    Z = np.zeros((n, len(Z0)))
    Z[0] = Z0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = h * np.array(f(t[i], Z[i], Q, A_hat, omega_hat))
        k2 = h * np.array(f(t[i] + h / 2, Z[i] + k1 / 2, Q, A_hat, omega_hat))
        k3 = h * np.array(f(t[i] + h / 2, Z[i] + k2 / 2, Q, A_hat, omega_hat))
        k4 = h * np.array(f(t[i] + h, Z[i] + k3, Q, A_hat, omega_hat))
        Z[i + 1] = Z[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return Z


Q = 2
omega_daszek = 2 / 3
cases = [
    {"A_hat": 0.5, "theta0": 0.01, "dtheta0": 0},
    {"A_hat": 0.5, "theta0": 0.3, "dtheta0": 0},
    {"A_hat": 1.35, "theta0": 0.3, "dtheta0": 0}
]

t = np.linspace(0, 50, 1000) 


for a in cases:
    A_daszek = a["A_hat"]
    theta0 = a["theta0"]
    dtheta0 = a["dtheta0"]
    
 
    Z0 = [theta0, dtheta0]
    params = (Q, A_daszek, omega_daszek)
    
    Z = rk4_dla_wielu_param(wahadlo, Z0, t, params)
    theta, dtheta = Z[:, 0], Z[:, 1]
    plt.figure(figsize=(15, 6))
    plt.subplot(1,2,1)
    plt.title(f"A_hat={A_daszek}, theta0={theta0}")
    plt.plot(t, theta)
    plt.subplot(1,2,2)
    plt.plot(theta, dtheta)
    plt.show()
