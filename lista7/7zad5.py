import numpy as np
import matplotlib.pyplot as plt

def system(x, Z):
    z1, z2 = Z
    dz1_dx = z2
    dz2_dx = -(1 - 0.2 * x) * z1**2
    return np.array([dz1_dx, dz2_dx])

def rk4_fixed_steps(f,y_0,x):
    h = (y_0[1] - y_0[0]) / len(x)
    y = np.zeros((len(x), len(y_0)))
    y[0] = y_0
    for i in range(len(x)-1):
        F1 = np.array(f(x[i],y[i]))
        F2 = np.array(f(x[i] + h/2,y[i] + h*F1/2))
        F3 = np.array(f(x[i] + h/2,y[i] + h*F2/2))
        F4 = np.array(f(x[i] + h,y[i] + h*F3))
        y[i+1] = y[i] + h / 6 *(F1 + 2 * F2 + 2 * F3 + F4) 
    return y

def metoda_strzelania(f, x, z1_0, z1_target, z2_initial_guess):
    tolerance = 1e-6  # Dokładność
    max_iter = 100  # Maksymalna liczba iteracji
    z2 = z2_initial_guess  # Początkowe zgadywanie z2(0)
    
    for iteration in range(max_iter):
        # Rozwiąż układ równań dla obecnego zgadywania
        Z0 = [z1_0, z2]
        Z = rk4_fixed_steps(f, Z0, x)
        z1_end = Z[-1, 0]  # z1 na końcu przedziału
        
        # Sprawdź, czy spełniono warunek brzegowy
        if abs(z1_end - z1_target) < tolerance:
            print(f"Znaleziono rozwiązanie po {iteration+1} iteracjach: z2(0) = {z2}")
            return Z
        
        # metoda stycznej (strzelanie proste)
        z2 += (z1_target - z1_end) / (x[-1] - x[0])  
    
    raise ValueError("Nie znaleziono rozwiązania w zadanej liczbie iteracji.")

x = np.linspace(0, np.pi / 2, 1000)  # Przedział [0, π/2]
z1_0 = 0
z1_target = 1  
z2 = 1


Z = metoda_strzelania(system, x, z1_0, z1_target, z2)


z1 = Z[:, 0]
z2 = Z[:, 1]

plt.plot(x, z1, label="y(x)")
plt.legend()
plt.show()