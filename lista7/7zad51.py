import numpy as np
import matplotlib.pyplot as plt

# Układ równań różniczkowych
def system(x, Z):
    z1, z2 = Z
    dz1_dx = z2
    dz2_dx = -(1 - 0.2 * x) * z1**2
    return np.array([dz1_dx, dz2_dx])

# Metoda Rungego-Kutty 4. rzędu
def rk4(f, Z0, x):
    n = len(x)
    Z = np.zeros((n, len(Z0)))
    Z[0] = Z0
    for i in range(n - 1):
        h = x[i + 1] - x[i]
        k1 = h * f(x[i], Z[i])
        k2 = h * f(x[i] + h / 2, Z[i] + k1 / 2)
        k3 = h * f(x[i] + h / 2, Z[i] + k2 / 2)
        k4 = h * f(x[i] + h, Z[i] + k3)
        Z[i + 1] = Z[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return Z

# Funkcja strzelająca
def shooting_method(f, x, z1_0, z1_target, z2_initial_guess):
    tolerance = 1e-6  # Dokładność
    max_iter = 100  # Maksymalna liczba iteracji
    z2 = z2_initial_guess  # Początkowe zgadywanie z2(0)
    
    for iteration in range(max_iter):
        # Rozwiąż układ równań dla obecnego zgadywania
        Z0 = [z1_0, z2]
        Z = rk4(f, Z0, x)
        z1_end = Z[-1, 0]  # z1 na końcu przedziału
        
        # Sprawdź, czy spełniono warunek brzegowy
        if abs(z1_end - z1_target) < tolerance:
            print(f"Znaleziono rozwiązanie po {iteration+1} iteracjach: z2(0) = {z2}")
            return Z
        
        # Poprawka z2(0) za pomocą metody stycznej (strzelanie proste)
        z2 += (z1_target - z1_end) / (x[-1] - x[0])  # Przybliżona poprawka
    
    raise ValueError("Nie znaleziono rozwiązania w zadanej liczbie iteracji.")

# Parametry
x = np.linspace(0, np.pi / 2, 100)  # Przedział [0, π/2]
z1_0 = 0  # y(0)
z1_target = 1  # y(π/2)
z2_initial_guess = 1.0  # Początkowe zgadywanie dla y'(0)

# Rozwiązanie
Z = shooting_method(system, x, z1_0, z1_target, z2_initial_guess)

# Wykresy
z1 = Z[:, 0]
z2 = Z[:, 1]

plt.figure(figsize=(10, 5))
plt.plot(x, z1, label="y(x)")
plt.title("Rozwiązanie zagadnienia brzegowego")
plt.xlabel("x")
plt.ylabel("y, y'")
plt.legend()
plt.grid()
plt.show()