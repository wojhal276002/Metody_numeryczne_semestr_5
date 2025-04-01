import numpy as np
from scipy.linalg import solve, det, inv, lu
import cProfile
import matplotlib.pyplot as plt

#dokladnosc
liczba = 1
while liczba+1 != 1:
    liczba_n = liczba/2
    liczba = liczba_n
dokladnosc = liczba

#tworzenie macierzy
mac = []
for p in range(5):
    mac.append([(1/(p+1)),(1/(p+2)),(1/(p+3)),(1/(p+4)),(1/(p+5))])
rozw = [[5],[4],[3],[2],[1]]


#metoda iteracyjnego 
P, L, U = lu(mac)
y = solve(L,rozw)
x =  solve(U,y)
x_0 = x.copy()
print(x)
iter = 0
wektor_reszt = (np.array(rozw,dtype=np.float64) - np.array(np.dot(mac,x),dtype=np.float64))
while np.linalg.norm(wektor_reszt,np.inf) > np.linalg.norm(np.dot(mac,x),np.inf)*2*dokladnosc*10**2:
    delta_x = solve(mac, wektor_reszt)
    x += delta_x
    iter += 1
    wektor_reszt = (np.array(rozw,dtype=np.float64) - np.array(np.dot(mac,x),dtype=np.float64))
print(x,iter)