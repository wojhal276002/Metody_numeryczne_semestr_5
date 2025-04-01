import numpy as np
from scipy.linalg import solve, det, inv
import matplotlib.pyplot as plt

n= 20
mac = []
for p in range(n):
    mac.append(np.zeros(n))
i = 0
j = 0
diag = 0.025
stop = False
while not stop:
    if i == j:
        mac[i][j] = diag
        diag+=0.025
    elif i + 1 == j:
        mac[i][j] = 5
    j+=1
    if j == 19:
        if i == j:
            mac[i][j] = diag
        elif i + 1 == j:
            mac[i][j] = 5
        if i == 19:
            stop = True
        else:
            i +=1
            j = 0
x_stale_0 = [[1] for i in range(20)]
x_stare = x_stale_0
x_nowe = []
normy = []
moment = 0
momentow = False
for iter in range(100):
    x_nowe = np.dot(mac,x_stare)
    normy.append(np.linalg.norm(x_nowe,ord=2)/np.linalg.norm(x_stale_0,ord=2))
    if np.linalg.norm(x_nowe,ord=2)<np.linalg.norm(x_stale_0,ord=2) and iter>0 and not momentow:
        moment = iter+1
        momentow = True
    x_stare, x_nowe = x_nowe, []
moment_spadku = normy.index(max(normy))+1
print('najmniejeze k:',moment,', malenie po iteracjach:',moment_spadku)
xs = np.linspace(1,100,100)
plt.plot(xs,normy)
plt.show()
    
