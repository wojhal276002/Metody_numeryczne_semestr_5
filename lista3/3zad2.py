import numpy as np
from scipy.linalg import solve, det, inv
import cProfile
def macierze(n):
    U = []
    D = []
    L = []
    for p in range(n):
        U.append(np.zeros(n))
        D.append(np.zeros(n))
        L.append(np.zeros(n))
    i = 0
    j = 0
    stop = False
    while not stop:
        if i == j:
            D[i][j] = 4
        elif i + 1 == j:
            U[i][j] = -1
        elif i == j+1:
            L[i][j] = -1
        elif i == 0 and j == n-1:
            U[i][j] = 1
        elif i == n-1 and j == 0:
            L[i][j] = 1
        j+=1
        if j == n-1:
            if i == j:
                D[i][j] = 4
            elif i + 1 == j:
                U[i][j] = -1
            elif i == j+1:
                L[i][j] = -1
            elif i == 0:
                U[i][j] = 1
            if i == n-1:
                stop = True
            else:
                i +=1
                j = 0
    rozw = []
    rozw2 = np.zeros(n)
    rozw2[n-1] = 100
    for p in range(n):
        rozw.append(np.zeros(1))
    rozw[n-1][0]=100
    return U,D,L,rozw, rozw2
n = 20

#do pierwszej metody

U,D,L,rozw, rozw2 = macierze(n)
D_inv = np.linalg.inv(D)
D_b = np.dot(D_inv,rozw)
min_D_L = -np.dot(D_inv,L)
min_D_U = -np.dot(D_inv,U)

#do drugiej metody 

N_0 = []
for i in range(len(D)):
    N_0.append(D[i] + L[i])
N = np.linalg.inv(N_0)
M = np.dot(-N, U)
zbieganie = max(np.abs(np.linalg.eig(M)[0])) < 1
print('czy rozwiązanie zbieżne:',zbieganie)



def main(n,D_b,min_D_L,min_D_U):
    stare_x = np.zeros(n)
    nowe_x = np.zeros(n)
    licz = 0
    stop = False
    while not stop:
        x = D_b[licz][0]
        dla_L = []
        dla_U = []
        for p in range(n):
            dla_L.append(min_D_L[licz][p]*nowe_x[p])
            dla_U.append(min_D_U[licz][p]*stare_x[p])
        x = x + sum(dla_L) + sum(dla_U)
        nowe_x[licz] = x
        licz+=1
        if licz == n:
            norma = np.linalg.norm(nowe_x-stare_x)
            if norma < 10**-4:
                stop = True
            else:
                stare_x = nowe_x
                nowe_x = np.zeros(n)
                licz = 0
    return stare_x

if __name__ == '__main__':
    cProfile.run('main(n,D_b,min_D_L,min_D_U)')
print(main(n,D_b,min_D_L,min_D_U))


#szybsza metoda

def main2(n,M,N,rozw):
    stare_x = np.zeros(n)
    nowe_x = np.dot(M,stare_x) + np.dot(N,rozw)
    while np.linalg.norm(nowe_x-stare_x) > 10**-4:
        stare_x = nowe_x
        nowe_x = np.dot(M, stare_x) + np.dot(N, rozw)
    return nowe_x

if __name__ == '__main__':
    cProfile.run('main2(n,M,N,rozw2)')
print(main2(n,M,N,rozw2))