import numpy as np
from scipy.linalg import solve, det, inv
import cProfile

#tworzenie macierzy A i b
n= 20
mac = []
for p in range(n):
    mac.append(np.zeros(n))
i = 0
j = 0
stop = False
while not stop:
    if i == j:
        mac[i][j] = 4
    elif i + 1 == j or i == j+1:
        mac[i][j] = -1
    elif i == 19 and j == 0:
        mac[i][j] = 1
    j+=1
    if j == 19:
        if i == j:
            mac[i][j] = 4
        elif i + 1 == j or i == j+1:
            mac[i][j] = -1
        elif i == 0:
            mac[i][j] = 1
        if i == 19:
            stop = True
        else:
            i +=1
            j = 0
rozw = []
for p in range(n):
    rozw.append(np.zeros(1))
rozw[19][0]=100



def gauSS(macierz_a,macierz_b):
    pierwszy_a = macierz_a.shape[0]
    if macierz_a.shape[1] != pierwszy_a:
        print('niespójny kształt macierzy kwadratowej')
        return
    if macierz_b.shape[1] > 1 or macierz_b.shape[0] != macierz_a.shape[0]:
            print('zła forma macierzy z wynikami')
            return
    n = len(macierz_b)
    m = n-1
    i = 0
    j = i-1 
    x = np.zeros(n)
    macierz_rozszerzona = np.concatenate((macierz_a,macierz_b),axis=1,dtype=float)
    while i < n:
        if macierz_rozszerzona[0][0] == 0.0:
            for c in range(i+1,n):
                nierzad = macierz_rozszerzona[c].copy()
                szukana_wartosc = macierz_rozszerzona[c][0]
                if szukana_wartosc != 0.0:
                    macierz_rozszerzona[c] = macierz_rozszerzona[0]
                    macierz_rozszerzona[0] = nierzad
                    break
                else:
                    if c == n-1:
                        print('Macierz jest osobliwa')
                        return 
        for j in range(i+1,n):
            czynnik = macierz_rozszerzona[j][i]/macierz_rozszerzona[i][i]
            macierz_rozszerzona[j] = macierz_rozszerzona[j] - (czynnik*macierz_rozszerzona[i])
        i+=1
    x[m] = macierz_rozszerzona[m][n]/macierz_rozszerzona[m][m]
    for k in range(n-2,-1,-1):
        x[k] = macierz_rozszerzona[k][n]
        for j in range(k+1,n):
            x[k] -= x[j]*macierz_rozszerzona[k][j]
        x[k] /= macierz_rozszerzona[k][k]
    print(x)
    return x
def main():  
    gauSS(np.array(mac),np.array(rozw))

if __name__ == '__main__':
    cProfile.run('main()')
print(solve(np.array(mac),np.array(rozw)))
