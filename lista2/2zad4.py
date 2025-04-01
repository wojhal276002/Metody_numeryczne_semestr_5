import numpy as np
from scipy.linalg import solve, det, inv

def gauSS(macierz_a,macierz_b,zadanie=None):
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
    for p in range(n):
        print(f'x{p}={x[p]}')
    if zadanie == 'zad6':
        detka = 1
        lista_ax = np.dot(macierz_a,x)
        lista_roznic = [c[0] for c in macierz_b]
        for c in range(n):
            detka*=macierz_rozszerzona[c][c]
        print('wyznacznik:',detka)
        print('Ax:',lista_ax)
        print('różnica:',lista_roznic-lista_ax)
    elif zadanie == 'odwrot':
        detka = 1
        for c in range(n):
            detka*=macierz_rozszerzona[c][c]
        return detka

#zad4
print('zad4')
gauSS(np.array([[0,0,2,1,2],[0,1,0,2,-1],[1,2,0,-2,0],[0,0,0,-1,1],[0,1,-1,1,-1]]),macierz_b=np.array([[1],[1],[-4],[-2],[-1]]))
print('scipy',solve(np.array([[0,0,2,1,2],[0,1,0,2,-1],[1,2,0,-2,0],[0,0,0,-1,1],[0,1,-1,1,-1]]),np.array([[1],[1],[-4],[-2],[-1]])))
#zad5
print('zad5')
gauSS(np.array([[1,0,0,0,0],[1,1,1,1,1],[1,3,9,27,81],[1,5,25,125,625],[1,6,36,216,1296]]),macierz_b=np.array([[-1],[1],[3],[2],[-2]]))
print('scipy',solve(np.array([[1,0,0,0,0],[1,1,1,1,1],[1,3,9,27,81],[1,5,25,125,625],[1,6,36,216,1296]]),np.array([[-1],[1],[3],[2],[-2]])))
#zad6
print('zad6')
gauSS(np.array([[3.5,2.77,-0.76,1.8],[-1.8,2.68,3.44,-0.09],[0.27,5.07,6.9,1.61],[1.71,5.45,2.68,1.71]]),macierz_b=np.array([[7.31],[4.23],[13.85],[11.55]]),zadanie='zad6')
print('scipy',solve(np.array([[3.5,2.77,-0.76,1.8],[-1.8,2.68,3.44,-0.09],[0.27,5.07,6.9,1.61],[1.71,5.45,2.68,1.71]]),np.array([[7.31],[4.23],[13.85],[11.55]])))

#zad7
print('zad7')
gauSS(np.array([[10,-2,-1,2,3,1,-4,7],[5,11,3,10,-3,3,3,-4],[7,12,1,5,3,-12,2,3],[8,7,-2,1,3,2,2,4],[2,-15,-1,1,4,-1,8,3],[4,2,9,1,12,-1,4,1],[-1,4,-7,-1,1,1,-1,-3],[-1,3,4,1,3,-4,7,6]]),macierz_b=np.array([[0],[12],[-5],[3],[-25],[-26],[9],[-7]]))
print('scipy',solve(np.array([[10,-2,-1,2,3,1,-4,7],[5,11,3,10,-3,3,3,-4],[7,12,1,5,3,-12,2,3],[8,7,-2,1,3,2,2,4],[2,-15,-1,1,4,-1,8,3],[4,2,9,1,12,-1,4,1],[-1,4,-7,-1,1,1,-1,-3],[-1,3,4,1,3,-4,7,6]]),np.array([[0],[12],[-5],[3],[-25],[-26],[9],[-7]])))




def gauSS_do_odwrotu(macierz_a,zadanie=None):
    pierwszy_a = macierz_a.shape[0]
    if macierz_a.shape[1] != pierwszy_a:
        print('niespójny kształt macierzy kwadratowej')
        return
    n = len(macierz_a[0])
    m = n-1
    i = 0
    j = i-1 
    macierz_rozszerzona = macierz_a
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
    print(macierz_rozszerzona[0],macierz_rozszerzona[1],macierz_rozszerzona[2],macierz_rozszerzona[3],macierz_rozszerzona[n-1])


def odwrotny_gauSS(macierz_a,zadanie=None):
    pierwszy_a = macierz_a.shape[0]
    if macierz_a.shape[1] != pierwszy_a:
        print('niespójny kształt macierzy kwadratowej')
        return
    macierz_a = np.array(macierz_a, dtype=float)
    n = len(macierz_a)
    m = n-1
    i = 0

    macierz_rozszerzona = macierz_a.copy()
    macierz_odwrocona2 = np.eye(n)
    macierz_odwrocona = macierz_odwrocona2.copy()
    while i < n:
        if macierz_rozszerzona[i][i] == 0.0:
            for c in range(i+1,n):
                nierzad = macierz_rozszerzona[c].copy()
                nierzad_odwr = macierz_rozszerzona[c].copy()
                szukana_wartosc = macierz_rozszerzona[c][0]
                if szukana_wartosc != 0.0:
                    macierz_rozszerzona[c] = macierz_rozszerzona[0]
                    macierz_rozszerzona[0] = nierzad
                    macierz_odwrocona[c] = macierz_odwrocona[0]
                    macierz_odwrocona[0] = nierzad_odwr
                    break
                else:
                    if c == n-1:
                        print('Macierz jest osobliwa, brak odwrotnej macierzy')
                        return 
        if i != n-1:
            czynnik = (1-macierz_rozszerzona[i][i])/macierz_rozszerzona[i+1][i]
            macierz_rozszerzona[i] += czynnik*macierz_rozszerzona[i+1]
            macierz_odwrocona[i] += czynnik*macierz_odwrocona[i+1]
        else:
            macierz_odwrocona[i] /= macierz_rozszerzona[i][i]
            macierz_rozszerzona[i] /= macierz_rozszerzona[i][i]
        for j in range(0,n):
            if j != i:
                czynnik = macierz_rozszerzona[j][i]/macierz_rozszerzona[i][i]
                macierz_rozszerzona[j] = (macierz_rozszerzona[j] - (czynnik*macierz_rozszerzona[i]))
                macierz_odwrocona[j] = (macierz_odwrocona[j] - (czynnik*macierz_odwrocona[i]))
        i+=1
    print(macierz_odwrocona)
    if zadanie == 'zad9':
        detka = 1
        for c in range(n):
            detka*=macierz_odwrocona[c][c]
        print('wyznacznik:',detka)



#zad8
print('zad8')
odwrotny_gauSS(np.array([[2, -1, 0, 0, 0, 0],[-1, 2, -1, 0, 0, 0],[0, -1, 2, -1, 0, 0],[0, 0, -1, 2, -1, 0],[0, 0, 0, -1, 2, -1],[0, 0, 0, 0, -1, 5]]))
print('odwrocona scipy:','\n',inv(np.array([[2, -1, 0, 0, 0, 0],[-1, 2, -1, 0, 0, 0],[0, -1, 2, -1, 0, 0],[0, 0, -1, 2, -1, 0],[0, 0, 0, -1, 2, -1],[0, 0, 0, 0, -1, 5]])))
#zad9
print('zad9')
print('wyznacznik:',det(np.array([[1, 3, -9, 6, 4],[2, -1, 6, 7, 1],[3, 2, -3, 15, 5],[8, -1, 1, 4, 2],[11, 1, -2, 18, 7]])))
print('odwrocona scipy:','\n',inv(np.array([[1, 3, -9, 6, 4],[2, -1, 6, 7, 1],[3, 2, -3, 15, 5],[8, -1, 1, 4, 2],[11, 1, -2, 18, 7]])))
odwrotny_gauSS(np.array([[1, 3, -9, 6, 4],[2, -1, 6, 7, 1],[3, 2, -3, 15, 5],[8, -1, 1, 4, 2],[11, 1, -2, 18, 7]]),'zad9')
print('wyznacznik scipy:',det(np.array([[1, 3, -9, 6, 4],[2, -1, 6, 7, 1],[3, 2, -3, 15, 5],[8, -1, 1, 4, 2],[11, 1, -2, 18, 7]])))
print('macierz elimiacji Gaussa dla zad9:','\n')
gauSS_do_odwrotu(np.array([[1, 3, -9, 6, 4],[2, -1, 6, 7, 1],[3, 2, -3, 15, 5],[8, -1, 1, 4, 2],[11, 1, -2, 18, 7]]))