import cProfile
import numpy as np
    
def wielomian_hornerr(wspolczynniki, x):
    wynik = wspolczynniki[0]
    for i in range(1, len(wspolczynniki)):
        wynik = wynik*x + wspolczynniki[i]
    return wynik

def main():
    lista = []
    float_ran = np.linspace(-10,10,200000)
    for i in float_ran:
        lista.append(wielomian_hornerr([1,1,-13,5,6],i))
    return lista

if __name__ == '__main__':
    cProfile.run('main()')


def wielomian_horner(x):
    return ((((6*x+5)*x-13)*x+1)*x+1)

def main():
    lista = []
    float_ran = np.linspace(-10,10,200000)
    for i in float_ran:
        lista.append(wielomian_horner(i))
    return lista

if __name__ == '__main__':
    cProfile.run('main()')