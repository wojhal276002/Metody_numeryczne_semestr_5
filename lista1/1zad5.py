import cProfile
import numpy as np

def wielomian_niehorner(wspolczynniki,x):
    wynik = 0
    power = len(wspolczynniki)
    for c in wspolczynniki:
        wynik+= c*x**power
        power-=1
    return wynik

wspolczynniki = [6, 5, -13, 1, 1]
def main():
    lista = []
    float_ran = np.linspace(-10,10,200000)
    for i in float_ran:
        lista.append(wielomian_niehorner(wspolczynniki,i))
    return lista

if __name__ == '__main__':
    cProfile.run('main()')

def wielomian_niehornerr(x):
    return 6*x**4+5*x**3-13*x**2+x+1

def main():
    lista = []
    float_ran = np.linspace(-10,10,200000)
    for i in float_ran:
        lista.append(wielomian_niehornerr(i))
    return lista

if __name__ == '__main__':
    cProfile.run('main()')
