import matplotlib.pyplot as plt
import numpy as np
import math

def pade_1(x):
    return (6-2*x)/(6+4*x+x**2)

def pade_2(x):
    return (6-4*x+x**2)/(6+2*x)

xs = np.linspace(0,5,1000)
pade1 = pade_1(xs)
pade2 = pade_2(xs)
exp= np.exp(-1*xs)

plt.plot(xs,pade1,label='z1',color='#873B00')
plt.plot(xs,pade2,label='z2',color='#2EC4A6')
plt.plot(xs,exp,label='exp(-x)',color='#FA1313')
plt.title('Przybliżenia Padégo')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.savefig('/Users/wojtek/Desktop/Metodynumeryczne/wszystkie.png')
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(xs,pade1,label='z1',linewidth=1.5)
plt.plot(xs,exp,label='exp(-x)',linewidth=1)
plt.title('Przybliżenia Padégo dla z1')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(xs,pade2,label='z2',linewidth=1.5)
plt.plot(xs,exp,label='exp(-x)',linewidth=1)
plt.title('Przybliżenia Padégo dla z2')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.savefig('/Users/wojtek/Desktop/Metodynumeryczne/laczny.png')
plt.show()

blad1 = abs(exp-pade1)
blad2 = abs(exp-pade2)

plt.plot(xs, blad1, label='błąd z1',linestyle='dotted')
plt.plot(xs,blad2,label='błąd z2',linestyle='dashed')
plt.title('Błąd przybliżeń z1 i z2')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.savefig('/Users/wojtek/Desktop/Metodynumeryczne/blad.png')
plt.show()