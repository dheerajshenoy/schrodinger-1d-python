import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.constants import hbar as hbar_actual

hbar = 1
L = 1
N = 1000
dx = L/N
m = 1
x = np.linspace(0, L, N + 1)

def V(x):
    return np.heaviside(x, 1)

pot = V(x)

diag = 2 * (1/dx**2 + m * V(x))[1: -1]
e = -1/(dx**2) * np.ones(len(diag) - 1)

E, PHI = eigh_tridiagonal(diag, e)
hbar = hbar_actual

plt.plot(x, V(x), 'b-')
for i in range(10):
    plt.plot(x[1:-1], PHI.T[i])
#plt.bar(np.arange(0, 10, 1), E[0: 10]/hbar)

#plt.plot(x, pot)
plt.show()
