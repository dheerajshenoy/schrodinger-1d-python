from matplotlib import animation
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from scipy.linalg import eigh_tridiagonal
from matplotlib.animation import FuncAnimation

class WavePacket:
    """
        grid_points: number of data points
        x0: center of the wave packet
        s0: standard deviation of the gaussian wave packet
        k0: wave number of the wave packet
        xi: x_lim
        xf: x_lim
        b_width: barrier width
        b_height: barrier height
    """
    def __init__(self, n_points, dt, x0 = -150.0, s0 = 5.0, k0 = 1.0, xi = -200.0,
                 xf = 200.0, b_width = 3.0, b_height = 1.0):
        self.n = n_points
        self.dt = dt
        self.x0 = x0
        self.s0 = s0
        self.k0 = k0
        self.xi = xi
        self.xf = xf
        self.b_width = b_width
        self.b_height = b_height

        self.x, self.dx = np.linspace(xi, xf, self.n, retstep=True)

        self.potential = np.array([b_height if 0.0 < x < b_width else 0.0 for x in self.x])

        norm = (2 * np.pi * s0**2)**(1/4)
        self.psi = norm * np.exp(-(self.x - self.x0)**2 /(4.0 * self.s0**2)) * np.exp(1.0j * k0 * self.x)

        h_diag = np.ones(n_points)/self.dx**2 + self.potential
        h_non_diag = np.ones(n_points - 1) * (-0.5 /self.dx**2)
        hamiltonian = sp.sparse.diags([h_diag, h_non_diag, h_non_diag], [0, 1, -1])

        imp = (sp.sparse.eye(n_points) - dt/2.0j * hamiltonian).tocsc()
        exp = (sp.sparse.eye(n_points) + dt/2.0j * hamiltonian).tocsc()
        self.evolution_matrix = sp.sparse.linalg.inv(imp).dot(exp).tocsr()

    def evolve(self):
        self.psi = self.evolution_matrix.dot(self.psi)
        self.prob = abs(self.psi)**2

        norm = sum(self.prob)
        self.prob /= norm
        self.psi /= norm**0.5

        return self.prob


w = WavePacket(n_points = 500, dt = 0.5, b_width =5, b_height = 1)

fig = plt.figure()
ax = plt.axes()
l, = plt.plot(w.x, w.potential * 0.1)
p, = plt.plot(w.x, w.evolve())

def animate(i):
    p.set_data(w.x, w.evolve())
    return p,


anim = FuncAnimation(fig, animate, frames = 1000, interval = 40)
anim.save("schrodinger-wave-packet.mp4")
plt.show()
