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

class Animator:
    def __init__(self, wave_packet):
        self.time = 0.0
        self.wave_packet = wave_packet
        self.fig, self.ax = plt.subplots()
        plt.plot(self.wave_packet.x, self.wave_packet.potential * 0.1, color='r')

        self.time_text = self.ax.text(0.05, 0.95, '', horizontalalignment='left',
                                      verticalalignment='top', transform=self.ax.transAxes)
        self.line, = self.ax.plot(self.wave_packet.x, self.wave_packet.evolve())
        self.ax.set_ylim(0, 0.2)
        self.ax.set_xlim(wave_packet.xi, wave_packet.xf)
        self.ax.set_xlabel('Position (a$_0$)')
        self.ax.set_ylabel('Probability density (a$_0$)')

    def update(self, data):
        self.line.set_ydata(data)
        return self.line,

    def time_step(self):
        while True:
            self.time += self.wave_packet.dt
            self.time_text.set_text(
                    'Elapsed time: {:6.2f} fs'.format(self.time * 2.419e-2))

            yield self.wave_packet.evolve()

    def animate(self):
        self.ani = FuncAnimation(self.fig, self.update, self.time_step, interval=5, blit=False)


w = WavePacket(n_points = 500, dt = 0.5, b_width =5, b_height = 1)
ani = Animator(w)
ani.animate()
plt.show()
