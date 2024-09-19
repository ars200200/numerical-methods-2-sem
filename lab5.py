import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Planck
from matplotlib.animation import ArtistAnimation


l = 10
T = 12

e = 2
x0 = 2.1

m = 1e-32
k = 10

sigma = 0.7
V0 = 1e-33

x1 = 4.5
x2 = 5.5

h = 0.01
tau = 0.05


x_array = np.arange(0, l + h, h)
t_array = np.arange(0,  T + tau, tau)
i = complex(0, 1)


def mu1(t):
    return 0


def mu2(t):
    return 0


def phi(x):
    return np.heaviside(e - np.abs(x - x0), 1) * np.cos(np.pi * (x - x0) / (2 * e)) ** 3 * np.exp(i * k * x)


def v(x):
    return V0 * np.heaviside(x - x1, 1) * np.heaviside(x2 - x, 1)


def thomas_algorithm(psi0):
    b[1] = 1 / tau + i * sigma * Planck / (m * h ** 2) + v(x_array[1]) * i * sigma / Planck
    c[1] = -i * Planck * sigma / (2 * m * h ** 2)
    f[1] = psi0[1] / tau + i * (1 - sigma) * (Planck / (2 * m) * (psi0[2] - 2 * psi0[1] + psi0[0]) / h ** 2 - v(x_array[1]) / Planck * psi0[1])

    B[1] = f[1] / b[1]
    A[1] = -c[1] / b[1]

    for j in range(2, len(x_array) - 1):
        a[j] = -i * Planck * sigma / (2 * m * h ** 2)
        c[j] = -i * Planck * sigma / (2 * m * h ** 2)
        b[j] = 1 / tau + i * sigma * Planck / (m * h ** 2) + v(x_array[j]) * i * sigma / Planck

        f[j] = psi0[j] / tau + i * (1 - sigma) * (Planck / (2 * m) * (psi0[j + 1] - 2 * psi0[j] + psi0[j - 1]) / h ** 2 - v(x_array[j]) / Planck * psi0[j])

        B[j] = (f[j] - a[j] * B[j - 1]) / (b[j] + a[j] * A[j - 1])
        A[j] = - c[j] / (b[j] + a[j] * A[j - 1])

    psi_new[-2] = B[-2]

    for j in range(len(x_array) - 3, 0, -1):
        psi_new[j] = B[j] + A[j] * psi_new[j + 1]

    psi_new[0] = 0
    psi_new[-1] = 0
    return psi_new


def psi_solution(psi0):
    global psi_global
    psi_global = thomas_algorithm(psi0)
    return psi_global


psi_global = phi(x_array)
psi_new = np.empty_like(x_array, dtype=np.cdouble)
a = np.empty_like(x_array, dtype=np.cdouble)
b = np.empty_like(x_array, dtype=np.cdouble)
c = np.empty_like(x_array, dtype=np.cdouble)
f = np.empty_like(x_array, dtype=np.cdouble)
A = np.empty_like(x_array, dtype=np.cdouble)
B = np.empty_like(x_array, dtype=np.cdouble)


frames = []
fig = plt.figure(figsize=[10, 6])
ax = fig.add_subplot(111, projection='3d')


for k in t_array:
    solution = psi_solution(psi_global)
    y = solution.real
    z = solution.imag
    x = x_array
    line = ax.plot(x, y, z, color='g')
    frames.append(line)


ani = ArtistAnimation(
    fig,
    frames,
    interval=100,
    blit=True,
    repeat=False)

v1 = np.linspace(-1, 1, 100)
v2 = np.linspace(-1, 1, 100)

V1, V2 = np.meshgrid(v1, v2)

surf = ax.plot_surface(x1, V1, V2, alpha=0.2)
surf2 = ax.plot_surface(x2, V1, V2, alpha=0.2)

plt.show()