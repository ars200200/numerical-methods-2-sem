import numpy as np
import matplotlib.pyplot as plt
from lab3 import mu, phi,  minmod

a = 10
l = 50
c = 0.5


def u_solution(x, t):
    return phi(x - a * t) * np.heaviside(x - a * t, 1) + mu(t - x / a) * np.heaviside(a * t - x, 0)

def f_upwind(u0):
    return a * u0[:-1]


def f_lax(u0):

    return a * (u0[:-1] * (1 / 2 + c / 2) + u0[1:] * (1 / 2 - c / 2))


def f_all(u0):
    r = (u0[1:-1] - u0[:-2]) / (u0[2:] - u0[1:-1] + h/10**8)

    upw = f_upwind(u0)
    lax = f_lax(u0)
    return upw[1:] + minmod(r) * (lax[1:] - upw[1:])

def u_numerical(u0, t, h, tau):
    fall = f_all(u0)
    global u0_global
    u0_global[0] = mu(t)
    u0_global[2:-1] = - (fall[1:] - fall[:-1]) / h * tau + u0[2:-1]
    u0_global[-1] = - a * (u0[-1] - u0[-2]) / h * tau + u0[-1]
    u0_global[1] = - a * (u0[1] - u0[0]) / h * tau + u0[1]
    return u0_global





h_array = np.array([0.01 * i for i in range(1, 10)])
max_diff = np.zeros_like(h_array)
counter = 0
for h in h_array:
    tau = c * h / a
    x_array = np.arange(0, l + h, h)
    t_array = np.arange(0, 5 + tau, tau)
    u0_global = phi(x_array)

    for t in t_array[1:]:
        u0_global = u_numerical(u0_global, t, h, tau)
        max_diff[counter] = max(max_diff[counter], np.max(np.abs(u0_global - u_solution(x_array, t))))

    counter += 1


print(max_diff)
plt.loglog(h_array, max_diff, marker='x')
plt.grid()

plt.show()
