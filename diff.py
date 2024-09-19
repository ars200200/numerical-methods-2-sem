import numpy as np
import matplotlib.pyplot as plt
from lab2_real import mu, phi, u_additional, u_solution

a = 10
l = 50
c = 0.5

def u_additional(u0):
    return (u0[1:] + u0[:-1]) / 2 - a / h * (u0[1:] - u0[:-1]) * tau / 2


def u_solution(x, t):
    return phi(x - a * t) * np.heaviside(x - a * t, 1) + mu(t - x / a) * np.heaviside(a * t - x, 0)


def u_numerical(u0, t):
    u0_additional = u_additional(u0)

    u0_global[0] = mu(t)
    u0_global[1:-1] = u0[1:-1] - a * (u0_additional[1:] - u0_additional[:-1]) / h * tau
    u0_global[-1] = - a * (u0[-1] - u0[-2]) / h * tau + u0[-1]

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
        u0_global = u_numerical(u0_global, t)
        max_diff[counter] = max(max_diff[counter], np.max(np.abs(u0_global - u_solution(x_array, t))))

    counter += 1


print(max_diff)
plt.loglog(h_array, max_diff, marker='x')
plt.grid()

plt.show()
