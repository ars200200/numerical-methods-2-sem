import numpy as np
import matplotlib.pyplot as plt
from lab3_voln import mu1, mu2, phi,  minmod

a = 10
l = 50
c = 0.5



def u_solution(x, t):
    boolean_array_inside = np.logical_and(x + a * t <= l,  x - a * t >= -l)
    boolean_array_outside = np.logical_and(x + a * t > l,  x - a * t < -l)

    u[boolean_array_inside] = (phi(x - a * t) + phi(x + a * t))[boolean_array_inside] / 2
    u[x + a * t > l] = (phi(x - a * t) - phi(2 * l - x - a * t))[x + a * t > l] / 2
    u[x - a * t < -l] = (phi(x + a * t) - phi(-2 * l - x + a * t))[x - a * t < -l] / 2
    u[boolean_array_outside] = (- phi(2 * l - x - a * t) - phi(-2 * l - x + a * t))[boolean_array_outside] / 2

    return u

def psi_f_upwind(psi0):
    return -a * psi0[1:]


def psi_f_lax(psi0, tau, h):
    alpha = -a * tau / h
    return -a * (psi0[:-1] * (1 / 2 + alpha / 2) + psi0[1:] * (1 / 2 - alpha / 2))


def u_f_upwind(u0):
    return a * u0[:-1]


def u_f_lax(u0, tau, h):
    alpha = a * tau / h
    return a * (u0[:-1] * (1 / 2 + alpha / 2) + u0[1:] * (1 / 2 - alpha / 2))

def psi_f_all(psi0, tau, h):
    r = (psi0[2:] - psi0[1:-1]) / (psi0[1:-1] - psi0[:-2] + h/10**8)
    upw = psi_f_upwind(psi0)
    lax = psi_f_lax(psi0, tau, h)
    return upw[1:] + minmod(r) * (lax[1:] - upw[1:])


def u_f_all(u0, tau, h):
    r = (u0[1:-1] - u0[:-2]) / (u0[2:] - u0[1:-1] + h/10**8)
    upw = u_f_upwind(u0)
    lax = u_f_lax(u0, tau, h)
    return upw[1:] + minmod(r) * (lax[1:] - upw[1:])


def psi_numerical(u0, psi0, tau, h):
    global psi0_global
    psifall = psi_f_all(psi0, tau, h)
    temp = np.array(-a * u0[-2] / h)
    psi0_global[-1] = temp
    psi0_global[2:-1] = - (psifall[1:] - psifall[:-1]) / h * tau + psi0[2:-1]
    psi0_global[0] = a * (psi0[1] - psi0[0]) / h * tau + psi0[0]
    psi0_global[1] = a * (psi0[2] - psi0[1]) / h * tau + psi0[1]

    return psi0_global


def u_numerical(u0, psi0, t, tau, h):
    u0_glob = np.empty_like(x_array)
    ufall = u_f_all(u0, tau, h)
    global u0_global

    u0_glob[0] = mu1(t)
    u0_glob[2:-1] = - (ufall[1:] - ufall[:-1]) / h * tau + u0[2:-1] + psi0[2:-1] * tau
    u0_glob[-1] = mu2(t)
    u0_glob[1] = - a * (u0[1] - u0[0]) / h * tau + u0[1] + psi0[1] * tau

    psi_numerical(u0, psi0, tau, h)
    u0_global = np.copy(u0_glob)
    return u0_global





h_array = np.array([2 ** -i for i in range(1, 7)])
max_diff = np.zeros_like(h_array)
counter = 0
for h in h_array:
    tau = c * h / a
    x_array = np.arange(-l, l + h, h)
    t_array = np.arange(0, 5 + tau, tau)
    u = np.empty_like(x_array)
    u0_global = phi(x_array)
    right_value = np.array((mu2(tau) - mu2(0)) / tau + a * (phi(l) - phi(l - h)) / h)
    psi0_global = np.hstack((a * (phi(x_array[:-1] + h) - phi(x_array[:-1])) / h, right_value))

    for t in t_array[1:]:
        u0_global = u_numerical(u0_global, psi0_global, t, tau, h)
        max_diff[counter] = max(max_diff[counter], np.max(np.abs(u0_global - u_solution(x_array, t))))


    counter += 1


print(max_diff)
plt.loglog(h_array, max_diff, marker='x')
plt.grid()

plt.show()
