2#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


a = 10
e = 3
l = 30

# $\tau \leq \frac{h}{a}$
#

# In[5]:


h, c = (0.05, 0.5)
tau = c * h / a

# In[6]:


x_array = np.arange(-l, l + h, h)
t_array = np.arange(0, 6 + tau, tau)


# In[7]:


def xi(x, x0, e):
    return np.abs(x - x0) / e


def phi1(x, x0, e):
    return np.heaviside(1 - xi(x, x0, e), 1)


def phi2(x, x0, e):
    return phi1(x, x0, e) * (1 - xi(x, x0, e))


def phi3(x, x0, e):
    return phi1(x, x0, e) * np.exp(-xi(x, x0, e) ** 2 / np.abs(1 - xi(x, x0, e)))


def phi4(x, x0, e):
    return phi1(x, x0, e) * (np.cos(np.pi * xi(x, x0, e) / 2) ** 3)


# In[8]:


def phi(x):
    return phi1(x, 0, e)


def mu1(t):
    return 0


def mu2(t):
    return 0


u = np.empty_like(x_array)


def u_solution(x, t):
    boolean_array_inside = np.logical_and(x + a * t <= l,  x - a * t >= -l)
    boolean_array_outside = np.logical_and(x + a * t > l,  x - a * t < -l)

    u[boolean_array_inside] = (phi(x - a * t) + phi(x + a * t))[boolean_array_inside] / 2
    u[x + a * t > l] = (phi(x - a * t) - phi(2 * l - x - a * t))[x + a * t > l] / 2
    u[x - a * t < -l] = (phi(x + a * t) - phi(-2 * l - x + a * t))[x - a * t < -l] / 2
    u[boolean_array_outside] = (- phi(2 * l - x - a * t) - phi(-2 * l - x + a * t))[boolean_array_outside] / 2

    return u


def psi_numerical(u0, psi0):
    global psi0_global
    temp = np.array(-a * u0[-2] / h)

    psi0_global[:-1] = a * (psi0[1:] - psi0[:-1]) / h * tau + psi0[:-1]
    psi0_global[-1] = temp

    return psi0_global


def u_numerical(u0, psi0, t):
    global u0_global
    u0_global[0] = np.array(mu1(t))
    u0_global[1:] = (psi0[1:] - a * (u0[1:] - u0[:-1]) / h) * tau + u0[1:]
    psi_numerical(u0, psi0)
    return u0_global


u0_global = phi(x_array)

right_value = np.array((mu2(tau) - mu2(0)) / tau + a * (phi(l) - phi(l - h)) / h)
psi0_global = np.hstack((a * (phi(x_array[:-1] + h) - phi(x_array[:-1])) / h, right_value))


fig, axis = plt.subplots()
plt.grid()
fig.set_figheight(10)
fig.set_figwidth(10)
line1, = axis.plot([], [], label='numerical')
line2, = axis.plot([], [], label='real')


def init():
    axis.set_xlim(-l, l)
    axis.set_ylim(-2, 2)
    return line1, line2


def update(frame):

    y2 = u_solution(x_array, frame)
    y1 = u_numerical(u0_global, psi0_global, frame)

    line1.set_data(x_array, y1)
    line2.set_data(x_array, y2)

    return line1, line2


ani = FuncAnimation(fig, update, frames=t_array,
                    init_func=init, blit=True, repeat=False, interval=1)

plt.legend()
plt.show()
