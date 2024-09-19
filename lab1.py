#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# In[2]:


a = 10  #float(input())

# $\tau \leq \frac{h}{a}$

h, c = (0.1, 0.1)
tao = c * h / a
l = 30
T = 3
x_array = np.arange(0, l + h, h)
t_array = np.arange(0, T + tao, tao)


def ksi(x, x0, e):
    return np.abs(x - x0) / e


def phi1(x, x0, e):
    return np.heaviside(1 - ksi(x, x0, e), 1)


def phi2(x, x0, e):
    return phi1(x, x0, e) * (1 - ksi(x, x0, e))


def phi3(x, x0, e):
    return phi1(x, x0, e) * np.exp(-ksi(x, x0, e) ** 2 / np.abs(1 - ksi(x, x0, e)))


def phi4(x, x0, e):
    return phi1(x, x0, e) * (np.cos(np.pi * ksi(x, x0, e) / 2) ** 3)


def phi(x):
    return phi1(x, 2, 1)


def mu(t):
    return phi1(t, 2, 1)


def psi(x, t):
    return 0


def u_solution(x, t):
    return phi(x - a * t) * np.heaviside(x - a * t, 1) + mu(t - x / a) * np.heaviside(a * t - x, 0)


def u_numerical(x, t):

    return np.hstack((np.array([mu(t)]), (psi(x, t) - (a * (u0[1:] - u0[:-1]) / h) * tao + u0[1:])))



def u_bad_approximation(x, t):
    return np.hstack((np.array([mu(t)]), (psi(x[1:-1], t) - (a * (u0_bad[2:] - u0_bad[:-2]) / h) * tao + u0_bad[1:-1]),
                     np.array(psi(x[-1], t) - (a * (u0_bad[-2] - u0_bad[-1]) / h) * tao + u0_bad[-1])))


u0 = phi(x_array)
u0_bad = phi(x_array)
fig, axis = plt.subplots()
fig.set_figheight(10)
fig.set_figwidth(10)
line1, = axis.plot([], [], label='numerical')
line2, = axis.plot([], [], label='real')
#line3, = axis.plot([], [], label='badapprox')


def init():
    axis.set_xlim(0, l)
    axis.set_ylim(0, 1.5)
    return line1, line2 #line3


def update(frame):
    y2 = u_solution(x_array, frame)
    y1 = u_numerical(x_array, frame)
    #y3 = u_bad_approximation(x_array, frame)
    line1.set_data(x_array, y1)
    line2.set_data(x_array, y2)
    #line3.set_data(x_array, y3)
    global u0
    global u0_bad
    u0 = np.copy(y1)
    #u0_bad = np.copy(y3)
    return line1, line2 #line3


ani = FuncAnimation(fig, update, frames=t_array,
                    init_func=init, blit=True, repeat=False, interval=5)

plt.legend()
plt.show()


