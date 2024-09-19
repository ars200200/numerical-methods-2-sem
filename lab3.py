import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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


# In[4]:


def phi(x):
    return phi4(x, 3, 3)


def mu(t):
    return phi4(t, 3, 1)


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


def minmod(r):
    return np.maximum(0, np.minimum(1, r))


def u_numerical(u0, t):
    fall = f_all(u0)
    global u0_global
    u0_global[0] = mu(t)
    u0_global[2:-1] = - (fall[1:] - fall[:-1]) / h * tau + u0[2:-1]
    u0_global[-1] = - a * (u0[-1] - u0[-2]) / h * tau + u0[-1]
    u0_global[1] = - a * (u0[1] - u0[0]) / h * tau + u0[1]
    return u0_global


def init():
    axis.set_xlim(0, l)
    axis.set_ylim(-1.5,  1.5)
    return line1, line2


def update(frame):
    y2 = u_solution(x_array, frame)
    y1 = u_numerical(u0_global, frame)

    line1.set_data(x_array, y1)
    line2.set_data(x_array, y2)


    return line1, line2


if __name__ == '__main__':
    a = 10
    e = 3
    l = 50



    h, c = (0.05, 0.5)
    tau = c * h / a
    x_array = np.arange(0, l + h, h)
    t_array = np.arange(0, 5 + tau, tau)
    u0_global = phi(x_array)
    f_wnd = np.empty(x_array.size - 2)
    fig, axis = plt.subplots()
    plt.grid()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    line1, = axis.plot([], [], label='numerical')
    line2, = axis.plot([], [], label='real')

    ani = FuncAnimation(fig, update, frames=t_array,
                        init_func=init, blit=True, repeat=False, interval=5)



    plt.legend()
    plt.show()