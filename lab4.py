import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


a = 10
e = 3
l = 30

# $\tau \leq \frac{h}{a}$
#

# In[5]:


h, c = (0.05, 0.1)
tau = c * h / a

# In[6]:


x_array = np.arange(0, l + h, h)
t_array = np.arange(0,  6 + tau, tau)


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


def phi(x):
    return phi4(x, 4, e)


def mu1(t):
    return 0


def mu2(t):
    return 0



def F_for_real_solution(x):
    return np.sign((x + l) % (2 * l) - l) * phi( np.abs((x + l) % (2 * l) - l))


def u_solution(x, t):
    return (F_for_real_solution(x - a * t) + F_for_real_solution(x + a * t)) / 2


def u_numerical(u0, u0_next, t):
    global u0_global
    global u0_global_2

    u0_global_next = ((a ** 2) * (tau ** 2) * (u0_next[2:] - 2 * u0_next[1:-1] + u0_next[:-2]) / (h ** 2) +
                      + 2 * u0_next[1:-1] - u0[1:-1])
    u0_global = np.copy(u0_next)
    u0_global_2[0] = mu1(t)
    u0_global_2[-1] = mu2(t)
    u0_global_2[1:-1] = np.copy(u0_global_next)

    return u0_global_2


def init():
    axis.set_xlim(-l, l)
    axis.set_ylim(-2, 2)
    return line1, line2


def update(frame):

    y2 = u_solution(x_array, frame)
    y1 = u_numerical(u0_global, u0_global_2, frame)

    line1.set_data(x_array, y1)
    line2.set_data(x_array, y2)

    return line1, line2




if __name__ == "__main__":

    u0_global = phi(x_array)
    u0_global_2 = np.copy(u0_global)
    fig, axis = plt.subplots()
    plt.grid()
    fig.set_figheight(10)
    fig.set_figwidth(10)

    line1, = axis.plot([], [], label='numerical')
    line2, = axis.plot([], [], label='real')

    ani = FuncAnimation(fig, update, frames=t_array,
                        init_func=init, blit=True, repeat=False, interval=1)

    plt.legend()
    plt.show()