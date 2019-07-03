import numpy as np
from basic_systems import *
from CBF_systems import *
from animator import *


class Tester(FeasibleInfimum, DoubleUnicycle):
    """docstring for Tester"""
    def __init__(self, x, p, gamma, a, dirc, Tlim):
        FeasibleInfimum.__init__(self, x, p, gamma, a, Tlim)
        self.dirc = dirc
        self.simdt = .001

    def nominal(self):
        u1 = 10*(1 - self.x[2])
        #u1 = 1
        u2 = 1*(np.sin((self.dirc - self.x[3])))
        return np.array([u1, u2])

    def input_cons(self):
        Ca = np.vstack((np.identity(2, dtype='float32'), -np.identity(2, dtype='float32'))).T
        ba = -np.ones((4,), dtype='float32')

        #Ca = np.vstack((np.identity(2*len(self.sys_list)+ 2, dtype='float32'), -np.identity(2*len(self.sys_list)+ 2, dtype='float32'))).T
        #ba = -np.ones((4*len(self.sys_list)+ 4,), dtype='float32')
        #self.G = np.identity(np.shape(self.g())[1]*(len(self.sys_list)+1))
        return (Ca, ba)

    # dynamics of control system
    def flow(self):
        u = self.u()
        #print(u)
        return self.f() + self.g() @ u

    #def u(self):
    #    return np.array([0, -1])


def h(x): return (x[0] + x[2]*np.sin(x[3]))**2 + (x[1] - x[2]*np.cos(x[3]))**2 - (1 + .1 + x[2])**2

#def h(x): return x[0]**2 + x[1]**2 - 1


#def ch(k, l): return (k[0] - l[0])**2 + (k[1] - l[1])**2 - 1

def ch(k, l):
    ds = .07
    return ((k[1] - k[2]*np.cos(k[3])) - (l[1] - l[2]*np.cos(l[3])))**2 + \
       ((k[0] + k[2]*np.sin(k[3])) - (l[0] + l[2]*np.sin(l[3])))**2 - (2*ds + k[2] + l[2])**2

def ach(ch): return ch

def a(h): return h

def p(x): return x[0]**2 + x[1]**2 - 1

def gamma(x): return [0, -1]

if __name__ == '__main__':
    # set up models
    dt = 1/10

    u1 = Tester(np.array([-3.0, 0, 1, 0]), p, gamma, a, 0, 5)
    #u2 = Tester(np.array([10, -2, 0, np.pi]), ch, h, a, ach, np.pi)
    #u3 = Tester(np.array([0, -3, 0, np.pi/2]), ch, h, a, None)
    u_list_ = [u1]
    #u1.sys_list = [u2]
    #u2.sys_list = [u1]
    #u3.sys_list = []

    #u1.step()
    #u3.step()
    anim = Animate_Uni(u_list_, True)
    anim.ax.add_artist(plt.Circle((0, 0), 1, fill=False))
    anim.animate_sys()