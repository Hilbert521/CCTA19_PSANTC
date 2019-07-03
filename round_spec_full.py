import numpy as np
from round import *


class ToUnicycle(SingleUnicycle):
    '''
    unicycle control system
    '''
    def __init__(self, x, system):
        SingleUnicycle.__init__(self, x)
        self.system = system
        self.v = 0

    def u(self):
        self.system.x = np.array([self.x[0], self.x[1], self.v, self.x[2]])
        inputs = self.system.u()
        self.v += inputs[0]*self.dt
        return np.array([self.v, inputs[1]]).T


class CarDrawer(DrawSystem, NetworkSystem):
    def __init__(self, x):
        DrawSystem.__init__(self, x)

    def draw_setup(self, axes):
        car = plt.imread('Car.png')
        self.drawings = [axes.imshow(car, animated=True)]
        self.drawings[0].set_transform(axes.transData)

    def draw_update(self, axes):
        tr = transforms.Affine2D().scale(.00015).rotate(self.x[3] + np.pi/2).translate(self.x[0], self.x[1])
        self.drawings[0].set_transform(tr + axes.transData)

    def input_cons(self):
        Ca = np.vstack((np.identity(2*(len(self.sys_list)+1), dtype='float32'), -np.identity(2*(len(self.sys_list)+1), dtype='float32'))).T
        ba = -np.ones((4*(len(self.sys_list) + 1),), dtype='float32')
        return (Ca, ba)


class Round_Testing(CoupleCBF, Round_Uni, CarDrawer):
    def __init__(self, x, yin, yout, h, gamma, ach, ch=None):
        CoupleCBF.__init__(self, x, h=h, gamma=gamma, ch=ch, a=lambda x: x, ach=ach)
        Round_Uni.__init__(self, x, yin, yout)
        CarDrawer.__init__(self, x)

    def nominal(self):
        return Round_Uni.nominal(self)

    def input_cons(self):
        return CarDrawer.input_cons(self)


class InRound(CoupleCBF, Round_Uni, CarDrawer):
    def __init__(self, x, yout, gamma):
        CoupleCBF.__init__(self, x, h=None, gamma=gamma, ch=None)
        Round_Uni.__init__(self, x, np.array([1, 0]), yout)
        CarDrawer.__init__(self, x)
        self.pd = .67
        self.k1 = 0
        self.k2 = .1
        self.vd = .08
        self.correct = np.array([0, -.03])
        self.G = np.identity(2*(len(self.sys_list)+1))

    def nominal(self):
        return Round_Uni.nominal(self)

    def u(self):
        return Round_Uni.nominal(self)

    def input_cons(self):
        return CarDrawer.input_cons(self)


def ch1(k, l):
    ds = .15
    a = 1

    return ((k[1] - k[2]*np.cos(k[3])/a) - (l[1] - l[2]*np.cos(l[3])/a))**2 + \
           ((k[0] + k[2]*np.sin(k[3])/a) - (l[0] + l[2]*np.sin(l[3])/a))**2 - (2*ds + k[2]/a + l[2]/a)**2


def ch2(k, l):
    ds = .15
    a = 1

    return ((k[1] + k[2]**2*np.sin(k[3])/4/a) - (l[1] + l[2]**2*np.sin(k[3])/4/a))**2 + \
           ((k[0] + k[2]**2*np.sin(k[3])/4/a) - (l[0] + l[2]**2*np.sin(k[3])/4/a))**2 - (2*ds + k[2]**2/4/a + l[2]**2/4/a)**2


def ch3(k, l):
    ds = .15
    a = 1

    return ((k[1] + k[2]**2*np.sin(k[3])/4/a) - (l[1] + l[2]**2*np.sin(k[3])/4/a))**2 + \
           ((k[0] + k[2]**2*np.sin(k[3])/4/a) - (l[0] + l[2]**2*np.sin(k[3])/4/a))**2 - (2*ds + k[2]**2/4/a + l[2]**2/4/a)**2


def in_h(rx, ry, rr):
    a = 1

    def h(x): return 10*((rr - x[2]/a)**2 - (x[0] + x[2]/a*np.sin(x[3]) - rx)**2 - (x[1] - x[2]/a*np.cos(x[3]) - ry)**2)
    return h


def in_h2(rx, ry, rr):
    a = 1

    def h(x): return 10*((rr - x[2]**2/4/a)**2 - (x[1] - ry + x[2]**2*np.sin(x[3])/4/a)**2 + - (x[0] - rx + x[2]**2*np.cos(x[3])/4/a)**2)
    return h


def in_h3(rx, ry, rr):
    a = 1

    def h(x): return 10*((rr - x[2]/a)**2 - (x[0] - x[2]/a*np.sin(x[3]) - rx)**2 - (x[1] + x[2]/a*np.cos(x[3]) - ry)**2)
    return h


def and_h(list_h, b=1):
    def h(x):
        sumexp = sum([np.exp(-b*h_i(x)) for h_i in list_h])
        if sumexp > 0:
            return -1/b*log(sumexp)
        else:
            return -1000000
    return h


def or_h(list_h, a=1):
    def h(x): return sum([h_i(x)*np.exp(a*h_i(x)) for h_i in list_h])/sum([np.exp(a*h_i(x)) for h_i in list_h])
    return h

def or_ch(list_ch, a=1):
    def ch(k, l): return sum([ch_i(k, l)*np.exp(a*ch_i(k, l)) for ch_i in list_ch])/sum([np.exp(a*ch_i(k, l)) for ch_i in list_ch])
    return ch


if __name__ == '__main__':

    # gamma
    def gamma(x): return [0, -1]
    
    yc = [[k, -0.1] for k in np.linspace(-1.2, -.7, 8)]
    xc = [[0.1, k] for k in np.linspace(1, .7, 4)]
    rc = [[.7*np.cos(t), .7*np.sin(t)] for t in np.linspace(-np.pi, np.pi/2, 40)]
    pts = np.concatenate((yc, xc, rc))


    # direction of roundabout
    yin = np.array([-1, 0])
    yout = np.array([0, 1])
    yout2 = np.array([1, 0])

    rad = .11
    h1 = or_h([in_h(p[0], p[1], rad) for p in pts], a=100)
    h2 = or_h([in_h3(p[0], p[1], rad) for p in pts], a=100)
    h = or_h([h1, h2], a=100)

    ch = or_ch([ch1])

    # initial positions for model
    st1 = np.array([-1.1, -.1, 0., 0.])
    st2 = np.array([-.7, .13, 0., -2*np.pi/3])
    st3 = np.array([-.3, .65, 0., -5*np.pi/6])

    sys1 = Round_Testing(st1, yin, yout, h, gamma, ch1)
    sys2 = InRound(st2, yout2, gamma)
    sys3 = InRound(st3, yout2, gamma)
    sys1.sys_list.append(sys2)
    sys1.sys_list.append(sys3)
    sys1.G = np.identity(np.shape(sys1.g())[1]*(len(sys1.sys_list) + 1))

    sys_list = [sys1, sys2, sys3]
    animator = Animate(sys_list, xlim=[-1.5, 1.5], ylim=[-1, 1])
    Roundabout = plt.imread('Roundabout.png')
    scale = 1
    animator.axes.imshow(Roundabout, extent=[scale*-1.5, scale*1.5, scale*-1, scale*1], zorder=0)

    for pt in pts:
        animator.axes.add_patch(patches.Circle(pt, rad, fill=False))
    animator.animate()