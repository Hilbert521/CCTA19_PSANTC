import sys
from systems import *
from animate import *
import matplotlib.transforms as transforms
from CBF_systems import *


class Round_Uni(DoubleUnicycle):
    '''
    Unicycle with field roundabout controller
    '''
    def __init__(self, x, yin, yout):
        DoubleUnicycle.__init__(self, x)
        self.yin = yin
        self.yout = yout
        self.pd = .7
        self.k1 = .2
        self.k2 = .2
        self.k3 = 1
        self.vd = .1
        self.p1 = .1
        self.p2 = 5
        self.correct = np.array([.03, 0])

    def nominal(self):
        pd = self.pd
        i_ = self.yin
        o_ = self.yout
        pos = self.x[0:2]
        vd = self.vd
        p1 = self.p1
        p2 = self.p2
        correct = self.correct
        ang_vec = np.array([np.cos(self.x[3]), np.sin(self.x[3])]).T

        off = (np.pi/2/pd)*(n2(pos) - pd)
        if off > np.pi/2:
            t_vec = -pos
        if ang(i_, pos) == 0:
            t_vec = -pos
        elif ang(o_, pos) == 0:
            t_vec = pos
        else:
            t_vec = -self.k1/ang(i_, pos)*pos + self.k2/ang(o_, pos - correct)*pos + \
                    self.k3*np.dot(R((np.pi/2 + off)), pos)
        td = np.arctan2(t_vec[1], t_vec[0])

        u1 = p1*(vd - self.x[2])
        u2 = p2*(np.sin((td - self.x[3])))

        return np.array([u1, u2])

    def u(self):
        return self.nominal()

    def draw_setup(self, axes):
        car = plt.imread('Car.png')
        self.drawings = [axes.imshow(car, animated=True)]
        self.drawings[0].set_transform(axes.transData)

    def draw_update(self, axes):
        tr = transforms.Affine2D().scale(.0002).rotate(self.x[3] + np.pi/2).translate(self.x[0], self.x[1])
        self.drawings[0].set_transform(tr + axes.transData)

if __name__ == '__main__':
    yin = np.array([-1, 0])
    yout = np.array([0, 1])
    sys = Round_Uni(np.array([-1.2, -.03, 0., 0.]), yin, yout)
    sys_list = [sys]
    animator = Animate(sys_list, xlim=[-1.5, 1.5], ylim=[-1, 1])
    Roundabout = plt.imread('Roundabout.PNG')
    scale = 1
    animator.axes.imshow(Roundabout, extent=[scale*-1.5, scale*1.5, scale*-1, scale*1], zorder=0)
    animator.animate()