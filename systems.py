import numpy as np


class ControlSystem:
    '''
    general control affine system
    '''
    def __init__(self, x, dt=.1, params=None,  t=0):
        self.x = x  # state
        self.t = t  # initial time
        self.dt = dt  # time step
        self.params = params  # define parameters

    # dynamics of control system
    def flow(self):
        return self.f() + self.g() @ self.u()

    # drift
    def f(self):
        raise NotImplementedError

    # affine term
    def g(self):
        raise NotImplementedError

    # input
    def u(self):
        raise NotImplementedError

    # euler approximation
    def step(self):
        self.x += self.flow()*self.dt
        self.t += self.dt


class SingleUnicycle(ControlSystem):
    '''
    unicycle control system with x, y, th
    '''
    def __init__(self, x):
        ControlSystem.__init__(self, x)

    # single unicycle dynamics
    def f(self):
        return np.array([0, 0, 0]).T

    def g(self):
        return np.array([[np.cos(self.x[2]), np.sin(self.x[2]), 0], [0, 0, 1]]).T

    # wrap theta
    def step(self):
        ControlSystem.step(self)
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))

    # 2 inputs: velocity, angular velocity
    def u(self):
        return np.array([0, 0])


class DoubleUnicycle(ControlSystem):
    '''
    unicycle control system with x, y, v, th
    '''
    def __init__(self, x):
        ControlSystem.__init__(self, x)

    # double unicycle dynamics
    def f(self):
        xdot = self.x[2]*np.cos(self.x[3])
        ydot = self.x[2]*np.sin(self.x[3])
        vdot = 0
        thdot = 0
        return np.array([xdot, ydot, vdot, thdot]).T

    def g(self):
        return np.array([[0, 0, 1, 0], [0, 0, 0, 1]]).T

    # wrap theta
    def step(self):
        ControlSystem.step(self)
        self.x[3] = np.arctan2(np.sin(self.x[3]), np.cos(self.x[3]))

    # 2 inputs: acceleration, angular velocity
    def u(self):
        return np.array([0, 0])


class NetworkSystem(ControlSystem):
    '''
    Parent class for systems with interactions
    '''
    def __init__(self, x, sys_list):
        ControlSystem.__init__(self, x)
        self.sys_list = sys_list


class DrawSystem(ControlSystem):
    '''
    Parent class for drawing simulations
    '''
    def __init__(self, x):
        ControlSystem.__init__(self, x)
        self.drawings = []

    # initiliaze drawings
    def draw_setup(self, axes=None):
        raise NotImplementedError

    # update drawings
    def draw_update(self, axes=None):
        raise NotImplementedError


# magnitude
def p(x, y): return (x**2 + y**2)**.5


# rotation matrix
def R(t):
    c, s = np.cos(t), np.sin(t)
    return np.array(((c, -s), (s, c)))


# norm
def n2(x): return np.dot(x, x)**.5


# angle between 2 vectors
def ang(x, y):
    if n2(x) == 0 or n2(y) == 0:
        return 0
    else:
        return np.arccos(np.dot(x, y)/n2(x)/n2(y))


# arccos with correct sign
def arccos2(px, py, p2):
    if py >= 0:
        return np.arccos(px/p2)
    else:
        return 2*np.pi - np.arccos(px/p2)