import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

from round_spec_full import *

import numpy as np
from time import time
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import os
import pickle

#os.rename('Round.jpeg', 'Round.png')


def init_pose(initial_pose):
    # define x initially
    x = r.get_poses()
    r.step()

    # Define goal points by removing orientation from poses
    goal_points = initial_pose

    # Get barrier certificates to avoid collisions
    uni_barrier_cert = create_unicycle_barrier_certificate(N, safety_radius=0.01)

    # While the number of robots at the required poses is less
    # than N...
    while(np.size(at_pose(x, goal_points, position_error=0.02, rotation_error=.1)) != N):

        # Get poses of agents
        x = r.get_poses()

        # Unicycle control inputs
        dxu = unicycle_pose_controller(x, goal_points)

        # Create safe input s
        dxu = uni_barrier_cert(dxu, x)

        r.set_velocities(np.arange(N), dxu)
        r.step()


def set_v(unicycles):
    inputs = np.array([uni.u() for uni in unicycles])
    uni_barrier_cert = create_unicycle_barrier_certificate(1, safety_radius=0.01)

    x = r.get_poses()
    for i in range(len(unicycles)):
        unicycles[i].x = x[:, i]

    if x[1, 0] > .95:
        inputs[0, :] = [0, 0]
    if x[0, 1] > 1.5:
        inputs[1, :] = [0, 0]
    if x[0, 2] > 1.35:
        inputs[2, :] = [0, 0]

    r.set_velocities(np.arange(N), inputs.T)


def run_roundabout(unicycles):
    array = []
    for i in range(2000):
        set_v(unicycles)
        r.step()
        array.append(r.get_poses().copy())
    return array


if __name__ == '__main__':
    # Instantiate Robotarium object
    N = 3
    dt = .1
    r = robotarium.Robotarium(number_of_agents=N, show_figure=True, save_data=False, update_time=dt)


    # circles
    yc = [[k, -0.1] for k in np.linspace(-1.2, -.7, 8)]
    xc = [[0.1, k] for k in np.linspace(1, .7, 4)]
    rc = [[.7*np.cos(t), .7*np.sin(t)] for t in np.linspace(-np.pi, np.pi/2, 40)]
    pts = np.concatenate((yc, xc, rc))
    #pts = np.array([[-1.2, -.1]])

    # direction of roundabout
    yin = np.array([-1, 0])
    yout = np.array([0, 1])
    yout2 = np.array([1, 0])

    # barrier
    rad = .11
    h1 = or_h([in_h2(p[0], p[1], rad) for p in pts], a=100)
    h2 = or_h([in_h(p[0], p[1], rad) for p in pts], a=100)
    h = or_h([h1, h2], a=100)

    # gamma
    def gamma(x): return [0, -1]

    # initial positions for model
    st1 = np.array([-1.2, -.03, 0., 0.])
    st2 = np.array([-.7, .13, 0., -2*np.pi/3])
    st3 = np.array([-.3, .65, 0., -5*np.pi/6])

    u1 = Round_Testing(st1, yin, yout, h, gamma, ch1)
    u2 = InRound(st2, None, yout2, gamma, None)
    u3 = InRound(st3, None, yout2, gamma, None)
    u1.u_list.append(u2)
    u1.u_list.append(u3)

    uni1 = ToUnicycle(st1[[0, 1, 3]], u1)
    uni2 = ToUnicycle(st1[[0, 1, 3]], u2)
    uni3 = ToUnicycle(st1[[0, 1, 3]], u3)
    unicycles = [uni1, uni2, uni3]

    img = plt.imread('Roundabount.PNG')
    scale = 1
    #r.axes.imshow(img, extent=[scale*-1.5, scale*1.5, scale*-1, scale*1], zorder=-1)

    # initial positions
    pos1 = st1[[0, 1, 3]]
    pos2 = st2[[0, 1, 3]]
    pos3 = st3[[0, 1, 3]]
    initial_pose = np.vstack((pos1, pos2, pos3)).T


    # set initial positions
    init_pose(initial_pose)

    # run experiment
    positions = run_roundabout(unicycles)

    # Always call this function at the end of your scripts!  It will accelerate the
    # execution of your experiment
    r.call_at_scripts_end()
