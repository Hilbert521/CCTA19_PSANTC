import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from time import time
from datetime import datetime
from systems import *
import json


class Animate():
    '''
    Parent Class for animating systems
    '''
    def __init__(self, sys_list, size=(18, 9), xlim=[-10, 10], ylim=[-10, 10],
                 showfig=True, saveData=False, inter=False):
        self.sys_list = sys_list
        self.fig = plt.figure()  # figure handle
        self.size = size  # size of figure
        self.axes = None  # axes handle
        self.xlim = xlim  # x axis limits
        self.ylim = ylim  # y axis limits
        self.interval = 0  # interval to run in real time
        self.drawings = []  # drawings
        self.data = []  # state history
        self.time_text = None  # time bar
        self.grid_on = False  # turn on grid on plot
        self.saveData = saveData  # option of saving traces of animation
        self.showfig = showfig  # show Figure or not
        self.time = 0
        self.dt = self.sys_list[0].dt

        # setup axes
        self.setup()
        if inter:
            self.set_interval()

    # initialization function, sets objects for axes
    def init_axes(self):
        for sys in self.sys_list:
            sys.draw_setup(self.axes)
            self.drawings += sys.drawings
        for artist in self.drawings:
            self.axes.add_artist(artist)
        self.axes.images = []  # flush duplicate images

        self.time_text.set_text('')
        if self.saveData:
            current = np.hstack(tuple([sys.x for sys in self.sys_list]))
            self.data = current

        return (self.time_text,) + tuple(self.drawings)

    # update function
    def update_frame(self, i):
        for sys in self.sys_list:
            sys.step()
            sys.draw_update(self.axes)

        self.time_text.set_text('time = %.1f' % self.time)
        self.time += self.dt
        if self.saveData:
            current = np.hstack(tuple([sys.x for sys in self.sys_list]))
            self.data = np.vstack((self.data, current))

        return (self.time_text,) + tuple(self.drawings)

    # setup figure handles
    def setup(self):
        self.axes = self.fig.add_subplot(111, aspect='equal', autoscale_on=False,
                                         xlim=self.xlim, ylim=self.ylim)
        self.fig.set_size_inches(self.size)

        if self.grid_on:
            self.axes.grid()
        self.time_text = self.axes.text(0.02, 0.95, '', transform=self.axes.transAxes)

        self.init_axes()

    # set up interval for approximately real time simulation
    def set_interval(self):

        t0 = time()
        self.update_frame(0)
        t1 = time()
        a_time = 1000*self.dt - (t1 - t0)
        if a_time > 0:
            self.interval = a_time

    # animate function, if name specified, saves file
    def animate(self, name=None, frames=500):

        # run animation
        anim_handle = animation.FuncAnimation(self.fig, self.update_frame, frames=frames,
                                              interval=self.interval, blit=True)

        if name:
            file_name = str(name)+'.mp4'
            anim_handle.save(file_name, fps=30, extra_args=['-vcodec', 'h264',
                             '-pix_fmt', 'yuv420p'])
        elif self.showfig:
            plt.show()

        if self.saveData:
            fn = 'Simulation_at_' + ''.join(e for e in str(datetime.now())[:20] if e.isalnum())
            with open(fn, 'w') as fh:
                json.dump(self.data.tolist(), fh)
