import time
import threading

import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

from gps.gui.config import common as gui_config
from gps.gui.action_axis import Action, ActionAxis
from gps.gui.output_axis import OutputAxis
from gps.gui.image_visualizer import ImageVisualizer
from gps.gui.real_time_plotter import RealTimePlotter
from gps.gui.mean_plotter import MeanPlotter
from gps.gui.plotter_3d import Plotter3D


fig = plt.figure()
gs = gridspec.GridSpec(2, 3)


# Action Axis
number = 0
def plus_1(event=None): global number; number += 1; print(number)
def plus_2(event=None): global number; number += 2; print(number)
def mult_4(event=None): global number; number *= 4; print(number)

actions_arr = [
    Action('plus1', 'plus1', plus_1, axis_pos=0, keyboard_binding='1', ps3_binding=None),
    Action('plus2', 'plus2', plus_2, axis_pos=1, keyboard_binding='2', ps3_binding=None),
    Action('print', 'print', mult_4, axis_pos=2, keyboard_binding='4', ps3_binding=None),
]
actions = {action._key: action for action in actions_arr}

action_axis = ActionAxis(fig, gs[0], 3, 1, actions,
        ps3_process_rate=gui_config['ps3_process_rate'],
        ps3_topic=gui_config['ps3_topic'],
        ps3_button=gui_config['ps3_button'],
        inverted_ps3_button=gui_config['inverted_ps3_button'])

# Output Axis
def demo_output_axis():
    max_i = 20
    for i in range(max_i):
        output_axis.append_text(str(i))
        c = 0.5 + 0.5*i/max_i
        output_axis.set_bgcolor((c, c, c))
        time.sleep(1)

output_axis = OutputAxis(fig, gs[1], max_display_size=10, log_filename=None)
output_axis_thread = threading.Thread(target=demo_output_axis)
output_axis_thread.daemon = True
output_axis_thread.start()

# Image Visualizer
def demo_image_visualizer():
    im = np.zeros((5, 5, 3))
    while True:
        i = random.randint(0, im.shape[0] - 1)
        j = random.randint(0, im.shape[1] - 1)
        k = random.randint(0, im.shape[2] - 1)
        im[i, j, k] = (im[i, j, k] + random.randint(0, 255)) % 256
        image_visualizer.update(im)
        time.sleep(5e-3)

image_visualizer = ImageVisualizer(fig, gs[2], cropsize=(3, 3))
image_visualizer_thread = threading.Thread(target=demo_image_visualizer)
image_visualizer_thread.daemon = True
image_visualizer_thread.start()

# Real Time Plotter
def demo_real_time_plotter():
    i, j = 0, 0
    while True:
        i += random.randint(-10, 10)
        j += random.randint(-10, 10)
        data = [i, j, i + j, i - j]
        mean = np.mean(data)
        real_time_plotter.update(data + [mean])
        time.sleep(5e-3)

real_time_plotter = RealTimePlotter(fig, gs[3],
        labels=['i', 'j', 'i+j', 'i-j', 'mean'],
        alphas=[0.15, 0.15, 0.15, 0.15, 1.0])
real_time_plotter_thread = threading.Thread(target=demo_real_time_plotter)
real_time_plotter_thread.daemon = True
real_time_plotter_thread.start()

# Mean Plotter
def demo_mean_plotter():
    i, j = 0, 0
    while True:
        i += random.randint(-10, 10)
        j += random.randint(-10, 10)
        data = [i, j, i + j, i - j]
        mean_plotter.update(data)
        time.sleep(1)

mean_plotter = MeanPlotter(fig, gs[4])
mean_plotter_thread = threading.Thread(target=demo_mean_plotter)
mean_plotter_thread.daemon = True
mean_plotter_thread.start()

# Plotter 3D
def demo_plotter_3d():
    xyzs = np.zeros((3, 1))
    while True:
        plotter_3d.clear_all()
        xyz = np.random.randint(-10, 10, size=3).reshape((3,1))
        xyzs = np.append(xyzs, xyz, axis=1)
        xs, ys, zs = xyzs
        plotter_3d.plot(0, xs, ys, zs)
        plotter_3d.draw()  # this must be called explicitly
        time.sleep(1)

plotter_3d = Plotter3D(fig, gs[5], num_plots=1, rows=1, cols=1)
plotter_3d_thread = threading.Thread(target=demo_plotter_3d)
plotter_3d_thread.daemon = True
plotter_3d_thread.start()


plt.show()