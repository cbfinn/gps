"""
GUI Demos

Demos how to use the six different GUI elements:

Action Panel
Textbox
Image Visualizer
Realtime Plotter
Mean Plotter
Plotter 3D
"""
import time
import random
import threading

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gps.gui.action_panel import Action, ActionPanel
from gps.gui.textbox import Textbox
from gps.gui.image_visualizer import ImageVisualizer
from gps.gui.realtime_plotter import RealtimePlotter
from gps.gui.mean_plotter import MeanPlotter
from gps.gui.plotter_3d import Plotter3D


def run_demo(demo_func):
    demo_thread = threading.Thread(target=demo_func)
    demo_thread.daemon = True
    demo_thread.start()

# Initialize Figure
fig = plt.figure()
gs = gridspec.GridSpec(2, 3)

# Action Panel
number = 0
def plus_1(event=None): global number; number += 1; print(number)
def plus_2(event=None): global number; number += 2; print(number)
def mult_4(event=None): global number; number *= 4; print(number)

actions_arr = [
    Action('plus1', 'plus1', plus_1, axis_pos=0, keyboard_binding='1', ps3_binding=None),
    Action('plus2', 'plus2', plus_2, axis_pos=1, keyboard_binding='2', ps3_binding=None),
    Action('print', 'print', mult_4, axis_pos=2, keyboard_binding='4', ps3_binding=None),
]
action_panel = ActionPanel(fig, gs[0], 3, 1, actions_arr)

# Textbox
def demo_textbox():
    max_i = 20
    for i in range(max_i):
        textbox.append_text(str(i))
        c = 0.5 + 0.5*i/max_i
        textbox.set_bgcolor((c, c, c))
        time.sleep(1)

textbox = Textbox(fig, gs[1], max_display_size=10, log_filename=None)
run_demo(demo_textbox)

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
run_demo(demo_image_visualizer)

# Realtime Plotter
def demo_realtime_plotter():
    i, j = 0, 0
    while True:
        i += random.randint(-10, 10)
        j += random.randint(-10, 10)
        data = [i, j, i + j, i - j]
        mean = np.mean(data)
        realtime_plotter.update(data + [mean])
        time.sleep(5e-3)

realtime_plotter = RealtimePlotter(fig, gs[3],
        labels=['i', 'j', 'i+j', 'i-j', 'mean'],
        alphas=[0.15, 0.15, 0.15, 0.15, 1.0])
run_demo(demo_realtime_plotter)

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
run_demo(demo_mean_plotter)

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
run_demo(demo_plotter_3d)

# Show Figure
plt.show()
