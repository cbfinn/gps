"""
Line plotter

Plots multiple sequences of 1-d data over time.
"""
import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

from gps.gui.util import buffered_axis_limits


class LinePlotter(object):
    def __init__(self, fig, gs, label='mean', color='black', num_plots=10, gui_on=True):
        self._fig = fig
        self._gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs)
        self._ax = plt.subplot(self._gs[0])
        self.gui_on = gui_on

        self._label = label
        self._color = color
        self.num_plots=num_plots

        self._ts = np.empty((1, 0))

        self._ax.set_xlim(0-0.5, 100)
        self._ax.set_ylim(0, 1)

        self._init = False

        if self.gui_on:
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend

    def init(self, t=100):
        """
        Initialize plots based off the length of the data array.
        """
        self._t = t
        self._init = True
        n = self.num_plots

        self._data = [np.zeros(t)]*n
        self._plots = [None]*n
        for i in range(n):
            self._plots[i] = self._ax.plot(self._data[i], linestyle='-')[0]

    def set_sequence(self, idx, x, style='-'):
        """
        Update the plots with new data x. Assumes x is a one-dimensional array.
        """
        #x = np.ravel([x])

        if not self._init:
            self.init(t=x.shape[0])

        assert x.shape[0] == self._t
        x = x.reshape(self._t)

        self._data[idx] = x
        self._plots[idx].set_data(np.arange(self._t), self._data[idx])
        self._plots[idx].set_linestyle(style)

        self._ax.set_xlim(0-0.5, self._t)
        y_min, y_max = np.amin(self._data), np.amax(self._data)
        self._ax.set_ylim(buffered_axis_limits(y_min, y_max, buffer_factor=1.1))
        if self.gui_on:
            self.draw()

    def draw(self):
        if not self.gui_on:
            return
        self._ax.draw_artist(self._ax.patch)
        for plot in self._plots:
            self._ax.draw_artist(plot)
        self._fig.canvas.update()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend

    def draw_ticklabels(self):
        """
        Redraws the ticklabels. Used to redraw the ticklabels (since they are
        outside the axis) when something else is drawn over them.
        """
        if not self.gui_on:
            return
        for item in self._ax.get_xticklabels() + self._ax.get_yticklabels():
            self._ax.draw_artist(item)
        self._fig.canvas.update()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend


class ScatterPlot(object):
    def __init__(self, fig, gs, xlabel='', ylabel='', gui_on=True):
        self._fig = fig
        self._gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs)
        self._ax = plt.subplot(self._gs[0])
        self.gui_on = gui_on

        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)

        self._ax.set_xlim(0-0.5, 100)
        self._ax.set_ylim(0, 1)

        self._init = False

        if self.gui_on:
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend

    def init(self):
        """
        Initialize plots based off the length of the data array.
        """
        self._init = True
        self._plots = []#self._ax.scatter(np.zeros(5),np.zeros(5))
        self._xdata = []
        self._ydata = []

    def clear(self):
        self._plots = []
        self._xdata = []
        self._ydata = []
        self._ax.clear()

    def add_data(self, x, y, color='blue'):
        #import pdb; pdb.set_trace()
        if not self._init:
            self.init()

        self._xdata.extend(x)
        self._ydata.extend(y)

        x_min, x_max = np.amin(self._xdata), np.amax(self._xdata)
        self._ax.set_xlim(buffered_axis_limits(x_min, x_max, buffer_factor=1.1))
        y_min, y_max = np.amin(self._ydata), np.amax(self._ydata)
        self._ax.set_ylim(buffered_axis_limits(y_min, y_max, buffer_factor=1.1))

        self._plots.append(self._ax.scatter(x,y, c=color))

        self.draw()

    def draw(self):
        if not self.gui_on:
            return
        self._ax.draw_artist(self._ax.patch)
        for plot in self._plots:
            self._ax.draw_artist(plot)
        self._fig.canvas.update()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend

    def draw_ticklabels(self):
        """
        Redraws the ticklabels. Used to redraw the ticklabels (since they are
        outside the axis) when something else is drawn over them.
        """
        if not self.gui_on:
            return
        for item in self._ax.get_xticklabels() + self._ax.get_yticklabels():
            self._ax.draw_artist(item)
        self._fig.canvas.update()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend

