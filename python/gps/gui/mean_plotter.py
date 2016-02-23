""" This file defines a mean data plotter. """
import random
import time

import numpy as np

import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec


class MeanPlotter:
    def __init__(self, fig, gs, label='mean', color='black', alpha=1.0, min_itr=10):
        self._fig = fig
        self._gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs)
        self._ax = plt.subplot(self._gs[0])

        self._label = label
        self._color = color
        self._alpha = alpha
        self._min_itr = min_itr

        self._ts = np.empty((1, 0))
        self._data_mean = np.empty((1, 0))
        self._plots_mean = self._ax.plot([], [], '-x', markeredgewidth=1.0, color=self._color,
                alpha=1.0, label=self._label)[0]

        self._ax.set_xlim(0-0.5, self._min_itr+0.5)
        self._ax.set_ylim(0, 1)
        self._ax.minorticks_on()
        self._ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

        self._init = False

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend

    def init(self, data_len):
        """ Initialize plots. """
        self._t = 0
        self._data_len = data_len
        self._data = np.empty((data_len, 0))
        self._plots = [
            self._ax.plot([], [], '.', markersize=4, color='black', alpha=self._alpha)[0]
            for _ in range(data_len)
        ]

        self._init = True

    def update(self, x, t=None):
        """ Update plots. """
        x = np.ravel([x])

        if not self._init:
            self.init(x.shape[0])

        if not t:
            t = self._t

        assert x.shape[0] == self._data_len
        t = np.array([t]).reshape((1, 1))
        x = x.reshape((self._data_len, 1))
        mean = np.mean(x).reshape((1, 1))

        self._t += 1
        self._ts = np.append(self._ts, t, axis=1)
        self._data = np.append(self._data, x, axis=1)
        self._data_mean = np.append(self._data_mean, mean, axis=1)

        for i in range(self._data_len):
            self._plots[i].set_data(self._ts, self._data[i, :])
        self._plots_mean.set_data(self._ts, self._data_mean[0, :])

        y_min, y_max = np.amin(self._data), np.amax(self._data)
        precision = np.power(10, np.floor(np.log10(np.amax(np.abs((y_min, y_max)) + 1e-100))) - 1)
        y_lim_min = np.floor(y_min/precision) * precision
        y_lim_max = np.ceil(y_max/precision) * precision

        self._ax.set_xlim(self._ts[0, 0]-0.5, max(self._ts[-1, 0], self._min_itr)+0.5)
        self._ax.set_ylim((y_lim_min, y_lim_max))
        self.draw()

    def draw(self):
        self._ax.draw_artist(self._ax.patch)
        [self._ax.draw_artist(plot) for plot in self._plots]
        self._ax.draw_artist(self._plots_mean)
        self._fig.canvas.update()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend

    def draw_ticklabels(self):
        """
        Redraws the ticklabels, for use when ticklabels are being drawn outside of the axis
        and need to be redrawn, since something else is drawing over them.
        """
        [self._ax.draw_artist(item) for item in self._ax.get_xticklabels() + self._ax.get_yticklabels()]
        self._fig.canvas.update()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend

if __name__ == "__main__":
    import matplotlib.gridspec as gridspec


    plt.ion()
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)

    plotter = MeanPlotter(fig, gs[0])

    i, j = 0, 0
    while True:
        i += random.randint(-10, 10)
        j += random.randint(-10, 10)
        data = [i, j, i + j, i - j]
        plotter.update(data)
        time.sleep(1)
