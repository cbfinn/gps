""" This file defines the real time plotter class. """
import random
import time

import numpy as np

import matplotlib.pylab as plt


class RealTimePlotter(object):
    """ Real time plotter class. """
    def __init__(self, fig, gs, time_window=500, labels=None, alphas=None):
        self._fig = fig
        self._gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs)
        self._ax = plt.subplot(self._gs[0])

        self._time_window = time_window
        self._labels = labels
        self._alphas = alphas
        self._init = False

        if self._labels:
            self.init(len(self._labels))

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend

    def init(self, data_len):
        """ Initialize plots. """
        self._t = 0
        self._data_len = data_len
        self._data = np.empty((0, data_len))

        cm = plt.get_cmap('spectral')
        self._plots = []
        for i in range(data_len):
            color = cm(1.0 * i / data_len)
            alpha = self._alphas[i] if self._alphas is not None else 1.0
            label = self._labels[i] if self._labels is not None else str(i)
            self._plots.append(
                self._ax.plot([], [], color=color, alpha=alpha, label=label)[0]
            )
        self._ax.set_xlim(0, self._time_window)
        self._ax.set_ylim(0, 1)
        self._ax.legend(loc='upper left', bbox_to_anchor=(0, 1.15))

        self._init = True

    #TODO: Any possible abstraction with MeanPlotter.update?
    def update(self, x):
        """ Update plots. """
        x = np.ravel([x])

        if not self._init:
            self.init(x.shape[0])

        assert x.shape[0] == self._data_len
        x = x.reshape((1, self._data_len))

        self._t += 1
        self._data = np.append(self._data, x, axis=0)

        t, tw = self._t, self._time_window
        t0, tf = (0, t) if t < tw else (t - tw, t)
        for i in range(self._data_len):
            self._plots[i].set_data(np.arange(t0, tf), self._data[t0:tf, i])

        x_range = (0, tw) if t < tw else (t - tw, t)
        self._ax.set_xlim(x_range)

        y_min, y_max = np.amin(self._data[t0:tf, :]), np.amax(self._data[t0:tf, :])
        y_mid, y_dif = (y_min + y_max) / 2.0, (y_max - y_min) / 2.0
        if y_dif == 0:
            y_dif = 1  # Make sure y_range does not have size 0.
        y_min, y_max = y_mid - 1.25 * y_dif, y_mid + 1.25 * y_dif
        precision = np.power(10, np.floor(np.log10(np.amax(np.abs((y_min, y_max)) + 1e-100))) - 1)
        y_lim_min = np.floor(y_min/precision) * precision
        y_lim_max = np.ceil(y_max/precision) * precision

        self._ax.set_ylim((y_lim_min, y_lim_max))

        self.draw()

    def draw(self):
        self._ax.draw_artist(self._ax.patch)
        [self._ax.draw_artist(plot) for plot in self._plots]
        self._fig.canvas.update()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend


if __name__ == "__main__":
    import matplotlib.gridspec as gridspec


    plt.ion()
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    plotter = RealTimePlotter(fig, gs[0],
        labels=['i', 'j', 'i+j', 'i-j', 'mean'],
        alphas=[0.15, 0.15, 0.15, 0.15, 1.0])

    i, j = 0, 0
    while True:
        i += random.randint(-10, 10)
        j += random.randint(-10, 10)
        data = [i, j, i + j, i - j]
        mean = np.mean(data)
        plotter.update(data + [mean])
        time.sleep(5e-3)
