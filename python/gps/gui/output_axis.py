""" This file defines the output axis. """
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ColorConverter


class OutputAxis:
    def __init__(self, fig, gs, log_filename=None, max_display_size=10,
        border_on=False, bgcolor=mpl.rcParams['figure.facecolor'], bgalpha=1.0,
        fontsize=12, font_family='sans-serif'):
        self._fig = fig
        self._gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs)
        self._ax = plt.subplot(self._gs[0])
        self._log_filename = log_filename

        self._text_box = self._ax.text(0.01, 0.95, '', color='black', fontsize=fontsize,
                va='top', ha='left', transform=self._ax.transAxes, family=font_family)
        self._text_arr = []
        self._max_display_size = max_display_size

        self._ax.set_xticks([])
        self._ax.set_yticks([])
        if not border_on:
            self._ax.spines['top'].set_visible(False)
            self._ax.spines['right'].set_visible(False)
            self._ax.spines['bottom'].set_visible(False)
            self._ax.spines['left'].set_visible(False)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()     # Fixes bug with Qt4Agg backend
        self.set_bgcolor(bgcolor, bgalpha)  # this must come after fig.canvas.draw()

    #TODO: Add docstrings here.
    def set_text(self, text):
        self._text_arr = [text]
        self._text_box.set_text('\n'.join(self._text_arr))
        self.log_text(text)
        self.draw()

    def append_text(self, text):
        self._text_arr.append(text)
        if len(self._text_arr) > self._max_display_size:
            self._text_arr = self._text_arr[-self._max_display_size:]
        self._text_box.set_text('\n'.join(self._text_arr))
        self.log_text(text)
        self.draw()

    def log_text(self, text):
        if self._log_filename is not None:
            with open(self._log_filename, 'a') as f:
                f.write(text + '\n')

    def set_bgcolor(self, color, alpha=1.0):
        self._ax.set_axis_bgcolor(ColorConverter().to_rgba(color, alpha))
        self.draw()

    def draw(self):
        color, alpha = self._ax.get_axis_bgcolor(), self._ax.get_alpha()
        self._ax.set_axis_bgcolor(mpl.rcParams['figure.facecolor'])
        self._ax.draw_artist(self._ax.patch)
        self._ax.set_axis_bgcolor(ColorConverter().to_rgba(color, alpha))

        self._ax.draw_artist(self._ax.patch)
        self._ax.draw_artist(self._text_box)
        self._fig.canvas.update()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend


if __name__ == "__main__":
    import matplotlib.gridspec as gridspec


    plt.ion()
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    output_axis = OutputAxis(fig, gs[0], max_display_size=10, log_filename=None)

    max_i = 20
    for i in range(max_i):
        output_axis.append_text(str(i))
        c = 0.5 + 0.5*i/max_i
        output_axis.set_bgcolor((c, c, c))
        time.sleep(1)

    plt.ioff()
    plt.show()
