import numpy as np
import matplotlib.pyplot as plt

class FPPlot(object):
    def __init__(self, fig=None, numfp=15, range=1):
        if fig is None:
            fig, axes = plt.subplots()
        else:
            axes = plt.subplot()

        self.range = range
        self.init_plot(fig, axes, numfp)

    def init_plot(self, fig, ax, numfp):
        self.fig = fig
        self.ax = ax
        self.ax.set_title('Feature points')
        # self.background = fig.canvas.copy_from_bbox(ax.bbox)
        self.img_data = np.zeros((numfp, 2))
        #self.img_data = np.random.randn(40,40)
        self.scat = self.ax.scatter(self.img_data[:,0],
            self.img_data[:,1])
        self.ax.axis([-self.range,self.range]*2)
        self.fig.show(False)
        self.fig.canvas.draw()

    def update_fp(self, X):
        N, _ = X.shape
        self.img_data = X
        self.scat.set_offsets(self.img_data)
        self.scat._sizes = 10*np.ones((N,))
        self.fig.canvas.draw()

    def destroy(self):
        plt.close(self.fig)