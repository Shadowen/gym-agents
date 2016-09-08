import pylab as plt
import numpy as np
import time

plt.ion()


class RealtimePlotter:
    def __init__(self):
        # Matplotlib stuff
        self.fig = plt.figure()
        self.ax = plt.gca()

    def new_line(self):
        self.graph = plt.plot([])[0]

    def update(self, data):
        self.graph.set_data(range(len(data)), data)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()


if __name__ == "__main__":
    p = RealtimePlotter()
    p.new_line()
    for i in range(100):
        x = np.arange(i)
        y = np.random.random(x.shape)
        p.update(y)
        time.sleep(0.25)
