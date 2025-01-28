import multiprocessing as mp
import time

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class ChildProcessPlotter:
    __slots__ = (
        'pipe', 'fig', 'ax',
        'x', 'ys', 'xlim', 'ylim',
        'plot_names', 'legend',
    )

    fig: Figure
    ax: Axes
    ys: list[list[float | None]]
    legend: bool

    def terminate(self):
        plt.close('all')

    def call_back(self):
        assert hasattr(self, 'pipe')
        assert not self.pipe.closed
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                self.ax.clear()
                for i in range(len(self.plot_names)):
                    y = self.ys[i]
                    y.pop(0)
                    y.append(command[i])
                    self.ax.plot(self.x, y, label=self.plot_names[i])
                    self.ax.set_ylim(*self.ylim)
                self.ax.grid(which='major', axis='y')
                self.ax.legend()
        self.fig.canvas.draw()
        return True

    def __call__(
        self,
        pipe,
        plot_names: tuple[str, ...],
        xlim: tuple[int, int],
        ylim: tuple[float, float],
        legend: bool,
    ):
        self.pipe = pipe
        self.plot_names = plot_names
        self.xlim = xlim
        self.ylim = ylim

        self.x = list(range(xlim[1]))
        self.ys = [[None] * xlim[1] for _ in range(len(plot_names))]

        self.fig, self.ax = plt.subplots()
        timer = self.fig.canvas.new_timer(interval=0)
        timer.add_callback(self.call_back)
        timer.start()
        plt.show()


class ProcessPlotter:
    __slots__ = (
        'plot_pipe', 'plotter_pipe',
        'plotter', 'plot_process',
        'closed',
    )

    def __init__(self, plot_names: tuple[str, ...], legend: bool = True):
        assert isinstance(legend, bool)
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ChildProcessPlotter()
        self.plot_process = mp.Process(
            target=self.plotter, args=(
                plotter_pipe, plot_names, (0, 200), (-1.1, 1.1), legend
            ), daemon=True
        )
        self.plot_process.start()
        self.closed = False

    def plot(self, data: tuple[float | None, ...]):
        assert not self.closed
        assert not self.plot_pipe.closed
        if self.plot_pipe.closed:
            self.closed = True
        self.plot_pipe.send(data)

    def finish(self) -> None:
        assert not self.closed
        self.closed = True
        self.plot_pipe.send(None)
