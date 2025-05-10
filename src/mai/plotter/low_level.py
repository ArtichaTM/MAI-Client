from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = (
    'ChildProcessPlotterBase',
    'PlotterPlot',
    'PlotterBarH'
)


class ChildProcessPlotterBase[T](ABC):
    __slots__ = (
        'pipe', 'fig', 'ax',
    )
    fig: Figure
    ax: Axes

    def __call__(self, pipe, *args, **kwargs) -> None:
        self.pipe = pipe
        self.prepare(*args, **kwargs)

        self.fig, self.ax = plt.subplots()
        timer = self.fig.canvas.new_timer(interval=0)
        timer.add_callback(self.call_back)
        timer.start()
        plt.show()

    def terminate(self) -> None:
        plt.close(self.fig)

    def call_back(self) -> bool:
        assert hasattr(self, 'pipe')
        assert not self.pipe.closed
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                self.draw(command)
        self.fig.canvas.draw()
        return True

    @abstractmethod
    def draw(self, command: T) -> None:
        raise NotImplementedError()

    @abstractmethod
    def prepare(self, *args, **kwargs) -> None:
        raise NotImplementedError()


class PlotterPlot(ChildProcessPlotterBase):
    __slots__ = (
        'x', 'ys', 'xlim', 'ylim',
        'plot_names', 'legend',
    )
    ys: list[list[float | None]]
    legend: bool

    def prepare(
        self,
        plot_names: tuple[str, ...],
        xlim: tuple[int, int],
        ylim: tuple[float, float],
        legend: bool,
    ) -> None:
        self.plot_names = plot_names
        self.xlim = xlim
        self.ylim = ylim
        self.legend = legend

        self.x = list(range(xlim[1]))
        self.ys = [[None] * xlim[1] for _ in range(len(plot_names))]

    def draw(self, command):
        self.ax.clear()
        for i in range(len(self.plot_names)):
            y = self.ys[i]
            y.pop(0)
            y.append(command[i])
            self.ax.plot(
                self.x,
                y,  # type: ignore
                label=self.plot_names[i]
            )
            self.ax.set_ylim(*self.ylim)
        self.ax.grid(which='major', axis='y')
        if self.legend:
            self.ax.legend()


class PlotterBarH(ChildProcessPlotterBase):
    __slots__ = (
        'bar_names', 'y',
    )
    bar_names: Sequence[str]

    def prepare(self, bar_names: Sequence[str]) -> None:
        assert all(isinstance(i, str) for i in bar_names)
        self.bar_names = bar_names
        self.y = np.arange(len(bar_names))

    def draw(self, command):
        self.ax.clear()
        self.ax.set_xlim(0., 1.)
        self.ax.barh(self.y, command)
        self.ax.set_yticks(self.y, labels=self.bar_names)
        self.ax.invert_yaxis()
        self.ax.set_xlabel('Влияние модуля')
        self.ax.set_title("Влияние модулей на вывод")
        self.fig.tight_layout()
