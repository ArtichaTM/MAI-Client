import multiprocessing as mp

from mai.plotter.low_level import ChildProcessPlotterBase


class ProcessPlotter[T]:
    __slots__ = (
        'plot_pipe', 'plotter_pipe',
        'plotter', 'plot_process',
        'closed',
    )
    plotter: ChildProcessPlotterBase[T]

    def __init__(
        self,
        cl: type[ChildProcessPlotterBase[T]],
        *args, **kwargs,
    ):
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = cl()
        self.plot_process = mp.Process(
            target=self.plotter, args=(
                plotter_pipe, *args
            ), kwargs=kwargs, daemon=True
        )
        self.plot_process.start()
        self.closed = False

    def plot(self, data: T):
        assert not self.closed
        assert not self.plot_pipe.closed
        if self.plot_pipe.closed:
            self.closed = True
        self.plot_pipe.send(data)

    def finish(self) -> None:
        assert not self.closed
        self.closed = True
        self.plot_pipe.send(None)
