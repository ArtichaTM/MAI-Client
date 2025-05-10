from .bases import BaseTactic
from mai.capnp.data_classes import MAIGameState, Vector
from mai.functions import create_dummy_controls, values_tracker
from mai.plotter.plotter import ProcessPlotter


class WatchGraph(BaseTactic):
    __slots__ = ('plotter', 'keep_running')
    keep_running: bool

    def __init__(self) -> None:
        super().__init__()
        self.keep_running = True

    def prepare(self) -> None:    
        def on_close() -> None:
            self.keep_running = False
        self.plotter = values_tracker(
            ('Speed magnitude',),
            on_close,
            legend=False
        )
        next(self.plotter)
        return super().prepare()

    def react_gen(self):
        """
        Presses key and waits one exchange before reset is complete
        """
        while self.keep_running:
            state, context = yield create_dummy_controls()
            state: MAIGameState
            car_vector = Vector.from_mai(state.car.velocity)
            self.plotter.send((car_vector.magnitude(),))

    def close(self) -> None:
        if not self.keep_running:
            try:
                self.plotter.close()
            except GeneratorExit:
                pass
            self.keep_running = False
