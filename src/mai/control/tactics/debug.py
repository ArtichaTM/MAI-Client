from .bases import BaseTactic
from mai.functions import create_dummy_controls
from mai.ai.networks.reachBallFloor import Module as reachBallFloor


class DebugTactic(BaseTactic):
    __slots__ = ('cl',)

    def __init__(self) -> None:
        super().__init__()

    def prepare(self) -> None:
        super().prepare()
        self.cl = reachBallFloor(None)

    def react_gen(self):
        while True:
            mapping, context = yield create_dummy_controls()
            print(context)
