from abc import ABC, abstractmethod

from mai.capnp.names import MAIControls, MAIGameState
from mai.capnp.data_classes import AdditionalContext

class BaseTactic(ABC):
    """Base class for all tactics

    finished: when set to true in react, this call will be the last
    """
    __slots__ = ('finished', )

    def __init__(self) -> None:
        super().__init__()
        self.finished = False

    def prepare(self) -> None:
        pass

    @abstractmethod
    def react(self, state: MAIGameState, context: AdditionalContext) -> MAIControls:
        raise NotImplementedError()
