from typing import Generator
from abc import ABC, abstractmethod

from mai.capnp.names import MAIControls, MAIGameState
from mai.capnp.data_classes import AdditionalContext
from mai.functions import create_dummy_controls


class BaseTactic(ABC):
    """Base class for all tactics

    finished: when set to true in react, this call will be the last
    """
    __slots__ = ('finished', '_gen')
    finished: bool
    _gen: Generator[MAIControls, tuple[
        MAIGameState, AdditionalContext
    ], None] | None

    def __init__(self) -> None:
        super().__init__()
        self.finished = False
        self._gen = None

    def prepare(self) -> None:
        self._gen = self.react_gen()
        try:
            next(self._gen)
        except StopIteration:
            self.finished = True

    def react(self, state: MAIGameState, context: AdditionalContext) -> MAIControls:
        assert not self.finished
        assert self._gen is not None
        try:
            output = self._gen.send((state, context))
        except StopIteration:
            self.finished = True
            return create_dummy_controls()
        return output if output else create_dummy_controls()

    @abstractmethod
    def react_gen(
        self
    ) -> Generator[MAIControls, tuple[MAIGameState, AdditionalContext], None]:
        raise NotImplementedError()

    def close(self) -> None:
        if self._gen is not None:
            self._gen.close()
