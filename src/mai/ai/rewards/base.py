from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mai.capnp.data_classes import AdditionalContext
    from mai.capnp.names import MAIGameState


class NNRewardBase(ABC):
    """
    Base class for all rewards functions
    """
    __slots__ = ('power',)
    power: float

    def __init__(self) -> None:
        self.power: float = 1.0
        self.reset()

    def __repr__(self) -> str:
        return (
            f"<NNR {self.name} power"
            f"={self.power: > 1.2f}>"
        )

    def __call__(
        self,
        state: MAIGameState,
        context: AdditionalContext
    ) -> float:
        assert self.enabled
        output = self._calculate(state, context)
        output *= self.power
        assert isinstance(output, float)
        return output 

    @abstractmethod
    def _calculate(
        self,
        state: MAIGameState,
        context: AdditionalContext
    ) -> float:
        raise NotImplementedError()

    def reset(self) -> None:
        """Called when environment is reset or initiated"""
        pass

    @property
    def enabled(self) -> bool:
        """
        This property shows if reward function used in generating rewards
        """
        assert isinstance(self.power, float)
        return self.power > 0.009

    @classmethod
    def get_name(cls) -> str:
        """
        Display name of model
        """
        return cls.__module__.split('.')[-1]

    @property
    def name(self) -> str:
        """
        Display name of model
        """
        return type(self).get_name()
