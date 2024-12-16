from .base import NNRewardBase

class NNReward(NNRewardBase):
    __slots__ = ()

    def _calculate(self, state, context) -> float:
        raise NotImplementedError()
