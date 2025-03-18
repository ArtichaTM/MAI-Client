from time import perf_counter

from .base import NNRewardBase
from mai.capnp.data_classes import Vector


class NNReward(NNRewardBase):
    __slots__ = (
        'multiplier',
        'latest_distance', 'latest_answer',
        'result',
    )
    _multiplier = 3
    _update_seconds: float = 0.2

    def __init__(self) -> None:
        super().__init__()
        self.latest_distance = None
        self.multiplier = self._multiplier

    def reset(self) -> None:
        super().reset()
        self.latest_distance = None

    def _calculate(self, state, context) -> float:
        if self.latest_distance is None:
            self.latest_distance = (
                Vector.from_mai(state.ball.position)-\
                Vector.from_mai(state.car.position)
            ).magnitude()
            self.latest_answer = perf_counter()-1
            self.result = 0
            return 0
        current = perf_counter()
        if (current - self.latest_answer) > self._update_seconds:
            distance = (
                Vector.from_mai(state.ball.position)-\
                Vector.from_mai(state.car.position)
            ).magnitude()
            # Bigger -> better
            # Positive -> we are closer to the ball
            difference = self.latest_distance - distance
            if difference < 0:
                self.result = difference * self.multiplier
            else:
                self.result = difference**0.7 * self.multiplier
            self.latest_distance = distance
            self.latest_answer = current
        return self.result
