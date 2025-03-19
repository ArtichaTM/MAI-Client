from time import perf_counter

from .base import NNRewardBase
from mai.capnp.data_classes import Vector


class NNReward(NNRewardBase):
    __slots__ = (
        'multiplier',
        'latest_distance', 'latest_answer', 
        'result',
    )
    _multiplier: float = 3.
    _update_seconds: float = 0.2

    def __init__(self) -> None:
        super().__init__()
        self.latest_distance = None
        self.multiplier = self._multiplier

    def reset(self) -> None:
        super().reset()
        self.latest_distance = None

    def _calculate(self, state, context) -> float:
        car = Vector.from_mai(state.car.position)
        ray = (
            Vector.from_mai(state.ball.position)-\
            car
        )
        distance = ray.magnitude()
        if self.latest_distance is None:
            self.latest_distance = distance
            self.latest_answer = perf_counter()-1
            self.result = 0
            return self.result
        current = perf_counter()
        if (current - self.latest_answer) > self._update_seconds:
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
