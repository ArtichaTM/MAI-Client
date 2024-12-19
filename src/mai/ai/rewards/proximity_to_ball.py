from time import perf_counter

from .base import NNRewardBase
from mai.capnp.data_classes import Vector

class NNReward(NNRewardBase):
    __slots__ = (
        'multiplier',
        'latest_distance', 'latest_answer',
        'result',
    )
    _divider = 0.7
    _update_seconds: float = 0.2

    def __init__(self) -> None:
        super().__init__()
        self.latest_distance = None
        self.multiplier = self._divider

    def _calculate(self, state, context) -> float:
        car_pos = Vector.from_mai(state.car.position)
        ball_pos = Vector.from_mai(state.ball.position)
        if self.latest_distance is None:
            self.latest_distance = (car_pos - ball_pos).magnitude()
            self.latest_answer = perf_counter()-1
            return 0

        current = perf_counter()
        if (current - self.latest_answer) > self._update_seconds:
            distance = (car_pos - ball_pos).magnitude()
            difference = self.latest_distance-distance
            self.latest_distance = distance
            self.result = difference * self.multiplier
            self.latest_answer = current

        return self.result
