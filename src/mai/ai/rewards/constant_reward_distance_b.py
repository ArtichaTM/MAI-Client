from time import perf_counter

from .base import NNRewardBase
from mai.capnp.data_classes import Vector

class NNReward(NNRewardBase):
    __slots__ = (
        'latest_distance',
    )
    _return_value = 0.3
    _update_seconds: float = 0.2

    def __init__(self) -> None:
        super().__init__()
        self.latest_distance = None

    def reset(self) -> None:
        super().reset()
        self.latest_distance = None

    def _calculate(self, state, context) -> float:
        car_pos = Vector.from_mai(state.car.position)
        ball_pos = Vector.from_mai(state.ball.position)
        if self.latest_distance is None:
            self.latest_distance = (car_pos - ball_pos).magnitude()
            return 0

        distance = (car_pos - ball_pos).magnitude()
        difference = distance - self.latest_distance
        self.latest_distance = distance
        if abs(difference) < 0.001:
            return 0
        elif difference < 0:
            return self._return_value
        else:
            return -self._return_value
