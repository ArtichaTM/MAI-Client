from .base import NNRewardBase

from mai.capnp.data_classes import Vector
from mai.settings import Settings

class NNReward(NNRewardBase):
    __slots__ = ('latest_distance',)
    _multiplier = 100

    def __init__(self) -> None:
        super().__init__()
        self.latest_distance = None

    def _calculate(self, state, context) -> float:
        car_pos = Vector.from_mai(state.car.position)
        ball_pos = Vector.from_mai(state.ball.position)
        distance = (car_pos - ball_pos).magnitude()
        if self.latest_distance is None:
            self.latest_distance = distance
            return 0
        difference = distance - self.latest_distance
        return difference * self._multiplier
