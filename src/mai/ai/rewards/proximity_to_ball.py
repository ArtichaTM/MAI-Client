from .base import NNRewardBase

from mai.capnp.data_classes import Vector
from mai.settings import Settings

class NNReward(NNRewardBase):
    __slots__ = ('multiplier', 'latest_distance', 'counter')
    _divider = 0.7

    def __init__(self) -> None:
        super().__init__()
        self.latest_distance = None
        self.multiplier = self._divider
        self.counter = 0

    def _calculate(self, state, context) -> float:
        car_pos = Vector.from_mai(state.car.position)
        ball_pos = Vector.from_mai(state.ball.position)
        distance = (car_pos - ball_pos).magnitude()
        if self.latest_distance is None:
            self.latest_distance = distance
            return 0

        if self.counter == 10:
            self.multiplier = Settings.get_current_eps() / self._divider
            self.counter = 0
        else:
            self.counter += 1

        difference = self.latest_distance-distance
        self.latest_distance = distance

        result = difference*self.multiplier
        return result
