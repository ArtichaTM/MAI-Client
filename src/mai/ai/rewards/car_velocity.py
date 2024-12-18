from .base import NNRewardBase

from mai.capnp.data_classes import Vector
from mai.settings import Settings

class NNReward(NNRewardBase):
    __slots__ = ('addition', 'multiplier')

    def __init__(self) -> None:
        super().__init__()
        self.addition = Settings.max_speed_magnitude / 10
        self.multiplier = 1

    def _calculate(self, state, context) -> float:
        car_velocity = Vector.from_mai(state.car.velocity)
        magnitude = car_velocity.magnitude() - context.magnitude_offsets['car']['v']
        magnitude -= self.addition
        magnitude *= self.multiplier
        return magnitude if magnitude > 0 else 0
