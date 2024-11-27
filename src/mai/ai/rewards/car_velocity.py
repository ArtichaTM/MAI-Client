from .base import NNRewardBase

from mai.capnp.data_classes import Vector

class NNReward(NNRewardBase):
    __slots__ = ()

    def _calculate(self, state, context) -> float:
        car_velocity = Vector.from_mai(state.car.velocity)
        magnitude = car_velocity.magnitude()
        return magnitude - context.magnitude_offsets['car']['v']
