from .base import NNRewardBase
from mai.capnp.data_classes import Vector

class NNReward(NNRewardBase):
    __slots__ = ()
    _update_seconds: float = 0.2
    _multiplier: float = 1.

    def __init__(self) -> None:
        super().__init__()

    def reset(self) -> None:
        super().reset()

    def _calculate(self, state, context) -> float:
        car_pos = Vector.from_mai(state.car.position)
        ball_pos = Vector.from_mai(state.ball.position)

        distance = (ball_pos-car_pos).magnitude() 
        distance = 1-(distance/2)
        return distance
