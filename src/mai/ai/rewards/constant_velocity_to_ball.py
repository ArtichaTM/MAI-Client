import numpy as np

from .base import NNRewardBase
from mai.capnp.data_classes import Vector

class NNReward(NNRewardBase):
    __slots__ = ('latest_result',)
    _update_seconds: float = 0.2
    _multiplier: float = 1.

    def __init__(self) -> None:
        super().__init__()
        self.latest_result: float = 0.

    def reset(self) -> None:
        super().reset()

    def _calculate(self, state, context) -> float:
        car_vel = Vector.from_mai(state.car.velocity)
        if car_vel.magnitude() < 0.001:
            return self.latest_result
        car_pos = Vector.from_mai(state.car.position)
        ball_pos = Vector.from_mai(state.ball.position)

        car_to_ball = (ball_pos-car_pos)
        vec1 = np.array([car_vel.x, car_vel.y, car_vel.z])
        vec2 = np.array([car_to_ball.x, car_to_ball.y, car_to_ball.z])
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        result = dot_product / (norm1 * norm2)
        self.latest_result = result
        return self.latest_result
