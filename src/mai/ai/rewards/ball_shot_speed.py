from time import perf_counter

from .base import NNRewardBase
from mai.capnp.data_classes import Vector, AdditionalContext

class NNReward(NNRewardBase):
    __slots__ = (
        'result', 'latest_answer', 'start_speed',
    )
    _update_seconds: float = 0.2

    def __init__(self) -> None:
        super().__init__()
        self.latest_answer = None
        self.start_speed: float = 0
        self.result: float = 0

    def is_ball_touched(self, context: AdditionalContext) -> bool:
        return context.exchanges_since_latest_message == 0 and \
            context.latest_message == 'ballTouched'

    def is_ball_touched_by_player(
        self, context: AdditionalContext,
        car_pos: Vector, ball_pos: Vector,
    ) -> bool:
        if not self.is_ball_touched(context):
            return False
        distance = (car_pos - ball_pos).magnitude()
        if distance < 1:
            return True
        return False

    def ball_touched(self, ball_vec: Vector) -> None:
        self.start_speed = ball_vec.magnitude()

    def _calculate(self, state, context) -> float:
        if self.latest_answer is None:
            self.latest_answer = perf_counter()-1
            return self.result

        car_pos = Vector.from_mai(state.car.position)
        ball_pos = Vector.from_mai(state.ball.position)

        if self.is_ball_touched_by_player(context, car_pos, ball_pos):
            ball_vec = Vector.from_mai(state.ball.velocity)
            self.ball_touched(ball_vec)

        if self.start_speed != 0:
            current = perf_counter()
            if (current - self.latest_answer) > self._update_seconds:
                self.start_speed *= 0.9
                if self.start_speed < 0.1:
                    self.start_speed = 0
                    self.result = 0
                else:
                    self.result = self.start_speed
            return self.result

        return self.result
