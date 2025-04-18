from torch import nn

from .base import NNModuleBase


class Module(NNModuleBase):
    __slots__ = ()
    input_types = (
        'state.car.position.x',
        'state.car.position.y',
        'state.car.position.z',
        'state.car.velocity.x',
        'state.car.velocity.y',
        'state.car.velocity.z',
        'state.car.rotation.pitch',
        'state.car.rotation.roll',
        'state.car.rotation.yaw',
        'state.ball.position.x',
        'state.ball.position.y',
        'state.ball.position.z',
    )
    output_types = (
        'controls.throttle',
        'controls.steer',
    )

    @classmethod
    def _create(cls):
        inner_1 = 16*2*2
        inner_2 = 32*2*2
        inner_3 = 16*2*2
        return nn.Sequential(
            nn.Linear(len(cls.input_types), inner_1),
            nn.ReLU(),
            nn.Linear(inner_1, inner_2),
            nn.ReLU(),
            nn.Linear(inner_2, inner_3),
            nn.ReLU(),
            nn.Linear(inner_3, len(cls.output_types))
        )

    def requires(self) -> set[str]:
        return set()

