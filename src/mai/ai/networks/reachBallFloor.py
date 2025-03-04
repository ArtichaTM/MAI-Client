from torch import nn

from .base import NNModuleBase


class NNModule(NNModuleBase):
    __slots__ = ()
    input_types = (
        'state.car.position.x',
        'state.car.position.y',
        'state.car.position.z',
        # 'state.car.velocity.x',
        # 'state.car.velocity.y',
        # 'state.car.velocity.z',
        # 'state.car.rotation.pitch',
        # 'state.car.rotation.roll',
        # 'state.car.rotation.yaw',
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
        inner_1 = 16*2*2*4
        inner_2 = 32*2*2*4
        inner_3 = 16*2*2*4
        inner_4 = 4*2*2*4
        return nn.Sequential(
            nn.Linear(len(cls.input_types), inner_1),
            nn.ReLU(),

            nn.Linear(inner_1, inner_2),
            nn.Dropout(),
            nn.ReLU(),

            nn.Linear(inner_2, inner_3),
            nn.Dropout(),
            nn.ReLU(),

            nn.GRUCell(inner_3, inner_4),
            nn.Dropout(0.2),

            nn.Linear(inner_4, len(cls.output_types)),
            nn.Tanh(),
        )

    def requires(self) -> set[str]:
        return set()

