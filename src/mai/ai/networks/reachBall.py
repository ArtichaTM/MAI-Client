from torch import nn

from .base import NNModuleBase


class NNModule(NNModuleBase):
    __slots__ = ()
    input_types = (
        'state.car.position',
        'state.car.velocity',
        'state.car.rotation',
        'state.car.angularVelocity',
        'state.ball.position',
        'state.ball.velocity',
    )
    output_types = (
        'controls.throttle',
        'controls.steer',
        'controls.pitch',
        'controls.yaw',
        'controls.roll',
        'controls.boost',
        'controls.jump',
        'controls.handbrake',
        'controls.dodgeVertical',
        'controls.dodgeHorizontal',
    )

    @classmethod
    def _create(cls):
        return nn.Sequential(
            nn.Linear(len(cls.input_types)*3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, len(cls.output_types))
        )

    def requires(self) -> set[str]:
        return {'state',}

