from typing import TYPE_CHECKING

from torch import nn

from .base import NNModuleBase

if TYPE_CHECKING:
    from ..controller import NNController


class NNModule(NNModuleBase):
    __slots__ = ()
    input_types = ()
    output_types = (
        'state.car.position',
        'state.car.velocity',
        'state.car.rotation',
        'state.car.angularVelocity',
        'state.ball.position',
        'state.ball.velocity',
        'state.ball.rotation',
        'state.ball.angularVelocity',
        'state.dead',
        'state.boostAmount',
    )

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def _create(cls):
        return nn.Sequential()

    def requires(self) -> set[str]:
        return set()

    def inference(self, nnc: 'NNController'):
        s = nnc.state
        d = nnc.current_dict
        for name, value in [
('state.car.position', nnc.vector_to_tensor(s.car.position)),
('state.car.velocity', nnc.vector_to_tensor(s.car.velocity)),
('state.car.rotation', nnc.rotator_to_tensor(s.car.rotation)),
('state.car.angularVelocity', nnc.vector_to_tensor(s.car.angularVelocity)),
('state.ball.position', nnc.vector_to_tensor(s.car.position)),
('state.ball.velocity', nnc.vector_to_tensor(s.car.velocity)),
('state.ball.rotation', nnc.rotator_to_tensor(s.car.rotation)),
('state.ball.angularVelocity', nnc.vector_to_tensor(s.car.angularVelocity)),
('state.dead', nnc.vector_to_tensor(s.car.angularVelocity)),
('state.boostAmount', nnc.vector_to_tensor(s.car.angularVelocity)),
        ]:
            d[name] = value
