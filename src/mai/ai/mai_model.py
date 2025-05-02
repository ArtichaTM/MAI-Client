from typing import Iterable, Sequence
from pathlib import Path

from torch import nn, load, save

from mai.functions import _init_weights


class MAINet(nn.Module):
    def __init__(self, modules_amount: int):
        assert isinstance(modules_amount, int)
        assert modules_amount > 1
        super().__init__()
        from mai.capnp.data_classes import STATE_KEYS
        self.fc1 = nn.Linear(len(STATE_KEYS), 256)
        self.fc2 = nn.Linear(256, modules_amount)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        return x

    @staticmethod
    def mai_name(modules: Iterable[str]) -> str:
        return f'mai_{'_'.join(i for i in modules)}'

    @classmethod
    def load[T: MAINet](
        cls: type[T],
        modules_folder: Path,
        modules: Sequence[str]
    ) -> T:
        mai_name = f'{cls.mai_name(modules)}.state_dict'
        mai_path = modules_folder / mai_name
        net = cls(len(modules))
        if mai_path.exists():
            state_dict = load(mai_path)
            net.load_state_dict(state_dict)
        else:
            _init_weights(net)
        return net
