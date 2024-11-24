from typing import Optional
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase

from mai.capnp.data_classes import FloatControls


__all__ = ('RocketLeagueEnv',)


class RocketLeagueEnv(EnvBase):
    __slots__ = ()

    def __init__(self, td_params: TensorDictBase = None, seed = None, device = 'cpu'):
        if td_params is None:
            td_params = self.gen_params()
        self.batch_locked = False

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError()

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        raise NotImplementedError()

    @staticmethod
    def _make_spec(td_params):
        raise NotImplementedError()

    @staticmethod
    def gen_params(g=10.0, batch_size = None) -> TensorDictBase:
        raise NotImplementedError()

    def _set_seed(self, seed: Optional[int]):
        self.rng = torch.manual_seed(seed)
