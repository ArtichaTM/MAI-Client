from typing import TYPE_CHECKING

from ..bases import BaseTactic
from mai.capnp.data_classes import RunParameters

if TYPE_CHECKING:
    from mai.ai.controller import NNController


__all__ = ('BaseTrainingTactic',)


class BaseTrainingTactic(BaseTactic):
    __slots__ = ('_nnc', '_run_parameters')

    def __init__(
        self,
        nnc: 'NNController',
        run_parameters: RunParameters
    ) -> None:
        super().__init__()
        self._nnc = nnc
        self._run_parameters = run_parameters


class ModuleTrainingTactic(BaseTrainingTactic):
    __slots__ = ()

    def prepare(self) -> None:
        super().prepare()
        modules = self._run_parameters.modules
        if len(modules) == 0:
            raise ValueError("Select modules to train")
        elif len(modules) == 1:
            pass
        else:
            raise ValueError("Only one module training allowed in ModuleTraining")

        _module = modules[0]
        module = self._nnc.get_module(_module)
        rewards = self._run_parameters.rewards

        self._nnc.unload_all_modules(_exceptions={module,})
        self._nnc.training = True
        if not module.enabled:
            self._nnc.module_enable(module)
            module.power = 1

        for reward in self._nnc.get_all_rewards():
            power = rewards.get(reward.name, 0.0)
            reward.power = power
