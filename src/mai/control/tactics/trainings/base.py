from typing import TYPE_CHECKING, Callable, Mapping
from time import perf_counter

from mai.ai.controller import NNController
from mai.capnp.data_classes import RunParameters

from ..bases import BaseTactic
from mai.functions import popup
from mai.capnp.data_classes import (
    RunParameters,
    MAIGameState,
    AdditionalContext,
    RestartReason
)

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
    __slots__ = ('reasons_to_func_map',)

    def prepare(self) -> None:
        super().prepare()
        modules = self._run_parameters.modules
        if len(modules) == 0:
            popup("Error during training startup", (
                "No modules selected for training"
            ))
            self.finished = True
            return
        elif len(modules) == 1:
            pass
        else:
            popup(
                "Error during training startup",
                "This training allows only 1 module to train"
            )
            self.finished = True
            return

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

        self.reasons_to_func_map: Mapping[RestartReason, Callable[[
            MAIGameState, AdditionalContext
        ], bool]] = {
            RestartReason.BALL_TOUCH: self._check_restart_hit,
            RestartReason.SCORE: self._check_restart_score,
            RestartReason.TIMEOUT: self._check_restart_timeout,
        }

    def _check_restart_hit(self, state, context: 'AdditionalContext', **kwargs) -> bool:
        return context.latest_message == 'ballTouched'

    def _check_restart_score(self, state, context: 'AdditionalContext', **kwargs) -> bool:
        return context.latest_message == 'ballExplode'

    def _check_restart_timeout(
        self,
        state,
        context: 'AdditionalContext',
        time_since_start: float | None = None
    ) -> bool:
        assert isinstance(time_since_start, float), "Pass time_since_start to kwargs"
        return (perf_counter() - time_since_start) > self._run_parameters.restart_timeout

    def is_restarting(self, state: 'MAIGameState', context: 'AdditionalContext', **kwargs) -> bool:
        for reason in self._run_parameters.restart_reasons:
            result = self.reasons_to_func_map[reason](state, context, **kwargs)
            if result:
                return True
        return False
