from typing import TYPE_CHECKING, Callable, Mapping
from time import perf_counter

from mai.ai.controller import ModulesController
from mai.capnp.data_classes import RunParameters

from ..bases import BaseTactic
from mai.functions import popup
from mai.capnp.data_classes import (
    RunParameters,
    MAIGameState,
    AdditionalContext,
    RestartReason
)
from mai.ai.rewards import build_rewards, NNRewardBase

if TYPE_CHECKING:
    from mai.ai.controller import ModulesController


__all__ = ('BaseTrainingTactic',)


class BaseTrainingTactic(BaseTactic):
    __slots__ = (
        '_run_parameters', '_rewards',
        '_prev_reward', '_trainer'
    )

    def __init__(
        self,
        run_parameters: RunParameters
    ) -> None:
        super().__init__()
        self._run_parameters = run_parameters
        self._prev_reward: float = 0.0
        self._rewards: list[NNRewardBase] = []

    def prepare(self) -> None:
        super().prepare()
        rewards = build_rewards()
        for reward_str, power in self._run_parameters.rewards.items():
            reward = rewards[reward_str]()
            reward.power = power
            self._rewards.append(reward)

    def calculate_reward(self, state: 'MAIGameState', context: 'AdditionalContext') -> float:
        reward: float = 0
        for reward_cl in self._rewards:
            r = reward_cl(state, context)
            reward += r
        difference = reward - self._prev_reward
        if abs(difference) > 0.2:
            reward = self._prev_reward + min(0.2, max(difference, -0.2))
        self._prev_reward = reward
        return reward

    def env_reset(self) -> None:
        self._prev_reward = 0.0
        for reward in self._rewards:
            reward.reset()


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

        self.reasons_to_func_map: Mapping[RestartReason, Callable[[
            MAIGameState, AdditionalContext
        ], bool]] = {
            RestartReason.BALL_TOUCH: self._check_restart_hit,
            RestartReason.BALL_EXPLODE: self._check_restart_explode,
            RestartReason.TIMEOUT: self._check_restart_timeout,
        }

    def _check_restart_hit(self, state, context: 'AdditionalContext', **kwargs) -> bool:
        return context.latest_message == 'ballTouched'

    def _check_restart_explode(self, state, context: 'AdditionalContext', **kwargs) -> bool:
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
