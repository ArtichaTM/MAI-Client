from typing import TYPE_CHECKING

from .bases import BaseTactic


if TYPE_CHECKING:
    from mai.ai.networks import NNModuleBase
    from mai.ai.controller import NNController


class RunModule(BaseTactic):
    __slots__ = ('_module', '_nnc', '_rewards')

    def __init__(
        self,
        module: 'NNModuleBase | str',
        rewards: dict[str, float],
        nnc: 'NNController'
    ) -> None:
        super().__init__()
        self._nnc = nnc
        self._module = nnc.get_module(module)
        self._rewards = rewards

    def prepare(self) -> None:
        self._nnc.unload_all_modules(_exceptions={self._module,})
        self._nnc.training = True
        if not self._module.enabled:
            self._nnc.module_enable(self._module)

        for reward in self._nnc.get_all_rewards():
            power = self._rewards.get(reward.name, 0.0)
            reward.power = power

    def react(self, state, context):
        controls = self._nnc.exchange(state, context)
        controls = controls.toNormalControls()
        controls = controls.toMAIControls()
        return controls
