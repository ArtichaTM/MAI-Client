from pathlib import Path
from typing import (
    Generator, Mapping, Callable,
    TYPE_CHECKING
)
import random
from itertools import pairwise

import torch

from mai.capnp.data_classes import (
    ModulesOutputMapping,
    STATE_KEYS,
    CONTROLS_KEYS
)
from .memory import ReplayMemory
from .rewards import NNRewardBase, build_rewards
from .controller import ModulesController

if TYPE_CHECKING:
    from mai.capnp.data_classes import RunParameters
    from mai.ai.networks.base import NNModuleBase


__all__ = (
    'Trainer',
)


class Critic(torch.nn.Module):
    def __init__(self, hidden_size: int= 32):
        assert isinstance(hidden_size, int)
        assert hidden_size > 8
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(len(STATE_KEYS)+len(CONTROLS_KEYS), hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class MCsController:
    __slots__ = ('_mc', '_target_mc')

    def __init__(
        self,
        mc: ModulesController,
        target_mc: ModulesController | None = None
    ) -> None:
        assert isinstance(mc, ModulesController)
        self._mc = mc
        self._target_mc = target_mc
        assert target_mc is None or self.__compare_assertions()

    def __iter__(self) -> Generator[ModulesController, None, None]:
        yield self._mc
        if self._target_mc:
            yield self._target_mc

    def __compare_assertions(self) -> bool:
        assert isinstance(self._target_mc, ModulesController)
        assert self._mc.training == self._target_mc.training
        for m_mc, m_t_mc in zip(
            self._mc.get_all_modules(), self._target_mc.get_all_modules()
        ):
            assert m_mc.enabled == m_t_mc.enabled, f"{m_mc}!={m_t_mc}"
            assert m_mc.loaded == m_t_mc.loaded, f"{m_mc}!={m_t_mc}"
            assert m_mc.file_name == m_t_mc.file_name, f"{m_mc}!={m_t_mc}"
            assert m_mc.name == m_t_mc.name, f"{m_mc}!={m_t_mc}"
            assert m_mc.power == m_t_mc.power, f"{m_mc}!={m_t_mc}"
            if m_mc.loaded:
                assert m_mc._model is not None
                assert m_t_mc._model is not None
                # Checking if parameters pointing to the same memory region
                for p1, p2 in zip(
                    m_mc._model.parameters(),
                    m_t_mc._model.parameters()
                ):
                    assert p1.data_ptr() != p2.data_ptr(), f"{p1}!={p2}"
        return True

    def _get_module(self, name: str) -> Generator['NNModuleBase', None, None]:
        yield self._mc.get_module(name)
        if self._target_mc is not None:
            yield self._target_mc.get_module(name)

    def _iter_modules(self) -> Generator[
        tuple['NNModuleBase', 'NNModuleBase | None'],
        None, None
    ]:
        if self._target_mc is None:
            for m in self._mc.get_all_modules():
                yield m, None
            return
        for m, target_m in zip(
            self._mc.get_all_modules(),
            self._target_mc.get_all_modules()
        ):
            yield m, target_m

    def module_apply(
        self,
        name: str,
        func: Callable[['NNModuleBase'], None]
    ):
        for module in self._get_module(name):
            func(module)

    @property
    def training(self) -> bool:
        return self._mc.training

    @training.setter
    def training(self, value: bool) -> None:
        assert isinstance(value, bool)
        self._mc.training = value
        if self._target_mc is not None:
            self._target_mc.training = value

    def enable(self, name: str) -> None:
        assert isinstance(name, str)
        for mc in self:
            mc.module_enable(name)

    def disable(self, name: str) -> None:
        assert isinstance(name, str)
        for mc in self:
            mc.module_disable(name)

    def load(self, name: str) -> None:
        assert isinstance(name, str)
        for mc in self:
            mc.module_load(name)

    def unload(self, name: str) -> None:
        assert isinstance(name, str)
        for mc in self:
            mc.module_unload(name)

    def unload_all(self, save: bool = True) -> None:
        assert isinstance(save, bool)
        for mc in self:
            mc.unload_all_modules(save=save)

    def loaded(self) -> Generator[
        tuple['NNModuleBase', 'NNModuleBase | None'],
        None, None
    ]:
        for module, module_t in self._iter_modules():
            if module.loaded:
                yield module, module_t

    def enabled(self) -> Generator[
        tuple['NNModuleBase', 'NNModuleBase | None'],
        None, None
    ]:
        for module, module_t in self._iter_modules():
            if module.enabled:
                yield module, module_t

class ModulesGetter:
    def __init__(self) -> None:
        pass

    def __get__(
        self,
        instance: 'Trainer',
        _: type['Trainer']
    ) -> MCsController:
        mcs = []
        if instance._mc is not None:
            mcs.append(instance._mc)
        if instance._target_mc is not None:
            mcs.append(instance._target_mc)
        return MCsController(*mcs)

    def __set__(self, instance: 'Trainer', value):
        raise RuntimeError("Modules can't be set")

    def __delete__(self, instance):
        raise RuntimeError("Modules can't be deleted")


class Trainer:
    __slots__ = (
        '_loaded', '_gen',
        '_all_rewards', '_loss',
        '_memory',
        'params',

        # Hyperparameters
        'batch_size',
        'lr',
        'critic_lr',
        'gamma',

        # Variables
        '_epoch_num',

        # Model fields
        '_mc', '_target_mc',
        '_mc_optimizer',

        # Critic fields
        '_critic', '_target_critic',
        '_critic_optimizer',
    )
    _instance: 'Trainer | None' = None
    _memory: ReplayMemory[ModulesOutputMapping]
    params: 'RunParameters'
    _all_rewards: Mapping[str, 'NNRewardBase']
    batch_size: int
    modules = ModulesGetter()

    _critic: Critic
    _target_critic: Critic
    _critic_optimizer: torch.optim.Optimizer

    _mc: ModulesController
    _target_mc: ModulesController | None
    _mc_optimizer: torch.optim.Optimizer

    def __init__(
        self,
        params: 'RunParameters'
    ) -> None:
        assert isinstance(params.modules, list)
        assert params.modules
        assert all((isinstance(i, str) for i in params.modules))
        self._all_rewards = {k: m() for k, m in build_rewards().items()}
        self._loaded = False
        self.params = params

    def __enter__[T: Trainer](self: T) -> T:
        type(self)._instance = self
        self.hyperparameters_init()
        self._memory = ReplayMemory()
        self._loss = torch.nn.MSELoss()
        self._gen = self.inference_gen()
        self._epoch_num = 0
        next(self._gen)

        self._target_mc = ModulesController(
            models_folder=Path()
        )
        self._mc = self._target_mc.copy(
            copy_device=True,
            copy_models_folder=False,
        )
        self._mc.models_folder = None

        self.modules.training = True
        def assign_power(m: 'NNModuleBase'):
            m.power = 1.
        for module in self.params.modules:
            self.modules.enable(module)
            self.modules.module_apply(module, assign_power)

        self._mc_optimizer = torch.optim.Adam(
            list(self.modules._mc.enabled_parameters()),
            lr=self.lr,
            amsgrad=True
        )

        critic_hidden_size = 32
        self._critic = Critic(critic_hidden_size)
        self._target_critic = Critic(critic_hidden_size)
        self._critic_optimizer = torch.optim.Adam(
            self._critic.parameters(recurse=True),
            lr=self.critic_lr,
            amsgrad=True
        )

        self._loaded = True
        return self

    def __exit__(self, *args) -> None:
        self._loaded = False
        type(self)._instance = None

    def hyperparameters_init(self) -> None:
        self.batch_size = 10
        self.lr = 1e-4
        self.critic_lr = 1e-4
        self.gamma = 0.1

    #
    # Training
    #

    def _select_action(self, state: ModulesOutputMapping) -> None:
        assert self._loaded
        assert isinstance(state, ModulesOutputMapping)
        assert state.has_state()
        assert not state.has_controls()
        if random.random() > self.params.random_threshold:
            self._mc(state)
        else:
            state.update(ModulesOutputMapping.create_random_controls(
                random_jump=self.params.random_jump
            ))
        assert state.has_any_controls()

    def inference(self, state_map: ModulesOutputMapping) -> None:
        assert self._loaded
        assert not state_map.has_controls()
        self._gen.send(state_map)
        assert state_map.has_controls()

    def inference_gen(self) -> Generator[None, ModulesOutputMapping, None]:
        state = yield
        assert self._loaded

        while True:
            assert state.has_all_state(), state.keys()
            assert not state.has_any_controls(), state.keys()
            self._select_action(state)
            self._memory.add(state)

            observations = yield
            state = observations

    def epoch_end(self) -> None:
        assert self._loaded
        self._epoch_num += 1

        # reward_sum = sum((i.reward for i in self._memory))
        # print(
        #     f'> Epoch {self._epoch_num} info: ',
        #     f'Reward summary: {reward_sum:.2f}',
        #     f'Reward average: {reward_sum/len(self._memory):.2f}',
        #     sep='\n'
        # )

        # batch = self._memory.sample(self.batch_size)
        # states, actions, next_states, rewards = batch.to_canonical()

        self._memory.clear()
