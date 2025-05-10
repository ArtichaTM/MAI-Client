from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Generator, Mapping, TYPE_CHECKING
)
import random
# from itertools import chain, pairwise

import torch

from mai.ai.mcs_controller import MCsController
from mai.ai.mai_model import MAINet
from mai.capnp.data_classes import (
    ModulesOutputMapping,
    # Vector,
    STATE_KEYS,
    CONTROLS_KEYS,
    RunParameters
)
from .memory import ReplayMemory, TensorReplayMemory
from .rewards import NNRewardBase, build_rewards
from .controller import ModulesController
from .utils import polyak_update

if TYPE_CHECKING:
    from mai.capnp.data_classes import RunParameters
    from mai.ai.networks.base import ModuleBase, NNModuleBase


__all__ = (
    'BaseTrainer',
    'ModuleTrainer',
    'MAITrainer',
)


class Critic(torch.nn.Module):
    def __init__(self, hidden_size: int= 32, modules_amount: int = 0):
        assert isinstance(hidden_size, int)
        assert isinstance(modules_amount, int)
        assert 0 <= modules_amount
        assert hidden_size > 8
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(
            len(STATE_KEYS)+len(CONTROLS_KEYS)+modules_amount,
            hidden_size
        )
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class ModulesGetter:
    def __init__(self) -> None:
        pass

    def __get__(
        self,
        instance: 'ModuleTrainer',
        _: type['ModuleTrainer']
    ) -> MCsController:
        mcs = []
        if instance._mc is not None:
            mcs.append(instance._mc)
        if instance._target_mc is not None:
            mcs.append(instance._target_mc)
        return MCsController(*mcs)

    def __set__(self, instance: 'ModuleTrainer', value):
        raise RuntimeError("Modules can't be set")

    def __delete__(self, instance):
        raise RuntimeError("Modules can't be deleted")


class BaseTrainer(ABC):
    _slots__ = (
        '_gen', '_loaded',
        'params'
    )
    _instance: 'BaseTrainer | None' = None

    def __init__(
        self,
        params: 'RunParameters'
    ) -> None:
        assert isinstance(params.modules, list)
        assert params.modules
        assert all((isinstance(i, str) for i in params.modules))
        super().__init__()
        self._loaded = False
        self._gen = None
        self.params = params

    def __enter__[T: BaseTrainer](self: T) -> T:
        BaseTrainer._instance = self
        if not self._loaded:
           self.prepare()
        return self

    def __exit__(self, *args) -> None:
        self._loaded = False
        BaseTrainer._instance = None

    def prepare(self) -> None:
        assert not self._loaded
        assert self._gen is None, self._gen
        self._gen = self.inference_gen()
        next(self._gen)
        self._loaded = True

    def inference(self, state_map: ModulesOutputMapping) -> None:
        assert self._loaded
        assert not state_map.has_controls()
        assert self._gen is not None
        self._gen.send(state_map)
        # assert state_map.has_controls(), state_map

    @abstractmethod
    def inference_gen(self) -> Generator[None, ModulesOutputMapping, None]:
        raise NotImplementedError()

    @abstractmethod
    def epoch_end(self) -> None:
        pass

class ModuleTrainer(BaseTrainer):
    __slots__ = (
        '_loaded', '_gen',
        '_all_rewards', '_loss',
        '_memory',

        # Hyperparameters
        'batch_size',
        'gamma',
        'tau',

        # Variables
        '_epoch_num',

        # Model fields
        '_mc', '_target_mc',
        '_mc_optimizer',
        'mc_lr',

        # Critic fields
        '_critic', '_target_critic',
        '_critic_optimizer',
        'critic_lr',
    )
    _memory: ReplayMemory[ModulesOutputMapping]
    _all_rewards: Mapping[str, 'NNRewardBase']
    batch_size: int
    modules = ModulesGetter()

    _critic: Critic
    _target_critic: Critic
    _critic_optimizer: torch.optim.Optimizer

    _mc: ModulesController
    _target_mc: ModulesController
    _mc_optimizer: torch.optim.Optimizer

    def __init__(
        self,
        params: 'RunParameters'
    ) -> None:
        super().__init__(params)
        self._all_rewards = {k: m() for k, m in build_rewards().items()}

    def __exit__(self, *args) -> None:
        super().__exit__(*args)
        self._target_mc.save()

    def prepare(self) -> None:
        super().prepare()
        self.batch_size = 10
        self.mc_lr = 1e-4
        self.critic_lr = 1e-4
        self.gamma = 0.97
        self.tau = 0.001

        self._memory = ReplayMemory()
        self._loss = torch.nn.MSELoss()
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
        def assign_power(m: 'ModuleBase'):
            m.power = 1.
        for module in self.params.modules:
            self.modules.enable(module)
            self.modules.module_apply(module, assign_power)

        self._mc_optimizer = torch.optim.Adam(
            list(self.modules._mc.enabled_parameters()),
            lr=self.mc_lr,
            amsgrad=True
        )

        critic_hidden_size = 256
        self._critic = Critic(critic_hidden_size)
        self._target_critic = Critic(critic_hidden_size)
        self._critic_optimizer = torch.optim.Adam(
            self._critic.parameters(recurse=True),
            lr=self.critic_lr,
            amsgrad=True
        )

        self._loaded = True

    #
    # Training
    #

    def _select_action(self, state: ModulesOutputMapping) -> None:
        assert self._loaded
        assert isinstance(state, ModulesOutputMapping)
        assert state.has_state()
        assert not state.has_controls()
        if random.random() > self.params.random_threshold:
            self._target_mc(state)
        else:
            state.update(ModulesOutputMapping.create_random_controls(
                random_jump=self.params.random_jump
            ))
        assert state.has_any_controls()

    def inference_gen(self) -> Generator[None, ModulesOutputMapping, None]:
        state = yield
        assert self._loaded

        print('Starting training')
        print('Epoch | Reward sum | Reward avg | Critic loss | MC loss')

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

        # batch = self._memory.sample(len(self._memory))
        batch = self._memory
        batch.rewards_differentiate()
        batch.rewards_affect_previous(percent=0.3, start_multiplier=0.3)
        batch.rewards_normalize()
        states, actions, next_states, rewards = batch.to_canonical()
        rewards = rewards.unsqueeze(1)

        # import numpy as np
        # with open('temp.csv', mode='w', encoding='utf-8') as f:
        #     keys = (*STATE_KEYS, *CONTROLS_KEYS)
        #     f.write(','.join(keys))
        #     f.write(',reward')
        #     for m in batch:
        #         f.write('\n')
        #         for key in keys:
        #             v = m._avg_from_dict(key, requires_grad=False)
        #             assert v is not None
        #             f.write(str(float(v)))
        #             f.write(',')
        #         f.write(str(m.reward))

        # Update critic network
        all_actions_m: list[ModulesOutputMapping] = self._target_mc(batch)
        next_actions = torch.stack([m.extract_controls().toTensor() for m in all_actions_m[1:]])
        assert states .shape == next_states .shape
        assert actions.shape == next_actions.shape
        assert states.shape[0] == actions.shape[0]
        assert next_states.shape[0] == actions.shape[0]
        assert states.shape[1] == len(STATE_KEYS)
        assert actions.shape[1] == len(CONTROLS_KEYS)
        target_q_values: torch.Tensor = self._target_critic(next_states, next_actions)
        real_q_values: torch.Tensor = self._critic(states, actions)
        assert target_q_values.shape == real_q_values.shape, f"{target_q_values.shape} {real_q_values.shape}"
        assert rewards.shape == target_q_values.shape, f"{rewards.shape} {target_q_values.shape}"
        expected_q_values: torch.Tensor =  + (self.gamma * target_q_values)

        assert real_q_values.shape == expected_q_values.shape
        critic_loss = self._loss(real_q_values, expected_q_values.detach())
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        # Update MC
        mc_loss = -self._critic(
            states,
            torch.stack([m.extract_controls().toTensor() for m in all_actions_m[:-1]])
        ).mean()
        self._mc_optimizer.zero_grad()
        mc_loss.backward()
        self._mc_optimizer.step()

        # Update target networks
        for t_p, p in zip(self._target_mc.enabled_parameters(), self._mc.enabled_parameters()):
            t_p.data.copy_(self.tau * p.data + (1.0 - self.tau) *t_p.data)
        for t_p, p in zip(self._target_critic.parameters(), self._critic.parameters()):
            t_p.data.copy_(self.tau * p.data + (1.0 - self.tau) *t_p.data)

        reward_sum = sum((i.reward for i in self._memory))
        print(
            f"{self._epoch_num: >5}"[:5],
            f"{reward_sum}"[:9],
            f"{reward_sum/len(self._memory)}"[:9],
            f"{critic_loss}"[:11],
            f"{mc_loss}"[:7],
            sep=' | '
        )
        self._memory.clear()

    def close(self) -> None:
        pass


class MAITrainer(BaseTrainer):
    __slots__ = (
        '_memory', '_mc', '_mse_loss',
        '_mai_net', '_epoch_num',

        'mai_lr', 'critic_lr',
        'tau', 'gamma',
    )

    def __init__(
        self,
        params: RunParameters
    ) -> None:
        super().__init__(params)

    def prepare(self) -> None:
        assert self._gen is None
        super().prepare()
        assert self._gen is not None
        self.critic_lr = 1e-4
        self.mai_lr = 1e-4
        # Rate at which critic weights transferred
        # to target critic
        self.tau = 0.001
        # Constant reward decrease
        self.gamma = 0.97
        critic_hidden_size = 256
        models_folder = Path()

        self._mse_loss = torch.nn.MSELoss()
        self._memory = TensorReplayMemory()

        self._mc = ModulesController(
            models_folder=models_folder
        )
        self._mc.training = False
        self._mc.unload_all_modules(save=False)
        for module_name in self.params.modules:
            module = self._mc.get_module(module_name)
            if not module.loaded:
                module.load()
            module.training = False
            module.power = 1.

        self._mai_net = MAINet.load(
            models_folder,
            self.params.modules
        )
        self._mai_optimizer = torch.optim.Adam(
            self._mai_net.parameters(recurse=True),
            lr=self.mai_lr,
            amsgrad=True
        )

        critic_args = (
            critic_hidden_size,
            len(self.params.modules)
        )
        self._critic = Critic(*critic_args)
        self._target_critic = Critic(*critic_args)
        self._critic_optimizer = torch.optim.Adam(
            self._critic.parameters(recurse=True),
            lr=self.critic_lr,
            amsgrad=True
        )

        self._loaded = True

    def inference_gen(self) -> Generator[None, ModulesOutputMapping, None]:
        state = yield
        assert self._loaded
        assert self._gen
        assert hasattr(self, '_mai_net')

        print('Epoch | Reward sum | Reward avg | Critic loss | MAI loss')
        for module_name in self.params.modules:
            self._mc.module_load(module_name)
            self._mc.module_enable(module_name)
            self._mc.module_power(module_name, 1.)

        while True:
            assert state.has_all_state(), state.keys()
            assert not state.has_any_controls(), state.keys()
            powers: torch.Tensor = self._mai_net(
                state
                    .extract_state(requires_grad=False)
                    .toTensor(requires_grad=True)
                )
            powers.clip(0., 1.)
            state.powers = powers
            assert powers.shape == (len(self.params.modules),)
            self._memory.add_v(
                state=state,
                powers=powers
            )
            assert all(0 <= i.item() <= 1 for i in powers), powers
            for module_name, power in zip(self.params.modules, powers):
                power = power.item()
                self._mc.module_power(
                    module_name,
                    power
                )

            assert not state.has_any_controls()
            self._mc(state)
            # from pprint import pp
            # pp(state)

            observations = yield
            state = observations

    def epoch_end(self) -> None:
        """Using TD3 from stable-baselines3
        https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/td3/td3.py
        """
        assert self._loaded
        self._epoch_num += 1

        # batch = self._memory.sample(len(self._memory))
        batch = self._memory
        batch.rewards_differentiate()
        batch.rewards_affect_previous(percent=0.3)
        batch.rewards_normalize()
        observations, actions, next_observations, rewards, powerss = batch.to_canonical()
        rewards = rewards.unsqueeze(1)
        actor_losses, critic_losses = [], []

        with torch.no_grad():
            # Select action according to policy
            next_actions = (self._mai_net(next_observations)).clamp(-1, 1)

            # Compute the next Q-values: min over all critics targets
            next_q_values = torch.cat(self._target_critic(next_observations, next_actions), dim=1)
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            target_q_values = rewards * self.gamma * next_q_values

            # # Update critic network
            # next_actions = actions[1:]
            # assert states     .shape    == next_states .shape
            # assert actions    .shape    == next_actions.shape
            # assert states     .shape[0] == actions.shape[0]
            # assert next_states.shape[0] == actions.shape[0]
            # assert states     .shape[1] == len(STATE_KEYS)
            # assert actions    .shape[1] == len(CONTROLS_KEYS)
            # target_q_values: torch.Tensor = self._target_critic(next_states, next_actions, powerss)
            # # TODO: Real_q_values are not real (not from game)
            # real_q_values: torch.Tensor = self._critic(states, actions, powerss)
            # assert target_q_values.shape == real_q_values.shape, f"{target_q_values.shape} {real_q_values.shape}"
            # assert rewards.shape == target_q_values.shape, f"{rewards.shape} {target_q_values.shape}"
            # expected_q_values: torch.Tensor =+ (self.gamma * target_q_values)

        # Get current Q-values estimates for each critic network
        current_q_values = self._critic(observations, actions)

        # Get current Q-values estimates for each critic network
        current_q_values = self._critic(observations, actions)

        # Compute critic loss
        critic_loss = sum(self._mse_loss(current_q, target_q_values) for current_q in current_q_values)
        assert isinstance(critic_loss, torch.Tensor)
        critic_losses.append(critic_loss.item())

        # Optimize the critics
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        critic_q1_forward = self._critic.q1_forward
        assert isinstance(critic_q1_forward, torch.Module)
        assert not isinstance(critic_q1_forward, torch.Tensor)
        mai_loss = -critic_q1_forward(observations, self._mai_net(observations)).mean()
        actor_losses.append(mai_loss.item())

        # Optimize the actor
        self._mai_optimizer.zero_grad()
        mai_loss.backward()
        self._mai_optimizer.step()

        polyak_update(self._critic.parameters(), self._target_critic.parameters(), self.tau)
        polyak_update(self._mai_net.parameters(), self._mai_net.parameters(), self.tau)
        # Copy running stats, see GH issue #996
        # polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
        # polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        print(
            f"{self._epoch_num: >5}"[:5],
            f"{self._memory.sum_reward()}"[:9],
            f"{self._memory.avg_reward()}"[:9],
            f"{critic_loss}"[:11],
            f"{mai_loss}"[:7],
            sep=' | '
        )
        self._memory.clear()
