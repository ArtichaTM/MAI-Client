from typing import Generator, Mapping, TYPE_CHECKING
import random
from itertools import pairwise

import torch

from mai.capnp.data_classes import (
    ModulesOutputMapping,
    STATE_KEYS,
    CONTROLS_KEYS
)
from .memory import ReplayMemory, Transition
from .rewards import NNRewardBase, build_rewards

if TYPE_CHECKING:
    from .controller import ModulesController
    from mai.capnp.data_classes import RunParameters


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


class Trainer:
    __slots__ = (
        '_loaded', '_gen', '_mc',
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
        '_model_optimizer',

        # Critic fields
        '_critic',
        '_critic_optimizer',
    )
    _instance: 'Trainer | None' = None
    _mc: 'ModulesController'
    _memory: ReplayMemory[ModulesOutputMapping]
    params: 'RunParameters'
    _all_rewards: Mapping[str, 'NNRewardBase']
    batch_size: int
    _critic: Critic
    _optimizer: torch.optim.Optimizer
    _critic_optimizer: torch.optim.Optimizer

    def __init__(
        self,
        mc: 'ModulesController',
        params: 'RunParameters'
    ) -> None:
        self._mc = mc
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

        self._model_optimizer = torch.optim.Adam(
            list(self._mc.enabled_parameters()),
            lr=self.lr,
            amsgrad=True
        )
        self._critic = Critic()
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

    def _select_action(self, state: ModulesOutputMapping) -> None:
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

    def _optimize_model(self) -> None:
        assert self._loaded

    def inference(self, state_map: ModulesOutputMapping) -> None:
        assert self._loaded
        assert not state_map.has_controls()
        self._gen.send(state_map)
        assert state_map.has_controls()

    def inference_gen(self) -> Generator[None, ModulesOutputMapping, None]:
        state = yield

        while True:
            assert state.has_all_state(), state.keys()
            assert not state.has_any_controls(), state.keys()
            self._select_action(state)
            self._memory.add(state)

            observations = yield
            state = observations

    def epoch_end(self) -> None:
        self._epoch_num += 1

        reward_sum = sum((i.reward for i in self._memory))
        print(
            f'> Epoch {self._epoch_num} info: ',
            f'Reward summary: {reward_sum:.2f}',
            f'Reward average: {reward_sum/len(self._memory):.2f}',
            sep='\n'
        )

        batch = self._memory.sample(self.batch_size)
        states, actions, next_states, rewards = batch.to_canonical()

        self._memory.clear()
