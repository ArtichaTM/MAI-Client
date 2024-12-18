from math import exp
import random
from typing import Generator, Mapping, TYPE_CHECKING

import torch

from mai.capnp.data_classes import ModulesOutputMapping
from .memory import ReplayMemory, Transition
from .rewards import NNRewardBase, build_rewards

if TYPE_CHECKING:
    from .controller import ModulesController

class Critic(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(state_size + action_size, 400)
        self.fc2 = torch.nn.Linear(400, 300)
        self.fc3 = torch.nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], 1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Trainer:
    __slots__ = (
        '_loaded', '_gen', '_mc',
        '_all_rewards', '_steps_done',

        # Hyperparameters
        'batch_size', 'gamma',
        'eps_start', 'eps_end', 'eps_decay',
        'tau', 'lr',

        # Training parameters
        '_optimizer',
    )
    _mc: 'ModulesController'
    _critic: Critic
    _all_rewards: Mapping[str, 'NNRewardBase']
    _batch_size: int
    _optimizer: torch.optim.Optimizer

    def __init__(self, mc: 'ModulesController') -> None:
        self._mc = mc
        self._all_rewards = {k: m() for k, m in build_rewards().items()}
        self._loaded = False

    def __enter__[T: Trainer](self: T) -> T:
        self.hyperparameters_init()
        self._loaded = True
        self._steps_done: int = 0
        self._optimizer = torch.optim.Adam(
            list(self._mc.enabled_parameters()),
            lr=self.lr,
            amsgrad=True
        )
        self._gen = self.inference_gen()
        next(self._gen)
        return self

    def __exit__(self, *args) -> None:
        self._loaded = False

    def hyperparameters_init(self) -> None:
        self.batch_size = 10
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.tau = 0.005
        self.lr = 1e-4

    def _select_action(self, state: ModulesOutputMapping) -> ModulesOutputMapping:
        eps_threshold = (
            self.eps_end + (self.eps_start - self.eps_end) *
            exp(-1. * self._steps_done / self.eps_decay)
        )
        self._steps_done += 1
        if random.random() > eps_threshold:
            return self._mc(state)
        else:
            return ModulesOutputMapping.create_random_controls()

    def _optimize_model(self) -> None:
        assert self._loaded

    def inference(self, state_map: ModulesOutputMapping, reward: float) -> ModulesOutputMapping:
        assert self._loaded
        return self._gen.send((state_map, reward))

    def reward_calculator(self) -> Generator[float, tuple[float, float], None]:
        difference = 0
        prev, current = yield difference
        while True:
            difference = current - prev
            prev, current = yield difference

    def inference_gen(self) -> Generator[ModulesOutputMapping, tuple[ModulesOutputMapping, float], None]:
        rewarder = self.reward_calculator()
        next(rewarder)

        state_map, prev_reward = yield ModulesOutputMapping.create_random_controls()
        while True:
            action = self._select_action(state_map)
            observation, reward = yield action

            # Optimize the model
            self._optimizer.zero_grad()
            loss = action.toTensor() * rewarder.send((prev_reward, reward))
            loss.mean().backward()
            # torch.nn.utils.clip_grad_value_(self._mc.enabled_parameters(), 100)
            self._optimizer.step()

            # Marking current values as previous
            state_map, prev_reward = observation, reward


    def epoch_end(self) -> None:
        pass
