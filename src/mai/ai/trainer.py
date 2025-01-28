from typing import Generator, Mapping, TYPE_CHECKING
import random

import torch

from mai.capnp.data_classes import ModulesOutputMapping
from .memory import ReplayMemory, Transition
from .rewards import NNRewardBase, build_rewards

if TYPE_CHECKING:
    from .controller import ModulesController

class Trainer:
    __slots__ = (
        '_loaded', '_gen', '_mc',
        '_all_rewards', '_loss',
        '_memory', '_prev_memory',

        # Hyperparameters
        'batch_size', 'random_threshold',
        'lr', '_steps_done',

        # Training parameters
        '_optimizer',
    )
    _instance: 'Trainer | None' = None
    _mc: 'ModulesController'
    _memory: ReplayMemory
    _prev_memory: ReplayMemory | None
    _all_rewards: Mapping[str, 'NNRewardBase']
    _batch_size: int
    _optimizer: torch.optim.Optimizer

    def __init__(self, mc: 'ModulesController') -> None:
        self._mc = mc
        self._all_rewards = {k: m() for k, m in build_rewards().items()}
        self._loaded = False

    def __enter__[T: Trainer](self: T) -> T:
        type(self)._instance = self
        self.hyperparameters_init()
        self._memory = ReplayMemory()
        self._prev_memory = None
        self._optimizer = torch.optim.Adam(
            list(self._mc.enabled_parameters()),
            lr=self.lr,
            amsgrad=True
        )
        self._loss = torch.nn.MSELoss()
        self._gen = self.inference_gen()
        next(self._gen)
        self.random_threshold = 0.1
        self._loaded = True
        return self

    def __exit__(self, *args) -> None:
        self._loaded = False
        type(self)._instance = None

    def hyperparameters_init(self) -> None:
        self.batch_size = 10
        self.lr = 1e-4

    def _select_action(self, state: ModulesOutputMapping) -> ModulesOutputMapping:
        if random.random() > self.random_threshold:
            return self._mc(state).extract_controls(requires_grad=True)
        else:
            return ModulesOutputMapping.create_random_controls()

    def _optimize_model(self) -> None:
        assert self._loaded

    def inference(self, state_map: ModulesOutputMapping, reward: float) -> ModulesOutputMapping:
        assert self._loaded
        return self._gen.send((state_map, reward))

    def inference_gen(self) -> Generator[ModulesOutputMapping, tuple[ModulesOutputMapping, float], None]:
        state_map, prev_reward = yield ModulesOutputMapping.create_random_controls()

        while True:
            action = self._select_action(state_map)
            observation, reward = yield action

            # Optimize the model
            # self._optimizer.zero_grad()
            # self._loss = action.toTensor() * reward
            # print(f"Reward = {reward:1.4f}")
            # self._loss.mean().backward()
            # self._optimizer.step()
            assert len(action) == 10, len(action)

            self._memory.add(Transition(state_map, action, observation, reward))

            # Marking current values as previous
            state_map, prev_reward = observation, reward

    def epoch_end(self) -> None:
        if self._prev_memory is None:
            self._prev_memory = self._memory
            self._memory = ReplayMemory()
            return

        prev_avg = self._prev_memory.avg_reward()
        curr_avg = self._memory.avg_reward()
        # print(f'prev:{prev_avg:1.2f}', end=', ')
        # print(f'curr:{curr_avg:1.2f}', end='')

        if prev_avg > curr_avg:
            best_mem, worst_mem = self._prev_memory, self._memory
            # print(', previous better', end='')
        else:
            best_mem, worst_mem = self._memory, self._prev_memory
            # print(', new      better', end='')

        while len(best_mem) > len(worst_mem):
            best_mem.q.pop(0)
        while len(worst_mem) > len(best_mem):
            worst_mem.q.pop(0)

        assert len(best_mem) == len(worst_mem)
        assert len(best_mem.q) == len(worst_mem.q)

        x1 = [t.action_t for t in worst_mem.q]
        x2 = [t.action_t for t in best_mem.q]

        assert len(set(i.shape for i in x1)) == len(set(i.shape for i in x2))
        assert x1[0].shape == x2[0].shape

        predicted = torch.cat(x1)
        target = torch.cat(x2)

        self._optimizer.zero_grad()
        loss = self._loss(predicted, target)
        assert isinstance(loss, torch.Tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._mc.enabled_parameters(), 100)
        self._optimizer.step()
        # print(f", loss: {loss.item():.1f}")

        self._prev_memory = best_mem
        self._memory = ReplayMemory()
