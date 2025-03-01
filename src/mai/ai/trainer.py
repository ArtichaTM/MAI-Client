from typing import Generator, Mapping, TYPE_CHECKING
import random
from itertools import pairwise

import torch

from mai.capnp.data_classes import ModulesOutputMapping
from .memory import ReplayMemory, Transition
from .rewards import NNRewardBase, build_rewards

if TYPE_CHECKING:
    from .controller import ModulesController
    from mai.capnp.data_classes import RunParameters

class Trainer:
    __slots__ = (
        '_loaded', '_gen', '_mc',
        '_all_rewards', '_loss',
        '_memory',
        'params',

        # Hyperparameters
        'batch_size', 'random_threshold',
        'lr', '_steps_done',

        # Training parameters
        '_optimizer',
    )
    _instance: 'Trainer | None' = None
    _mc: 'ModulesController'
    _memory: ReplayMemory
    params: 'RunParameters'
    _all_rewards: Mapping[str, 'NNRewardBase']
    _batch_size: int
    _optimizer: torch.optim.Optimizer

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
        assert isinstance(state, ModulesOutputMapping)
        if random.random() > self.random_threshold:
            return self._mc(state).extract_controls(requires_grad=True)
        else:
            return ModulesOutputMapping.create_random_controls(
                random_jump=not self.params.random_jump
            )

    def _optimize_model(self) -> None:
        assert self._loaded

    def inference(self, state_map: ModulesOutputMapping, reward: float) -> ModulesOutputMapping:
        assert self._loaded
        return self._gen.send((state_map, reward))

    def inference_gen(self) -> Generator[ModulesOutputMapping, tuple[ModulesOutputMapping, float], None]:
        state_map, prev_reward = yield ModulesOutputMapping.create_random_controls(
            random_jump=not self.params.random_jump
        )

        while True:
            action = self._select_action(state_map)
            observation, reward = yield action

            assert len(action) == 10, len(action)

            self._memory.add(Transition(state_map, action, observation, reward))

            # Marking current values as previous
            state_map, prev_reward = observation, reward

    def epoch_end(self) -> None:
        self._loss = None

        for prev, current in pairwise(self._memory):
            pass
