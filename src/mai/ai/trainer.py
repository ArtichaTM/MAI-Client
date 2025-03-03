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
        'batch_size', 'lr',
        'gamma',

        # Training parameters
        '_optimizer', '_steps_done',
    )
    _instance: 'Trainer | None' = None
    _mc: 'ModulesController'
    _memory: ReplayMemory[ModulesOutputMapping]
    params: 'RunParameters'
    _all_rewards: Mapping[str, 'NNRewardBase']
    batch_size: int
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
        self._loaded = True
        return self

    def __exit__(self, *args) -> None:
        self._loaded = False
        type(self)._instance = None

    def hyperparameters_init(self) -> None:
        self.batch_size = 10
        self.lr = 1e-4
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
        self._loss = None

        # for mapping in self._memory:
        #     print(
        #         mapping.has_state(),
        #         mapping.has_reward(),
        #         mapping.has_controls(),
        #         mapping.reward if mapping.has_reward() else None,
        #         mapping.is_complete(),
        #         sep='\t',
        #     )

        self._memory.clear()
