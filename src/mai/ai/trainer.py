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
    _memory: ReplayMemory
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
                random_jump=not self.params.random_jump
            ))
        assert state.has_controls()

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
            assert state.has_state()
            assert state.has_reward()
            assert not state.has_controls()
            self._select_action(state)
            assert state.has_controls()
            observations = yield

            t = Transition(
                state=state,
                next_state=observations,
            )
            self._memory.add(t)

            # Marking current values as previous
            state = observations

    def epoch_end(self) -> None:
        self._loss = None

        for prev, current in pairwise(self._memory):
            assert isinstance(prev, Transition)
            assert isinstance(current, Transition)
            # print(
            #     prev.action.has_state(),
            #     prev.action.has_controls(),
            #     current.action.has_state(),
            #     current.action.has_controls(),
            #     '|',
            #     prev.state.has_state(),
            #     prev.state.has_controls(),
            #     current.state.has_state(),
            #     current.state.has_controls(),
            #     sep='\t',
            # )

        self._memory.clear()
